#!/usr/bin/env python3
"""Offline open-loop evaluation of DM0-base on RoboChallenge turn_on_faucet.

Comprehensive logging of ALL inputs and outputs:
  - Inputs : prompt string, tokenized input_ids (decoded), image file paths,
             state vector (sampled), image_masks
  - Outputs: generated token string (with CoT/bbox/action/trajectory tags preserved),
             generated token ids (sampled), action prediction (sampled),
             action MSE, timing metrics
"""

import argparse
import json
import os
import sys
import time
from glob import glob
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
from dexbotic.tokenization.process import DM0Tokenization
from dexbotic.data.dataset.transform.common import ToNumpy


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_norm_stats(ckpt_path):
    norm_file = os.path.join(ckpt_path, "norm_stats.json")
    if os.path.exists(norm_file):
        with open(norm_file, "r") as f:
            data = json.load(f)
        if "norm_stats" in data:
            data = data["norm_stats"]
        return ToNumpy()(data)
    return None


def build_aloha_state(left_state, right_state):
    left_qpos = np.array(left_state["qpos"], dtype=np.float32)[:6]
    left_grip = np.float32(left_state["gripper"])
    right_qpos = np.array(right_state["qpos"], dtype=np.float32)[:6]
    right_grip = np.float32(right_state["gripper"])
    return np.concatenate([left_qpos, [left_grip], right_qpos, [right_grip]])


def read_video_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def sample_array(arr, max_items=10):
    """Return a JSON-safe sampled representation of a numpy array."""
    if arr is None:
        return None
    flat = arr.flatten()
    if len(flat) <= max_items:
        return arr.tolist()
    step = max(1, len(flat) // max_items)
    return {"shape": list(arr.shape), "sampled_values": flat[::step].tolist()[:max_items]}


def parse_generated_tags(text):
    """Extract tagged segments from generated text: <subtask>, <bbox>, <traj>, <act>, <action>N</action>."""
    import re
    tags = {}
    for tag in ["subtask", "bbox", "traj", "act"]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            tags[tag] = matches
    action_tokens = re.findall(r"<action>(\d+)</action>", text)
    if action_tokens:
        tags["discrete_action_tokens"] = action_tokens
    return tags if tags else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--frame_interval", type=int, default=30)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = os.path.basename(args.checkpoint.rstrip("/"))
    log_file = os.path.join(args.output_dir, f"eval_{ckpt_name}_{run_id}.jsonl")
    summary_file = os.path.join(args.output_dir, f"summary_{ckpt_name}_{run_id}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Load model ---
    logger.info(f"Loading model from {args.checkpoint}")
    model = DM0ForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": device},
    )
    model.tie_lm_head()
    model.model.action_out_proj = model.model.action_out_proj.float()
    model.model.action_in_proj = model.model.action_in_proj.float()
    model.model.action_time_mlp_in = model.model.action_time_mlp_in.float()
    model.model.action_time_mlp_out = model.model.action_time_mlp_out.float()
    model.eval()
    logger.info(f"lm_head tied: {model.lm_head.weight.data_ptr() == model.model.llm.embed_tokens.weight.data_ptr()}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False, trust_remote_code=True)
    orig_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = max(2048, orig_max_len)
    logger.info(f"tokenizer.model_max_length: {orig_max_len} -> {tokenizer.model_max_length}")

    tokenization_func = DM0Tokenization(tokenizer)
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    logger.info(f"eos_token_id: {eos_token_id}")

    # --- Norm stats (DM0-base may not have them) ---
    norm_stats = load_norm_stats(args.checkpoint)
    has_norm = norm_stats is not None
    logger.info(f"norm_stats available: {has_norm}")

    prompt = "grasp the faucet switch and turn it on"
    non_delta_mask = [6, 13]

    # --- Find episodes ---
    data_root = args.data_dir
    episode_dirs = sorted(glob(os.path.join(data_root, "data", "episode_*")))
    if not episode_dirs:
        episode_dirs = sorted(glob(os.path.join(data_root, "episode_*")))
    if args.max_episodes > 0:
        episode_dirs = episode_dirs[:args.max_episodes]
    logger.info(f"Evaluating {len(episode_dirs)} episodes")

    # --- Evaluate ---
    results = []
    stats = {"gen_tokens": [], "gen_ms": [], "act_ms": [], "mse": []}

    with open(log_file, "w") as log_f:
        for ep_idx, ep_dir in enumerate(episode_dirs):
            ep_name = os.path.basename(ep_dir)
            logger.info(f"[{ep_idx+1}/{len(episode_dirs)}] {ep_name}")

            states_dir = os.path.join(ep_dir, "states")
            left_path = os.path.join(states_dir, "left_states.jsonl")
            right_path = os.path.join(states_dir, "right_states.jsonl")
            if not os.path.exists(left_path):
                logger.warning(f"  Skipping: no left_states.jsonl")
                continue
            left_states = load_jsonl(left_path)
            right_states = load_jsonl(right_path)
            n_states = min(len(left_states), len(right_states))

            videos_dir = os.path.join(ep_dir, "videos")
            cam_files = ["cam_high_rgb.mp4", "cam_wrist_left_rgb.mp4", "cam_wrist_right_rgb.mp4"]
            caps = {}
            for cf in cam_files:
                vp = os.path.join(videos_dir, cf)
                if os.path.exists(vp):
                    caps[cf] = cv2.VideoCapture(vp)
            if not caps:
                logger.warning(f"  Skipping: no videos")
                continue

            n_video = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps.values()))
            n_frames = min(n_states, n_video)
            frame_indices = list(range(0, n_frames, args.frame_interval))
            logger.info(f"  {n_frames} frames -> {len(frame_indices)} samples")

            for fidx in frame_indices:
                # ---- INPUTS ----
                images_pil = []
                image_paths = []
                for cf in cam_files:
                    if cf in caps:
                        img = read_video_frame(caps[cf], fidx)
                        if img is not None:
                            images_pil.append(img)
                            image_paths.append(f"{videos_dir}/{cf}#frame={fidx}")

                if not images_pil:
                    continue

                state = build_aloha_state(left_states[fidx], right_states[fidx])

                tok_result = tokenization_func([{"from": "human", "value": prompt}])
                input_ids = torch.tensor(tok_result["input_ids"]).unsqueeze(0).to(device)
                attn_mask = (input_ids != tokenizer.pad_token_id).bool()

                non_pad_ids = input_ids[0][attn_mask[0]]
                prompt_decoded = tokenizer.decode(non_pad_ids, skip_special_tokens=False)

                num_images = 3
                images_tensor = model.process_images(images_pil).to(dtype=torch.bfloat16)
                if images_tensor.shape[0] < num_images:
                    pad = torch.zeros(num_images - images_tensor.shape[0], *images_tensor.shape[1:],
                                      dtype=images_tensor.dtype, device=images_tensor.device)
                    images_tensor = torch.cat([images_tensor, pad], dim=0)
                images_tensor = images_tensor[:num_images].unsqueeze(0).to(device)
                image_masks = torch.tensor(
                    [[True] * min(len(images_pil), num_images)
                     + [False] * max(0, num_images - len(images_pil))],
                    device=device,
                )

                # ---- TOKEN GENERATION ----
                t0 = time.monotonic()
                try:
                    gen_tokens = model.generate(
                        input_ids=input_ids, attention_mask=attn_mask,
                        images=images_tensor, image_masks=image_masks,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample, temperature=args.temperature,
                        eos_token_id=eos_token_id,
                    )
                    gen_time = time.monotonic() - t0
                    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=False)
                    gen_ids = gen_tokens[0].cpu().tolist()
                    n_gen = gen_tokens.shape[1]
                    gen_error = None
                except Exception as e:
                    gen_time = time.monotonic() - t0
                    gen_text = ""
                    gen_ids = []
                    n_gen = 0
                    gen_error = str(e)
                    logger.error(f"  generate() error at frame {fidx}: {e}")

                parsed_tags = parse_generated_tags(gen_text)

                # ---- ACTION GENERATION ----
                t1 = time.monotonic()
                action_raw_sample = None
                action_error = None
                try:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    state_padded = torch.zeros(1, 32, device=device, dtype=torch.float32)
                    state_padded[0, :state_tensor.shape[1]] = state_tensor

                    actions = model.inference_action(
                        input_ids=input_ids, attention_mask=attn_mask,
                        states=state_padded, images=images_tensor, image_masks=image_masks,
                        diffusion_steps=args.diffusion_steps,
                    )
                    act_time = time.monotonic() - t1
                    action_np = actions[0].detach().cpu().float().numpy()
                    action_raw_sample = action_np[:5, :args.action_dim].tolist()
                except Exception as e:
                    act_time = time.monotonic() - t1
                    action_np = None
                    action_error = str(e)
                    logger.error(f"  inference_action() error at frame {fidx}: {e}")

                # ---- GT & MSE ----
                action_mse = None
                if action_np is not None:
                    gt_actions = []
                    horizon = min(35, n_frames - fidx)
                    for t in range(fidx, fidx + horizon):
                        gt_actions.append(build_aloha_state(left_states[t], right_states[t]))
                    if len(gt_actions) < horizon:
                        gt_actions += [gt_actions[-1]] * (horizon - len(gt_actions))
                    gt = np.array(gt_actions, dtype=np.float32)
                    pred = action_np[:horizon, :args.action_dim]
                    if pred.shape[0] == gt.shape[0]:
                        action_mse = float(np.mean((pred - gt) ** 2))

                # ---- BUILD RECORD ----
                record = {
                    "episode": ep_name,
                    "frame_idx": fidx,
                    "n_total_frames": n_frames,
                    # --- inputs ---
                    "input_prompt": prompt,
                    "input_prompt_tokenized": prompt_decoded,
                    "input_ids_len": int(input_ids.shape[1]),
                    "input_ids_non_pad": int(attn_mask.sum().item()),
                    "input_image_paths": image_paths,
                    "input_image_masks": image_masks[0].cpu().tolist(),
                    "input_state_14d": state.tolist(),
                    # --- token generation output ---
                    "gen_text": gen_text,
                    "gen_parsed_tags": parsed_tags,
                    "gen_token_ids_sampled": gen_ids[:20] if len(gen_ids) > 20 else gen_ids,
                    "gen_num_tokens": n_gen,
                    "gen_time_ms": round(gen_time * 1000, 1),
                    "gen_error": gen_error,
                    # --- action generation output ---
                    "action_raw_sample_5x14": action_raw_sample,
                    "action_mse_vs_gt": action_mse,
                    "action_time_ms": round(act_time * 1000, 1),
                    "action_error": action_error,
                }
                results.append(record)
                log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_f.flush()

                stats["gen_tokens"].append(n_gen)
                stats["gen_ms"].append(gen_time * 1000)
                stats["act_ms"].append(act_time * 1000)
                if action_mse is not None:
                    stats["mse"].append(action_mse)

                if len(results) % 10 == 0:
                    logger.info(
                        f"  [{ep_name}] f={fidx} gen={n_gen}tok/{gen_time*1000:.0f}ms "
                        f"act={act_time*1000:.0f}ms mse={action_mse} "
                        f"tags={list(parsed_tags.keys()) if parsed_tags else 'none'}"
                    )

            for c in caps.values():
                c.release()

    # --- Summary ---
    n_errors = sum(1 for r in results if r["gen_error"] or r["action_error"])
    summary = {
        "run_id": run_id,
        "checkpoint": args.checkpoint,
        "checkpoint_name": ckpt_name,
        "data_dir": args.data_dir,
        "total_episodes": len(episode_dirs),
        "total_frames": len(results),
        "errors": n_errors,
        "frame_interval": args.frame_interval,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "diffusion_steps": args.diffusion_steps,
        "avg_gen_tokens": float(np.mean(stats["gen_tokens"])) if stats["gen_tokens"] else 0,
        "avg_gen_ms": float(np.mean(stats["gen_ms"])) if stats["gen_ms"] else 0,
        "avg_act_ms": float(np.mean(stats["act_ms"])) if stats["act_ms"] else 0,
        "avg_action_mse": float(np.mean(stats["mse"])) if stats["mse"] else None,
        "median_action_mse": float(np.median(stats["mse"])) if stats["mse"] else None,
        "p90_action_mse": float(np.percentile(stats["mse"], 90)) if stats["mse"] else None,
        "n_frames_with_tags": sum(1 for r in results if r["gen_parsed_tags"]),
        "tag_types_found": list(set(
            tag for r in results if r["gen_parsed_tags"] for tag in r["gen_parsed_tags"]
        )),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Evaluation complete!")
    logger.info(f"  Checkpoint: {ckpt_name}")
    logger.info(f"  Episodes: {summary['total_episodes']}, Frames: {summary['total_frames']}, Errors: {n_errors}")
    logger.info(f"  Avg gen tokens: {summary['avg_gen_tokens']:.1f}, Avg gen time: {summary['avg_gen_ms']:.0f} ms")
    logger.info(f"  Avg action time: {summary['avg_act_ms']:.0f} ms")
    logger.info(f"  Action MSE  avg={summary['avg_action_mse']}  median={summary['median_action_mse']}  p90={summary['p90_action_mse']}")
    logger.info(f"  Frames with parsed tags: {summary['n_frames_with_tags']}, Types: {summary['tag_types_found']}")
    logger.info(f"  Log:     {log_file}")
    logger.info(f"  Summary: {summary_file}")


if __name__ == "__main__":
    main()
