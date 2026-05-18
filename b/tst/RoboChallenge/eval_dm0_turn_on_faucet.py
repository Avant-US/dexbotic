#!/usr/bin/env python3
"""Offline open-loop evaluation of DM0 on RoboChallenge turn_on_faucet.

Evaluates both:
  1. Token generation (CoT / bbox / action tokens) via model.generate()
  2. Continuous action prediction via model.inference_action()

Records all generated semantic tokens and action quality metrics.
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
from transformers import AutoTokenizer, DynamicCache

# Ensure dexbotic is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
from dexbotic.tokenization.process import DM0Tokenization
from dexbotic.data.dataset.transform.action import ActionNorm, PadState
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy, ToTensor
from dexbotic.data.dataset.transform.output import ActionDenorm, AbsoluteAction


# ---------- ALOHA data helpers ----------

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
    logger.warning(f"norm_stats.json not found at {ckpt_path}, using default")
    return {"min": -1, "max": 1}


def build_aloha_state(left_state, right_state):
    """Build 14-dim ALOHA state: [left_qpos(6)+left_gripper(1), right_qpos(6)+right_gripper(1)]."""
    left_qpos = np.array(left_state["qpos"], dtype=np.float32)[:6]
    left_grip = np.float32(left_state["gripper"])
    right_qpos = np.array(right_state["qpos"], dtype=np.float32)[:6]
    right_grip = np.float32(right_state["gripper"])
    return np.concatenate([left_qpos, [left_grip], right_qpos, [right_grip]])


def read_video_frame(cap, frame_idx):
    """Read a specific frame from a cv2 VideoCapture, return PIL Image or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def build_gt_action_trajectory(left_states, right_states, frame_idx, horizon=50):
    """Build ground-truth action trajectory from frame_idx for `horizon` steps."""
    actions = []
    n = min(len(left_states), len(right_states))
    for t in range(frame_idx, min(frame_idx + horizon, n)):
        s = build_aloha_state(left_states[t], right_states[t])
        actions.append(s)
    if len(actions) < horizon:
        actions += [actions[-1]] * (horizon - len(actions))
    return np.array(actions, dtype=np.float32)


# ---------- Main evaluation ----------

def main():
    parser = argparse.ArgumentParser(description="DM0 RoboChallenge turn_on_faucet evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to unpacked turn_on_faucet/ directory")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=-1,
                        help="-1 = all episodes")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--frame_interval", type=int, default=30,
                        help="Sample every N frames per episode")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=14,
                        help="ALOHA=14 (dual-arm)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"eval_{run_id}.jsonl")
    summary_file = os.path.join(args.output_dir, f"summary_{run_id}.json")

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
    logger.info(f"lm_head.weight shape: {model.lm_head.weight.shape}")
    logger.info(f"lm_head tied to embed_tokens: {model.lm_head.weight.data_ptr() == model.model.llm.embed_tokens.weight.data_ptr()}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False, trust_remote_code=True)
    original_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = 2048
    logger.info(f"tokenizer.model_max_length overridden: {original_max_len} → {tokenizer.model_max_length}")

    tokenization_func = DM0Tokenization(tokenizer)
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    logger.info(f"eos_token_id (<|im_end|>): {eos_token_id}")

    # --- Normalization ---
    norm_stats = load_norm_stats(args.checkpoint)
    input_transform = Pipeline([
        PadState(ndim=model.model.config.action_dim, axis=-1),
        ActionNorm(statistic_mapping=norm_stats, strict=False, use_quantiles=True),
        ToTensor(),
    ])
    output_transform = Pipeline([
        ToNumpy(),
        ActionDenorm(statistic_mapping=norm_stats, strict=False, use_quantiles=True),
        AbsoluteAction(),
    ])

    prompt = "grasp the faucet switch and turn it on"
    non_delta_mask = [6, 13]  # ALOHA gripper indices

    # --- Find episodes ---
    data_root = args.data_dir
    episode_pattern = os.path.join(data_root, "data", "episode_*")
    episode_dirs = sorted(glob(episode_pattern))
    if not episode_dirs:
        episode_pattern = os.path.join(data_root, "episode_*")
        episode_dirs = sorted(glob(episode_pattern))
    if not episode_dirs:
        for root, dirs, files in os.walk(data_root):
            if "left_states.jsonl" in files or "states.jsonl" in files:
                episode_dirs.append(os.path.dirname(os.path.dirname(root)))
                break
        if not episode_dirs:
            logger.error(f"No episodes found under {data_root}")
            sys.exit(1)
        episode_dirs = sorted(set(episode_dirs))

    if args.max_episodes > 0:
        episode_dirs = episode_dirs[:args.max_episodes]
    logger.info(f"Found {len(episode_dirs)} episodes to evaluate")

    # --- Evaluate ---
    results = []
    total_gen_tokens = []
    total_gen_times = []
    total_action_times = []

    with open(log_file, "w") as log_f:
        for ep_idx, ep_dir in enumerate(episode_dirs):
            ep_name = os.path.basename(ep_dir)
            logger.info(f"[{ep_idx+1}/{len(episode_dirs)}] Processing {ep_name}")

            # Load states
            states_dir = os.path.join(ep_dir, "states")
            left_states_path = os.path.join(states_dir, "left_states.jsonl")
            right_states_path = os.path.join(states_dir, "right_states.jsonl")
            single_states_path = os.path.join(states_dir, "states.jsonl")

            if os.path.exists(left_states_path):
                left_states = load_jsonl(left_states_path)
                right_states = load_jsonl(right_states_path)
            elif os.path.exists(single_states_path):
                single_states = load_jsonl(single_states_path)
                left_states = single_states
                right_states = single_states
            else:
                logger.warning(f"No states found in {states_dir}, skipping")
                continue

            n_states = min(len(left_states), len(right_states))

            # Load videos
            videos_dir = os.path.join(ep_dir, "videos")
            cam_names = ["cam_high_rgb.mp4", "cam_wrist_left_rgb.mp4", "cam_wrist_right_rgb.mp4"]
            caps = {}
            for cn in cam_names:
                vpath = os.path.join(videos_dir, cn)
                if os.path.exists(vpath):
                    caps[cn] = cv2.VideoCapture(vpath)

            if len(caps) == 0:
                logger.warning(f"No videos found in {videos_dir}, skipping")
                continue

            n_video_frames = int(min(
                cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps.values()
            ))
            n_frames = min(n_states, n_video_frames)

            frame_indices = list(range(0, n_frames, args.frame_interval))
            logger.info(f"  {n_frames} frames, evaluating {len(frame_indices)} samples")

            for fidx in frame_indices:
                # Read images
                images_pil = []
                for cn in cam_names:
                    if cn in caps:
                        img = read_video_frame(caps[cn], fidx)
                        if img is not None:
                            images_pil.append(img)

                if len(images_pil) == 0:
                    continue

                # Build state
                state = build_aloha_state(left_states[fidx], right_states[fidx])

                # Tokenize prompt
                tok_result = tokenization_func([
                    {"from": "human", "value": prompt},
                ])
                input_ids = torch.tensor(tok_result["input_ids"]).unsqueeze(0).to(device)
                attn_mask = (input_ids != tokenizer.pad_token_id).bool()

                # Process images
                images_tensor = model.process_images(images_pil).to(dtype=model.dtype)
                # Pad to 3 images if needed
                num_images = 3
                if images_tensor.shape[0] < num_images:
                    pad = torch.zeros(
                        num_images - images_tensor.shape[0],
                        *images_tensor.shape[1:],
                        dtype=images_tensor.dtype,
                        device=images_tensor.device,
                    )
                    images_tensor = torch.cat([images_tensor, pad], dim=0)
                images_tensor = images_tensor[:num_images].unsqueeze(0).to(device)
                image_masks = torch.tensor(
                    [[True] * min(len(images_pil), num_images)
                     + [False] * max(0, num_images - len(images_pil))],
                    device=device,
                )

                # === Token Generation ===
                t0 = time.monotonic()
                try:
                    gen_tokens = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        images=images_tensor,
                        image_masks=image_masks,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        eos_token_id=eos_token_id,
                    )
                    gen_time = time.monotonic() - t0
                    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=False)
                    gen_ids = gen_tokens[0].cpu().tolist()
                    n_gen = gen_tokens.shape[1]
                except Exception as e:
                    logger.error(f"  generate() failed at frame {fidx}: {e}")
                    gen_time = time.monotonic() - t0
                    gen_text = f"ERROR: {e}"
                    gen_ids = []
                    n_gen = 0

                # === Action Generation ===
                t1 = time.monotonic()
                try:
                    state_input = input_transform({
                        "state": state.copy(),
                        "meta_data": {"non_delta_mask": np.array(non_delta_mask)},
                    })
                    state_tensor = state_input["state"].unsqueeze(0).to(device)

                    actions = model.inference_action(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        states=state_tensor,
                        images=images_tensor,
                        image_masks=image_masks,
                        diffusion_steps=args.diffusion_steps,
                    )
                    action_time = time.monotonic() - t1
                    action_np = actions.detach().cpu().float().numpy()

                    # Denormalize
                    out_data = {
                        "action": action_np.copy(),
                        "state": state_input["state"].numpy()[None],
                        "meta_data": {"non_delta_mask": np.array(non_delta_mask)},
                    }
                    out_data = output_transform(out_data)
                    action_denorm = out_data["action"][0, :, :args.action_dim]

                except Exception as e:
                    logger.error(f"  inference_action() failed at frame {fidx}: {e}")
                    action_time = time.monotonic() - t1
                    action_denorm = None
                    action_np = None

                # GT action trajectory for MSE
                gt_traj = build_gt_action_trajectory(
                    left_states, right_states, fidx,
                    horizon=min(35, n_frames - fidx),
                )

                action_mse = None
                if action_denorm is not None and gt_traj is not None:
                    horizon = min(action_denorm.shape[0], gt_traj.shape[0])
                    if horizon > 0:
                        pred = action_denorm[:horizon]
                        gt = gt_traj[:horizon]
                        action_mse = float(np.mean((pred - gt) ** 2))

                result = {
                    "episode": ep_name,
                    "frame_idx": fidx,
                    "generated_text": gen_text,
                    "generated_token_ids": gen_ids,
                    "num_generated_tokens": n_gen,
                    "generation_time_ms": round(gen_time * 1000, 1),
                    "action_time_ms": round(action_time * 1000, 1),
                    "action_mse": action_mse,
                    "action_sample_5x7": (
                        action_denorm[:5, :7].tolist()
                        if action_denorm is not None and action_denorm.shape[0] >= 5
                        else None
                    ),
                }
                results.append(result)
                log_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                log_f.flush()

                total_gen_tokens.append(n_gen)
                total_gen_times.append(gen_time * 1000)
                total_action_times.append(action_time * 1000)

                if len(results) % 5 == 0:
                    logger.info(
                        f"  [{ep_name}] frame={fidx} "
                        f"gen_tokens={n_gen} gen_ms={gen_time*1000:.0f} "
                        f"act_ms={action_time*1000:.0f} mse={action_mse}"
                    )

            # Release videos
            for cap in caps.values():
                cap.release()

    # --- Summary ---
    summary = {
        "run_id": run_id,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "total_episodes": len(episode_dirs),
        "total_frames_evaluated": len(results),
        "frame_interval": args.frame_interval,
        "max_new_tokens": args.max_new_tokens,
        "avg_generated_tokens": float(np.mean(total_gen_tokens)) if total_gen_tokens else 0,
        "avg_generation_time_ms": float(np.mean(total_gen_times)) if total_gen_times else 0,
        "avg_action_time_ms": float(np.mean(total_action_times)) if total_action_times else 0,
        "action_mse_values": [r["action_mse"] for r in results if r["action_mse"] is not None],
        "avg_action_mse": float(np.mean([r["action_mse"] for r in results if r["action_mse"] is not None])) if any(r["action_mse"] is not None for r in results) else None,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Evaluation complete!")
    logger.info(f"  Episodes: {summary['total_episodes']}")
    logger.info(f"  Frames evaluated: {summary['total_frames_evaluated']}")
    logger.info(f"  Avg generated tokens: {summary['avg_generated_tokens']:.1f}")
    logger.info(f"  Avg generation time: {summary['avg_generation_time_ms']:.0f} ms")
    logger.info(f"  Avg action time: {summary['avg_action_time_ms']:.0f} ms")
    logger.info(f"  Avg action MSE: {summary['avg_action_mse']}")
    logger.info(f"  Log: {log_file}")
    logger.info(f"  Summary: {summary_file}")


if __name__ == "__main__":
    main()
