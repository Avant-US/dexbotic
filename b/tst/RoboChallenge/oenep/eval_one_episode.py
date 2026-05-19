#!/usr/bin/env python3
"""Single-episode DM0 evaluation with customizable prompts.

Accepts:
  - An episode data path (e.g. /mnt/g/DATA/RoboChallenge/.../episode_001004)
  - Custom system prompt, user prompt, assistant prefix
  - Runs DM0 inference on every sampled frame in the episode
  - Logs detailed input/output and ground truth comparison to JSONL

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_one_episode.py \
        --checkpoint /path/to/DM0-base \
        --episode_path /mnt/g/DATA/RoboChallenge/task_table30_turn_on_faucet/turn_on_faucet/data/episode_001004 \
        --system_prompt "You are a helpful robot assistant." \
        --user_prompt "grasp the faucet switch and turn it on" \
        --assistant_prefix "" \
        --frame_interval 30
Detect the  faucet switch. Ground each element with [xmin, ymin, xmax, ymax]. coordinates are normalized to 0-1000.
Detect the  faucet switch. Ground each element with [xmin, ymin, xmax, ymax]. coordinates are normalized to 0-1000. You are on the <robot_aloha> platform utilizing the <eef_control> interface.
Grasp the faucet switch and turn it on. Proceed to execute the corresponding action sequence on the <robot_aloha> platform utilizing the <eef_control> interface.
Let's think step by step. Detect the  faucet switch. Ground each element with [xmin, ymin, xmax, ymax]. coordinates are normalized to 0-1000.
You are a senior expert in the fields of Artificial Intelligence, Robotics, Vision-Language Models (VLMs), and Vision-Language-Action models (VLAs). When asked a question, you will first engage in a step-by-step, deep-thinking process to analyze its core nature and the underlying intent. You will decompose the problem into sub-problems to tackle them sequentially. Following this, you will identify the objects relevant to solving the problem in the provided image and enclose them in bounding boxes, using the format [[minX, minY, maxX, maxY]]  (scaled from 0 to 1000, with the top-left corner as origin). Your thinking, reasoning and analysis process must be enclosed between <think> and </think>. Once the thinking phase is complete, you will output the final solution along with the 2D trajectories and actions required to execute the task. Let's think step by step. Detect the  faucet switch. Ground each element with [xmin, ymin, xmax, ymax]. coordinates are normalized to 0-1000.
    CUDA_VISIBLE_DEVICES=7 python eval_one_episode.py \
        --checkpoint /mnt/g/CKPT/dexbotic/DM0-base \
        --episode_path /mnt/g/DATA/RoboChallenge/task_table30_turn_on_faucet/turn_on_faucet/data/episode_001004 \
        --system_prompt "" \
        --user_prompt "Grasp the faucet switch and turn it on. Proceed to execute the corresponding action sequence on the <robot_aloha> platform utilizing the <eef_control> interface." \
        --assistant_prefix "" \
        --frame_interval 600

CUDA_VISIBLE_DEVICES=7 python eval_one_episode.py \
        --checkpoint /mnt/g/CKPT/dexbotic/DM0-base \
        --episode_path /mnt/g/DATA/RoboChallenge/task_table30_turn_on_faucet/turn_on_faucet/data/episode_001004 \
        --system_prompt "" \
        --user_prompt "You are a senior expert in the fields of Artificial Intelligence, Robotics, Vision-Language Models (VLMs), and Vision-Language-Action models (VLAs). When asked a question, you will first engage in a step-by-step, deep-thinking process to analyze its core nature and the underlying intent. You will decompose the problem into sub-problems to tackle them sequentially. Following this, you will identify the objects relevant to solving the problem in the provided image and enclose them in bounding boxes, using the format [[minX, minY, maxX, maxY]]  (scaled from 0 to 1000, with the top-left corner as origin). Your thinking, reasoning and analysis process must be enclosed between <think> and </think>. Once the thinking phase is complete, you will output the final solution along with the 2D trajectories and actions required to execute the task. Grasp the faucet switch and turn it on. Proceed to execute the corresponding action sequence on the <robot_aloha> platform utilizing the <eef_control> interface." \
        --assistant_prefix "<think>" \
        --frame_interval 600        

"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
from dexbotic.tokenization import conversation as conversation_lib


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_aloha_state(left_state, right_state):
    """Build 14-dim ALOHA state: [left_qpos(6)+gripper(1), right_qpos(6)+gripper(1)]."""
    lq = np.array(left_state["qpos"], dtype=np.float32)[:6]
    lg = np.float32(left_state["gripper"])
    rq = np.array(right_state["qpos"], dtype=np.float32)[:6]
    rg = np.float32(right_state["gripper"])
    return np.concatenate([lq, [lg], rq, [rg]])


def read_video_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def parse_tags(text):
    """Extract structured tags from generated text."""
    tags = {}
    for tag in ["subtask", "bbox", "traj", "act"]:
        m = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if m:
            tags[tag] = m
    at = re.findall(r"<action>(\d+)</action>", text)
    if at:
        tags["discrete_action_tokens"] = at
    coord_pairs = re.findall(
        r"\(\s*<action>(\d+)</action>\s*,\s*<action>(\d+)</action>\s*\)", text
    )
    if coord_pairs:
        tags["xy_coords"] = [[int(x), int(y)] for x, y in coord_pairs]
    return tags or None


def build_prompt_old(
    user_prompt,
    system_prompt=None,
    assistant_prefix="",
    robot_tag="<robot_aloha>",
    control_tag="<eef_control>",
    num_image_placeholders=3,
    task_prompt_template="What action should the robot take to {prompt}?",
    chat_template="step",
):
    """Build full prompt string with SYSTEM/USER/ASSISTANT structure.

    Format:
        SYSTEM: {system_prompt}<sep>
        USER: <image_1> ... <image_N>
        <robot_tag> <control_tag>
        What action should the robot take to {user_prompt}?<sep>
        ASSISTANT: {assistant_prefix}
    """
    conv = conversation_lib.conv_templates[chat_template].copy()
    if system_prompt is not None:
        conv.system = system_prompt
    sep = conv.sep

    image_line = " ".join(
        f"<image_{idx + 1}>" for idx in range(max(num_image_placeholders, 0))
    )
    tag_line = " ".join(tag for tag in [robot_tag, control_tag] if tag)
    task_line = task_prompt_template.format(prompt=user_prompt)
    user_content = "\n".join(part for part in [image_line, tag_line, task_line] if part)

    suffix = assistant_prefix or ""
    full = f"{conv.system}{sep}{conv.roles[0]}: {user_content}{sep}{conv.roles[1]}: {suffix}"
    return full

def build_prompt(
    user_prompt,
    system_prompt=None,
    assistant_prefix="",
    robot_tag="<robot_aloha>",
    control_tag="<eef_control>",
    num_image_placeholders=3,
    task_prompt_template="What action should the robot take to {prompt}?",
    chat_template="step",
):
    """Build full prompt string with SYSTEM/USER/ASSISTANT structure.

    Format:
        SYSTEM: {system_prompt}<sep>
        USER: <image_1> ... <image_N>
        <robot_tag> <control_tag>
        What action should the robot take to {user_prompt}?<sep>
        ASSISTANT: {assistant_prefix}
    """
    fullp = "<|im_start|>";
    if system_prompt:
        fullp = f"{fullp}system\n{system_prompt}<|im_end|>\n"
    fullp = f"{fullp}user\n{user_prompt}<|im_end|>\n"
    suffix = assistant_prefix or ""
    fullp = f"{fullp}<|im_start|>assistant\n{suffix}"
    return fullp

def tokenize_prompt(tokenizer, prompt_str, max_len=2048):
    """Tokenize prompt string directly (bypass DM0Tokenization pop logic)."""
    ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    n_real = len(ids)
    if len(ids) > max_len:
        logger.warning(f"Prompt truncated: {len(ids)} -> {max_len}")
        ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    return np.asarray(ids), n_real


@torch.no_grad()
def generate_with_min_tokens(
    model, tokenizer, input_ids, attention_mask, images, image_masks,
    max_new_tokens=128, min_new_tokens=0,
    do_sample=False, temperature=0.7, top_p=0.95, top_k=50,
    eos_token_id=None,
):
    """Generate tokens with min_new_tokens enforcement via eos suppression."""
    from transformers import DynamicCache
    from dexbotic.model.dm0.dm0_utils import make_attn_mask_2d, make_attn_mask_4d

    batch_size = input_ids.shape[0]
    device = input_ids.device

    prefix_hs, prefix_pad_mask, prefix_attn_mask = (
        model.get_prefix_hidden_states(input_ids, attention_mask, images, image_masks)
    )
    prefix_pad_mask = prefix_pad_mask.bool()
    prefix_attn_2d = make_attn_mask_2d(prefix_pad_mask, prefix_attn_mask)
    prefix_attn_4d = make_attn_mask_4d(prefix_attn_2d, dtype=prefix_hs.dtype)
    prefix_positions = torch.cumsum(prefix_pad_mask.long(), dim=1) - 1

    if model.model.config.bf16:
        prefix_hs = prefix_hs.to(torch.bfloat16)

    module_list = [model.model.llm, model.model.action_expert.model]
    (prefix_out, _), kv_cache = model._merged_attention_forward(
        module_list=module_list,
        attention_mask=prefix_attn_4d,
        position_ids=prefix_positions,
        past_key_values=DynamicCache(),
        input_embeds_list=[prefix_hs, None],
        use_cache=True,
    )

    context_mask = prefix_pad_mask.clone()
    generated_tokens = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    logits = model.lm_head(prefix_out[:, -1:])
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        cur_logits = logits.squeeze(1).clone().float()

        if eos_token_id is not None and step < min_new_tokens:
            cur_logits[:, eos_token_id] = float("-inf")

        if do_sample and temperature > 0:
            scaled = cur_logits / temperature
            if top_k and top_k < scaled.shape[-1]:
                v, _ = torch.topk(scaled, top_k, dim=-1)
                kth = v[:, -1].unsqueeze(-1)
                scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
                cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
            probs = torch.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = cur_logits.argmax(dim=-1, keepdim=True)

        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_token_id)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        context_mask = torch.cat(
            [context_mask, torch.ones((batch_size, 1), dtype=torch.bool, device=device)],
            dim=1,
        )

        if finished.all():
            break

        token_embeds = model.model.embed_language_tokens(next_token)
        if model.model.config.bf16:
            token_embeds = token_embeds.to(torch.bfloat16)

        decode_position = context_mask.long().sum(dim=1, keepdim=True) - 1
        decode_mask_2d = context_mask[:, None, :]
        decode_mask_4d = make_attn_mask_4d(decode_mask_2d, dtype=token_embeds.dtype)

        (decode_out, _), kv_cache = model._merged_attention_forward(
            module_list=module_list,
            attention_mask=decode_mask_4d,
            position_ids=decode_position,
            past_key_values=kv_cache,
            input_embeds_list=[token_embeds, None],
            use_cache=True,
        )
        logits = model.lm_head(decode_out[:, -1:])

    return generated_tokens


def compute_gt_trajectory(left_states, right_states, frame_idx, horizon, n_total):
    """Build ground-truth state trajectory starting at frame_idx."""
    gt = []
    for t in range(frame_idx, min(frame_idx + horizon, n_total)):
        gt.append(build_aloha_state(left_states[t], right_states[t]))
    if len(gt) < horizon:
        gt += [gt[-1]] * (horizon - len(gt))
    return np.array(gt, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Run DM0 evaluation on a single episode with custom prompts"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DM0 model checkpoint")
    parser.add_argument("--episode_path", type=str, required=True,
                        help="Path to episode directory (e.g. .../episode_001004)")
    parser.add_argument("--system_prompt", type=str,
                        default="You are a helpful robot assistant.",
                        help="System prompt for the model")
    parser.add_argument("--user_prompt", type=str,
                        default="grasp the faucet switch and turn it on",
                        help="User prompt (task instruction)")
    parser.add_argument("--assistant_prefix", type=str, default="",
                        help="Prefix appended after 'ASSISTANT: ' to prime generation")
    parser.add_argument("--robot_tag", type=str, default="<robot_aloha>")
    parser.add_argument("--control_tag", type=str, default="<eef_control>")
    parser.add_argument("--num_image_placeholders", type=int, default=3)
    parser.add_argument("--task_prompt_template", type=str,
                        default="What action should the robot take to {prompt}?")
    parser.add_argument("--frame_interval", type=int, default=30,
                        help="Sample every N frames")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--min_new_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--tie_lm_head", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--action_horizon", type=int, default=35,
                        help="Number of future steps for action prediction & GT comparison")
    args = parser.parse_args()
    print("==="*10)
    print(args)
    print("==="*10)

    os.makedirs(LOG_DIR, exist_ok=True)

    ep_name = os.path.basename(args.episode_path.rstrip("/"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"one_ep_{ep_name}_{run_id}.jsonl")
    summary_file = os.path.join(LOG_DIR, f"one_ep_{ep_name}_{run_id}_summary.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Episode: {args.episode_path}")

    # --- Validate episode ---
    states_dir = os.path.join(args.episode_path, "states")
    videos_dir = os.path.join(args.episode_path, "videos")
    left_path = os.path.join(states_dir, "left_states.jsonl")
    right_path = os.path.join(states_dir, "right_states.jsonl")

    if not os.path.exists(left_path):
        logger.error(f"left_states.jsonl not found at: {left_path}")
        sys.exit(1)

    left_states = load_jsonl(left_path)
    right_states = load_jsonl(right_path)
    n_states = min(len(left_states), len(right_states))
    logger.info(f"Loaded states: left={len(left_states)}, right={len(right_states)}")

    cam_files = ["cam_high_rgb.mp4", "cam_wrist_left_rgb.mp4", "cam_wrist_right_rgb.mp4"]
    caps = {}
    for cf in cam_files:
        vp = os.path.join(videos_dir, cf)
        if os.path.exists(vp):
            caps[cf] = cv2.VideoCapture(vp)
    if not caps:
        logger.error(f"No video files found in: {videos_dir}")
        sys.exit(1)

    n_video = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps.values()))
    n_frames = min(n_states, n_video)
    frame_indices = list(range(0, n_frames, args.frame_interval))
    logger.info(f"Video frames: {n_video}, states: {n_states}, usable: {n_frames}")
    logger.info(f"Frame interval: {args.frame_interval} -> {len(frame_indices)} samples")

    # --- Load model ---
    logger.info(f"Loading model: {args.checkpoint}")
    model = DM0ForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        # sliding_window=None,    在 DM0-base checkpoint 的config.json 中加了slid的disable项()"sliding_window": null, "use_sliding_window": false,
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, use_fast=False, trust_remote_code=True
    )
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    logger.info(f"eos_token_id: {eos_id}, vocab_size: {len(tokenizer)}")

    # --- Build prompt ---
    full_prompt = build_prompt(
        user_prompt=args.user_prompt,
        system_prompt=args.system_prompt if args.system_prompt else None,
        assistant_prefix=args.assistant_prefix,
        robot_tag=args.robot_tag,
        control_tag=args.control_tag,
        num_image_placeholders=args.num_image_placeholders,
        task_prompt_template=args.task_prompt_template,
    )
    input_ids_np, n_real = tokenize_prompt(tokenizer, full_prompt, max_len=args.prompt_max_len)
    input_ids = torch.tensor(input_ids_np).unsqueeze(0).to(device)
    attn_mask = (input_ids != tokenizer.pad_token_id).bool()

    logger.info(f"=== Prompt Configuration ===")
    logger.info(f"  system_prompt: {args.system_prompt!r}")
    logger.info(f"  user_prompt: {args.user_prompt!r}")
    logger.info(f"  assistant_prefix: {args.assistant_prefix!r}")
    logger.info(f"  robot_tag: {args.robot_tag}, control_tag: {args.control_tag}")
    logger.info(f"  full_prompt: {full_prompt!r}")
    logger.info(f"  prompt tokens: real={n_real}, padded_to={args.prompt_max_len}")
    logger.info(f"=== Generation Config ===")
    logger.info(f"  max_new_tokens={args.max_new_tokens}, min_new_tokens={args.min_new_tokens}")
    logger.info(f"  do_sample={args.do_sample}, temperature={args.temperature}")
    logger.info(f"  top_p={args.top_p}, top_k={args.top_k}")
    logger.info(f"  diffusion_steps={args.diffusion_steps}, action_dim={args.action_dim}")

    # --- Evaluate ---
    results = []
    stats = {"gen_tokens": [], "gen_ms": [], "act_ms": [], "mse": []}

    with open(log_file, "w") as logf:
        # Write header record with config
        header = {
            "_type": "config",
            "episode_path": args.episode_path,
            "episode_name": ep_name,
            "checkpoint": args.checkpoint,
            "system_prompt": args.system_prompt,
            "user_prompt": args.user_prompt,
            "assistant_prefix": args.assistant_prefix,
            "robot_tag": args.robot_tag,
            "control_tag": args.control_tag,
            "task_prompt_template": args.task_prompt_template,
            "num_image_placeholders": args.num_image_placeholders,
            "full_prompt": full_prompt,
            "prompt_token_count": n_real,
            "prompt_max_len": args.prompt_max_len,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "diffusion_steps": args.diffusion_steps,
            "action_dim": args.action_dim,
            "action_horizon": args.action_horizon,
            "frame_interval": args.frame_interval,
            "n_total_frames": n_frames,
            "n_samples": len(frame_indices),
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        }
        logf.write(json.dumps(header, ensure_ascii=False) + "\n")
        logf.flush()

        for sample_idx, fidx in enumerate(frame_indices):
            logger.info(f"[{sample_idx+1}/{len(frame_indices)}] frame={fidx}")

            # --- Read images ---
            imgs_pil = []
            img_paths = []
            for cf in cam_files:
                if cf in caps:
                    img = read_video_frame(caps[cf], fidx)
                    if img is not None:
                        imgs_pil.append(img)
                        img_paths.append(f"{videos_dir}/{cf}#frame={fidx}")

            if not imgs_pil:
                logger.warning(f"  No images at frame {fidx}, skipping")
                continue

            # --- Build state ---
            state = build_aloha_state(left_states[fidx], right_states[fidx])

            # --- Process images ---
            n_img = args.num_image_placeholders
            img_t = model.process_images(imgs_pil).to(dtype=torch.bfloat16)
            if img_t.shape[0] < n_img:
                img_t = torch.cat([
                    img_t,
                    torch.zeros(n_img - img_t.shape[0], *img_t.shape[1:],
                                dtype=img_t.dtype, device=img_t.device)
                ], dim=0)
            img_t = img_t[:n_img].unsqueeze(0).to(device)
            img_m = torch.tensor(
                [[True] * min(len(imgs_pil), n_img)
                 + [False] * max(0, n_img - len(imgs_pil))],
                device=device,
            )

            # --- Token Generation ---
            t0 = time.monotonic()
            gen_error = None
            try:
                gen = generate_with_min_tokens(
                    model, tokenizer, input_ids, attn_mask, img_t, img_m,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    eos_token_id=eos_id,
                )
                gen_time = time.monotonic() - t0
                gen_text = tokenizer.decode(gen[0], skip_special_tokens=False)
                gen_ids = gen[0].cpu().tolist()
                n_gen = gen.shape[1]
            except Exception as e:
                gen_time = time.monotonic() - t0
                gen_text, gen_ids, n_gen = "", [], 0
                gen_error = str(e)
                logger.error(f"  generate error: {e}")

            tags = parse_tags(gen_text)

            # --- Action Generation ---
            t1 = time.monotonic()
            act_error = None
            act_sample = None
            act_full = None
            try:
                s_t = torch.zeros(1, 32, device=device, dtype=torch.float32)
                s_t[0, :len(state)] = torch.tensor(state, dtype=torch.float32)
                acts = model.inference_action(
                    input_ids=input_ids, attention_mask=attn_mask,
                    states=s_t, images=img_t, image_masks=img_m,
                    diffusion_steps=args.diffusion_steps,
                )
                act_time = time.monotonic() - t1
                act_np = acts[0].detach().cpu().float().numpy()
                act_sample = act_np[:5, :args.action_dim].tolist()
                act_full = act_np[:, :args.action_dim]
            except Exception as e:
                act_time = time.monotonic() - t1
                act_error = str(e)
                logger.error(f"  action error: {e}")

            # --- Ground Truth & MSE ---
            horizon = min(args.action_horizon, n_frames - fidx)
            gt_traj = compute_gt_trajectory(
                left_states, right_states, fidx, horizon, n_frames
            )
            mse = None
            per_step_mse = None
            per_dim_mse = None
            if act_full is not None and gt_traj is not None:
                h = min(act_full.shape[0], gt_traj.shape[0])
                if h > 0:
                    pred = act_full[:h]
                    gt = gt_traj[:h]
                    diff_sq = (pred - gt) ** 2
                    mse = float(np.mean(diff_sq))
                    per_step_mse = diff_sq.mean(axis=1).tolist()
                    per_dim_mse = diff_sq.mean(axis=0).tolist()

            # --- Build record ---
            record = {
                "_type": "frame",
                "sample_idx": sample_idx,
                "frame_idx": fidx,
                "n_total_frames": n_frames,
                # Inputs
                "input_image_paths": img_paths,
                "input_image_masks": img_m[0].cpu().tolist(),
                "input_state_14d": state.tolist(),
                # Token generation
                "gen_text": gen_text,
                "gen_token_ids": gen_ids,
                "gen_num_tokens": n_gen,
                "gen_time_ms": round(gen_time * 1000, 1),
                "gen_parsed_tags": tags,
                "gen_error": gen_error,
                # Action generation
                "action_sample_5x14": act_sample,
                "action_time_ms": round(act_time * 1000, 1),
                "action_error": act_error,
                # Ground truth comparison
                "gt_state_at_frame": state.tolist(),
                "gt_trajectory_first5": gt_traj[:5].tolist() if gt_traj is not None else None,
                "action_mse": mse,
                "action_per_step_mse": per_step_mse,
                "action_per_dim_mse": per_dim_mse,
            }
            results.append(record)
            logf.write(json.dumps(record, ensure_ascii=False) + "\n")
            logf.flush()

            stats["gen_tokens"].append(n_gen)
            stats["gen_ms"].append(gen_time * 1000)
            stats["act_ms"].append(act_time * 1000)
            if mse is not None:
                stats["mse"].append(mse)

            logger.info(
                f"  gen={n_gen}tok/{gen_time*1000:.0f}ms "
                f"act={act_time*1000:.0f}ms mse={mse:.6f}" if mse else
                f"  gen={n_gen}tok/{gen_time*1000:.0f}ms "
                f"act={act_time*1000:.0f}ms mse=N/A"
            )
            if tags:
                logger.info(f"  tags: {list(tags.keys())}")
            logger.info(f"  text[:150]: {gen_text[:150]!r}")

    # --- Release videos ---
    for c in caps.values():
        c.release()

    # --- Summary ---
    n_errors = sum(1 for r in results if r.get("gen_error") or r.get("action_error"))
    summary = {
        "run_id": run_id,
        "episode_path": args.episode_path,
        "episode_name": ep_name,
        "checkpoint": args.checkpoint,
        "system_prompt": args.system_prompt,
        "user_prompt": args.user_prompt,
        "assistant_prefix": args.assistant_prefix,
        "full_prompt": full_prompt,
        "n_total_frames": n_frames,
        "n_samples": len(results),
        "n_errors": n_errors,
        "frame_interval": args.frame_interval,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "diffusion_steps": args.diffusion_steps,
        "action_dim": args.action_dim,
        "action_horizon": args.action_horizon,
        "avg_gen_tokens": float(np.mean(stats["gen_tokens"])) if stats["gen_tokens"] else 0,
        "avg_gen_ms": float(np.mean(stats["gen_ms"])) if stats["gen_ms"] else 0,
        "avg_act_ms": float(np.mean(stats["act_ms"])) if stats["act_ms"] else 0,
        "avg_mse": float(np.mean(stats["mse"])) if stats["mse"] else None,
        "median_mse": float(np.median(stats["mse"])) if stats["mse"] else None,
        "min_mse": float(np.min(stats["mse"])) if stats["mse"] else None,
        "max_mse": float(np.max(stats["mse"])) if stats["mse"] else None,
        "p90_mse": float(np.percentile(stats["mse"], 90)) if stats["mse"] else None,
        "all_mse_values": stats["mse"],
        "n_frames_with_tags": sum(1 for r in results if r.get("gen_parsed_tags")),
        "tag_types_found": list(set(
            t for r in results if r.get("gen_parsed_tags") for t in r["gen_parsed_tags"]
        )),
        "log_file": log_file,
        "summary_file": summary_file,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Single-episode evaluation complete!")
    logger.info(f"  Episode: {ep_name}")
    logger.info(f"  Samples: {len(results)}, Errors: {n_errors}")
    logger.info(f"  Avg gen tokens: {summary['avg_gen_tokens']:.1f}")
    logger.info(f"  Avg gen time: {summary['avg_gen_ms']:.0f} ms")
    logger.info(f"  Avg act time: {summary['avg_act_ms']:.0f} ms")
    logger.info(f"  Action MSE  avg={summary['avg_mse']}  median={summary['median_mse']}")
    logger.info(f"  Tags found: {summary['n_frames_with_tags']} frames, types={summary['tag_types_found']}")
    logger.info(f"  Log:     {log_file}")
    logger.info(f"  Summary: {summary_file}")


if __name__ == "__main__":
    main()
