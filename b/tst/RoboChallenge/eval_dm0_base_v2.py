#!/usr/bin/env python3
"""DM0-base evaluation v2: append 'ASSISTANT: ' prefix to prompt to elicit longer CoT.

The default DM0Tokenization pops the empty assistant turn, so the prompt ends with
'... USER: <text> ' which makes the model immediately sample <|im_end|>.
Here we manually append 'ASSISTANT: ' so the model knows to start generating an
assistant response.

Also adds:
  --min_new_tokens     : refuse to emit eos_token until this many tokens are generated
  --no_eos_for_first_n : suppress eos_token logits for the first N steps
"""

import argparse
import json
import os
import re
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
from dexbotic.tokenization import conversation as conversation_lib


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_aloha_state(ls, rs):
    lq = np.array(ls["qpos"], dtype=np.float32)[:6]
    lg = np.float32(ls["gripper"])
    rq = np.array(rs["qpos"], dtype=np.float32)[:6]
    rg = np.float32(rs["gripper"])
    return np.concatenate([lq, [lg], rq, [rg]])


def read_video_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else None


def parse_tags(text):
    tags = {}
    for tag in ["subtask", "bbox", "traj", "act"]:
        m = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if m:
            tags[tag] = m
    at = re.findall(r"<action>(\d+)</action>", text)
    if at:
        tags["discrete_action_tokens"] = at
    coord_pairs = re.findall(r"\(\s*<action>(\d+)</action>\s*,\s*<action>(\d+)</action>\s*\)", text)
    if coord_pairs:
        tags["xy_coords"] = [[int(x), int(y)] for x, y in coord_pairs]
    coord_quads = re.findall(
        r"\[\s*<action>(\d+)</action>\s*,\s*<action>(\d+)</action>\s*,\s*<action>(\d+)</action>\s*,\s*<action>(\d+)</action>\s*\]",
        text,
    )
    if coord_quads:
        tags["bbox_xyxy"] = [[int(a), int(b), int(c), int(d)] for a, b, c, d in coord_quads]
    return tags or None


def build_prompt_with_assistant(
    user_prompt,
    chat_template="step",
    robot_tag=None,
    control_tag=None,
    scaffold_prefix=None,
    prompt_style="paper",
    num_image_placeholders=3,
    task_prompt_template="What action should the robot take to {prompt}?",
    system_prompt=None,
):
    """Build full prompt string with cross-embodiment tags + 'ASSISTANT: ' suffix.

    Optional 'scaffold_prefix' is appended after 'ASSISTANT: ' to prime the model
    to follow DM0's 4-level Embodied Spatial Scaffolding output format
    (subtask -> bbox -> 2D EEF traj -> discrete action tokens).

    prompt_style='paper' follows the DM0 paper-style layout:
        USER: <image_1> <image_2> <image_3>
        <robot_aloha> <eef_control>
        What action should the robot take to ...

    prompt_style='legacy' keeps the earlier experiment layout:
        USER: <robot_aloha><joint_control> grasp ...

    Scaffolding examples:
        scaffold_prefix='subtask: '         -> Hint subtask layer
        scaffold_prefix='main arm gripper: ' -> Hint 2D trajectory layer
    """
    conv = conversation_lib.conv_templates[chat_template].copy()
    if system_prompt is not None:
        conv.system = system_prompt
    sep = conv.sep
    suffix = scaffold_prefix or ""

    if prompt_style == "legacy":
        tag_prefix = ""
        if robot_tag:
            tag_prefix += robot_tag
        if control_tag:
            tag_prefix += control_tag
        if tag_prefix:
            tag_prefix += " "
        user_content = f"{tag_prefix}{user_prompt}"
    elif prompt_style == "paper":
        image_line = " ".join(
            f"<image_{idx + 1}>" for idx in range(max(num_image_placeholders, 0))
        )
        tag_line = " ".join(tag for tag in [robot_tag, control_tag] if tag)
        task_line = task_prompt_template.format(prompt=user_prompt)
        user_content = "\n".join(
            part for part in [image_line, tag_line, task_line] if part
        )
    else:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")

    full = f"{conv.system}{sep}{conv.roles[0]}: {user_content}{sep}{conv.roles[1]}: {suffix}"
    return full


def append_cot_to_input_ids(
    base_input_ids,
    base_attn_mask,
    cot_tokens,
    pad_token_id,
    eos_token_id,
    max_len,
):
    """Append generated CoT tokens to the prompt input_ids for CoT-then-Action.

    Builds new input_ids = [original non-pad prompt tokens] + [cot tokens (eos stripped)],
    re-padded to max_len.

    Returns: (new_input_ids [1, max_len], new_attn_mask [1, max_len], new_real_len)
    """
    import torch as _t
    real_ids = base_input_ids[0][base_attn_mask[0]]  # [n_real]
    cot_seq = cot_tokens[0].clone()
    if eos_token_id is not None and len(cot_seq) > 0 and cot_seq[-1].item() == eos_token_id:
        cot_seq = cot_seq[:-1]
    combined = _t.cat([real_ids, cot_seq])
    if len(combined) > max_len:
        combined = combined[:max_len]
    n_real_new = len(combined)
    pad_len = max_len - n_real_new
    if pad_len > 0:
        pad = _t.full((pad_len,), pad_token_id, dtype=combined.dtype, device=combined.device)
        combined = _t.cat([combined, pad])
    new_input_ids = combined.unsqueeze(0)
    new_attn_mask = (new_input_ids != pad_token_id).bool()
    return new_input_ids, new_attn_mask, n_real_new


def tokenize_prompt(tokenizer, prompt_str, max_len=2048):
    """Tokenize the full prompt without DM0Tokenization's pop logic.
    
    Returns: input_ids tensor [1, T] padded to max_len.
    """
    ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    n_real = len(ids)
    if len(ids) > max_len:
        logger.warning(f"prompt truncated: {len(ids)} -> {max_len}")
        ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    return np.asarray(ids), n_real


@torch.no_grad()
def generate_with_min_tokens(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    images,
    image_masks,
    max_new_tokens=128,
    min_new_tokens=0,
    do_sample=False,
    temperature=0.0,
    top_p=0.95,
    top_k=50,
    eos_token_id=None,
    suppress_tokens=None,
):
    """Custom generate that enforces min_new_tokens by masking eos before that.
    
    This is built on the DM0 generate() infrastructure but adds:
      - min_new_tokens: do not emit eos until step >= min_new_tokens
      - top_p / top_k sampling
      - suppress_tokens: list of token ids to mask (set logits to -inf)
    """
    from transformers import DynamicCache
    from dexbotic.model.dm0.dm0_utils import make_attn_mask_2d, make_attn_mask_4d

    batch_size = input_ids.shape[0]
    device = input_ids.device

    # Encode prefix
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
        cur_logits = logits.squeeze(1).clone().float()  # [B, V]

        # Suppress unwanted tokens
        if suppress_tokens:
            for tid in suppress_tokens:
                cur_logits[:, tid] = float("-inf")

        # Block eos until min_new_tokens reached
        if eos_token_id is not None and step < min_new_tokens:
            cur_logits[:, eos_token_id] = float("-inf")

        if do_sample and temperature > 0:
            scaled = cur_logits / temperature
            # top-k
            if top_k and top_k < scaled.shape[-1]:
                v, _ = torch.topk(scaled, top_k, dim=-1)
                kth = v[:, -1].unsqueeze(-1)
                scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
            # top-p
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_new_tokens", type=int, default=32,
                        help="Force model to generate at least this many tokens before allowing eos")
    parser.add_argument("--frame_interval", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--prompt_max_len", type=int, default=2048)
    parser.add_argument("--robot_tag", type=str, default="<robot_aloha>",
                        help="Cross-embodiment robot tag (e.g. '<robot_aloha>')")
    parser.add_argument("--control_tag", type=str, default="<eef_control>",
                        help="Control type tag (e.g. '<joint_control>' or '<eef_control>')")
    parser.add_argument("--prompt_style", type=str, default="paper",
                        choices=["paper", "legacy"],
                        help="Prompt layout. 'paper' matches the DM0 paper-style USER turn; "
                             "'legacy' keeps the earlier compact tag prefix.")
    parser.add_argument("--num_image_placeholders", type=int, default=3,
                        help="Number of textual image placeholders in paper-style prompt.")
    parser.add_argument("--task_prompt_template", type=str,
                        default="What action should the robot take to {prompt}?",
                        help="Task sentence template used in paper-style prompt.")
    parser.add_argument("--system_prompt", type=str,
                        default="You are a helpful robot assistant.",
                        help="System prompt. Use empty string to keep the template default.")
    parser.add_argument("--scaffold_prefix", type=str, default="",
                        help="Scaffolding prefix appended after 'ASSISTANT: ' "
                             "(e.g. 'subtask: ', 'main arm gripper: ')")
    parser.add_argument("--cot_then_action", action="store_true",
                        help="If set, append generated CoT tokens to input_ids before action inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = os.path.basename(args.checkpoint.rstrip("/"))
    log_file = os.path.join(args.output_dir, f"eval_{ckpt_name}_v2_{run_id}.jsonl")
    summary_file = os.path.join(args.output_dir, f"summary_{ckpt_name}_v2_{run_id}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model: {args.checkpoint}")
    model = DM0ForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": device},
    )
    model.model.action_out_proj = model.model.action_out_proj.float()
    model.model.action_in_proj = model.model.action_in_proj.float()
    model.model.action_time_mlp_in = model.model.action_time_mlp_in.float()
    model.model.action_time_mlp_out = model.model.action_time_mlp_out.float()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False, trust_remote_code=True)
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    logger.info(f"eos_token_id: {eos_id}")

    user_prompt = "grasp the faucet switch and turn it on"
    full_prompt = build_prompt_with_assistant(
        user_prompt, chat_template="step",
        robot_tag=args.robot_tag, control_tag=args.control_tag,
        scaffold_prefix=args.scaffold_prefix,
        prompt_style=args.prompt_style,
        num_image_placeholders=args.num_image_placeholders,
        task_prompt_template=args.task_prompt_template,
        system_prompt=args.system_prompt or None,
    )
    logger.info(f"prompt_style={args.prompt_style}, robot_tag={args.robot_tag}, control_tag={args.control_tag}")
    logger.info(f"num_image_placeholders={args.num_image_placeholders}, system_prompt={args.system_prompt!r}")
    logger.info(f"scaffold_prefix={args.scaffold_prefix!r}, cot_then_action={args.cot_then_action}")
    logger.info(f"Full prompt: {repr(full_prompt)}")

    input_ids_np, n_real = tokenize_prompt(tokenizer, full_prompt, max_len=args.prompt_max_len)
    logger.info(f"prompt tokens: real={n_real}, padded_to={args.prompt_max_len}")

    input_ids = torch.tensor(input_ids_np).unsqueeze(0).to(device)
    attn_mask = (input_ids != tokenizer.pad_token_id).bool()

    logger.info(f"min_new_tokens={args.min_new_tokens}, max_new_tokens={args.max_new_tokens}")
    logger.info(f"do_sample={args.do_sample}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")

    episode_dirs = sorted(glob(os.path.join(args.data_dir, "data", "episode_*")))[:args.max_episodes]
    logger.info(f"Evaluating {len(episode_dirs)} episodes")

    results = []
    stats = {"gen_tokens": [], "gen_ms": [], "act_ms": [], "mse": []}

    with open(log_file, "w") as logf:
        for ep_idx, ep_dir in enumerate(episode_dirs):
            ep_name = os.path.basename(ep_dir)
            left_path = os.path.join(ep_dir, "states", "left_states.jsonl")
            right_path = os.path.join(ep_dir, "states", "right_states.jsonl")
            if not os.path.exists(left_path):
                continue
            left_states = load_jsonl(left_path)
            right_states = load_jsonl(right_path)
            n_states = min(len(left_states), len(right_states))

            vdir = os.path.join(ep_dir, "videos")
            cams = ["cam_high_rgb.mp4", "cam_wrist_left_rgb.mp4", "cam_wrist_right_rgb.mp4"]
            caps = {}
            for c in cams:
                p = os.path.join(vdir, c)
                if os.path.exists(p):
                    caps[c] = cv2.VideoCapture(p)
            if not caps:
                continue
            n_video = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps.values()))
            n_frames = min(n_states, n_video)
            frame_indices = list(range(0, n_frames, args.frame_interval))
            logger.info(f"[{ep_idx+1}/{len(episode_dirs)}] {ep_name}: {n_frames}f -> {len(frame_indices)} samples")

            for fidx in frame_indices:
                imgs_pil = [read_video_frame(caps[c], fidx) for c in cams if c in caps]
                imgs_pil = [i for i in imgs_pil if i is not None]
                if not imgs_pil:
                    continue
                state = build_aloha_state(left_states[fidx], right_states[fidx])
                img_paths = [f"{vdir}/{c}#frame={fidx}" for c in cams if c in caps]

                img_t = model.process_images(imgs_pil).to(dtype=torch.bfloat16)
                n_img = 3
                if img_t.shape[0] < n_img:
                    img_t = torch.cat(
                        [img_t, torch.zeros(n_img - img_t.shape[0], *img_t.shape[1:],
                                            dtype=img_t.dtype, device=img_t.device)],
                        dim=0,
                    )
                img_t = img_t[:n_img].unsqueeze(0).to(device)
                img_m = torch.tensor(
                    [[True] * min(len(imgs_pil), n_img)
                     + [False] * max(0, n_img - len(imgs_pil))],
                    device=device,
                )

                # Generate
                t0 = time.monotonic()
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
                    gen_err = None
                except Exception as e:
                    gen_time = time.monotonic() - t0
                    gen_text, gen_ids, n_gen, gen_err = "", [], 0, str(e)

                tags = parse_tags(gen_text)

                # Action — possibly with CoT-then-Action chaining
                if args.cot_then_action and n_gen > 0:
                    new_input_ids, new_attn_mask, n_real_new = append_cot_to_input_ids(
                        input_ids, attn_mask, gen,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_id,
                        max_len=args.prompt_max_len,
                    )
                    action_input_ids = new_input_ids
                    action_attn_mask = new_attn_mask
                    cot_extended_len = n_real_new
                else:
                    action_input_ids = input_ids
                    action_attn_mask = attn_mask
                    cot_extended_len = None

                t1 = time.monotonic()
                try:
                    s_t = torch.zeros(1, 32, device=device, dtype=torch.float32)
                    s_t[0, :len(state)] = torch.tensor(state, dtype=torch.float32)
                    acts = model.inference_action(
                        input_ids=action_input_ids, attention_mask=action_attn_mask,
                        states=s_t, images=img_t, image_masks=img_m,
                        diffusion_steps=args.diffusion_steps,
                    )
                    act_time = time.monotonic() - t1
                    act_np = acts[0].detach().cpu().float().numpy()
                    act_sample = act_np[:5, :args.action_dim].tolist()
                    act_err = None
                except Exception as e:
                    act_time = time.monotonic() - t1
                    act_np, act_sample, act_err = None, None, str(e)

                # Also compute baseline MSE without CoT (for direct comparison)
                act_np_baseline = None
                if args.cot_then_action and act_err is None:
                    try:
                        s_t2 = torch.zeros(1, 32, device=device, dtype=torch.float32)
                        s_t2[0, :len(state)] = torch.tensor(state, dtype=torch.float32)
                        acts2 = model.inference_action(
                            input_ids=input_ids, attention_mask=attn_mask,
                            states=s_t2, images=img_t, image_masks=img_m,
                            diffusion_steps=args.diffusion_steps,
                        )
                        act_np_baseline = acts2[0].detach().cpu().float().numpy()
                    except Exception:
                        pass

                # MSE
                mse = None
                mse_baseline = None
                if act_np is not None:
                    horizon = min(35, n_frames - fidx)
                    gt = np.array([
                        build_aloha_state(
                            left_states[min(t, n_states - 1)],
                            right_states[min(t, n_states - 1)],
                        )
                        for t in range(fidx, fidx + horizon)
                    ], dtype=np.float32)
                    pred = act_np[:horizon, :args.action_dim]
                    if pred.shape[0] == gt.shape[0]:
                        mse = float(np.mean((pred - gt) ** 2))
                    if act_np_baseline is not None:
                        pred_b = act_np_baseline[:horizon, :args.action_dim]
                        if pred_b.shape[0] == gt.shape[0]:
                            mse_baseline = float(np.mean((pred_b - gt) ** 2))

                rec = {
                    "episode": ep_name,
                    "frame_idx": fidx,
                    "n_total_frames": n_frames,
                    "input_prompt_style": args.prompt_style,
                    "input_robot_tag": args.robot_tag,
                    "input_control_tag": args.control_tag,
                    "input_num_image_placeholders": args.num_image_placeholders,
                    "input_task_prompt_template": args.task_prompt_template,
                    "input_system_prompt": args.system_prompt,
                    "input_scaffold_prefix": args.scaffold_prefix,
                    "input_cot_then_action": args.cot_then_action,
                    "input_cot_extended_len": cot_extended_len,
                    "input_prompt_user": user_prompt,
                    "input_prompt_full": full_prompt,
                    "input_prompt_token_count": n_real,
                    "input_image_paths": img_paths,
                    "input_image_masks": img_m[0].cpu().tolist(),
                    "input_state_14d": state.tolist(),
                    "gen_text": gen_text,
                    "gen_parsed_tags": tags,
                    "gen_token_ids_first_30": gen_ids[:30],
                    "gen_num_tokens": n_gen,
                    "gen_time_ms": round(gen_time * 1000, 1),
                    "gen_error": gen_err,
                    "action_raw_sample_5x14": act_sample,
                    "action_mse_vs_gt": mse,
                    "action_mse_baseline_no_cot": mse_baseline,
                    "action_time_ms": round(act_time * 1000, 1),
                    "action_error": act_err,
                }
                results.append(rec)
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()

                stats["gen_tokens"].append(n_gen)
                stats["gen_ms"].append(gen_time * 1000)
                stats["act_ms"].append(act_time * 1000)
                if mse is not None:
                    stats["mse"].append(mse)
                if mse_baseline is not None:
                    stats.setdefault("mse_baseline", []).append(mse_baseline)

                if len(results) % 10 == 0:
                    cot_marker = "[CoT->Act]" if args.cot_then_action else ""
                    logger.info(
                        f"  {cot_marker}[{ep_name}] f={fidx} gen={n_gen}tok/{gen_time*1000:.0f}ms "
                        f"act={act_time*1000:.0f}ms mse={mse} mse_base={mse_baseline} "
                        f"tags={list(tags.keys()) if tags else None}"
                    )
                    logger.info(f"  text: {repr(gen_text[:200])}")

            for c in caps.values():
                c.release()

    # Summary
    summary = {
        "run_id": run_id,
        "checkpoint": args.checkpoint,
        "note": "v2 with ASSISTANT prefix + min_new_tokens, optional CoT-then-Action and Scaffolding",
        "prompt_style": args.prompt_style,
        "num_image_placeholders": args.num_image_placeholders,
        "task_prompt_template": args.task_prompt_template,
        "system_prompt": args.system_prompt,
        "robot_tag": args.robot_tag,
        "control_tag": args.control_tag,
        "scaffold_prefix": args.scaffold_prefix,
        "cot_then_action": args.cot_then_action,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "total_episodes": len(episode_dirs),
        "total_frames": len(results),
        "errors": sum(1 for r in results if r["gen_error"] or r["action_error"]),
        "avg_gen_tokens": float(np.mean(stats["gen_tokens"])) if stats["gen_tokens"] else 0,
        "avg_gen_ms": float(np.mean(stats["gen_ms"])) if stats["gen_ms"] else 0,
        "avg_act_ms": float(np.mean(stats["act_ms"])) if stats["act_ms"] else 0,
        "avg_action_mse": float(np.mean(stats["mse"])) if stats["mse"] else None,
        "median_action_mse": float(np.median(stats["mse"])) if stats["mse"] else None,
        "p90_action_mse": float(np.percentile(stats["mse"], 90)) if stats["mse"] else None,
        "avg_action_mse_baseline": (
            float(np.mean(stats["mse_baseline"])) if stats.get("mse_baseline") else None
        ),
        "median_action_mse_baseline": (
            float(np.median(stats["mse_baseline"])) if stats.get("mse_baseline") else None
        ),
        "n_frames_with_tags": sum(1 for r in results if r["gen_parsed_tags"]),
        "tag_types_found": list(set(t for r in results if r["gen_parsed_tags"] for t in r["gen_parsed_tags"])),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Evaluation complete (v2)")
    logger.info(f"  Episodes: {summary['total_episodes']}, Frames: {summary['total_frames']}, Errors: {summary['errors']}")
    logger.info(f"  Avg gen tokens: {summary['avg_gen_tokens']:.1f}, Avg gen time: {summary['avg_gen_ms']:.0f} ms")
    logger.info(f"  Avg action time: {summary['avg_act_ms']:.0f} ms")
    logger.info(f"  Action MSE  avg={summary['avg_action_mse']} median={summary['median_action_mse']}")
    if summary["avg_action_mse_baseline"] is not None:
        logger.info(f"  Baseline (no CoT) MSE  avg={summary['avg_action_mse_baseline']} median={summary['median_action_mse_baseline']}")
    logger.info(f"  Frames with tags: {summary['n_frames_with_tags']}, Types: {summary['tag_types_found']}")
    logger.info(f"  Log: {log_file}")
    logger.info(f"  Summary: {summary_file}")


if __name__ == "__main__":
    main()
