#!/usr/bin/env python3
"""generate_sft_waypoints.py — 生成 M1 脚本化轨迹的 waypoint 序列。

纯计算脚本，不连接机器人。输出 JSON 文件供 Step 3 使用。
"""
import json
import numpy as np

# ── 常量（来源见 §A.2）──
home_q  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
target_q = np.array([0.5, 0.5, 0.0, -1.2, 0.0, 1.5, 0.0], dtype=np.float64)
q_min = np.array([-4.35, -3.04, -2.26, -1.99, -2.26, -0.95, -1.47], dtype=np.float64)
q_max = np.array([ 1.21,  0.07,  2.26,  0.25,  2.26,  0.95,  1.47], dtype=np.float64)

N_WAYPOINTS = 20  # 每段的插值步数

def q_to_norm(q: np.ndarray) -> np.ndarray:
    return (2.0 * (q - q_min) / (q_max - q_min) - 1.0).astype(np.float64)

def make_segment(q_start: np.ndarray, q_end: np.ndarray, n: int):
    """返回 list of dict，每个 dict 包含 waypoint 的弧度值和归一化动作。"""
    segment = []
    for k in range(n + 1):
        alpha = k / n
        q_k = q_start + alpha * (q_end - q_start)
        a_k = q_to_norm(q_k)
        # 安全检查：归一化后必须在 [-1, 1] 内
        assert np.all(a_k >= -1.0 - 1e-6) and np.all(a_k <= 1.0 + 1e-6), \
            f"Waypoint {k} out of [-1,1]: {a_k}"
        segment.append({
            "step": k,
            "q_rad": q_k.tolist(),
            "action_norm": np.clip(a_k, -1.0, 1.0).tolist(),
        })
    return segment

# 正向 home → target（跳过 k=0 的 home 本身，从 k=1 开始才是真正的 action）
forward = make_segment(home_q, target_q, N_WAYPOINTS)
# 反向 target → home
backward = make_segment(target_q, home_q, N_WAYPOINTS)

episode = {
    "task": "Move the right arm to the target joint configuration.",
    "forward": forward,
    "backward": backward,
    "metadata": {
        "home_q": home_q.tolist(),
        "target_q": target_q.tolist(),
        "q_min": q_min.tolist(),
        "q_max": q_max.tolist(),
        "n_waypoints": N_WAYPOINTS,
        "env_step_hz": 10,
    },
}

output_path = "sft_waypoints.json"
with open(output_path, "w") as f:
    json.dump(episode, f, indent=2)
print(f"Saved {len(forward) + len(backward)} waypoints to {output_path}")
print(f"Forward segment: q change per joint = {(target_q - home_q).tolist()}")
print(f"Max per-step q change = {np.max(np.abs(target_q - home_q)) / N_WAYPOINTS:.4f} rad")