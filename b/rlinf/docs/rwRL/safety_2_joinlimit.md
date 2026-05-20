# Galaxea R1 Pro 关节角反馈与关节安全限制：深度分析

> **定位**：本文是 [safety_2.md](safety_2.md) 的补充专题，聚焦于 L2 per-joint limit 涉及的两个核心问题：
> 1. R1 Pro 的关节角反馈接口到底在哪里？为什么设计文档说"没有直接的关节角反馈"？
> 2. mobiman SDK 内部已有关节保护，RLinf 的 L2 是否多余？
>
> 阅读对象：负责真机联调的工程师、安全系统设计者。
> 版本：safety_2_joinlimit · 日期：2026-04-27

---

## 目录

1. [问题背景](#1-问题背景)
2. [R1 Pro 关节角反馈接口](#2-r1-pro-关节角反馈接口)
3. [RLinf 中关节反馈数据的完整链路](#3-rlinf-中关节反馈数据的完整链路)
4. ["没有直接的关节角反馈"的真实含义](#4-没有直接的关节角反馈的真实含义)
5. [mobiman 内部的关节保护机制](#5-mobiman-内部的关节保护机制)
6. [为什么 RLinf 仍需要 L2](#6-为什么-rlinf-仍需要-l2)
7. [L2 实现方案建议](#7-l2-实现方案建议)
8. [参考资料](#8-参考资料)

---

## 1. 问题背景

设计文档 [r1pro5op47.md](r1pro5op47.md) §6.4.2 定义了五级安全闸门，其中 L2 per-joint limit 负责关节空间的位置和速度约束。`SafetyConfig` 中已经定义了完整的参数（`r1_pro_safety.py:61-67`）：

```python
# ── L2: per-joint limits (right arm; left arm uses same) ────
arm_q_min: np.ndarray = field(default_factory=lambda: np.array(
    [-2.7, -1.8, -2.7, -3.0, -2.7, -0.1, -2.7], dtype=np.float32))
arm_q_max: np.ndarray = field(default_factory=lambda: np.array(
    [2.7, 1.8, 2.7, 0.0, 2.7, 3.7, 2.7], dtype=np.float32))
arm_qvel_max: np.ndarray = field(default_factory=lambda: np.array(
    [3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0], dtype=np.float32))
```

**7 个关节的位置极限含义**：

| 关节 | q_min (rad) | q_max (rad) | 含义 |
|------|------------|------------|------|
| J1 | -2.7 | 2.7 | 基座旋转，±155° |
| J2 | -1.8 | 1.8 | 肩俯仰，±103° |
| J3 | -2.7 | 2.7 | 上臂旋转，±155° |
| J4 | -3.0 | 0.0 | 肘关节，只能向一侧弯 |
| J5 | -2.7 | 2.7 | 前臂旋转，±155° |
| J6 | -0.1 | 3.7 | 腕关节，不对称范围 |
| J7 | -2.7 | 2.7 | 腕旋转，±155° |

**角速度极限**：近端关节（J1-J4）限 3.0 rad/s，远端关节（J5-J7）限 5.0 rad/s — 远端力臂短、惯量小，可以转得更快。

但实际实现中，L2 只有注释占位（`r1_pro_safety.py:196-199`）：

```python
# ── L2 per-joint limit ──────────────────────────────────────
# We do not have raw qpos in the action (only deltas), so L2
# acts as a guard against absurdly large velocity demands.
# Predicted next q is checked through the EE clip + step cap.
```

设计文档第 2037 行的解释是：

> 当前控制模式是笛卡尔空间（EE pose），**没有直接的关节角反馈**。

这引出两个问题：R1 Pro 真的没有关节角反馈吗？如果有，为什么 L2 没实现？

---

## 2. R1 Pro 关节角反馈接口

### 2.1 HDAS 驱动层提供完整反馈

R1 Pro 的底层驱动包叫 **HDAS**（Hardware Data Acquisition System），通过 ROS2 话题发布标准 `sensor_msgs/JointState` 消息。根据 [Galaxea 官方文档](https://docs.galaxea-dynamics.com/Guide/R1Pro/software_introduction/R1Pro_Software_Guide_ROS2/)，反馈话题包括：

| ROS2 话题 | 消息类型 | JointState 字段 | 内容 |
|-----------|---------|-----------------|------|
| `/hdas/feedback_arm_right` | `sensor_msgs/JointState` | position[0:7], velocity[0:7], effort[0:7] | 右臂 7 关节：角度(rad) / 角速度(rad/s) / 力矩(N·m) |
| `/hdas/feedback_arm_left` | `sensor_msgs/JointState` | position[0:7], velocity[0:7], effort[0:7] | 左臂 7 关节：同上 |
| `/hdas/feedback_gripper_right` | `sensor_msgs/JointState` | position[0], velocity[0] | 右夹爪行程(mm) / 速度 |
| `/hdas/feedback_gripper_left` | `sensor_msgs/JointState` | position[0], velocity[0] | 左夹爪行程(mm) / 速度 |
| `/hdas/feedback_torso` | `sensor_msgs/JointState` | position[0:4], velocity[0:4] | 4 关节躯干 |
| `/hdas/feedback_chassis` | `sensor_msgs/JointState` | position[0:3], velocity[0:3] | 3 自由度底盘 |

HDAS 驱动通过以下命令启动：

```bash
ros2 launch HDAS r1pro.py
```

启动后，上述话题以 `sensor_data` QoS（best-effort，低延迟）持续发布。

### 2.2 关节控制命令话题

下发关节目标的话题：

| ROS2 话题 | 消息类型 | 用途 |
|-----------|---------|------|
| `/motion_target/target_joint_state_arm_right` | `sensor_msgs/JointState` | 右臂关节目标位置 |
| `/motion_target/target_joint_state_arm_left` | `sensor_msgs/JointState` | 左臂关节目标位置 |
| `/motion_target/target_joint_state_torso` | `sensor_msgs/JointState` | 躯干关节目标 |
| `/motion_target/target_position_gripper_right` | `sensor_msgs/JointState` | 右夹爪目标 |
| `/motion_target/target_position_gripper_left` | `sensor_msgs/JointState` | 左夹爪目标 |

### 2.3 末端位姿控制话题（当前 RLinf 使用的模式）

RLinf 当前使用的是末端位姿控制模式，而非直接的关节控制：

| ROS2 话题 | 消息类型 | 用途 |
|-----------|---------|------|
| `/motion_target/target_pose_arm_right` | `geometry_msgs/PoseStamped` | 右臂 EE 目标位姿 [xyz + quat] |
| `/motion_target/target_pose_arm_left` | `geometry_msgs/PoseStamped` | 左臂 EE 目标位姿 [xyz + quat] |

使用此模式时，由 mobiman 内部的 Relaxed IK 解算器将笛卡尔空间目标转换为关节角命令。

**关键结论：R1 Pro 硬件层面提供了完整的关节角反馈（位置、速度、力矩），通过标准 ROS2 `sensor_msgs/JointState` 消息发布。**

---

## 3. RLinf 中关节反馈数据的完整链路

RLinf 的控制器已经订阅了所有关节反馈话题，数据在系统中的流向是完整的：

### 3.1 数据流图

```
┌──────────────────────┐
│  HDAS 驱动           │
│  /hdas/feedback_arm_* │──── sensor_msgs/JointState ────┐
│  /hdas/feedback_torso │                                │
│  /hdas/feedback_chassis│                                │
└──────────────────────┘                                 │
                                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  GalaxeaR1ProController._init_subscribers()                     │
│  (r1_pro_controller.py:286-348)                                 │
│                                                                 │
│  JointState, "/hdas/feedback_arm_right"  → _on_arm_right_feedback│
│  JointState, "/hdas/feedback_arm_left"   → _on_arm_left_feedback │
│  JointState, "/hdas/feedback_gripper_*"  → _on_gripper_*_feedback│
│  JointState, "/hdas/feedback_torso"      → _on_torso_feedback   │
│  JointState, "/hdas/feedback_chassis"    → _on_chassis_feedback │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  GalaxeaR1ProRobotState  (r1_pro_robot_state.py:80-104)        │
│                                                                 │
│  right_arm_qpos: np.ndarray (7,)   ← msg.position[:7]         │
│  right_arm_qvel: np.ndarray (7,)   ← msg.velocity[:7]         │
│  right_arm_qtau: np.ndarray (7,)   ← msg.effort[:7]           │
│  left_arm_qpos:  np.ndarray (7,)                               │
│  left_arm_qvel:  np.ndarray (7,)                               │
│  left_arm_qtau:  np.ndarray (7,)                               │
│  torso_qpos:     np.ndarray (4,)                               │
│  torso_qvel:     np.ndarray (4,)                               │
│  chassis_qpos:   np.ndarray (3,)                               │
│  chassis_qvel:   np.ndarray (3,)                               │
│                                                                 │
│  get_arm_qpos(side) → right_arm_qpos / left_arm_qpos          │
│  get_arm_qvel(side) → right_arm_qvel / left_arm_qvel          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  GalaxeaR1ProSafetySupervisor.validate(action, state, schema)  │
│  (r1_pro_safety.py:170-282)                                     │
│                                                                 │
│  state.right_arm_qpos   ← 数据已经在这里了！                      │
│  state.right_arm_qvel   ← 数据已经在这里了！                      │
│                                                                 │
│  但 L2 只有注释占位，没有使用这些数据。                              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 回调函数代码

控制器的回调函数将 ROS2 消息解析为 numpy 数组（`r1_pro_controller.py:417-487`）：

```python
def _on_arm_right_feedback(self, msg) -> None:
    self._stamp_first_seen("arm_right")
    with self._state_lock:
        if msg.position:
            self._state.right_arm_qpos = np.asarray(
                msg.position[:7], dtype=np.float32,
            )
        if msg.velocity:
            self._state.right_arm_qvel = np.asarray(
                msg.velocity[:7], dtype=np.float32,
            )
        if msg.effort:
            self._state.right_arm_qtau = np.asarray(
                msg.effort[:7], dtype=np.float32,
            )
```

左臂（`_on_arm_left_feedback`）、夹爪、躯干、底盘的回调结构完全相同。

### 3.3 已有的关节角使用场景

虽然 L2 安全检查尚未使用关节角数据，但系统中已有两处使用了它：

**1. `go_to_rest()` — 复位时的关节收敛检查**（`r1_pro_controller.py:648-667`）：

```python
def go_to_rest(self, side: str, qpos: list, timeout_s: float = 5.0) -> bool:
    self.send_arm_joints(side, qpos)
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        with self._state_lock:
            cur = self._state.get_arm_qpos(side)
        target = np.asarray(qpos, dtype=np.float32).reshape(-1)[:7]
        if cur.size and target.size and np.allclose(target, cur, atol=0.03):
            return True  # 所有关节误差 < 0.03 rad，认为到位
        time.sleep(0.05)
    return False
```

**2. `clear_errors()` — 错误恢复时发送当前关节角**（`r1_pro_controller.py:669-681`）：

```python
def clear_errors(self, side: str) -> None:
    with self._state_lock:
        cur_q = self._state.get_arm_qpos(side).tolist()
    self.send_arm_joints(side, cur_q)  # 发送当前位置让 mobiman 重新锚定
```

这两处证明关节角反馈数据是可靠且可用的。

---

## 4. "没有直接的关节角反馈"的真实含义

设计文档说 "没有直接的关节角反馈" 容易引起误解。它的含义**不是**硬件层面没有反馈，而是在 L2 安全检查的特定上下文中存在以下挑战：

### 4.1 动作空间是笛卡尔增量，不是关节角增量

RLinf 当前的策略动作空间定义为：

```
action ∈ [-1, 1]^D

经 ActionSchema.split() 分解为：
  right_xyz:     [dx, dy, dz]       × action_scale[0]  → EE 位置增量 (米)
  right_rpy:     [drx, dry, drz]    × action_scale[1]  → EE 姿态增量 (rad)
  right_gripper: [gripper]          × action_scale[2]  → 夹爪开合
```

策略输出的是**笛卡尔空间增量**，不是关节空间增量。

### 4.2 控制管线走 EE pose 模式

RLinf 通过 `send_arm_pose()` 发布 `PoseStamped` 消息到 `/motion_target/target_pose_arm_*`，由 mobiman 内部的 Relaxed IK 解算器转换为关节角命令。RLinf 本身不执行逆运动学。

### 4.3 问题的本质

L2 想做的事情是：**"如果执行这个动作，各关节的位置和速度是否会超限？"**

但从笛卡尔增量 `[dx, dy, dz, drx, dry, drz]` 到关节角增量 `[dq1, dq2, ..., dq7]` 需要 **Jacobian 逆**计算：

```
dq = J^{-1}(q) · [dx, dy, dz, drx, dry, drz]^T
```

当前实现中没有维护 Jacobian 矩阵，也没有集成 IK 库，因此无法在**动作下发前**预测动作对关节角的影响。

但关节角的**事后反馈**是完整可用的 — `state.right_arm_qpos` 在每个控制周期都被 HDAS 更新。

---

## 5. mobiman 内部的关节保护机制

### 5.1 三层内部保护

mobiman SDK 管线中存在至少三层保护：

#### 5.1.1 Relaxed IK 解算器的隐式约束

Galaxea 使用的 Relaxed IK 算法（源自 [Rakita et al., RSS 2018](https://github.com/uwgraphics/relaxed_ik)）将逆运动学建模为加权非线性优化问题。其目标函数包含多个加权项：

- **末端精度**：最小化 EE 目标位姿与实际位姿的误差
- **关节极限惩罚**：关节角接近极限时，惩罚项梯度增大，将解推回可行域
- **运动平滑性**：最小化关节角速度的抖动 (minimum-jerk)
- **自碰撞避免**：基于碰撞球的自碰撞检测

关键行为：当 EE 目标位姿要求的关节角会超限时，Relaxed IK 不会输出违反限位的解，而是**降级精度**找一个近似解 — 即 EE 不会精确到达目标位姿，但关节角保持在安全范围内。"Relaxed" 的含义就是允许末端精度被"放松"以换取关节可行性。

#### 5.1.2 JointTracker 的电机保护

Galaxea 官方文档明确说：

> "If the interface is called to follow a big angle jump, it will trigger motors' protection."

这是驱动层保护 — 当 JointTracker 检测到目标关节角与当前关节角差异过大（大角度跳变）时，会触发电机保护机制，类似急停。

#### 5.1.3 驱动器层的硬件限位

每个关节的电机驱动器内置硬件限位（mechanical end-stop + 电气限位），这是最底层的物理保护。

### 5.2 保护的局限性

尽管 mobiman 有三层保护，但它们存在重要局限：

1. **Relaxed IK 的降级是不透明的**：RLinf 无法知道 IK 是否降级了、降级了多少。策略可能以为 EE 到达了目标位置，实际上因为关节极限约束，EE 停在了一个意外的位置。

2. **电机保护是断路器式的**：一旦触发，机械臂停止运动，需要人工介入恢复。在 RL 训练中，一个 episode 中间触发电机保护意味着该 episode 中断，数据浪费。

3. **保护参数与 RL 安全需求不同**：mobiman 的关节限位对应硬件物理极限，而 RL 训练需要的安全工作区域通常比硬件极限窄得多。

---

## 6. 为什么 RLinf 仍需要 L2

### 6.1 防御纵深原则

正确的安全设计是**多层防御**（defense in depth），每层各司其职：

```
┌─────────────────────────────────────────────────────────────────┐
│ L2 (RLinf SafetySupervisor)                                     │
│ RL 安全工作区 — 最保守                                            │
│ 检查关节角是否接近极限，提前裁剪动作幅度                              │
│ 效果：平滑裁剪，不中断训练                                         │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ Relaxed IK (mobiman)                                      │   │
│  │ IK 可行域 — 较保守                                         │   │
│  │ 关节角接近极限时降级 EE 精度                                 │   │
│  │ 效果：EE 偏离目标，但关节安全                                │   │
│  │                                                           │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │ JointTracker 电机保护                                │   │   │
│  │  │ 驱动层 — 硬件级                                      │   │   │
│  │  │ 大角度跳变时触发                                      │   │   │
│  │  │ 效果：急停，需人工恢复                                 │   │   │
│  │  │                                                     │   │   │
│  │  │  ┌───────────────────────────────────────────────┐   │   │   │
│  │  │  │ 机械限位                                       │   │   │   │
│  │  │  │ 物理端点 — 最后防线                              │   │   │   │
│  │  │  │ 效果：物理碰撞，可能损坏                          │   │   │   │
│  │  │  └───────────────────────────────────────────────┘   │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 三个关键差异

#### 差异一：保护粒度 — 事前预防 vs 事后补救

```
RLinf L2 的意图:
  策略输出 action → 检查当前关节角 → 发现接近极限 → 裁剪 action → 发送安全的目标
                     ↑ 在这里拦截，平滑裁剪

mobiman 的保护:
  收到目标 → IK 求解 → 解可能不精确 → 发给电机 → 电机保护触发 → 急停
                                                    ↑ 在这里才拦截，代价高
```

L2 在动作下发前检查，能平滑裁剪动作幅度，RL 训练不受中断。mobiman 的保护在动作执行阶段或之后才触发，代价更高。

#### 差异二：训练连续性

JointTracker 的电机保护一旦触发，机械臂**停止运动**（类似断路器跳闸）。在 RL 训练中：
- 一个 episode 中间突然急停 → 需要人工介入重启
- 该 episode 的数据被浪费
- 训练流程中断，影响数据收集效率

RLinf 的 L2 应该像 L3a 那样做**平滑裁剪**（rewrite action），让训练在安全边界内继续运行，而不是触发底层急停。

#### 差异三：安全边界不同

- **硬件极限**：机械结构允许的最大范围（由 HDAS/电机驱动器强制执行）
- **mobiman 极限**：IK 解算器的约束范围（可能比硬件稍窄）
- **RL 安全极限**：考虑到策略探索的安全余量，应该比上面两者都窄

`SafetyConfig` 中的 `arm_q_min/arm_q_max` 代表的是 **RL 训练中的安全工作区域**。例如，硬件 J4 的极限可能是 [-3.14, 0.0]，但 RL 训练可能只需要 [-2.5, -0.3] 的子区间。这种缩窄只有 RLinf 自己能做，mobiman 不了解 RL 训练的上下文。

### 6.3 类比

mobiman 的内部保护是**安全网**（safety net），RLinf 的 L2 应该是**安全带**（seatbelt）— 你不希望依赖安全网才停下来，而是通过安全带在一开始就避免坠落。

---

## 7. L2 实现方案建议

### 7.1 方案对比

| 方案 | 思路 | 优点 | 缺点 | 复杂度 |
|------|------|------|------|--------|
| **A. 事后校验** | 每步读取 `state.arm_qpos`，接近极限时裁剪下一步的动作幅度 | 简单，数据已可用，无额外依赖 | 有一帧延迟（裁剪的是下一步，当前步已执行） | 低 |
| **B. 逆运动学预测** | 调用 IK 将预测的 EE 目标转换为关节角目标，再与极限比较 | 事前精确预测 | 需要 URDF + IK 库（pinocchio / relaxed_ik），每步多一次 IK 求解 | 高 |
| **C. Jacobian 近似** | 用当前关节角的 Jacobian 逆估算 dq = J⁻¹ dxyz | 较快的事前估算 | 需要维护 Jacobian，近似精度有限，奇异点附近不稳定 | 中 |

### 7.2 推荐方案 A — 事后校验

方案 A 最实际 — 关节角反馈数据已经存在于 `GalaxeaR1ProRobotState` 中，只需在 `validate()` 方法的 L2 位置添加检查逻辑。

#### 7.2.1 设计思路

```
每个控制步:
  1. 获取当前关节角 state.get_arm_qpos(side)
  2. 计算各关节到极限的距离 margin = min(q - q_min, q_max - q)
  3. 如果 margin < warning_threshold:
       按 margin/warning_threshold 的比例缩小 action 幅度
  4. 如果 margin < critical_threshold:
       将该臂动作置零（soft_hold）
  5. 记录 reason 和 metrics
```

#### 7.2.2 代码示意

在 `r1_pro_safety.py` 的 `validate()` 方法中，替换 L2 注释占位：

```python
# ── L2 per-joint limit ──────────────────────────────────────
for side in ("right", "left"):
    if side == "right" and not action_schema.has_right_arm:
        continue
    if side == "left" and not action_schema.has_left_arm:
        continue

    qpos = state.get_arm_qpos(side)
    qvel = state.get_arm_qvel(side)

    # L2a: 关节位置接近极限时缩小动作幅度
    margin_low = qpos - self._cfg.arm_q_min
    margin_high = self._cfg.arm_q_max - qpos
    margin = np.minimum(margin_low, margin_high)
    min_margin = float(np.min(margin))

    if min_margin < self._cfg.l2_critical_margin:
        # 关节已经非常接近极限，冻结该臂动作
        info.safe_action = self._zero_arm_action(
            info.safe_action, side, action_schema,
        )
        info.soft_hold = True
        info.reason.append(f"L2:{side}_qpos_critical")
    elif min_margin < self._cfg.l2_warning_margin:
        # 接近极限，按比例缩小动作幅度
        scale = min_margin / self._cfg.l2_warning_margin
        info.safe_action = self._scale_arm_action(
            info.safe_action, side, action_schema, scale,
        )
        info.reason.append(f"L2:{side}_qpos_warning")

    # L2b: 关节角速度超限检查
    qvel_excess = np.abs(qvel) - self._cfg.arm_qvel_max
    if np.any(qvel_excess > 0):
        info.safe_action = self._scale_arm_action(
            info.safe_action, side, action_schema, 0.5,
        )
        info.reason.append(f"L2:{side}_qvel_excess")
```

#### 7.2.3 需要新增的 SafetyConfig 参数

```python
# ── L2: margin thresholds ────
l2_warning_margin: float = 0.15    # rad，约 8.6°，开始缩小动作
l2_critical_margin: float = 0.05   # rad，约 2.9°，冻结该臂动作
```

#### 7.2.4 一帧延迟的影响评估

方案 A 的裁剪基于**上一帧的关节角**，对当前帧的动作进行调整。这意味着存在一帧延迟 — 如果某一帧动作将关节推到了极限附近，要到下一帧才会触发裁剪。

在 R1 Pro 的 10Hz 控制频率下（100ms/步），一帧延迟 = 100ms。考虑到：
- `action_scale[0]` 默认 0.05m/步，即使满幅动作在 100ms 内的位移也只有 5cm
- 关节极限的 `warning_margin` 设为 0.15 rad（约 8.6°），有足够的缓冲区
- L3a 的 TCP 安全框提供了笛卡尔空间的事前保护

一帧延迟在实际操作中是可接受的。`warning_margin` 的设定正是为了提供这个缓冲。

### 7.3 方案 B — 逆运动学预测（长期方案）

如果需要更精确的事前保护，可以引入 IK 库（如 pinocchio）：

```python
# 概念性代码
import pinocchio as pin

# 加载 R1 Pro URDF
model = pin.buildModelFromUrdf("galaxea_r1_pro.urdf")
data = model.createData()

def predict_joint_target(current_qpos, ee_target_pose):
    """用 IK 预测目标关节角"""
    q_target = pin.computeFrameJacobian(...)  # 简化表示
    return q_target

# 在 L2 中使用
q_target = predict_joint_target(state.get_arm_qpos(side), ee_target)
if np.any(q_target < arm_q_min) or np.any(q_target > arm_q_max):
    # 裁剪或拒绝
```

此方案的额外依赖（pinocchio + URDF）和计算开销使其更适合作为长期演进方向。

---

## 8. 参考资料

### 8.1 Galaxea 官方文档

- [R1 Pro Software Guide ROS2](https://docs.galaxea-dynamics.com/Guide/R1Pro/software_introduction/R1Pro_Software_Guide_ROS2/) — 完整的 ROS2 话题参考、mobiman 架构图
- [Galaxea ATC ROS2 SDK V2.1.4 Changelog](https://docs.galaxea-dynamics.com/Guide/sdk_change_log/ros2/v2.1.4/) — 最新 SDK 更新，含躯干电机校验、夹爪速度接口
- [R1 Pro 软件介绍 (中文)](https://docs.galaxea-ai.com/zh/Guide/R1Pro/software_introduction/ros2/R1Pro_Software_Introduction_ROS2/)

### 8.2 Relaxed IK 论文与代码

- Rakita, D., Mutlu, B., & Gleicher, M. (2018). [RelaxedIK: Real-time Synthesis of Accurate and Feasible Robot Arm Motion.](https://www.roboticsproceedings.org/rss14/p43.pdf) *RSS 2018.*
- Rakita, D., Mutlu, B., & Gleicher, M. (2020). [An analysis of RelaxedIK.](https://link.springer.com/article/10.1007/s10514-020-09918-9) *Autonomous Robots, 44, 1341–1358.*
- GitHub: [uwgraphics/relaxed_ik](https://github.com/uwgraphics/relaxed_ik) — Rust 核心实现，含 ROS/MuJoCo/Unity 封装

### 8.3 RLinf 源码参考

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `r1_pro_safety.py` | 61-67 | `SafetyConfig` 中 L2 参数定义 |
| `r1_pro_safety.py` | 196-199 | L2 注释占位 |
| `r1_pro_controller.py` | 286-348 | 关节反馈话题订阅 |
| `r1_pro_controller.py` | 417-487 | 关节反馈回调函数 |
| `r1_pro_controller.py` | 588-600 | `send_arm_joints()` 关节控制 |
| `r1_pro_controller.py` | 648-667 | `go_to_rest()` 关节收敛检查 |
| `r1_pro_robot_state.py` | 80-104 | 关节状态字段定义 |
| `r1_pro_robot_state.py` | 146-150 | `get_arm_qpos()` / `get_arm_qvel()` |
| `r1_pro_action_schema.py` | 119-141 | `predict_arm_ee_pose()` EE 预测 |

### 8.4 相关设计文档

- [r1pro5op47.md](r1pro5op47.md) §6.4.2 — 五级闸门设计，L2 定义
- [safety_2.md](safety_2.md) §5.2 / §9 — L2 实现状态分析与差距评估
