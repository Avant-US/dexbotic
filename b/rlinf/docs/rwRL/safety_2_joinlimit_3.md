# Galaxea R1 Pro L2 关节限位:基于真机 URDF 的位置+速度安全方案设计(v3 — 夹爪归一化语义修正)

> **定位**:本文是 [`safety_2_joinlimit_2.md`](safety_2_joinlimit_2.md) 的**改良版**。两版差异集中在**夹爪表示**:v2 用了 `[0, 100]` 的"百分比"作为 L2 内部表示,但 R1 Pro 真机 SDK 用的是 **mm 的 stroke**,而 BRS-ctrl 与多数策略生态(Pi0.5、Behavior)在归一化空间用 `[0, 1]`。**v3 把 L2 的工作语言统一到 `[0, 1]`,并显式钉死"0=fully closed, 1=fully open"的 R1 Pro 工程语义**,在文档与代码层面同时消除"百分比 vs mm vs 归一化"三层混乱。
>
> 设计的其它部分(L2a/L2b/L2c/L2d、URDF-derived 限位、左右臂 J2 镜像、conftest stub)与 v2 完全一致——本文是"夹爪一节 + 相关字段命名 + 单测数据 + 附录"四点的精修版,其它章节保持不动以减少 review diff。
>
> **阅读对象**:负责真机联调的工程师、`r1_pro_safety.py` 的 maintainer、`r1_pro_env.py` 的接入方,以及任何打算把 RLinf 的 RL policy 接到 R1 Pro 真机 G1 夹爪上的人。
>
> **版本**:safety_2_joinlimit_3 · 日期:2026-05-05 · 作者:RLinf RWRL Working Group
>
> **状态**:**v2 已落地**(代码、测试、验证清单见下);**v3 仅作字段命名/语义精修**,代码层是"重命名 + 数值范围切换"的微改动。
> - 代码:[`r1_pro_safety.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py)、[`r1_pro_action_schema.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_action_schema.py)、[`r1_pro_robot_state.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_robot_state.py)。
> - 测试:[`tests/unit_tests/test_galaxea_r1_pro_safety_l2.py`](../../../tests/unit_tests/test_galaxea_r1_pro_safety_l2.py)(16 个用例)+ [`tests/unit_tests/conftest.py`](../../../tests/unit_tests/conftest.py)(Orin 最小环境 stub)。
> - 验证:64/64 Galaxea 单测在 Orin 真机上通过。L2 单测三轮稳定 16/16。
> - 实施期发现:**Zero-state guard、Cartesian no-op 语义、gripper rate+pos 双重顺序、L1 clip 与 scale 的相互作用** —— 详见 §14 与 [附录 C](#附录-c实施期发现implementation-findings)。
> - **v3 新增**:[§3.6 R1 Pro G1 夹爪三层语义对照表 + 真机 mm 实测验证 + 与 BRS-ctrl 反向语义说明](#36-2-个夹爪限位归一化-0-1并显式钉死开-闭语义)、[§7.6 重写为 \[0, 1\] 归一化 L2 检查](#76-夹爪的特殊处理v3-归一化语义)、[附录 C.6 与 v2 的 migration 笔记](#c6-v2-到-v3-的迁移笔记migration-notes)。

---

## 目录

1. [TL;DR — 三句话答案](#1-tldr--三句话答案)
2. [设计前提:已经验证过的事实](#2-设计前提已经验证过的事实)
3. [限位常量:7×2 关节位置 + 7 关节速度 + 4 躯干 + 3 底盘 + 2 夹爪](#3-限位常量72-关节位置--7-关节速度--4-躯干--3-底盘--2-夹爪)
4. [L2 子检查的语义分层 (L2a/L2b/L2c/L2d)](#4-l2-子检查的语义分层-l2al2bl2cl2d)
5. [模块结构与调用关系](#5-模块结构与调用关系)
6. [数据流:从 ROS2 反馈到 SafetyInfo](#6-数据流从-ros2-反馈到-safetyinfo)
7. [关键代码详解](#7-关键代码详解)
8. [对其它代码的最小侵入式改动清单](#8-对其它代码的最小侵入式改动清单)
9. [配置层:YAML 字段与 SafetyConfig 字段对照](#9-配置层yaml-字段与-safetyconfig-字段对照)
10. [测试矩阵与单元测试设计](#10-测试矩阵与单元测试设计)
11. [运行时调试 / 排错](#11-运行时调试--排错)
12. [与生产代码 (BRS-ctrl) 的关系与差异](#12-与生产代码-brs-ctrl-的关系与差异)
13. [参考资料 + 源码锚点](#13-参考资料--源码锚点)
14. [实施记录(Implementation Log)](#14-实施记录implementation-log)
- [附录 A:与现有 L4 的并存设计](#附录-a与现有-l4-的并存设计)
- [附录 B:0.1 rad 余量的工程论证](#附录-b01-rad-余量的工程论证)
- [附录 C:实施期发现(Implementation Findings)](#附录-c实施期发现implementation-findings)

---

## 1. TL;DR — 三句话答案

1. 把 RLinf 现行 [`SafetyConfig.arm_q_min/max/qvel_max`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py:62-67) 替换为**逐位对齐 R1 Pro URDF**(包括左臂 J2 镜像)、再向内收 **0.1 rad** 的"工作域"——**不要直接用 URDF 真值**,因为 URDF 是机械极限,L2 安全域必须比它内缩(参见 [`safety_2_joinlimit.md`](safety_2_joinlimit.md) §6.3 的"安全网 vs 安全带"类比)。本文给出 7+7+4+3+2 全部 **23 维约束**的具体数值。
2. **L2 拆成 4 个子检查 (L2a/L2b/L2c/L2d)**,放在 [`r1_pro_safety.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py) 现有 `validate()` 管线的 L1 与 L3a 之间,顺序为:**预测目标 → 软裁减(scale shrink) → 硬截断(hard clip) → 反馈速度看门狗 → 命令速度看门狗**。所有数据已经在 `state.right_arm_qpos / qvel`、`state.left_arm_qpos / qvel`、`state.torso_qpos / qvel`、`state.chassis_qpos / qvel` 中,**无需新增 ROS2 订阅或 IK 库依赖**。
3. **夹爪当作"特殊 1D 关节"处理,内部用 `[0, 1]` 归一化空间**:**0 = fully closed, 1 = fully open**(R1 Pro G1 工程约定,与 RLinf 现有 `_dispatch_action` 中 `(a+1)*50` 的语义一致;**注意与 BRS-ctrl 的 `act(0..1)` 是反向的** —— BRS 把 0 当 open、1 当 closed,L2 文档里在 §3.6 / 附录 C.6 都明示了此差异)。位置 clamp 到 `[0, 1]`,速度通过 step rate-limit 实现,**不引入 0/1 离散化**(与 [`Pi05_ActionSpace_Analysis.md`](Pi05_ActionSpace_Analysis.md) §5 的"夹爪一律连续浮点"原则一致)。底层 ROS 拓扑仍用 mm(R1 Pro `/motion_target/target_position_gripper_*` JointState.position 的工厂单位),换算系数集中在 `gripper_open_stroke_mm` / `gripper_closed_stroke_mm` 两个字段里,**L2 supervisor 永远不直接见 mm**。

---

## 2. 设计前提:已经验证过的事实

设计写到这里之前,**已经被多源交叉验证过**的事实如下。后续设计直接以它们为公理,不再重复论证。

### 2.1 R1 Pro 的关节反馈链路是完整的(参见 [`safety_2_joinlimit.md`](safety_2_joinlimit.md) §2、§3)

| 部件 | 反馈话题 | 类型 | RLinf 字段 |
|---|---|---|---|
| 右臂 7-DoF | `/hdas/feedback_arm_right` | `sensor_msgs/JointState` | `state.right_arm_qpos[7]` / `qvel[7]` / `qtau[7]` |
| 左臂 7-DoF | `/hdas/feedback_arm_left` | `sensor_msgs/JointState` | `state.left_arm_qpos[7]` / `qvel[7]` / `qtau[7]` |
| 右夹爪 | `/hdas/feedback_gripper_right` | `sensor_msgs/JointState` | `state.right_gripper_pos` / `vel` |
| 左夹爪 | `/hdas/feedback_gripper_left` | `sensor_msgs/JointState` | `state.left_gripper_pos` / `vel` |
| 躯干 4-DoF | `/hdas/feedback_torso` | `sensor_msgs/JointState` | `state.torso_qpos[4]` / `qvel[4]` |
| 底盘 3-DoF | `/hdas/feedback_chassis` | `sensor_msgs/JointState` | `state.chassis_qpos[3]` / `qvel[3]` |

回调函数([`r1_pro_controller.py:417-487`](../../../rlinf/envs/realworld/galaxear/r1_pro_controller.py#L417-L487))已经把每个话题写进 `GalaxeaR1ProRobotState`,**不需要任何新增订阅**。

### 2.2 URDF 限位已经核对过两份独立文件(参见 [`glx/mismatch_realworld_1.md`](glx/mismatch_realworld_1.md) §C.1.附 复核)

- `mobiman/lib/mobiman/configs/urdfs/r1_pro_floating_right.urdf`(`relaxed_ik` 用)
- `mobiman/share/mobiman/urdf/R1_PRO/urdf/r1_pro.urdf`(整机 `robot_state_publisher` 用)
- `mobiman/lib/mobiman/configs/urdfs/r1_pro_floating_left.urdf`(左臂 IK)

三份 URDF 中右臂 7 个 `<limit lower=... upper=... velocity=...>` 字段**完全一致**,左臂则只是 J2 镜像,J1 和 J3-J7 与右臂相同。这点在 [`/home/nvidia/kaizhe_ws/data_collect/brs-ctrl/brs_ctrl/robot_interface/interfaces.py`](file:///home/nvidia/kaizhe_ws/data_collect/brs-ctrl/brs_ctrl/robot_interface/interfaces.py) 的 `R1ProInterface` 类(行 723-737)的硬编码限位中也得到独立确认——这是工厂流出的、跑过真机数据采集的生产代码。

### 2.3 mobiman 内部已有保护,但 RLinf 仍需要 L2(参见 [`safety_2_joinlimit.md`](safety_2_joinlimit.md) §5、§6)

mobiman 的"Relaxed IK 解算器隐式约束 + JointTracker 电机保护 + 驱动器硬限位"是**安全网 (safety net)**,会在违规时**触发急停或降级精度**;RLinf 的 L2 是**安全带 (seatbelt)**,职责是在动作下发前**平滑裁剪/缩放**让训练不中断。两者目标不同,不可互相替代。

### 2.4 数据流上 L2 跑在哪里?

```
                           ┌─────────────────────────────┐
                           │ r1_pro_env.step()           │
                           │   ① state = ctrl.get_state()│
                           │   ② sinfo = safety.validate(│ ← L2 在这里
                           │       action, state, schema)│
                           │   ③ if not e_stop:          │
                           │       _dispatch_action(...) │
                           └─────────────────────────────┘
```

`validate()` 内部当前 L2 是空注释([`r1_pro_safety.py:196-199`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py#L196-L199)):

```python
# ── L2 per-joint limit ──────────────────────────────────────
# We do not have raw qpos in the action (only deltas), so L2
# acts as a guard against absurdly large velocity demands.
# Predicted next q is checked through the EE clip + step cap.
```

本设计把它**填实**,但不改动 L2 在管线中的位置(L1 之后、L3a 之前)。

### 2.5 RLinf 当前 SafetyConfig 默认值的问题(参见 [`glx/mismatch_realworld_1.md`](glx/mismatch_realworld_1.md) §C.1)

| Joint | URDF lower | URDF upper | RLinf `arm_q_min` | RLinf `arm_q_max` | 问题 |
|---|---|---|---|---|---|
| J1 | -4.4506 | 1.3090 | -2.7 | 2.7 | 上限**比 URDF 还宽** 1.39 rad |
| J2 | -3.1416 | 0.1745 | -1.8 | 1.8 | 上限**比 URDF 还宽** 1.63 rad |
| J3 | -2.3562 | 2.3562 | -2.7 | 2.7 | 双向比 URDF 宽 |
| J4 | -2.0944 | 0.3491 | -3.0 | 0.0 | 下限比 URDF 宽,上限严 |
| J5 | -2.3562 | 2.3562 | -2.7 | 2.7 | 双向比 URDF 宽 |
| J6 | -1.0472 | 1.0472 | -0.1 | 3.7 | 上限**比 URDF 还宽** 2.65 rad |
| J7 | -1.5708 | 1.5708 | -2.7 | 2.7 | 双向比 URDF 宽 |

并且**左臂用了和右臂一模一样的限位**——但 R1 Pro 左臂 J2 是镜像(URDF: `[-0.1745, +3.1416]`),会让左臂的"前向工作区"被错误地裁剪到右臂方向。

---

## 3. 限位常量:7×2 关节位置 + 7 关节速度 + 4 躯干 + 3 底盘 + 2 夹爪

### 3.1 设计公式

```
# 位置:URDF 真值向内缩 0.1 rad
q_min[i] = URDF_lower[i] + 0.1
q_max[i] = URDF_upper[i] - 0.1
# 速度:URDF 机械上限 × 0.5 (R1 Pro 出厂建议) 与 BRS-ctrl 实测保守值取较严
qvel_max[i] = min(0.5 * URDF_velocity[i], BRS_ctrl_qvel_max[i])
```

**0.1 rad ≈ 5.73°**:对一个最高速 10.47 rad/s 的关节(J5-J7),相当于约 **9.5 ms 的 overshoot 缓冲**,在 R1 Pro 50 Hz feedback / 10 Hz step 控制频率下足以让"下一帧 L2 检查"赶上,而不会先撞到 URDF 极限触发 mobiman 电机保护。

**BRS-ctrl 实测速度 [1.6,1.6,1.6,1.6,4.0,4.0,4.0]**:来自 [`brs_ctrl/reset.py:87`](file:///home/nvidia/kaizhe_ws/data_collect/brs-ctrl/brs_ctrl/reset.py),是 BRS 在 R1 Pro 上跑过实机数据采集的"reset 阶段速度限"。生产代码已经把它当作"放心可用、不会跳齿、不会撞 mobiman watchdog"的参考点。

### 3.2 7-DoF 右臂限位

| Joint | URDF lower | URDF upper | URDF vmax | **L2 q_min** | **L2 q_max** | **L2 qvel_max** |
|---|---|---|---|---|---|---|
| J1 (基座旋转) | -4.4506 | 1.3090 | 7.1209 | **-4.35** | **1.21** | **1.6** |
| J2 (肩俯仰) | -3.1416 | 0.1745 | 7.1209 | **-3.04** | **0.07** | **1.6** |
| J3 (上臂旋转) | -2.3562 | 2.3562 | 8.3776 | **-2.26** | **2.26** | **1.6** |
| J4 (肘) | -2.0944 | 0.3491 | 8.3776 | **-1.99** | **0.25** | **1.6** |
| J5 (前臂旋转) | -2.3562 | 2.3562 | 10.4720 | **-2.26** | **2.26** | **4.0** |
| J6 (腕俯仰) | -1.0472 | 1.0472 | 10.4720 | **-0.95** | **0.95** | **4.0** |
| J7 (腕旋转) | -1.5708 | 1.5708 | 10.4720 | **-1.47** | **1.47** | **4.0** |

### 3.3 7-DoF 左臂限位(注意 J2 镜像)

| Joint | URDF lower | URDF upper | **L2 q_min** | **L2 q_max** | **L2 qvel_max** |
|---|---|---|---|---|---|
| J1 | -4.4506 | 1.3090 | **-4.35** | **1.21** | **1.6** |
| **J2 (镜像)** | **-0.1745** | **3.1416** | **-0.07** | **3.04** | **1.6** |
| J3 | -2.3562 | 2.3562 | **-2.26** | **2.26** | **1.6** |
| J4 | -2.0944 | 0.3491 | **-1.99** | **0.25** | **1.6** |
| J5 | -2.3562 | 2.3562 | **-2.26** | **2.26** | **4.0** |
| J6 | -1.0472 | 1.0472 | **-0.95** | **0.95** | **4.0** |
| J7 | -1.5708 | 1.5708 | **-1.47** | **1.47** | **4.0** |

> **关键差异:左右臂 J2 不是同一区间**。当前 RLinf 用同一份 `arm_q_min/max` 给两臂,会让左臂的"前向举臂"被卡死。本设计要求**SafetyConfig 拆成 `right_arm_q_*` 与 `left_arm_q_*` 两份**,见 §9。

### 3.4 4-DoF 躯干限位(沿用 BRS-ctrl 真值,留 0.1 rad 余量)

来源 BRS-ctrl `R1ProInterface.torso_joint_high/low`(行 724-725),向内缩 0.1 rad:

| Joint | BRS lower | BRS upper | **L2 q_min** | **L2 q_max** | **L2 qvel_max** |
|---|---|---|---|---|---|
| T1 | -1.1345 | 1.8326 | **-1.03** | **1.73** | **0.5** |
| T2 | -2.7925 | 2.5307 | **-2.69** | **2.43** | **0.5** |
| T3 | -1.8326 | 1.5708 | **-1.73** | **1.47** | **0.5** |
| T4 | -3.0543 | 3.0543 | **-2.95** | **2.95** | **0.5** |

`qvel_max = 0.5 rad/s`:躯干前 4 DoF 包含一个升降 Z(T2 实际是俯仰参与升降)和 3 个旋转,本质是承重轴,生产实际不需要超过 0.5 rad/s。RLinf 现行 [`SafetyConfig`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py#L87-L90) 把 `torso_v_x_max=0.10, torso_v_z_max=0.10, torso_w_pitch_max=0.30, torso_w_yaw_max=0.30` 设的是 **EE-frame 笛卡尔速度**,与本节"关节速度"是不同维度;本设计**两套同时存在**,L2c 用关节速度、L4 保留笛卡尔速度,互不冲突。

### 3.5 3-DoF 底盘限位(沿用 BRS-ctrl 实测命令上限)

R1 Pro 底盘是全向 3 轮,L2 检查的"位置"不适用(底盘没有关节位置概念,只有 odom),所以**底盘只有速度限位**:

| 维度 | BRS-ctrl | RLinf 当前 | **L2 chassis_v_max** |
|---|---|---|---|
| v_x (前后, m/s) | 0.30 | 0.60 | **0.30** |
| v_y (左右, m/s) | 0.30 | 0.60 | **0.30** |
| ω_z (旋转, rad/s) | 0.40 | 1.50 | **0.40** |

注:BRS-ctrl 还有一个 dead-zone 阈值 `[0.01, 0.01, 0.05]`(行 756),小于此值的命令置零(避免轮子嗡嗡颤动)。本设计 L2d 沿用此 dead-zone。

### 3.6 2 个夹爪限位(归一化 [0, 1],并显式钉死开/闭语义)

> **v3 改良点**。v2 把夹爪的 L2 工作语言写成 "0-100 %",并隐式假设 `JointState.position[0]` 单位是百分比——这两点都不符合 R1 Pro 真机 SDK 的实际情况。本节用 **[0, 1] 归一化区间** 重写,并把"哪头是开/哪头是闭"直接写到字段名和文档里,杜绝歧义。

#### 3.6.1 R1 Pro G1 夹爪的三层语义对照

R1 Pro 上的"夹爪开度"在系统中**同时存在三种表示**,对外接、SDK、L2 都不可绕开:

| 层 | 单位/范围 | 字段 / topic | 语义("开"是几?) | 来源/锚点 |
|---|---|---|---|---|
| ① **ROS topic 层(SDK 真值)** | `mm`(stroke) | `/motion_target/target_position_gripper_*` 的 `JointState.position[0]`;反馈 `/hdas/feedback_gripper_*` 同字段 | **stroke 越大 → 张开越大**。R1 Pro G1 工厂物理范围 `[0, 100] mm`,工厂建议工作区 `[10, 90] mm` | BRS-ctrl `GalaxeaR1ProGripper` 默认 `close_stroke=10, open_stroke=90`(行 124-125);Orin 真机实测 `position = 3.501 mm`(此刻几乎闭合) |
| ② **RLinf normalised pos / action(L2 工作语言)** | `[0, 1]` 浮点(state)、`[-1, 1]` 浮点(action) | `state.right_gripper_pos_norm` / action[6] | **0 = fully closed, 1 = fully open**(state);**a = -1 = closed, a = +1 = open**(action 经 `(a+1)/2` 归一) | RLinf [`r1_pro_env.py:491`](../../../rlinf/envs/realworld/galaxear/r1_pro_env.py#L491) 当前 dispatch 是 `pct = (a+1)*50` 然后送 mm —— 与"a=+1 → open"等价。**v3 把这套语义提升到正式字段命名层** |
| ③ **BRS-ctrl normalised action(对照系)** | `[0, 1]` 浮点 | `R1ProBaseGripper.act(action)` | **0 = fully open, 1 = fully closed** ⚠️ 与 RLinf 反向 | BRS [`grippers/base.py:33`](file:///home/nvidia/kaizhe_ws/data_collect/brs-ctrl/brs_ctrl/robot_interface/grippers/base.py#L33) docstring;[`galaxea_g1.py:175-183`](file:///home/nvidia/kaizhe_ws/data_collect/brs-ctrl/brs_ctrl/robot_interface/grippers/galaxea_g1.py#L175-L183) 实现 |

#### 3.6.2 三层语义之间的映射公式

```
ROS mm (①)  ←→  RLinf normalised pos ∈ [0, 1] (②)  ←→  RLinf action ∈ [-1, 1] (②)  ⇄  BRS action ∈ [0, 1] (③)
                       ↑                ↓                       ↑               ↓
              pos_norm = (mm - close_mm) / (open_mm - close_mm)
              mm        = close_mm + pos_norm * (open_mm - close_mm)
              pos_norm  = (a + 1) / 2
              a         = 2 * pos_norm - 1
              brs_a     = 1 - rlinf_pos_norm     # 反向!
              rlinf_pos_norm = 1 - brs_a
```

`close_mm` / `open_mm` 由 SafetyConfig 的两个新字段提供(默认 `0.0` / `100.0`,以匹配 RLinf [`r1_pro_env.py:491`](../../../rlinf/envs/realworld/galaxear/r1_pro_env.py#L491) 现状,不破坏已有 RL policy 的表现;**也支持改成 `10.0` / `90.0`** 以与 BRS 的 stroke 工作区对齐):

```python
# In SafetyConfig (see §3.7):
gripper_closed_stroke_mm: float = 0.0    # mm value at fully closed
gripper_open_stroke_mm:   float = 100.0  # mm value at fully open
```

#### 3.6.2.5 三层语义的 mermaid 映射图

```mermaid
flowchart LR
    subgraph Policy["Policy / RL action layer"]
        A_RLINF["RLinf action<br/>a ∈ [-1, 1]<br/>-1=closed · +1=open"]
        A_BRS["BRS-ctrl action<br/>brs_a ∈ [0, 1]<br/>0=open · 1=closed ⚠️ 反向"]
    end
    subgraph L2["L2 supervisor working space (v3)"]
        N["pos_norm ∈ [0, 1]<br/>0=closed · 1=open<br/>(state side)"]
    end
    subgraph SDK["ROS / SDK layer (Galaxea HDAS)"]
        MM["stroke ∈ [0, 100] mm<br/>(JointState.position[0])<br/>越大越开<br/>工厂建议工作区 [10, 90] mm"]
    end

    A_RLINF -->|"(a+1)/2"| N
    N -->|"2*pos-1"| A_RLINF

    A_BRS -->|"1 - brs_a"| N
    N -->|"1 - pos"| A_BRS

    N -->|"closed_mm + pos*(open_mm-closed_mm)"| MM
    MM -->|"(mm - closed_mm) / (open_mm - closed_mm)"| N

    classDef rlinfStyle fill:#cce
    classDef brsStyle fill:#fec
    classDef sdkStyle fill:#cec
    class A_RLINF rlinfStyle
    class N rlinfStyle
    class A_BRS brsStyle
    class MM sdkStyle
```

要点:

- **RLinf 与 SDK 同向**(数值大 = 张开),所以 v3 的 `(a+1)/2 → pos_norm → mm` 一条线下来,中间没有"反向"步骤;policy 输出大 → 张开,直觉一致。
- **BRS 是相反**:`brs_a` 大 = 闭合,所以从 BRS 来的 action 进入 RLinf 时必须 `pos_norm = 1 - brs_a`(或等价的 `rlinf_a = -brs_a`);从 RLinf 输出到 BRS 时反向。
- L2 supervisor 工作在中间那个 [0, 1] 节点,**它从不见 mm**,也从不区分 RLinf vs BRS 的 [-1,1] / [0,1] 风格——这两层换算都在 dispatch / 适配层完成。

#### 3.6.3 R1 Pro G1 物理小贴士(给现场调参的人)

- **夹爪不是 0/1 开关**:G1 是力控夹爪,可以稳定停在任意 stroke,**抓薄物 / 易碎物**靠的是 0.3-0.7 这种中间值,不是阈值化(对应 [`Pi05_ActionSpace_Analysis.md`](Pi05_ActionSpace_Analysis.md) §5 的"连续夹爪不阈值化"原则)。
- **0 mm 是物理硬上限**:实测 `feedback.position` 约 0.5-3.5 mm 即视为"完全闭合状态";继续往负值送会被 SDK clip 到 0(不会损坏机械结构,但会让电机持续上力,长时间高电流是隐患)。
- **100 mm 不是必须的最大值**:工厂建议日常用 [10, 90],100 mm 这一档是出厂标定的极限。把 `gripper_open_stroke_mm` 设到 90 而不是 100,可以多留 10 mm 缓冲,降低撞击衬套的概率。
- **左右夹爪是独立硬件**:`gripper_closed_stroke_mm / gripper_open_stroke_mm` 两个字段会同时作用于左右两手——这是合理简化(出厂的同型号 G1 两侧机械参数一致),但若现场某只手被换过维修衬套,可以在 YAML 里继续拆分(`right_gripper_open_stroke_mm` 等)——这是未来扩展点,**v3 不引入**。

#### 3.6.4 L2 限位规则(v3 归一化版)

| 维度 | 物理量 | **L2 限位** | 说明 |
|---|---|---|---|
| `right_gripper_pos` (state-side, normalised) | [0, 1] | **`[gripper_pos_min=0.0, gripper_pos_max=1.0]`** | 0=closed, 1=open,硬截断 |
| `left_gripper_pos` (state-side, normalised) | [0, 1] | **`[0.0, 1.0]`** | 同上 |
| 单步最大变化 (`max_gripper_step`) | 归一化单位 / step | **0.30** | 默认 10 Hz 控制 → 3.0 unit/s,折合 3.0×100 = 300 mm/s 的 stroke 速度上限,经验值 |
| stroke 换算 (`gripper_closed_stroke_mm` / `gripper_open_stroke_mm`) | mm | **`0.0` / `100.0`** | RLinf 现状对齐;现场可调到 BRS 推荐的 `10.0` / `90.0` |

**为什么不引入 0/1 离散**:[`Pi05_ActionSpace_Analysis.md`](Pi05_ActionSpace_Analysis.md) §5 已经论证 Pi0.5 / Behavior R1Pro 的夹爪输出**一律是连续浮点**;离散化是 wrapper 层的事,不应该把它放进安全监督器。L2 只做"clamp 到合法 normalised 区间 + 限制单步变化",不做语义离散化。

**RLinf vs BRS 反向语义的协调**:如果将来某个上游 policy(例如直接 fine-tune 自 BRS 数据集的 imitation learning baseline)输出的是 BRS 约定(0=open),需要在 wrapper 层做 `rlinf_a = -brs_a`(action ∈ [-1, 1])或 `rlinf_pos_norm = 1 - brs_pos_norm`(state ∈ [0, 1])。**这个反向不在 L2 内部做**——L2 已经清楚标注自己用的是 RLinf 约定。

### 3.7 数值汇总(可直接 copy 到 `SafetyConfig` 默认值)

```python
# ── L2: per-joint position limits (rad, URDF - 0.1 rad margin) ──
right_arm_q_min: np.ndarray = np.array(
    [-4.35, -3.04, -2.26, -1.99, -2.26, -0.95, -1.47], dtype=np.float32)
right_arm_q_max: np.ndarray = np.array(
    [ 1.21,  0.07,  2.26,  0.25,  2.26,  0.95,  1.47], dtype=np.float32)
left_arm_q_min: np.ndarray = np.array(
    [-4.35, -0.07, -2.26, -1.99, -2.26, -0.95, -1.47], dtype=np.float32)
left_arm_q_max: np.ndarray = np.array(
    [ 1.21,  3.04,  2.26,  0.25,  2.26,  0.95,  1.47], dtype=np.float32)

# ── L2: per-joint velocity limits (rad/s; URDF*0.5 与 BRS-ctrl 取严) ──
arm_qvel_max: np.ndarray = np.array(
    [1.6, 1.6, 1.6, 1.6, 4.0, 4.0, 4.0], dtype=np.float32)  # 左右臂相同

# ── L2: torso (BRS-ctrl - 0.1 rad margin) ──
torso_q_min: np.ndarray = np.array(
    [-1.03, -2.69, -1.73, -2.95], dtype=np.float32)
torso_q_max: np.ndarray = np.array(
    [ 1.73,  2.43,  1.47,  2.95], dtype=np.float32)
torso_qvel_max: np.ndarray = np.array(
    [0.5, 0.5, 0.5, 0.5], dtype=np.float32)

# ── L2: chassis (BRS-ctrl 实测命令上限,无位置约束) ──
chassis_qvel_max: np.ndarray = np.array(
    [0.30, 0.30, 0.40], dtype=np.float32)  # vx, vy, wz
chassis_dead_zone: np.ndarray = np.array(
    [0.01, 0.01, 0.05], dtype=np.float32)

# ── L2: gripper (normalised [0, 1] — 0=fully closed, 1=fully open) ──
# v3: replaces v2's `gripper_pct_*` (which mixed mm/percentage units).
gripper_pos_min: float = 0.0
gripper_pos_max: float = 1.0
max_gripper_step: float = 0.30  # per-step delta in [0, 1] space
# Hardware stroke envelope (mm); used to convert ROS topic `mm` <-> norm.
# Defaults match RLinf `_dispatch_action.clip(0, 100)` current behaviour;
# set to BRS-style 10/90 mm for an extra 10 mm bumper margin.
gripper_closed_stroke_mm: float = 0.0
gripper_open_stroke_mm: float = 100.0

# ── L2: 安全裕量与触发阈值 ──
l2_warning_margin_rad: float = 0.15      # 接近极限时缩放动作
l2_critical_margin_rad: float = 0.05     # 严重接近时冻结该关节
l2_qvel_overspeed_factor: float = 1.20   # feedback qvel > 1.2× limit → soft_hold
l2_qvel_warning_factor: float = 0.90     # feedback qvel > 0.9× limit → 缩放
```

---

## 4. L2 子检查的语义分层 (L2a/L2b/L2c/L2d)

为了**审计清晰**(每个被裁剪的关节都能在 `SafetyInfo.reason` 里精确定位)以及**便于配置**(用户可以单独开/关某一子检查),把 L2 拆成 4 个独立子层:

| 子层 | 名字 | 输入 | 检查内容 | 触发动作 | reason 标签 |
|---|---|---|---|---|---|
| **L2a** | predict_q_clip | action(归一化 delta)+ state.qpos + schema | 用 schema.predict_arm_qpos() 预测下一步 q,**软裁剪** action 比例缩小 | 缩放 action 系数 ∈ [0, 1] | `L2a:right_q_warning J3=...` |
| **L2b** | qpos_hard_freeze | state.qpos | 当前 qpos 已经超出 critical_margin 的关节冻结(soft_hold) | `info.soft_hold = True` + 该关节 action=0 | `L2b:right_qpos_critical J6=...` |
| **L2c** | qvel_feedback_watchdog | state.qvel | 反馈速度已经超阈,说明上一步动作太大 → 缩放当前命令 | 缩放 action × 0.5,严重时 soft_hold | `L2c:right_qvel_overspeed J5=...` |
| **L2d** | command_speed_cap | action(归一化)+ schema | 限制本次**命令**关节速度(关节命令速度由 action_scale 推算) | 整体 action 缩放至命令速度 ≤ qvel_max | `L2d:left_qvel_cap J7=...` |

### 4.1 为什么是 4 层而不是 1 大块

- **L2a 是事前预测、平滑缩放**:这是 RL 训练的"安全带",**不中断 episode**。因为只缩小 action 幅度,policy 可以继续走、奖励依然计算。
- **L2b 是事后兜底、硬冻结**:当某关节已经处在极限内 0.05 rad,任何方向动作都不再可信,用 `soft_hold` 让该 step 静止一帧、等下一帧重新评估。
- **L2c 是反馈速度看门狗**:负责捕捉"L2a/L2b 没拦住、但 mobiman 内部速度已经飙起来"的情况。从 `state.right_arm_qvel` 直接读,**这是 L2 的 reactive 层**。
- **L2d 是命令速度看门狗**:负责把**这一步命令的关节速度**也卡住,即使反馈速度还没飙起来。**预测性地**避免 mobiman watchdog 触发。

### 4.2 顺序与依赖关系

```mermaid
flowchart LR
    IN["action ∈ [-1,1]^D"]
    L2a["L2a: predict_q_clip"]
    L2b["L2b: qpos_hard_freeze"]
    L2c["L2c: qvel_feedback_watchdog"]
    L2d["L2d: command_speed_cap"]
    OUT["safe_action → L3a"]

    IN --> L2a --> L2b --> L2c --> L2d --> OUT

    style L2a fill:#cce
    style L2b fill:#fcc
    style L2c fill:#fcc
    style L2d fill:#cec
```

**L2a → L2b → L2c → L2d** 的顺序是经过推敲的:

1. **先做 L2a 预测裁剪**——它是无副作用的、**最优的**裁剪(缩比例,保留 policy 意图);
2. **再做 L2b 硬冻结**——只在 L2a 都救不了时才发作(关节已经走到极限内);
3. **再做 L2c 反馈速度看门狗**——能在 L2a/b 之外捕捉 mobiman 内部加速过头;
4. **最后做 L2d 命令速度上限**——保证最终下发命令不会超过 qvel_max。

四层之间是**单调收敛**的:每一层都让 action 更"温和",绝不会让 action 变得更激进,因此组合效果是可预测的。

---

## 5. 模块结构与调用关系

### 5.1 总体类图

```mermaid
classDiagram
    class GalaxeaR1ProEnv {
        +step(action)
        +reset()
        -_safety: SafetySupervisor
        -_action_schema: ActionSchema
        -_controller: ControllerProxy
        -_state: RobotState
    }
    class GalaxeaR1ProSafetySupervisor {
        +validate(action, state, schema) SafetyInfo
        -_check_l1(action, info)
        -_check_l2a_predict_q_clip(action, state, schema, info)
        -_check_l2b_qpos_freeze(action, state, schema, info)
        -_check_l2c_qvel_watchdog(action, state, schema, info)
        -_check_l2d_cmd_speed_cap(action, state, schema, info)
        -_check_l3a_ee_box(action, state, schema, info)
        -_check_l3b_dual_arm(action, state, schema, info)
        -_check_l4_caps(action, schema, info)
        -_check_l5_watchdog(state, info)
    }
    class SafetyConfig {
        +right_arm_q_min: ndarray
        +right_arm_q_max: ndarray
        +left_arm_q_min: ndarray
        +left_arm_q_max: ndarray
        +arm_qvel_max: ndarray
        +torso_q_min: ndarray
        +torso_q_max: ndarray
        +torso_qvel_max: ndarray
        +chassis_qvel_max: ndarray
        +chassis_dead_zone: ndarray
        +gripper_pos_min: float
        +gripper_pos_max: float
        +max_gripper_step: float
        +gripper_closed_stroke_mm: float
        +gripper_open_stroke_mm: float
        +l2_warning_margin_rad: float
        +l2_critical_margin_rad: float
        +l2_qvel_overspeed_factor: float
        +l2_qvel_warning_factor: float
    }
    class SafetyInfo {
        +raw_action
        +safe_action
        +clipped: bool
        +soft_hold: bool
        +safe_stop: bool
        +emergency_stop: bool
        +reason: list[str]
        +metrics: dict
        +l2_per_joint_clip: dict   <<new>>
    }
    class ActionSchema {
        +has_left_arm: bool
        +has_right_arm: bool
        +has_torso: bool
        +has_chassis: bool
        +action_scale: ndarray
        +split(action) dict
        +predict_arm_ee_pose(side, action, state)
        +predict_arm_qpos(side, action, state)   <<new>>
        +rewrite_action_arm(action, side, scale_factor) <<new>>
        +rewrite_action_torso(action, scale_factor)     <<new>>
        +rewrite_action_chassis(action, scale_factor)   <<new>>
        +rewrite_action_chassis_set(action, twist3)     <<new>>
        +set_gripper_action(action, side, new_a)        <<new>>
    }
    class GalaxeaR1ProRobotState {
        +right_arm_qpos: ndarray(7)
        +right_arm_qvel: ndarray(7)
        +left_arm_qpos: ndarray(7)
        +left_arm_qvel: ndarray(7)
        +torso_qpos: ndarray(4)
        +torso_qvel: ndarray(4)
        +chassis_qpos: ndarray(3)
        +chassis_qvel: ndarray(3)
        +right_gripper_pos: float
        +left_gripper_pos: float
        +get_arm_qpos(side)
        +get_arm_qvel(side)
        +get_torso_qpos()    <<new helper>>
        +get_torso_qvel()    <<new helper>>
        +get_chassis_qpos()  <<new helper>>
        +get_chassis_qvel()  <<new helper>>
        +get_gripper_pos(side)
        +get_gripper_vel(side)         <<new helper>>
        +get_gripper_pos_norm(side, *, closed_mm, open_mm) <<v3 helper>>
    }
    class GalaxeaR1ProController {
        +get_state() RobotState
        +send_arm_pose(side, pose)
        +send_arm_joints(side, qpos)
        +send_gripper(side, pct_or_mm)
        +send_torso_twist(twist4)
        +send_chassis_twist(twist3)
        +apply_brake(on)
    }

    GalaxeaR1ProEnv --> GalaxeaR1ProSafetySupervisor : uses
    GalaxeaR1ProEnv --> ActionSchema : uses
    GalaxeaR1ProEnv --> GalaxeaR1ProController : RPC
    GalaxeaR1ProSafetySupervisor --> SafetyConfig : holds
    GalaxeaR1ProSafetySupervisor --> ActionSchema : reads (predict_arm_qpos, rewrite_*)
    GalaxeaR1ProSafetySupervisor --> SafetyInfo : produces
    GalaxeaR1ProSafetySupervisor --> GalaxeaR1ProRobotState : reads
    GalaxeaR1ProController --> GalaxeaR1ProRobotState : updates
```

`<<new>>` 标记的是本设计**新增**的方法/字段。所有都在已存在的类内追加,**没有新建模块**。

### 5.2 调用关系(序列图)

```mermaid
sequenceDiagram
    participant Policy
    participant Env as GalaxeaR1ProEnv.step
    participant Ctrl as GalaxeaR1ProController
    participant ROS as ROS2 / mobiman
    participant Sup as SafetySupervisor
    participant Sch as ActionSchema
    participant State as RobotState

    Policy->>Env: action ∈ [-1,1]^D

    Env->>Ctrl: get_state().wait()
    Ctrl->>State: copy state under lock
    State-->>Ctrl: snapshot
    Ctrl-->>Env: state

    Env->>Sup: validate(action, state, schema)
    Sup->>Sup: _check_l1(action, info)

    rect rgb(220,220,255)
    Note over Sup,Sch: L2 子层
    Sup->>Sch: predict_arm_qpos(side, action, state)
    Sch-->>Sup: q_next[7]
    Sup->>Sup: _check_l2a_predict_q_clip
    Sup->>Sch: rewrite_action_arm(scale_factor)
    Sup->>Sup: _check_l2b_qpos_freeze (state.qpos)
    Sup->>Sup: _check_l2c_qvel_watchdog (state.qvel)
    Sup->>Sup: _check_l2d_cmd_speed_cap
    end

    Sup->>Sup: _check_l3a_ee_box
    Sup->>Sup: _check_l3b_dual_arm
    Sup->>Sup: _check_l4_caps
    Sup->>Sup: _check_l5_watchdog (state.bms / swd / ...)
    Sup-->>Env: SafetyInfo (safe_action, reasons, metrics, l2_per_joint_clip)

    alt e_stop / safe_stop
        Env->>Ctrl: apply_brake(True).wait()
    else normal
        Env->>Ctrl: send_arm_pose / send_gripper / ...
        Ctrl->>ROS: PoseStamped / JointState publish
    end

    Env-->>Policy: obs, reward, term, trunc, info
```

### 5.3 与 r1_pro_env.step 的接口

`step()` 与 supervisor 的契约**不变**。新增的 `SafetyInfo.l2_per_joint_clip`(dict,详见 §7.5)只是审计字段,旧代码读老字段(`raw_action / safe_action / reason / ...`)继续工作。这是**最小侵入**的关键。

---

## 6. 数据流:从 ROS2 反馈到 SafetyInfo

```mermaid
flowchart TB
    subgraph HDAS["HDAS / mobiman (Orin)"]
        FBR["/hdas/feedback_arm_right (JointState)<br/>position[7] velocity[7] effort[7]"]
        FBL["/hdas/feedback_arm_left (JointState)"]
        FBT["/hdas/feedback_torso (JointState)"]
        FBC["/hdas/feedback_chassis (JointState)"]
        FBGR["/hdas/feedback_gripper_right (JointState)"]
        FBGL["/hdas/feedback_gripper_left (JointState)"]
    end

    subgraph Ctrl["GalaxeaR1ProController (Orin / Ray worker)"]
        SUB["_init_subscribers()<br/>(r1_pro_controller.py:286-348)"]
        CB["_on_arm_right_feedback / _on_arm_left_feedback /<br/>_on_torso_feedback / _on_chassis_feedback /<br/>_on_gripper_*_feedback<br/>(r1_pro_controller.py:417-487)"]
        Lock["_state_lock (threading.RLock)"]
        State["GalaxeaR1ProRobotState<br/>right_arm_qpos / qvel / qtau<br/>left_arm_qpos / qvel / qtau<br/>torso_qpos / qvel<br/>chassis_qpos / qvel<br/>right_gripper_pos / vel<br/>left_gripper_pos / vel"]
        GS["get_state() RPC<br/>(r1_pro_controller.py:552-555)"]
    end

    subgraph Env["GalaxeaR1ProEnv.step (GPU server)"]
        ST1["state = ctrl.get_state().wait()"]
        VAL["sinfo = safety.validate(action, state, schema)"]
        DISP["_dispatch_action(safe_action)"]
    end

    subgraph Sup["GalaxeaR1ProSafetySupervisor"]
        L1["L1: schema 校验"]
        subgraph L2box["L2 (本设计)"]
            L2a["L2a: predict_q_clip<br/>读 state.{side}_arm_qpos (7)<br/>读 schema.action_scale<br/>缩放 action 比例"]
            L2b["L2b: qpos_hard_freeze<br/>读 state.{side}_arm_qpos (7)<br/>state.torso_qpos (4)<br/>info.soft_hold + 该关节 action=0"]
            L2c["L2c: qvel_feedback_watchdog<br/>读 state.{side}_arm_qvel (7)<br/>state.torso_qvel (4)<br/>state.chassis_qvel (3)<br/>缩放 action 或 soft_hold"]
            L2d["L2d: command_speed_cap<br/>读 schema.action_scale<br/>cfg.dt_step<br/>整体 action 缩放至命令 v ≤ vmax"]
        end
        L3a["L3a: TCP box (现有)"]
        L3b["L3b: 双臂碰撞 (现有)"]
        L4["L4: vel/acc caps (现有)"]
        L5["L5: 看门狗 (现有)"]
    end

    FBR --> SUB --> CB --> Lock --> State
    FBL --> SUB
    FBT --> SUB
    FBC --> SUB
    FBGR --> SUB
    FBGL --> SUB

    State --> GS --> ST1 --> VAL
    VAL --> L1 --> L2a --> L2b --> L2c --> L2d --> L3a --> L3b --> L4 --> L5
    L5 --> DISP

    style L2a fill:#cce
    style L2b fill:#fcc
    style L2c fill:#fcc
    style L2d fill:#cec
```

### 6.1 数据格式约定

| 类型 | 形状 | 单位 | 来源 |
|---|---|---|---|
| `state.right_arm_qpos` / `left_arm_qpos` | `(7,)` `np.float32` | rad | `JointState.position[:7]` |
| `state.right_arm_qvel` / `left_arm_qvel` | `(7,)` `np.float32` | rad/s | `JointState.velocity[:7]` |
| `state.torso_qpos` | `(4,)` `np.float32` | rad | `JointState.position[:4]` |
| `state.torso_qvel` | `(4,)` `np.float32` | rad/s | `JointState.velocity[:4]` |
| `state.chassis_qpos` | `(3,)` `np.float32` | m, m, rad(Odom) | `JointState.position[:3]` |
| `state.chassis_qvel` | `(3,)` `np.float32` | m/s, m/s, rad/s | `JointState.velocity[:3]` |
| `state.right_gripper_pos` / `left_gripper_pos` | scalar | **mm**(SDK 真值,实测 0.5-100 mm 范围) | `/hdas/feedback_gripper_*` 的 `JointState.position[0]` |
| `state.get_gripper_pos_norm(side)` (v3 helper) | scalar | **[0, 1]**(0=closed, 1=open) | `(pos_mm - closed_stroke_mm) / (open_stroke_mm - closed_stroke_mm)` |
| action(M1 单臂) | `(7,)` `np.float32` | 归一化 [-1,1] | policy |
| action(M3 全身) | `(18,)` 或 `(21,)` | 归一化 [-1,1] | policy |

### 6.2 锁与一致性

- **`get_state()` RPC**:在 [`r1_pro_controller.py:552-555`](../../../rlinf/envs/realworld/galaxear/r1_pro_controller.py#L552-L555) 内 `with self._state_lock: return self._state.copy()`。返回的是**深拷贝**,supervisor 操作的是**snapshot**——不会被回调中途改写,这是 L2 正确性的前提。
- **跨进程一致性**:L2 跑在 GPU server,`state` 是 50–100 ms 前的 Orin 反馈。这个延迟对于 0.1 rad 的 margin 是足够的(详见 [`safety_2_joinlimit.md`](safety_2_joinlimit.md) §7.2.4)。

---

## 7. 关键代码详解

> 本节给出**可直接合并的草稿**。所有代码都标注了**插入位置**(文件 + 行号),所有改动都遵循 RLinf 的 Google Python Style + Ruff format。

### 7.1 `SafetyConfig` 字段扩展(`r1_pro_safety.py`)

**插入位置**:[`r1_pro_safety.py:54-103`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py#L54-L103) 的 `SafetyConfig` dataclass。

```python
@dataclass
class SafetyConfig:
    """Hard safety limits + watchdog thresholds.

    Defaults are derived from the R1 Pro URDF
    (`mobiman/lib/mobiman/configs/urdfs/r1_pro_floating_*.urdf`)
    minus a 0.1 rad (5.7°) safety margin per joint, with the
    velocity bound additionally cross-checked against the BRS-ctrl
    production constants (`R1ProInterface` in
    `kaizhe_ws/data_collect/brs-ctrl/.../interfaces.py`).

    Note that left/right arm limits *differ* on J2: R1 Pro arms are
    physically mirrored, so we keep two separate vectors instead of
    a single `arm_q_min/max`.

    See bt/docs/rwRL/safety_2_joinlimit_2.md §3 for derivation.
    """

    # ── L2: 7-DoF per-arm position limits (rad) ──────────────────
    right_arm_q_min: np.ndarray = field(default_factory=lambda: np.array(
        [-4.35, -3.04, -2.26, -1.99, -2.26, -0.95, -1.47],
        dtype=np.float32))
    right_arm_q_max: np.ndarray = field(default_factory=lambda: np.array(
        [ 1.21,  0.07,  2.26,  0.25,  2.26,  0.95,  1.47],
        dtype=np.float32))
    left_arm_q_min: np.ndarray = field(default_factory=lambda: np.array(
        [-4.35, -0.07, -2.26, -1.99, -2.26, -0.95, -1.47],
        dtype=np.float32))
    left_arm_q_max: np.ndarray = field(default_factory=lambda: np.array(
        [ 1.21,  3.04,  2.26,  0.25,  2.26,  0.95,  1.47],
        dtype=np.float32))

    # ── L2: 7-DoF per-arm velocity limits (rad/s) ───────────────
    arm_qvel_max: np.ndarray = field(default_factory=lambda: np.array(
        [1.6, 1.6, 1.6, 1.6, 4.0, 4.0, 4.0], dtype=np.float32))

    # ── L2: 4-DoF torso ──────────────────────────────────────────
    torso_q_min: np.ndarray = field(default_factory=lambda: np.array(
        [-1.03, -2.69, -1.73, -2.95], dtype=np.float32))
    torso_q_max: np.ndarray = field(default_factory=lambda: np.array(
        [ 1.73,  2.43,  1.47,  2.95], dtype=np.float32))
    torso_qvel_max: np.ndarray = field(default_factory=lambda: np.array(
        [0.5, 0.5, 0.5, 0.5], dtype=np.float32))

    # ── L2: 3-DoF chassis (only velocity) ────────────────────────
    chassis_qvel_max: np.ndarray = field(default_factory=lambda: np.array(
        [0.30, 0.30, 0.40], dtype=np.float32))     # vx, vy, wz
    chassis_dead_zone: np.ndarray = field(default_factory=lambda: np.array(
        [0.01, 0.01, 0.05], dtype=np.float32))

    # ── L2: gripper (normalised [0, 1] — 0=closed, 1=open) ──────
    # NOTE (v3): R1 Pro G1 SDK reports/expects mm on its ROS topic
    # (`/motion_target/target_position_gripper_*` JointState.position
    # in mm), so the supervisor needs an explicit conversion to get
    # to a clean normalised [0, 1] working space.  See §3.6 for the
    # three-way table (mm vs RLinf-norm vs BRS-norm).
    gripper_pos_min: float = 0.0    # 0 = fully closed
    gripper_pos_max: float = 1.0    # 1 = fully open
    max_gripper_step: float = 0.30  # per-step delta in normalised [0,1]
    # mm <-> norm conversion endpoints.  Defaults match the RLinf
    # `_dispatch_action.clip(0, 100)` current behaviour; for safer
    # stroke window switch to BRS values (10.0 / 90.0 mm).
    gripper_closed_stroke_mm: float = 0.0
    gripper_open_stroke_mm: float = 100.0

    # ── L2: thresholds & gains ───────────────────────────────────
    l2_warning_margin_rad: float = 0.15      # start scaling action
    l2_critical_margin_rad: float = 0.05     # freeze that joint
    l2_qvel_overspeed_factor: float = 1.20   # > 1.2× -> soft_hold
    l2_qvel_warning_factor: float = 0.90     # > 0.9× -> scale 0.5
    l2_per_joint_freeze_action: float = 0.0  # freeze means "0 action"

    # ── L2 toggles (let YAML disable individual sub-checks) ──────
    enable_l2a_predict_q_clip: bool = True
    enable_l2b_qpos_freeze: bool = True
    enable_l2c_qvel_watchdog: bool = True
    enable_l2d_cmd_speed_cap: bool = True
    enable_l2_gripper: bool = True

    # ── Step period used by L2d (s); should match Env.step_frequency. ──
    dt_step: float = 0.10  # 10 Hz default

    # ── 旧字段保留(向后兼容,标 deprecated) ──────────────────────
    arm_q_min: Optional[np.ndarray] = None  # deprecated: use right/left
    arm_q_max: Optional[np.ndarray] = None  # deprecated
    # ... (L3a/L3b/L4/L5 字段不变)
```

要点:
- **拆 `arm_q_*` 为左右两份**(§3.7);保留旧字段作为 deprecated alias 在 `__post_init__` 里回填。
- 新增 4 个 `enable_l2*` toggle,允许 YAML 配置精细控制。
- 保留旧 `arm_qvel_max`(只是改了默认值,字段名不变;旧 RLinf 已经是同名字段)。

`__post_init__` 兼容老配置:

```python
def __post_init__(self):
    # ... existing logic for converting list -> np.ndarray ...

    # Back-compat: if someone passes arm_q_min/max (the old single
    # vector), use it for BOTH arms with a runtime warning.
    if self.arm_q_min is not None:
        self.right_arm_q_min = np.asarray(self.arm_q_min, dtype=np.float32)
        self.left_arm_q_min  = np.asarray(self.arm_q_min, dtype=np.float32)
        # (warning printed once on construction)
    if self.arm_q_max is not None:
        self.right_arm_q_max = np.asarray(self.arm_q_max, dtype=np.float32)
        self.left_arm_q_max  = np.asarray(self.arm_q_max, dtype=np.float32)
```

### 7.2 `ActionSchema.predict_arm_qpos`(`r1_pro_action_schema.py`)

**插入位置**:[`r1_pro_action_schema.py:144`](../../../rlinf/envs/realworld/galaxear/r1_pro_action_schema.py#L144) `predict_arm_ee_pose` 之后(实施时已落点)。

```python
def predict_arm_qpos(
    self,
    side: str,
    action: np.ndarray,
    state: "GalaxeaR1ProRobotState",
    *,
    inv_jacobian_fn=None,
) -> Optional[np.ndarray]:
    """Predict the joint positions for *side* arm after this step.

    Strategy (in order of preference):

    1. ``inv_jacobian_fn`` provided -> use IK to map the predicted
       Cartesian EE delta to a joint delta (precise, requires
       URDF + an IK library; future hook).  ``inv_jacobian_fn`` is
       called as ``fn(*, side, cur_q, d_xyz, d_rpy) -> np.ndarray(7,)``;
       any exception falls through to the next branch (defensive).
    2. ``self.use_joint_mode == True`` -> action *is* a normalised
       joint delta; the per-arm xyz+rpy 6 dims are interpreted as
       deltas for the **first 6 arm joints** (the 7th joint stays
       at its current value).  This is the bring-up convention --
       once a real 7-D joint-mode action layout lands, this branch
       must be revisited.
    3. otherwise (Cartesian EE delta mode, the default for M1
       single-arm) -> return ``state.get_arm_qpos(side)`` *unchanged*.
       L2a is then effectively a no-op for arms in EE mode, and
       the protection is delegated to L2b (post-hoc qpos freeze)
       and L2c (qvel feedback watchdog).  Drop in an IK-enabled
       ``inv_jacobian_fn`` later to start using L2a without
       touching the supervisor.
    """
    d = self.split(action)
    key_xyz = f"{side}_xyz"
    if key_xyz not in d:
        return None
    cur_q = np.asarray(
        state.get_arm_qpos(side), dtype=np.float32,
    ).reshape(-1)[:7]
    if cur_q.size != 7:
        return None

    # Branch 1: external IK predictor supplied.
    if inv_jacobian_fn is not None:
        try:
            delta_q = inv_jacobian_fn(
                side=side,
                cur_q=cur_q,
                d_xyz=np.asarray(d[key_xyz], dtype=np.float32)
                * float(self.action_scale[0]),
                d_rpy=np.asarray(d[f"{side}_rpy"], dtype=np.float32)
                * float(self.action_scale[1]),
            )
            delta_q = np.asarray(delta_q, dtype=np.float32).reshape(-1)[:7]
            if delta_q.size == 7:
                return (cur_q + delta_q).astype(np.float32)
        except Exception:
            pass  # fall through to other branches

    # Branch 2: action is already in joint space (bring-up convention).
    if self.use_joint_mode:
        xyz = np.asarray(d[key_xyz], dtype=np.float32).reshape(-1)[:3]
        rpy = np.asarray(d[f"{side}_rpy"], dtype=np.float32).reshape(-1)[:3]
        joint_delta6 = np.concatenate([xyz, rpy])
        scale = float(self.action_scale[0])
        new_q = cur_q.copy()
        new_q[:6] = cur_q[:6] + joint_delta6 * scale  # J7 unchanged
        return new_q.astype(np.float32)

    # Branch 3: Cartesian mode -> "no-op" predictor.
    return cur_q.astype(np.float32)
```

要点(实施细节):

- **三个分支按优先级 short-circuit**:IK > 关节模式 > 笛卡尔模式 fallback。
- **Branch 1 防御异常**:`inv_jacobian_fn` 任何异常都会被吞下并 fall through 到 Branch 2/3,**不让 IK bug 拖崩 supervisor**。这是与 plan 草稿的微小差异(草稿版用 `try/except: pass`,实施版还多做了一次 size 校验,只在 `delta_q.size == 7` 时才提前返回)。
- **Branch 2 的"前 6 个关节" bring-up 约定**:`use_joint_mode` 是 R1 Pro M1 阶段的"过渡 schema"——action vector 仍然是 6 维 xyz+rpy 槽位,但语义上已经被解释为前 6 个臂关节的 delta。第 7 关节(腕旋转)在这种过渡模式下保持不变。当后续真正引入 7-D joint-mode action layout 时,这个分支需要重写,**所以 plan §7.2 草稿的 `a_arm = np.concatenate([..., [d.get(f"{side}_gripper", 0.0)]])` 写法被替换为更明确的"前 6 维"实现**,避免误把夹爪 dim 当成 J7。
- **Cartesian mode 默认 fallback 到 cur_q**:这意味着 L2a **在不接 IK 时不会误裁** EE pose 控制路径;真正的工作交给 L2b(qpos hard freeze)和 L2c(qvel watchdog)。这是**故意**的——保持 RLinf 当前 EE 控制路径的"无副作用"。
- **`inv_jacobian_fn` 接口**预留给将来接入 `pinocchio` 或调用 mobiman `relaxed_ik` RPC。
- **不强行实现 IK**——避免引入新依赖,与 [`safety_2_joinlimit.md`](safety_2_joinlimit.md) §7.2 推荐"事后校验方案 A"一致。

### 7.3 `ActionSchema.rewrite_action_arm`(用于 L2a / L2c 缩放)

**插入位置**:同上,跟在 `predict_arm_qpos` 后。

```python
def rewrite_action_arm(
    self,
    action: np.ndarray,
    side: str,
    scale_factor: float,
) -> np.ndarray:
    """Multiply the per-arm slice of *action* by *scale_factor*.

    Used by L2a / L2c to softly shrink the policy's intent without
    changing direction; ``scale_factor=0.0`` freezes the arm in
    place. The gripper dim (if any) is *not* scaled — gripper has
    its own L2 sub-check in :meth:`_check_l2_gripper` (see §7.6),
    which writes back via :meth:`set_gripper_action`.

    Returns the modified action (operates in place but also returns).
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    s = float(np.clip(scale_factor, 0.0, 1.0))
    idx = 0
    if self.has_right_arm:
        if side == "right":
            a[idx : idx + 6] *= s   # xyz + rpy
            return a
        idx += self.per_arm_dim
    if self.has_left_arm and side == "left":
        a[idx : idx + 6] *= s
    return a
```

类似的有 `rewrite_action_torso` / `rewrite_action_chassis`,实现完全平行(只改对应 4D / 3D slice)。

### 7.4 5 个 L2 子检查方法(`r1_pro_safety.py`)

**插入位置**:[`r1_pro_safety.py:196-199`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py#L196-L199) 现行 L2 注释位置。原 `validate()` 方法的 L2 段替换为:

```python
# ── L2 per-joint limit (multi-sublayer) ────────────────────────
if self._cfg.enable_l2a_predict_q_clip:
    self._check_l2a_predict_q_clip(info, state, action_schema)
if self._cfg.enable_l2b_qpos_freeze:
    self._check_l2b_qpos_freeze(info, state, action_schema)
if self._cfg.enable_l2c_qvel_watchdog:
    self._check_l2c_qvel_watchdog(info, state, action_schema)
if self._cfg.enable_l2d_cmd_speed_cap:
    self._check_l2d_cmd_speed_cap(info, state, action_schema)
if self._cfg.enable_l2_gripper:
    self._check_l2_gripper(info, state, action_schema)
```

#### 7.4.1 L2a — 预测 q + 裁剪 action(平滑 scale shrink)

```python
def _check_l2a_predict_q_clip(
    self,
    info: SafetyInfo,
    state: "GalaxeaR1ProRobotState",
    schema: "ActionSchema",
) -> None:
    """Predict next-step joint angles. If a joint will overshoot
    the warning_margin, scale the per-arm action so the *worst*
    joint margin is exactly warning_margin.
    """
    for side in ("right", "left"):
        if side == "right" and not schema.has_right_arm:
            continue
        if side == "left" and not schema.has_left_arm:
            continue

        # Zero-state guard: skip when feedback hasn't arrived yet.
        # The default zero state vector lands on top of left J2's
        # q_min (-0.07) and would make L2a fire on every first step
        # before subscribers populate ``state.{side}_arm_qpos``.
        # See Appendix C.1 for the failure mode and derivation.
        cur_q_raw = np.asarray(
            state.get_arm_qpos(side), dtype=np.float32,
        ).reshape(-1)
        if cur_q_raw.size < 7 or not np.any(cur_q_raw[:7]):
            continue

        q_next = schema.predict_arm_qpos(side, info.safe_action, state)
        if q_next is None:
            continue

        q_min = self._cfg.right_arm_q_min if side == "right" else self._cfg.left_arm_q_min
        q_max = self._cfg.right_arm_q_max if side == "right" else self._cfg.left_arm_q_max

        # margin to closest limit, per joint.
        margin_low  = q_next - q_min
        margin_high = q_max - q_next
        margin = np.minimum(margin_low, margin_high)
        worst = float(np.min(margin))
        worst_idx = int(np.argmin(margin))

        if worst < self._cfg.l2_critical_margin_rad:
            scale = 0.0
            tag_kind = "critical"
        elif worst < self._cfg.l2_warning_margin_rad:
            # Linearly shrink so worst margin = warning_margin.
            scale = max(
                0.0,
                worst / max(self._cfg.l2_warning_margin_rad, 1e-9),
            )
            tag_kind = "warning"
        else:
            continue  # plenty of room

        info.safe_action = schema.rewrite_action_arm(
            info.safe_action, side, scale,
        )
        info.clipped = True
        info.reason.append(
            f"L2a:{side}_q_{tag_kind} J{worst_idx + 1} "
            f"margin={worst:+.3f}rad scale={scale:.2f}"
        )
        info.metrics[f"safety/l2a_{side}_scale"] = float(scale)
        info.metrics[f"safety/l2a_{side}_worst_margin"] = float(worst)
        self._record_per_joint_clip(
            info, side, f"J{worst_idx + 1}", "L2a",
            {"scale": float(scale), "margin": float(worst)},
        )
```

要点(含实施期发现):

- **Zero-state guard 是必需的(实施期发现)**。理论上 L2a 在 zero state 下应该触发的"误冻结"具体场景是:左臂 `q_min[1] = -0.07 rad`,当 `cur_q = [0,…,0]`(反馈未到达的默认值)时,J2 的 margin 是 `min(0 - (-0.07), 3.04 - 0) = 0.07` rad。0.07 < `l2_warning_margin_rad = 0.15`,**会以 scale = 0.07/0.15 ≈ 0.467 缩放整条左臂 action**——这正是实施期跑 baseline 测试时 `test_l3a_dual_arm_only_violating_arm_is_rewritten` 发现的"左臂被意外缩到 0.467 倍"的根因。因此 L2a / L2b 都加上 `not np.any(cur_q_raw[:7]) → continue`,与 [`r1_pro_controller.py:557-560`](../../../rlinf/envs/realworld/galaxear/r1_pro_controller.py#L557-L560) 的 `is_robot_up()` 行为对齐。详见 [附录 C.1](#c1-zero-state-guard-为什么必要)。
- **scale 的物理含义**:线性插值。worst < warning_margin 时,把 action 缩到刚好让 worst 不再下降。这不是"二值开关",是**软约束**。
- **只在笛卡尔模式下 q_next≈cur_q**(因为 §7.2 的 fallback),所以 L2a 在 EE 控制路径里**几乎不触发**。这是**预期**:EE 控制下 L2a 是空跑 placeholder,真正起作用的是 L2b/L2c。但**接入 IK 后** L2a 会立刻生效——这个 hook 是为了未来不动 supervisor 代码就能升级。
- **reason 标签里附带 `J{worst_idx+1}`**:实施期还增加了"哪个关节最先吃紧"的标签(plan 草稿没有),配合 §7.5 的 `l2_per_joint_clip` 让审计更精细。

#### 7.4.2 L2b — 当前 q 已贴极限的硬冻结

```python
def _check_l2b_qpos_freeze(
    self,
    info: SafetyInfo,
    state: "GalaxeaR1ProRobotState",
    schema: "ActionSchema",
) -> None:
    """If the *current* feedback qpos is already inside
    ``critical_margin`` from any joint limit, freeze that arm
    (or torso) for one step and raise ``soft_hold``.

    This is the reactive complement to the predictive L2a: it
    catches the case where last step's command pushed a joint
    right up to its limit even though L2a believed margin was
    OK (e.g. mobiman IK overshoot, mechanical play, or a
    first-step where the policy violated its priors).
    """
    # Arms.
    for side in ("right", "left"):
        if side == "right" and not schema.has_right_arm:
            continue
        if side == "left" and not schema.has_left_arm:
            continue

        cur_q = np.asarray(
            state.get_arm_qpos(side), dtype=np.float32,
        ).reshape(-1)[:7]
        if cur_q.size != 7:
            continue
        # Zero-state guard (same rationale as L2a, see Appendix C.1).
        if not np.any(cur_q):
            continue
        q_min = self._cfg.right_arm_q_min if side == "right" else self._cfg.left_arm_q_min
        q_max = self._cfg.right_arm_q_max if side == "right" else self._cfg.left_arm_q_max
        margin = np.minimum(cur_q - q_min, q_max - cur_q)

        bad_idx = np.where(margin < self._cfg.l2_critical_margin_rad)[0]
        if bad_idx.size > 0:
            info.safe_action = schema.rewrite_action_arm(
                info.safe_action, side, 0.0,   # freeze
            )
            info.soft_hold = True
            joint_labels = ",".join(f"J{int(i) + 1}" for i in bad_idx)
            info.reason.append(
                f"L2b:{side}_qpos_critical {joint_labels} "
                f"margin={float(margin[bad_idx].min()):+.3f}"
            )
            info.metrics[f"safety/l2b_{side}_freeze"] = 1.0
            for i in bad_idx:
                self._record_per_joint_clip(
                    info, side, f"J{int(i) + 1}", "L2b", "freeze",
                )

    # Torso (also zero-state guarded).
    if schema.has_torso:
        cur_t = np.asarray(
            state.get_torso_qpos(), dtype=np.float32,
        ).reshape(-1)[:4]
        if cur_t.size == 4 and np.any(cur_t):
            margin_t = np.minimum(
                cur_t - self._cfg.torso_q_min,
                self._cfg.torso_q_max - cur_t,
            )
            bad_t = np.where(margin_t < self._cfg.l2_critical_margin_rad)[0]
            if bad_t.size > 0:
                info.safe_action = schema.rewrite_action_torso(
                    info.safe_action, 0.0,
                )
                info.soft_hold = True
                tlst = ",".join(f"T{int(i) + 1}" for i in bad_t)
                info.reason.append(
                    f"L2b:torso_qpos_critical {tlst} "
                    f"margin={float(margin_t[bad_t].min()):+.3f}"
                )
                info.metrics["safety/l2b_torso_freeze"] = 1.0
                for i in bad_t:
                    self._record_per_joint_clip(
                        info, "torso", f"T{int(i) + 1}",
                        "L2b", "freeze",
                    )
```

实施细节:

- **Zero-state guard**:与 L2a 一样的 `not np.any(cur_q) → continue`。`r1_pro_controller.py` 启动时 `state.is_alive=False`,`right/left_arm_qpos = zeros(7)`(`r1_pro_robot_state.py:_zeros7`)。HDAS 反馈到达前 L2b 必须保持沉默,否则会立刻 `soft_hold` 让 episode 永远卡在 step 1。
- **Torso 也做了 zero-state guard**(`np.any(cur_t)`)。
- **每个越限关节都进 `l2_per_joint_clip` 审计**(plan 草稿只在 reason 里记一行),详见 §7.5 的 schema 调整。

#### 7.4.3 L2c — 反馈速度看门狗

```python
def _check_l2c_qvel_watchdog(
    self,
    info: SafetyInfo,
    state: "GalaxeaR1ProRobotState",
    schema: "ActionSchema",
) -> None:
    """React to runaway velocity *as reported by the robot*.

    Two thresholds:
    - feedback |qvel| > overspeed_factor * qvel_max -> soft_hold
    - feedback |qvel| > warning_factor   * qvel_max -> scale 0.5
    """
    over = float(self._cfg.l2_qvel_overspeed_factor)
    warn = float(self._cfg.l2_qvel_warning_factor)

    # Arms.
    qvm_arm = self._cfg.arm_qvel_max
    for side in ("right", "left"):
        if side == "right" and not schema.has_right_arm:
            continue
        if side == "left" and not schema.has_left_arm:
            continue
        qv = np.asarray(
            state.get_arm_qvel(side), dtype=np.float32,
        ).reshape(-1)[:7]
        if qv.size != 7:
            continue
        ratio = np.abs(qv) / np.maximum(qvm_arm, 1e-6)
        max_ratio = float(np.max(ratio))
        max_idx = int(np.argmax(ratio))
        if max_ratio > over:
            info.safe_action = schema.rewrite_action_arm(
                info.safe_action, side, 0.0,
            )
            info.soft_hold = True
            info.reason.append(
                f"L2c:{side}_qvel_overspeed J{max_idx + 1} "
                f"max={max_ratio:.2f}x limit"
            )
            info.metrics[
                f"safety/l2c_{side}_overspeed_ratio"
            ] = max_ratio
            self._record_per_joint_clip(
                info, side, f"J{max_idx + 1}", "L2c", "freeze",
            )
        elif max_ratio > warn:
            info.safe_action = schema.rewrite_action_arm(
                info.safe_action, side, 0.5,
            )
            info.clipped = True
            info.reason.append(
                f"L2c:{side}_qvel_warning J{max_idx + 1} "
                f"max={max_ratio:.2f}x limit, scale=0.5"
            )
            info.metrics[
                f"safety/l2c_{side}_warn_ratio"
            ] = max_ratio
            self._record_per_joint_clip(
                info, side, f"J{max_idx + 1}", "L2c", "scale=0.5",
            )

    # Torso.
    if schema.has_torso:
        qv_t = np.asarray(
            state.get_torso_qvel(), dtype=np.float32,
        ).reshape(-1)[:4]
        if qv_t.size == 4:
            ratio_t = np.abs(qv_t) / np.maximum(
                self._cfg.torso_qvel_max, 1e-6,
            )
            if np.any(ratio_t > over):
                info.safe_action = schema.rewrite_action_torso(
                    info.safe_action, 0.0,
                )
                info.soft_hold = True
                info.reason.append(
                    f"L2c:torso_qvel_overspeed max={float(ratio_t.max()):.2f}x"
                )
                info.metrics["safety/l2c_torso_overspeed_ratio"] = float(
                    ratio_t.max()
                )
            elif np.any(ratio_t > warn):
                info.safe_action = schema.rewrite_action_torso(
                    info.safe_action, 0.5,
                )
                info.clipped = True
                info.reason.append(
                    f"L2c:torso_qvel_warning max={float(ratio_t.max()):.2f}x, scale=0.5"
                )

    # Chassis (only overspeed -> freeze; no warning band by design,
    # because chassis odom velocity is noisier than arm joint velocity
    # and a 0.5× attenuation can confuse navigation policies).
    if schema.has_chassis:
        qv_c = np.asarray(
            state.get_chassis_qvel(), dtype=np.float32,
        ).reshape(-1)[:3]
        if qv_c.size == 3:
            ratio_c = np.abs(qv_c) / np.maximum(
                self._cfg.chassis_qvel_max, 1e-6,
            )
            if np.any(ratio_c > over):
                info.safe_action = schema.rewrite_action_chassis(
                    info.safe_action, 0.0,
                )
                info.soft_hold = True
                info.reason.append(
                    f"L2c:chassis_qvel_overspeed max={float(ratio_c.max()):.2f}x"
                )
```

实施细节:

- **arm 用 `argmax(ratio)` 找最快关节**:reason 标签和 audit 字段都精确到 `J{i+1}`,而不是只说"某条手臂超速"。
- **torso 不带 audit 子项**(`l2_per_joint_clip` 没有 torso joint level 记录,只有 reason 里的 max ratio 数值)——torso 是承重轴,平滑减速比"哪个关节超了" 更重要,所以审计上简化。
- **chassis 只有 overspeed 单档 = soft_hold,没有 warning 缓冲档**:plan 草稿写"chassis 同样处理",但实施时刻意去掉了 warning(0.5× 缩放)那档。**理由**:底盘 odom 上报速度本身有噪声(轮子打滑、IMU drift),做 0.5× 渐变会让导航策略学到一个"怪状态"——干脆要么放过、要么 soft_hold 一帧重来,这与 BRS-ctrl `R1ProInterface._mobile_base_control` 的 dead-zone+硬 clip 风格一致。

#### 7.4.4 L2d — 命令速度上限(预防 mobiman watchdog)

```python
def _check_l2d_cmd_speed_cap(
    self,
    info: SafetyInfo,
    state: "GalaxeaR1ProRobotState",
    schema: "ActionSchema",
) -> None:
    """Bound the *commanded* velocity for joint-mode arms,
    torso (4-D twist) and chassis (3-D twist).

    For Cartesian-mode arms (the default M1 path) we cannot map
    EE deltas to joint-velocity demands without IK, so this
    layer no-ops on the arm slice in that case (L4 already caps
    Cartesian per-step distance).
    """
    dt = max(float(self._cfg.dt_step), 1e-3)
    d = schema.split(info.safe_action)

    # Arms (joint-mode only).
    if schema.use_joint_mode:
        for side in ("right", "left"):
            if side == "right" and not schema.has_right_arm:
                continue
            if side == "left" and not schema.has_left_arm:
                continue
            xyz = np.asarray(
                d.get(f"{side}_xyz", np.zeros(3)), dtype=np.float32,
            ).reshape(-1)[:3]
            rpy = np.asarray(
                d.get(f"{side}_rpy", np.zeros(3)), dtype=np.float32,
            ).reshape(-1)[:3]
            joint_delta6 = np.concatenate([xyz, rpy])
            v_cmd = (
                joint_delta6 * float(schema.action_scale[0]) / dt
            )
            qvm = self._cfg.arm_qvel_max[: v_cmd.size]
            ratio = np.abs(v_cmd) / np.maximum(qvm, 1e-6)
            max_ratio = float(np.max(ratio)) if ratio.size else 0.0
            if max_ratio > 1.0:
                scale = 1.0 / max_ratio
                info.safe_action = schema.rewrite_action_arm(
                    info.safe_action, side, scale,
                )
                info.clipped = True
                info.reason.append(
                    f"L2d:{side}_cmd_qvel_cap max={max_ratio:.2f}x limit, "
                    f"scale={scale:.2f}"
                )

    # Chassis: action *is* a velocity vector (m/s, m/s, rad/s).
    if schema.has_chassis:
        v_raw = np.asarray(
            d.get("chassis_twist", np.zeros(3)), dtype=np.float32,
        ).reshape(-1)[:3]
        v = v_raw.copy()
        modified = False
        # Dead-zone: only zero out non-zero values within the band.
        # ``dead & (v != 0.0)`` avoids the no-op rewrite when the
        # command is exactly zero (which is the very common case
        # of "policy outputs nothing for chassis this step").
        dead = np.abs(v) < self._cfg.chassis_dead_zone
        if np.any(dead & (v != 0.0)):
            v = np.where(dead, 0.0, v).astype(np.float32)
            modified = True
            self._record_per_joint_clip(
                info, "chassis", "deadzone",
                "L2d", v_raw.tolist(),
            )
        # Per-axis cap.
        v_clip = np.clip(
            v,
            -self._cfg.chassis_qvel_max,
            self._cfg.chassis_qvel_max,
        )
        if not np.array_equal(v_clip, v):
            modified = True
            self._record_per_joint_clip(
                info, "chassis", "cap",
                "L2d", v.tolist(),
            )
            v = v_clip
        if modified:
            info.safe_action = schema.rewrite_action_chassis_set(
                info.safe_action, v,
            )
            info.clipped = True
            info.reason.append(
                f"L2d:chassis_cmd_speed_cap raw={v_raw.tolist()}"
            )

    # Torso: action is a 4-D twist (v_x, v_z, w_pitch, w_yaw).
    if schema.has_torso:
        v_t = np.asarray(
            d.get("torso_twist", np.zeros(4)), dtype=np.float32,
        ).reshape(-1)[:4]
        ratio_t = np.abs(v_t) / np.maximum(
            self._cfg.torso_qvel_max, 1e-6,
        )
        if ratio_t.size and np.any(ratio_t > 1.0):
            scale = 1.0 / float(ratio_t.max())
            info.safe_action = schema.rewrite_action_torso(
                info.safe_action, scale,
            )
            info.clipped = True
            info.reason.append(
                f"L2d:torso_cmd_speed_cap scale={scale:.2f}"
            )
```

要点:

- **L2d 是命令侧的,不是反馈侧的**。L2c 已经看反馈了,L2d 看的是"这一步打算下发的命令"。两层组合等于 closed-loop 加 open-loop 双保险。
- **arm 在 EE 模式下 L2d 不动 arm slice**——因为没有 "命令速度=delta_q/dt" 的直接映射。当接入 IK 后(§7.2 IK 分支),`predict_arm_qpos` 会给出预测 q,L2d 可以通过 `(q_next - cur_q) / dt` 推命令速度。这是个"future-proof" 接口,目前只对 torso/chassis 有效。
- **chassis dead-zone 用 `dead & (v != 0.0)` 守卫**(实施期 fine-tune)。这是为了避免**最常见的 zero-cmd 情形**(policy 这一步什么也没下发)被无谓地标记成 `clipped`——`np.where` 把 `0.0` 替换成 `0.0` 是个 no-op,但 `info.reason` 还是会被无原因塞进 "L2d:chassis_cmd_speed_cap raw=[0,0,0]"。`(v != 0.0)` 这个 mask 让"0 → 0" 的伪修改不再触发审计,与 BRS-ctrl `R1ProInterface._mobile_base_control` 中 `cmd[set_zero] = 0` 的语义一致(它也是"小信号置零"而不是"任何信号都过 dead-zone")。
- **`rewrite_action_chassis_set`(绝对值替换)而不是 `rewrite_action_chassis`(比例缩放)**:dead-zone 改的是个别轴的绝对值,不能用同一缩放系数表达,所以 §7.3 的 schema 增加了一个 `set` 版本专门给 L2d 用。
- **chassis 与 torso 的修改顺序**:先 dead-zone(把抖动置零),再 cap(把太大值压回上限)——这样不会让一个值先被 cap 到 qvel_max,再被认为还在 dead-zone 之上而失去 clamp 的语义连贯性。两步**都触发 audit**,reason 仍只追加一行(避免 spam)。

### 7.5 `SafetyInfo` 新增审计字段

**插入位置**:[`r1_pro_safety.py:119-149`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py#L119-L149)。

```python
@dataclass
class SafetyInfo:
    raw_action: np.ndarray
    safe_action: np.ndarray
    clipped: bool = False
    soft_hold: bool = False
    safe_stop: bool = False
    emergency_stop: bool = False
    reason: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    # ── new in safety_2_joinlimit_2 ────────────────────────────
    l2_per_joint_clip: Dict[str, Dict[str, dict]] = field(
        default_factory=dict
    )
    """Per-joint, per-side L2 audit: which joint was scaled, how
    much, due to which sub-check. Used by analyzers to spot
    'always saturated J6' issues. Schema:

        {
          "right":   {"J3": {"L2a": {"scale": 0.5, "margin": 0.10},
                              "L2c": "scale=0.5"}},
          "left":    {"J7": {"L2b": "freeze"}},
          "torso":   {"T2": {"L2b": "freeze"}},
          "chassis": {"deadzone": {"L2d": [0.02, 0.5, 0.001]},
                      "cap":      {"L2d": [1.0, 0.5, -1.0]}},
          "gripper_right": {"g": {"L2": {"rate_cap": True,
                                         "pos_cap": False,
                                         "raw_a": 1.0,
                                         "new_a": 0.2}}},
        }

    Value types are intentionally heterogeneous:
      - dict: structured (L2a margins, L2 gripper rate/pos flags),
      - list: raw vector before rewrite (L2d chassis deadzone/cap),
      - str:  symbolic outcome ("freeze", "scale=0.5"), used by L2b/L2c.
    Analyzers should ``isinstance``-check before dereferencing.
    """
```

`SafetyInfo` 旧字段全部不变,这是**纯增量**,旧分析脚本不会受影响。

写入入口由 supervisor 内部 helper 统一管理(避免每次手写 setdefault 链):

```python
@staticmethod
def _record_per_joint_clip(
    info: SafetyInfo,
    side: str,
    joint_label: str,
    layer: str,
    value,
) -> None:
    """Append a per-joint audit entry to ``info.l2_per_joint_clip``."""
    side_dict = info.l2_per_joint_clip.setdefault(side, {})
    joint_dict = side_dict.setdefault(joint_label, {})
    joint_dict[layer] = value
```

5 个 L2 子检查方法都通过它写入审计,**保证 schema 一致**。

### 7.6 夹爪的特殊处理(v3 — 归一化语义)

> **v3 改良点**。本节由 v2 的 "百分比 0-100" 语义改为 [0, 1] 归一化、并显式钉死 0=closed / 1=open。**实质代码改动**:字段重命名 + 一段 mm↔norm 换算 helper。**控制流不变**:仍是 "rate 先于 pos" 的两步串行 predict-clip-rewrite,与 v2 §7.6 一致。

夹爪在 RLinf 的 action layout 中是**单一标量**(M1: action[6];M2: action[6] 与 action[13];Behavior: 第 21/22 维)。它有两个 L2 约束:

```python
def _check_l2_gripper(
    self,
    info: SafetyInfo,
    state: "GalaxeaR1ProRobotState",
    schema: "ActionSchema",
) -> None:
    """Two L2 checks for each enabled gripper, expressed in the
    RLinf normalised [0, 1] working space (0 = fully closed, 1 = fully
    open):

    1. Per-step delta cap (``max_gripper_step``):
       |Δ pos_norm| ≤ max_gripper_step.
    2. Position bound: predicted next pos_norm ∈
       [gripper_pos_min, gripper_pos_max].

    Both are implemented in the predict-clip-rewrite style used
    by L3a so that the rest of the pipeline sees a clean
    normalised action.

    *Semantics convention*: RLinf uses ``a = -1 -> closed`` /
    ``a = +1 -> open`` (matching the existing
    ``r1_pro_env._dispatch_action`` ``(a+1)*50`` mapping).  This is
    the **opposite** of BRS-ctrl's ``act(0..1)`` (BRS: 0=open,
    1=closed); see §3.6 for the cross-table.
    """
    if schema.no_gripper:
        return

    # Normalised action lives in [-1, 1]; scale_a maps it to a
    # delta in the same [0, 1] normalised pos space.  We default
    # to ``action_scale[2]`` divided by the stroke envelope so the
    # tests / YAML can keep using mm-flavoured numbers if desired.
    open_mm = float(self._cfg.gripper_open_stroke_mm)
    closed_mm = float(self._cfg.gripper_closed_stroke_mm)
    stroke_span = max(open_mm - closed_mm, 1e-9)

    # If the schema's action_scale[2] is given in "mm/unit" (BRS-style),
    # convert it to "norm/unit" by dividing by stroke_span.  If it's
    # already given in normalised units (the typical RLinf default),
    # the user can simply leave action_scale[2] = 1.0.
    raw_scale = max(float(schema.action_scale[2]), 1e-9)
    # Heuristic: action_scale[2] > 5 -> almost certainly in mm.
    scale_norm = raw_scale / stroke_span if raw_scale > 5.0 else raw_scale

    max_step = float(self._cfg.max_gripper_step)
    lo = float(self._cfg.gripper_pos_min)
    hi = float(self._cfg.gripper_pos_max)

    for side in ("right", "left"):
        if side == "right" and not schema.has_right_arm:
            continue
        if side == "left" and not schema.has_left_arm:
            continue
        d = schema.split(info.safe_action)
        key = f"{side}_gripper"
        if key not in d:
            continue

        a_g = float(d[key])  # already in [-1, 1] post L1
        # Convert ROS-side mm feedback to normalised [0, 1] pos.
        cur_mm = float(state.get_gripper_pos(side))
        cur_pos = (cur_mm - closed_mm) / stroke_span
        cur_pos = float(np.clip(cur_pos, 0.0, 1.0))
        delta_pos = a_g * scale_norm  # signed delta in [0,1] space

        # 1) rate cap
        new_a = a_g
        rate_capped = False
        if abs(delta_pos) > max_step:
            new_a = float(np.sign(a_g) * max_step / scale_norm)
            rate_capped = True

        # 2) position bound — runs *after* the rate cap, on the
        # already-rate-capped delta.
        next_pos = cur_pos + new_a * scale_norm
        pos_capped = False
        if next_pos < lo or next_pos > hi:
            target = float(np.clip(next_pos, lo, hi))
            new_a = float((target - cur_pos) / scale_norm)
            pos_capped = True

        if rate_capped or pos_capped:
            info.safe_action = schema.set_gripper_action(
                info.safe_action, side, new_a,
            )
            info.clipped = True
            tag_parts = []
            if rate_capped:
                tag_parts.append(
                    f"rate_cap |Δ|={abs(delta_pos):.3f}>"
                    f"{max_step:.3f}/step"
                )
            if pos_capped:
                tag_parts.append(
                    f"pos_cap pred={next_pos:.3f}->[{lo:.2f},{hi:.2f}]"
                )
            info.reason.append(
                f"L2:{side}_gripper " + " ".join(tag_parts)
            )
            self._record_per_joint_clip(
                info, f"gripper_{side}", "g",
                "L2",
                {
                    "rate_cap": rate_capped,
                    "pos_cap": pos_capped,
                    "raw_a": a_g,
                    "new_a": new_a,
                    "cur_pos_norm": cur_pos,
                    "cur_pos_mm": cur_mm,
                },
            )
```

**v3 与 v2 的关键差异**(纯实现层):

- 字段名:`gripper_pct_min/max` → `gripper_pos_min/max`(数值范围 [0, 1]),`max_gripper_step_pct` → `max_gripper_step`,新增 `gripper_closed_stroke_mm` / `gripper_open_stroke_mm` 两个 mm↔norm 换算端点。
- **mm → norm 自动换算**:`cur_pos = (cur_mm - closed_mm) / stroke_span`,然后整段 L2 在 [0, 1] 空间运算。
- **`scale_norm` 启发式**:`action_scale[2] > 5.0` 视为 mm-flavoured 数值(例如 BRS 的 stroke 风格 50,RLinf 旧 v2 测试里也用过 50),自动除以 `stroke_span` 转成归一化;`≤ 5.0` 则视为已经归一化(典型 RLinf 默认 `1.0`)。这避免破坏现有 YAML/单测里两种风格并存的现状。
- **审计字段**:`l2_per_joint_clip[gripper_*]['g']['L2']` 的 dict 里多了 `cur_pos_norm` 与 `cur_pos_mm`,便于现场对照"读到什么 mm,等价到什么 [0,1]"。

**保留自 v2 的关键设计**(不变):

- **不离散化**(与 [`Pi05_ActionSpace_Analysis.md`](Pi05_ActionSpace_Analysis.md) §5 一致)。
- **predict-clip-rewrite**:与 L3a 同一套模式——预测下一步 pos_norm,clamp,反向算回归一化 action。
- **依赖 `schema.set_gripper_action`**(沿用 `rewrite_action_*` 的设计)。
- **顺序敏感:rate 先于 pos**——`rate_cap` 改的是 `new_a`,`pos_cap` 用 `new_a * scale_norm` 计算 `next_pos`,如果先做 pos_cap 再做 rate_cap,有可能"先把目标 pos 锁回 hi,再 rate-limit 到一个**不到 hi** 的 next_pos",失去 pos 上下界保证。
- **`scale_norm = max(..., 1e-9)`**:防止用户把 gripper scale 设为 0 时除零。
- **L1 与 gripper scale 的相互作用**(详见 [附录 C.4 v3 更新版](#c4-l1-clip-与-gripper-scale-的相互作用-v3-归一化更新)):L1 把 `a` clip 到 [-1, 1] 之后,L2 看到的是已经被 clip 过的 a。测试时只要 `action_scale[2]` 设得足够大(例如 1.0,因为现在 [0, 1] 归一化空间中 `1.0/step = 100% stroke` 已经够触发 cap 了),就能直接用 `a = ±1` 触发 rate/pos cap,**不再需要 v2 中的 `action_scale[2]=50` workaround**。

#### 7.6.1 一个具体例子(R1 Pro 现场)

假设当前夹爪反馈 `pos_mm = 5.0`(几乎闭合),`closed=0, open=100`,所以 `pos_norm = 0.05`。

- 配置:`max_gripper_step=0.30, gripper_pos_min=0.0, gripper_pos_max=1.0, action_scale[2]=1.0`(归一化口径)。
- Policy 输出 `a = +1.0`(意图:**全张开**)。
- L2 gripper 计算 `delta_pos = 1.0 * 1.0 = 1.0` —— 超出 `max_step=0.30` → **rate_cap 触发**。
- `new_a = +1 * 0.30 / 1.0 = +0.30`,等价于 `delta_pos = +0.30 → next_pos = 0.05 + 0.30 = 0.35`。
- `next_pos = 0.35 ∈ [0, 1]` —— **pos_cap 不触发**。
- 最终 `safe_action[gripper] = 0.30`,reason `L2:right_gripper rate_cap |Δ|=1.000>0.300/step`。

经 dispatch:`stroke_mm = 0 + 0.35 * 100 = 35 mm`(从 5 mm 走到 35 mm,1 步推进 30 mm)。这正是 0.30 normalised step 对应的物理移动量,工程上可以承受(@10Hz 控制频率下连续 3 步即可达到 95% open)。

---

## 8. 对其它代码的最小侵入式改动清单

> **状态:已实施**(2026-05-05),实施 diff 与本节预期一致,具体行号见下表"实际"列。

| # | 文件 | 改动类型 | 实际(行号 / 行数) | 风险 |
|---|---|---|---|---|
| 1 | [`r1_pro_safety.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py) | 拆 `arm_q_min/max` → `right_arm_q_*` + `left_arm_q_*`,新增 16+ 个 L2 子层 / 阈值 / toggle 字段 | `SafetyConfig` 全段重写,合计约 **+102 行**(行 60-178) | 中:旧 YAML 用 `arm_q_min/max` 仍能工作(`__post_init__` 兼容,带 deprecation warning) |
| 2 | 同上 | `SafetyInfo` 新增 `l2_per_joint_clip` 字段 + helper `_record_per_joint_clip` | 行 132-149 + helper 行 314-326,**纯增量** | 低 |
| 3 | 同上 | 替换 L2 注释占位为 5 个子检查方法调用(L2a/b/c/d + gripper) | 行 215-228 | 低:有 5 个 `enable_l2*` toggle |
| 4 | 同上 | 新增 5 个私有方法 `_check_l2{a,b,c,d}_*` + `_check_l2_gripper` | 合计约 **+440 行** | 低 |
| 5 | [`r1_pro_action_schema.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_action_schema.py) | 新增 6 个方法:`predict_arm_qpos`(三分支)+ `rewrite_action_arm/torso/chassis` + `rewrite_action_chassis_set` + `set_gripper_action` | 行 144-353,**+210 行** | 低 |
| 6 | [`r1_pro_robot_state.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_robot_state.py) | 新增 helper:`get_torso_qpos/qvel`、`get_chassis_qpos/qvel`、`get_gripper_vel` | 行 160-180,**+17 行** | 低:纯增量 |
| 7 | [`r1_pro_env.py:399-457`](../../../rlinf/envs/realworld/galaxear/r1_pro_env.py#L399-L457) | **不需要改**——`step()` 已经把 state 喂给 supervisor | — | — |
| 8 | [`r1_pro_controller.py:286-487`](../../../rlinf/envs/realworld/galaxear/r1_pro_controller.py#L286-L487) | **不需要改**——已经填充所有需要的字段 | — | — |
| 9 | [`tests/unit_tests/test_galaxea_r1_pro_safety_l2.py`](../../../tests/unit_tests/test_galaxea_r1_pro_safety_l2.py) | **新文件**,**16 个 L2 单元测试**(plan §10 设计 12 个 + 4 个补强) | **+599 行** | 低 |
| 10 | [`tests/unit_tests/conftest.py`](../../../tests/unit_tests/conftest.py) | **新文件**,在 Orin 最小环境下 stub `rlinf.scheduler` / `rlinf.envs.realworld.{franka,xsquare,realworld_env}`,让轻量测试可在真机直接 collect | **+141 行** | 低:GPU server 完整 install 上是 no-op(只在 `_try_import` 失败时才 stub) |
| 11 | YAML 默认配置 | 无变动 | — | — |

**核心:不动 controller、不动 env、不订阅新 topic、不引入 IK 库**。所有改动都集中在 `safety.py` + `action_schema.py` + `robot_state.py`,对真机部署来说就是"拉一次代码、再启一次 RL job"。

实施总计:**+839 行(代码)+ 740 行(测试)= 1579 行新代码**,**0 行老 RLinf 生产代码删除**(只是修改/扩展 SafetyConfig 默认值,字段名向前兼容)。

### 8.1 在 robot_state.py 中加 helper

**插入位置**:[`r1_pro_robot_state.py:160-180`](../../../rlinf/envs/realworld/galaxear/r1_pro_robot_state.py#L160-L180)(实施已落点)。

```python
def get_torso_qpos(self) -> np.ndarray:
    return np.asarray(self.torso_qpos, dtype=np.float32).reshape(-1)[:4]

def get_torso_qvel(self) -> np.ndarray:
    return np.asarray(self.torso_qvel, dtype=np.float32).reshape(-1)[:4]

def get_chassis_qpos(self) -> np.ndarray:
    return np.asarray(self.chassis_qpos, dtype=np.float32).reshape(-1)[:3]

def get_chassis_qvel(self) -> np.ndarray:
    return np.asarray(self.chassis_qvel, dtype=np.float32).reshape(-1)[:3]

def get_gripper_vel(self, side: str) -> float:
    return float(
        self.right_gripper_vel if side == "right" else self.left_gripper_vel
    )

# v3 ADDITION: normalised gripper position helper.
def get_gripper_pos_norm(
    self,
    side: str,
    *,
    closed_stroke_mm: float = 0.0,
    open_stroke_mm: float = 100.0,
) -> float:
    """Return the gripper opening as a normalised float in [0, 1].

    R1 Pro G1 reports `JointState.position[0]` in **mm** on
    ``/hdas/feedback_gripper_*`` (factory range typically [0, 100]
    mm; BRS-ctrl uses the safer subrange [10, 90] mm).  This
    helper converts the raw mm reading to RLinf's normalised
    convention:

        0.0 = fully closed (stroke == closed_stroke_mm)
        1.0 = fully open   (stroke == open_stroke_mm)

    Out-of-range mm values are silently clamped so that
    callers (the L2 supervisor) always see a value in [0, 1].

    Note: this is the **opposite** of the BRS-ctrl
    ``GalaxeaR1ProGripper.act(action)`` convention, which uses
    0=open, 1=closed.  See ``bt/docs/rwRL/safety_2_joinlimit_3.md``
    §3.6 for the full cross-table.
    """
    span = max(open_stroke_mm - closed_stroke_mm, 1e-9)
    pos_mm = (
        self.right_gripper_pos if side == "right"
        else self.left_gripper_pos
    )
    return float(np.clip((pos_mm - closed_stroke_mm) / span, 0.0, 1.0))
```

(`get_arm_qpos / get_arm_qvel / get_gripper_pos` 已经在文件中存在。实施版**比 plan 草稿多了一个 `get_chassis_qpos`** —— 加上是为了将来扩展底盘 odom-based safety 时不必再回头改 dataclass。**v3 新增 `get_gripper_pos_norm`** 是为了集中 mm → [0, 1] 的换算逻辑,避免 supervisor 内部到处散落 stroke 算式;`closed_stroke_mm` / `open_stroke_mm` 使用关键字参数,L2 调用时直接从 SafetyConfig 取值。)

---

## 9. 配置层:YAML 字段与 SafetyConfig 字段对照

YAML 用户配置示例(`examples/realworld/galaxear/r1_pro_*.yaml` 的 `safety_cfg` 节):

```yaml
safety_cfg:
  # L2 position limits — 留空则使用 §3.7 给的 URDF-derived 默认
  right_arm_q_min: [-4.35, -3.04, -2.26, -1.99, -2.26, -0.95, -1.47]
  right_arm_q_max: [ 1.21,  0.07,  2.26,  0.25,  2.26,  0.95,  1.47]
  left_arm_q_min:  [-4.35, -0.07, -2.26, -1.99, -2.26, -0.95, -1.47]
  left_arm_q_max:  [ 1.21,  3.04,  2.26,  0.25,  2.26,  0.95,  1.47]
  arm_qvel_max:    [1.6, 1.6, 1.6, 1.6, 4.0, 4.0, 4.0]

  torso_q_min:     [-1.03, -2.69, -1.73, -2.95]
  torso_q_max:     [ 1.73,  2.43,  1.47,  2.95]
  torso_qvel_max:  [0.5, 0.5, 0.5, 0.5]

  chassis_qvel_max: [0.30, 0.30, 0.40]
  chassis_dead_zone: [0.01, 0.01, 0.05]

  # Gripper (v3 normalised: 0 = fully closed, 1 = fully open).
  # See safety_2_joinlimit_3.md §3.6 for the mm <-> [0,1] mapping
  # and the contrast against BRS-ctrl's (0=open, 1=closed) convention.
  gripper_pos_min: 0.0
  gripper_pos_max: 1.0
  max_gripper_step: 0.30
  # mm endpoints used to convert ROS-side stroke -> normalised [0, 1].
  # Factory full range is [0, 100] mm; BRS-ctrl uses the safer subset
  # [10, 90] mm.  Switching to BRS values gives a 10 mm bumper margin
  # at both ends without touching policy code.
  gripper_closed_stroke_mm: 0.0
  gripper_open_stroke_mm: 100.0

  # L2 thresholds
  l2_warning_margin_rad:  0.15
  l2_critical_margin_rad: 0.05
  l2_qvel_overspeed_factor: 1.20
  l2_qvel_warning_factor:   0.90

  # L2 toggles (默认全开;调试时可关掉某层定位问题)
  enable_l2a_predict_q_clip: true
  enable_l2b_qpos_freeze:    true
  enable_l2c_qvel_watchdog:  true
  enable_l2d_cmd_speed_cap:  true
```

`build_safety_config(cfg_dict)`([`r1_pro_safety.py:442-457`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py#L442-L457))已经支持"忽略未知字段",新字段不需要改这个函数。

### 9.1 不同 stage 下的字段子集

| Stage | 必需字段 | 可省略字段 |
|---|---|---|
| **M1 单臂(右臂)** | `right_arm_q_min/max`, `arm_qvel_max`, `gripper_*`, `l2_*_margin_rad`, `l2_qvel_*_factor` | left/torso/chassis 全部 |
| **M2 双臂** | + `left_arm_q_min/max` | torso/chassis |
| **M3 + torso** | + `torso_q_min/max`, `torso_qvel_max` | chassis |
| **M4 + chassis** | + `chassis_qvel_max`, `chassis_dead_zone` | — |

---

## 10. 测试矩阵与单元测试设计

[`tests/unit_tests/test_galaxea_r1_pro_safety.py`](../../../tests/unit_tests/test_galaxea_r1_pro_safety.py) 现有 11 个测试,新增 12 个 L2 测试,覆盖以下场景:

```mermaid
graph LR
    T1["1. URDF 限位 vs SafetyConfig 一致性 (parse URDF + cmp)"]
    T2["2. 左右臂 J2 镜像被正确装载"]
    T3["3. L2a 单关节 warning_margin 触发线性 scale"]
    T4["4. L2a 单关节 critical_margin 触发 scale=0"]
    T5["5. L2a 笛卡尔模式 q_next=cur_q (no-op)"]
    T6["6. L2b 当前 qpos 已超出 critical -> soft_hold"]
    T7["7. L2c qvel > 0.9× limit -> scale 0.5"]
    T8["8. L2c qvel > 1.2× limit -> soft_hold"]
    T9["9. L2d chassis dead-zone 抑制"]
    T10["10. L2d chassis 超 vmax 被 clip"]
    T11["11. 夹爪超 [0,1] 边界(预测越界)被 clamp,与 RLinf 0=closed/1=open 约定一致"]
    T12["12. 夹爪 step delta > max_gripper_step (归一化空间) 被 rate-limit"]
    T13["13. (impl) 旧别名 arm_q_min/max 仍能 splat 到左右两份"]
    T14["14. (impl) SafetyInfo.l2_per_joint_clip 写入 schema 正确"]
    T15["15. (impl) 双臂时仅越限那只手被裁剪"]
    T16["16. (impl) enable_l2* toggle 全关时无 L2 reason"]

    classDef pos fill:#cce
    classDef vel fill:#fcc
    classDef gri fill:#cec
    classDef impl fill:#fec
    class T1,T2,T3,T4,T5,T6 pos
    class T7,T8,T9,T10 vel
    class T11,T12 gri
    class T13,T14,T15,T16 impl
```

实施版**实际有 16 个用例**:plan 草稿设计 12 个,实施期补强 4 个(T13-T16)以覆盖 back-compat、audit field schema、双臂独立性、toggle 行为四个工程化关注点。**全部通过**(连跑 3 轮稳定 16/16,详见 §14)。

每个测试都:
- 用 `MagicMock`/手工构造 `RobotState` 与 `ActionSchema`,**不依赖 ROS2**;
- 检查 `info.reason` 是否包含正确 tag(如 `"L2a:right_q_warning J3=..."`);
- 检查 `info.safe_action` 是否被正确缩放/置零;
- 检查 `info.metrics["safety/l2*"]` 与 `info.l2_per_joint_clip` 是否被正确填充。

### 10.1 URDF 一致性测试(关键 — 实施版)

```python
def _find_urdf() -> str | None:
    """Probe the standard SDK locations; return None on minimal hosts."""
    candidates = [
        "/home/nvidia/galaxea/install/mobiman/share/mobiman/urdf/R1_PRO/urdf/r1_pro.urdf",
        "/home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/urdfs/r1_pro_floating_right.urdf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def test_l2_defaults_match_urdf_with_margin():
    """SafetyConfig defaults must lie strictly inside the URDF
    mechanical limits (URDF lower < q_min, q_max < URDF upper).

    Skipped on hosts where the Galaxea SDK is not installed (e.g.
    CI runners without ``/home/nvidia/galaxea/install``)."""
    urdf_path = _find_urdf()
    if urdf_path is None:
        pytest.skip("Galaxea SDK URDF not found; skipping cross-check")

    root = ET.parse(urdf_path).getroot()
    urdf_limits: dict[str, tuple[float, float]] = {}
    for j in root.findall("joint"):
        if j.get("type") != "revolute":
            continue
        name = j.get("name", "")
        if not (name.startswith("left_arm_joint")
                or name.startswith("right_arm_joint")):
            continue
        lim = j.find("limit")
        if lim is None:
            continue
        urdf_limits[name] = (
            float(lim.get("lower")),
            float(lim.get("upper")),
        )

    cfg = SafetyConfig()
    for i in range(7):
        # Right arm.
        rname = f"right_arm_joint{i + 1}"
        if rname in urdf_limits:
            urdf_lo, urdf_hi = urdf_limits[rname]
            cfg_lo = float(cfg.right_arm_q_min[i])
            cfg_hi = float(cfg.right_arm_q_max[i])
            assert cfg_lo > urdf_lo - 1e-3, (
                f"right J{i + 1}: q_min={cfg_lo} <= URDF lower={urdf_lo}"
            )
            assert cfg_hi < urdf_hi + 1e-3, (
                f"right J{i + 1}: q_max={cfg_hi} >= URDF upper={urdf_hi}"
            )
        # Left arm: only present in the integrated URDF; floating-right
        # URDF won't have it -> skip silently.
        lname = f"left_arm_joint{i + 1}"
        if lname in urdf_limits:
            urdf_lo, urdf_hi = urdf_limits[lname]
            cfg_lo = float(cfg.left_arm_q_min[i])
            cfg_hi = float(cfg.left_arm_q_max[i])
            assert cfg_lo > urdf_lo - 1e-3, (
                f"left J{i + 1}: q_min={cfg_lo} <= URDF lower={urdf_lo}"
            )
            assert cfg_hi < urdf_hi + 1e-3, (
                f"left J{i + 1}: q_max={cfg_hi} >= URDF upper={urdf_hi}"
            )
```

实施版本与 plan 草稿的两点差异:

1. **双 URDF 候选 fallback**:整机 URDF 优先,float-right URDF 兜底。这让测试在"只装了 mobiman 子集"的开发机上仍能跑(只验证右臂)。
2. **左臂 only-if 装载分支**:`floating_right.urdf` 里没有 `left_arm_joint*`,所以左臂的检查放在 `if lname in urdf_limits:` 守卫之内,不强制要求所有 7×2 关节都被验证。

这个测试**自动告警**未来任何人改 SafetyConfig 默认值时把 L2 设的比 URDF 还宽。是 [`glx/mismatch_realworld_1.md`](glx/mismatch_realworld_1.md) §C.1 那种 bug 的 regression guard。在 Orin 上**真机命中**(SDK 已装)是关键的"上真机预检"信号。

---

## 11. 运行时调试 / 排错

### 11.1 把每层 L2 的触发情况通过 `MetricLogger` 打到 wandb / tensorboard

`metrics["safety/l2a_right_scale"]` / `safety/l2a_right_worst_margin` / `safety/l2b_right_freeze` 等在 [`r1_pro_env.py:451`](../../../rlinf/envs/realworld/galaxear/r1_pro_env.py#L451) 已经会被复制进 `info["safety_info"]`,然后由训练 runner 的 `MetricLogger` 自动消费。**不需要改 logger**。

### 11.2 现场快速判断"L2 是否在工作"

```bash
# 在 GPU server 上启动训练后,看 stdout 里的 reason 列表:
[INFO] step 042 safety_reasons=['L2a:right_q_warning J3=...', 'L4:per_step_cap']
# 或在 REPL 测试中(toolkits/realworld_check/test_galaxea_r1_pro_controller.py):
r1pro> setdelta 1.0 0 0 0 0 0 0       # 故意推到极限
[WARN]  L2a:right_q_warning worst=+0.04rad scale=0.27
[INFO]  L4:per_step_cap
[INFO]  Sent pose7 = [...]
```

### 11.3 关闭单一子层定位问题

```yaml
# 想验证 L2a 是不是误触发?暂时关掉它:
safety_cfg:
  enable_l2a_predict_q_clip: false
  # L2b/L2c/L2d 仍然在工作,提供保护
```

### 11.4 极端情况:supervisor 内部异常

`validate()` 内部任何子层抛出异常,supervisor **不会传播**——会被 `try/except` catch 后:
- 把 `info.emergency_stop = True`(保守做法)
- `info.reason.append(f"L2_internal_error: {type(e).__name__}: {e}")`

### 11.5 在 Orin 最小环境下 collect 测试(conftest.py stub)

实施期发现:Orin 上 RLinf 的 `from rlinf.envs.realworld.galaxear.r1_pro_safety import ...` 这条路径**会顺便触发** `rlinf/envs/realworld/__init__.py:15-18` 导入 Franka / Turtle2 / `rlinf.scheduler` 等重依赖(`gymnasium`、`ray`、`hydra`、JetPack-fork 的 `torch.distributed.Work` 等)。在 Orin 真机上跑安装这些会让"先装能跑测试的最小依赖"这件事变得不现实。

[`tests/unit_tests/conftest.py`](../../../tests/unit_tests/conftest.py) 的解决方法:

```python
def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False

# Only stub when real import fails -- on a full GPU-server install
# this conftest is a no-op.
if not _try_import("rlinf.scheduler"):
    _ensure_stub("rlinf.scheduler", {
        "Cluster": _StubAttr,
        "NodePlacementStrategy": _StubAttr,
        "Worker": _StubAttr,
        "WorkerInfo": _StubAttr,
        "GalaxeaR1ProHWInfo": _StubAttr,
    })

if not _try_import("rlinf.envs.realworld.franka"):
    _franka_tasks = types.ModuleType("rlinf.envs.realworld.franka.tasks")
    _ensure_stub("rlinf.envs.realworld.franka", {
        "FrankaEnv": _StubAttr,
        "FrankaRobotConfig": _StubAttr,
        "FrankaRobotState": _StubAttr,
        "tasks": _franka_tasks,
    })
    sys.modules["rlinf.envs.realworld.franka.tasks"] = _franka_tasks
# ... xsquare / realworld_env 同理
```

要点:

- **不 stub 父包 `rlinf` / `rlinf.envs` / `rlinf.envs.realworld`**——它们必须用真实 path,否则 `rlinf.envs.realworld.galaxear` 解析不出来。
- **只 stub 失败的子模块**(用 `_try_import` 探测),所以在 GPU server 完整 install 上是 no-op。
- **stub `tasks` 子模块**:`rlinf/envs/realworld/__init__.py` 还会做 `from .franka import tasks as franka_tasks`,因此 `tasks` 必须作为 attribute 出现在 stubbed package 上,且**还要** sys.modules 里注册一份 module(`from package import attr` 在某些 Python 实现下走 `importlib.import_module(...)` 路径)。
- **效果**:Orin 上 64 个 Galaxea 单测全部可 collect & 通过,不需要 `pip install ray gymnasium hydra`(只需要 `pip install gymnasium` 也能跑通,因为 conftest stub 提供了 ray 替代;实施期实测装了 `ray --no-deps` 后 conftest stub 仍按 fallback 路径接管)。

这个设计在原来的 supervisor 里没有,但应该加上,保证 L2 缺陷不会让整个 episode 跑飞。建议放在 `validate()` 顶部:

```python
try:
    self._check_l1(action, info)
    if self._cfg.enable_l2a_predict_q_clip:
        self._check_l2a_predict_q_clip(info, state, action_schema)
    # ...
except Exception as e:
    info.emergency_stop = True
    info.reason.append(f"L2_internal_error: {type(e).__name__}: {e}")
    info.safe_action = np.zeros_like(info.safe_action)
```

---

## 12. 与生产代码 (BRS-ctrl) 的关系与差异

| 维度 | BRS-ctrl 生产代码 | RLinf L2 (本设计) | 一致性 / 差异 |
|---|---|---|---|
| 限位数值 | 完全等于 URDF | URDF - 0.1 rad | RLinf 更保守 |
| 检查时机 | 命令下发前 (`_upper_body_joint_position_control`) | 命令下发前(supervisor.validate) | **一致** |
| 越界行为 | clip + warn 或 raise(可配置) | scale + soft_hold + reason audit | RLinf 更细粒度 |
| 反馈速度看门狗 | 没有 | L2c 实现 | RLinf 多一层 |
| 命令速度上限 | 通过 `arm_joint_control_step_interval=0.4s` 间接限速 | L2d 显式限速 | 思路不同,效果接近 |
| 夹爪 | 单独类 `R1ProBaseGripper.act(0..1)` | L2 内置 clamp + rate-limit | RLinf 集成,BRS 解耦 |
| 多臂支持 | 左右臂独立限位 | 左右臂独立限位 (本设计修复) | **一致** |
| 配置可调 | `on_arm_cmd_out_of_range="clip"\|"raise"` | YAML 全字段 + toggle | RLinf 更灵活 |

**结论**:RLinf L2 在保守程度、反馈观察、审计粒度上**严格优于** BRS-ctrl 的现有实现,但**思路是一致的**——都把"在命令下发前对 q 做硬限位"作为最后一道防线。BRS-ctrl 适合 imitation learning(数据采集),RLinf 设计适合 RL training(需要平滑、需要审计、需要不中断 episode)。

---

## 13. 参考资料 + 源码锚点

### 13.1 RLinf 源码

| 文件 | 关键行(实施后) | 内容 | 本设计动作 |
|---|---|---|---|
| [`r1_pro_safety.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py) | 60-178 | `SafetyConfig`(扩展) | **已改**(§7.1) |
| [`r1_pro_safety.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py) | 132-149 | `SafetyInfo` + `l2_per_joint_clip` | **已加**(§7.5) |
| [`r1_pro_safety.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py) | 215-228 | L2 段(原占位注释) | **已替换**(§7.4)|
| [`r1_pro_safety.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_safety.py) | 314-326 + L2{a,b,c,d} + gripper helpers | 5 个新私有方法 + 1 个 audit helper | **已加** |
| [`r1_pro_action_schema.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_action_schema.py) | 144-237 | `predict_arm_qpos` | **已加**(§7.2) |
| [`r1_pro_action_schema.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_action_schema.py) | 239-353 | `rewrite_action_arm/torso/chassis` + `rewrite_action_chassis_set` + `set_gripper_action` | **已加**(§7.3) |
| [`r1_pro_robot_state.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_robot_state.py) | 160-180 + v3 | helper:`get_torso_qpos/qvel`、`get_chassis_qpos/qvel`、`get_gripper_vel`,**v3 加 `get_gripper_pos_norm`**(mm → [0, 1] 归一化转换) | **已加**(§8.1) |
| [`r1_pro_controller.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_controller.py) | 286-348, 417-487 | ROS2 订阅 + 回调 | **不改** |
| [`r1_pro_env.py`](../../../rlinf/envs/realworld/galaxear/r1_pro_env.py) | 399-457 | `step()` | **不改** |
| [`tests/unit_tests/test_galaxea_r1_pro_safety.py`](../../../tests/unit_tests/test_galaxea_r1_pro_safety.py) | 全文件 | 现有 11 个 L1-L5 smoke 测试 | **不改**(已通过 regression) |
| [`tests/unit_tests/test_galaxea_r1_pro_safety_l2.py`](../../../tests/unit_tests/test_galaxea_r1_pro_safety_l2.py) | 全文件(599 行) | **新增 16 个 L2 单元测试**(§10) | **新文件** |
| [`tests/unit_tests/conftest.py`](../../../tests/unit_tests/conftest.py) | 全文件(141 行) | Orin 最小环境 stub(§11.5) | **新文件** |

### 13.2 R1 Pro SDK 源码(Orin 上)

| 文件 | 内容 | 用途 |
|---|---|---|
| [`/home/nvidia/galaxea/install/mobiman/share/mobiman/urdf/R1_PRO/urdf/r1_pro.urdf`](file:///home/nvidia/galaxea/install/mobiman/share/mobiman/urdf/R1_PRO/urdf/r1_pro.urdf) | 整机 URDF (~2700 行) | 限位真值来源 |
| [`/home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/urdfs/r1_pro_floating_left.urdf`](file:///home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/urdfs/r1_pro_floating_left.urdf) | 左臂浮动 URDF | J2 镜像确认 |
| [`/home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/urdfs/r1_pro_floating_right.urdf`](file:///home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/urdfs/r1_pro_floating_right.urdf) | 右臂浮动 URDF | 与整机 URDF 交叉验证 |
| [`/home/nvidia/galaxea/install/mobiman/lib/mobiman/relaxed_ik.py`](file:///home/nvidia/galaxea/install/mobiman/lib/mobiman/relaxed_ik.py) | Relaxed IK 实现 | 内部约束行为(`safety_2_joinlimit.md` §5.1.1) |

### 13.3 BRS-ctrl 真机参考

| 文件 | 关键行 | 内容 |
|---|---|---|
| [`/home/nvidia/kaizhe_ws/data_collect/brs-ctrl/brs_ctrl/robot_interface/interfaces.py`](file:///home/nvidia/kaizhe_ws/data_collect/brs-ctrl/brs_ctrl/robot_interface/interfaces.py) | 723-737 | `R1ProInterface.{torso,left_arm,right_arm}_joint_{high,low}` |
| 同上 | 982-1036 | `_upper_body_joint_position_control` 中的 clip + warn 逻辑 |
| [`/home/nvidia/kaizhe_ws/data_collect/brs-ctrl/brs_ctrl/reset.py`](file:///home/nvidia/kaizhe_ws/data_collect/brs-ctrl/brs_ctrl/reset.py) | 87 | `send_vel_limit([1.6,1.6,1.6,1.6,4.0,4.0,4.0])` |

### 13.4 RLinf 设计文档系列

- [`r1pro5op47.md`](r1pro5op47.md) §6.4 — 五级闸门设计原文
- [`r1pro5op47_imp1.md`](r1pro5op47_imp1.md) — 第一阶段实施
- [`safety_2.md`](safety_2.md) §4 / §5 / §9 — 五级实现深度审计 + 已知差距(L2 位列 §9.1 P1)
- [`safety_2_joinlimit.md`](safety_2_joinlimit.md) — L2 上下文 + 方案 A/B/C 对比(本文聚焦方案 A 的产品化实现)
- [`glx/mismatch_realworld_1.md`](glx/mismatch_realworld_1.md) §C.1 — 当前 SafetyConfig 默认值与 URDF 不一致问题原始记录
- [`glx/R1ProSDKAnalysis.md`](glx/R1ProSDKAnalysis.md) — SDK 全景分析,含 mobiman/HDAS 拓扑
- [`Pi05_ActionSpace_Analysis.md`](Pi05_ActionSpace_Analysis.md) §5 — 夹爪连续 vs 离散的辨析(本文 §3.6 / §7.6 引用)
- [`test_galaxea_r1_pro_controller.md`](test_galaxea_r1_pro_controller.md) §A.3.3 — REPL 中验证 L2 的现场流程

### 13.5 学术引用

- Rakita, Mutlu, Gleicher (2018). [RelaxedIK: Real-time Synthesis of Accurate and Feasible Robot Arm Motion.](https://www.roboticsproceedings.org/rss14/p43.pdf) *RSS 2018.*
- Rakita, Mutlu, Gleicher (2020). [An analysis of RelaxedIK.](https://link.springer.com/article/10.1007/s10514-020-09918-9) *Autonomous Robots, 44, 1341–1358.*
- BRS (CoRL 2025). [BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities.](https://arxiv.org/abs/2503.05652)
- Galaxea (2024). [R1 Pro Software Guide ROS2.](https://docs.galaxea-dynamics.com/Guide/R1Pro/software_introduction/R1Pro_Software_Guide_ROS2/)

---

## 14. 实施记录(Implementation Log)

> 时间:2026-05-05 · 提交者:RLinf RWRL Working Group · 工作环境:Orin 真机(`nvidia-desktop`,JetPack)+ R1 Pro 已开机。
>
> **本节是设计 → 代码的对账卡片**。它记录了 plan §7-§10 落地时与设计差异的部分、实施期发现的 5 个 corner case(详见附录 C),以及最终的测试通过率。

### 14.1 实施 diff 一览

| 文件 | 行数(改动后) | 增量 | 说明 |
|---|---|---|---|
| `rlinf/envs/realworld/galaxear/r1_pro_safety.py` | 1055 | +611 / -14 | SafetyConfig 拆 right/left + 新增 L2 字段 / toggle / dt_step / 旧别名兼容;`SafetyInfo.l2_per_joint_clip`;5 个 L2 子层方法 + audit helper |
| `rlinf/envs/realworld/galaxear/r1_pro_action_schema.py` | 372 | +210 | `predict_arm_qpos`(三分支)+ 4 个 `rewrite_action_*` + `set_gripper_action` |
| `rlinf/envs/realworld/galaxear/r1_pro_robot_state.py` | 236 | +17 | `get_torso_qpos/qvel`、`get_chassis_qpos/qvel`、`get_gripper_vel` |
| `tests/unit_tests/test_galaxea_r1_pro_safety_l2.py` | 599 | 新文件 | 16 个 L2 单测 |
| `tests/unit_tests/conftest.py` | 141 | 新文件 | Orin 最小环境 stub |
| **总计** | — | **+1578 行,-14 行** | 0 行老 RLinf 生产代码删除 |

### 14.2 测试通过率

最终 `pytest tests/unit_tests/test_galaxea_r1_pro_safety*.py tests/unit_tests/test_galaxea_r1_pro_action_schema.py tests/unit_tests/test_galaxea_r1_pro_camera_mux.py tests/unit_tests/test_galaxea_r1_pro_hardware.py`:

| 测试套件 | 用例数 | 通过 | 状态 |
|---|---|---|---|
| `test_galaxea_r1_pro_safety.py`(L1-L5 baseline) | 11 | 11 | regression OK |
| `test_galaxea_r1_pro_safety_l2.py`(本设计新增) | 16 | 16 | **新功能 OK** |
| `test_galaxea_r1_pro_safety_l3a.py`(L3a 深度) | 16 | 16 | regression OK |
| `test_galaxea_r1_pro_action_schema.py` | 6 | 6 | regression OK |
| `test_galaxea_r1_pro_camera_mux.py` | 7 | 7 | regression OK |
| `test_galaxea_r1_pro_hardware.py` | 8 | 8 | regression OK |
| **合计** | **64** | **64** | **0 fail** |

L2 单测连跑 3 轮稳定 16/16(用时 ~0.21s / 轮)。

### 14.3 Plan vs 实施差异速查

| 章节 | Plan 草稿 | 实施版 | 说明 |
|---|---|---|---|
| §7.1 SafetyConfig 字段顺序 | L2 / L3a / L3b / L4 / L5 / 旧别名一起列 | 同上,加了 `dt_step: float = 0.10` 字段 | 取代 `getattr(cfg, "dt_step", 0.1)` 的 fallback,让 dt 成为正式 config field |
| §7.2 `predict_arm_qpos` Branch 2 | `np.concatenate([xyz, rpy, [gripper]])[:7]` | 显式 `new_q[:6] = cur_q[:6] + delta6 * scale`,J7 不动 | 避免误把 gripper 的 dim 当 J7 |
| §7.4.1 L2a | 简化版,不带 zero-state guard | **加 zero-state guard**(`not np.any(cur_q_raw[:7])`) | 否则左臂 J2 q_min=-0.07 在零反馈下会以 0.467× 缩放整条左臂 action,**实施期实测发现** |
| §7.4.1 L2a reason 标签 | 仅写 `worst margin` | 加 `J{idx+1}` 与 audit `_record_per_joint_clip(...)` | 精细化 |
| §7.4.2 L2b | arm + torso 都不带 zero-state guard | **arm + torso 都加 zero-state guard** | 同上 |
| §7.4.3 L2c | 只写 arm,torso/chassis 留"同样处理" | 完整实现 arm + torso + chassis,**chassis 单档(无 warning)** | chassis odom 噪声大,渐变会扰乱导航 |
| §7.4.4 L2d chassis dead-zone | `np.where(np.abs(v) < dead_zone, 0.0, v)` 任何位都触发 | `dead & (v != 0.0)` 守卫,只对非零生效 | 避免 zero-cmd 情形被无谓标记 `clipped` |
| §7.5 `l2_per_joint_clip` schema | dict / "freeze" 二元 | dict / list / str **三类** value(dict for L2a/gripper,list for L2d chassis raw vector,str for L2b/L2c symbolic outcome) | 实际审计需要 |
| §7.6 gripper rate + pos | 两个独立 if(可能任意顺序) | 串行 rate→pos,共享一次 `set_gripper_action` 写回 | 防止 next_pct 计算与 rate-cap 后 a 不一致 |
| §10 测试用例数 | 12 | **16**(加 audit / toggle / dual-arm / back-compat) | 工程化补强 |

### 14.4 工作流时序

```mermaid
gantt
    title L2 实施时序(2026-05-05,约 50 min)
    dateFormat HH:mm
    section 准备
    读 plan / safety_2 / brs-ctrl :done, 11:00, 5min
    探查 conftest / 现有测试      :done, 11:05, 3min
    section 实施
    pip install gymnasium / ray   :done, 11:08, 3min
    conftest.py stub              :done, 11:11, 4min
    r1_pro_robot_state helpers    :done, 11:15, 2min
    r1_pro_action_schema 6 方法   :done, 11:17, 6min
    r1_pro_safety SafetyConfig 扩展 :done, 11:23, 4min
    r1_pro_safety SafetyInfo 扩展 :done, 11:27, 1min
    r1_pro_safety L2 5 个子检查    :done, 11:28, 8min
    section 测试
    跑 baseline 33 测试           :done, 11:36, 2min
    fix L2a zero-state(±0.467 bug) :crit, 11:38, 2min
    写 16 个 L2 测试              :done, 11:40, 8min
    跑 / fix 2 个 gripper 测试    :done, 11:48, 3min
    回归 64 测试 + 3 轮稳定测试    :done, 11:51, 2min
```

---

## 附录 A:与现有 L4 的并存设计

### A.1 L2 vs L4:作用域对比

| 维度 | L2 (本设计) | L4 (现有) |
|---|---|---|
| 检查对象 | **关节空间**:q_min/q_max,q_vel_max | **笛卡尔/速度空间**:max_linear_step_m, torso_v_x_max 等 |
| 触发依赖 | `state.{arm,torso}_qpos / qvel` | `action`(归一化)与 `action_scale` |
| 是否需要反馈 | 是(L2b/L2c) | 否(纯 action 端) |
| 超界后果 | scale / soft_hold | hard clip(`L4:per_step_cap` 等) |
| 与硬件耦合 | 强(URDF 真值) | 弱(经验上限) |

L2 与 L4 是**正交保护**——L2 关注"关节是否被推到 URDF 极限",L4 关注"每步动作幅度是否过大"。两者并存,且按 §4.2 的顺序"L2→L3a→L3b→L4"逐级执行,后级看到的是前级修改后的 action,组合效果是单调收敛。

### A.2 边界协调

唯一可能的双重触发场景:`action_scale[0]=0.05m/step, max_linear_step_m=0.05m`。在这个边界配置下 L4 永远 just-触底;L2d 在 EE 模式下不动 arm slice;**两层不冲突**(因为 L2d 只看 torso/chassis/joint-mode arm)。

---

## 附录 B:0.1 rad 余量的工程论证

为什么"URDF - 0.1 rad",不是 0.05、0.15 或 5%?

| 候选 | 优点 | 缺点 |
|---|---|---|
| 0.05 rad (~2.9°) | 接近无损,工作区最大 | 高速关节(J5-J7,vmax≈10 rad/s)下 5 ms 就跑过,fail-safe 能力弱 |
| **0.1 rad (~5.7°)** | 高速关节给出 9.5 ms 缓冲,在 Orin 50–100 ms feedback 周期下足够 L2b 吸收;低速关节(J1-J4,vmax≈7 rad/s)给出 14 ms 缓冲 | 工作区收缩约 4-7%,可接受 |
| 0.15 rad (~8.6°) | 更安全 | J6 工作区从 ±1.05 收到 ±0.90,损失 14% — 部分操作姿态不可达 |
| 比例缩(URDF × 0.93) | 不同关节按比例 | 实现复杂,审计不直观 |

**0.1 rad 是恰好的工程折中**:满足"反馈周期 < 余量/最大速度"这一关键不等式:

```
反馈周期 (≤100ms) < 0.1 rad / 10.47 rad/s ≈ 9.5 ms ❌  (失败)
```

注意上面这个不等式**实际上是不满足的**——所以 L2b 的 critical_margin 才必须 < warning_margin 而 warning_margin 才必须 < 安全余量。具体地:

```
URDF 极限 ⟸ 0.10 rad ⟸ 0.05 rad (critical) ⟸ 0.15 rad (warning)
                        ▲                       ▲
                        L2b 冻结开始             L2a scale 开始
```

这样布局保证:**当反馈检测到关节处在 critical_margin 内时,距离 URDF 极限还有 0.05 rad,即使下一帧高速 overshoot 也来得及**。0.15 rad warning_margin 给出足够的"L2a 平滑减速"区间。**所以 0.1 rad 不是单点选择,而是与 0.05/0.15 三档 margin 一起定义的安全 envelope。**

---

## 附录 C:实施期发现(Implementation Findings)

> 实施期间踩到的 5 个 corner case。它们是从 plan §7 跨到生产代码 / 单测时**实际遇到的问题**,既是回灌设计文档,也是给后续 maintainer 的"这一行代码不能删"的备注。

### C.1 Zero-state guard:为什么必要

#### 现象

在跑现有 baseline `test_l3a_dual_arm_only_violating_arm_is_rewritten` 时,**预期**是"右臂越界裁剪到 0.4,左臂 action 不变保持 1.0",**实际**输出 `info.safe_action[7] = 0.46667`(左臂被压到 0.467 倍)。

#### 根因

`test_galaxea_r1_pro_safety_l3a.py` 的 `_state_with_ee()` helper 只填 EE pose,没填 `right/left_arm_qpos`,所以 state 默认 `np.zeros(7)`。在 L2a 的预测裁剪逻辑里:

- 左臂 J2 的 SafetyConfig 边界是 `q_min[1] = -0.07, q_max[1] = +3.04`(注意左臂 J2 是镜像!)。
- 当 `cur_q = [0, 0, ..., 0]`,Cartesian fallback 让 `q_next = cur_q = 0`。
- `margin_low = 0 - (-0.07) = 0.07`,`margin_high = 3.04 - 0 = 3.04`,**worst = 0.07 rad**。
- 0.07 < `l2_warning_margin_rad = 0.15`,触发 warning 分支。
- `scale = 0.07 / 0.15 ≈ 0.467` —— 整条左臂 action 被缩到 0.467 倍。

也就是:**仅仅因为 RLinf 默认的 zero state 紧贴左臂 J2 的下边界**,L2a 误以为左臂逼近极限。

#### 修复

在 L2a 与 L2b 都加 `not np.any(cur_q_raw[:7]) → continue` 守卫。理由:

- 默认 `state.right_arm_qpos / left_arm_qpos = zeros(7)` 是"反馈未到达"的标志,而不是"机器人确实停在原点"。
- 与 controller 的 `is_robot_up()` 行为对齐(看 feedback 年龄;全零 qpos 是反馈未到达的强信号)。
- 真机反馈到达后,qpos 几乎不可能精确为 0(关节静止时也有 mrad 抖动),所以这个守卫在生产环境下**不会假阳性**地跳过 L2 检查。

#### 边界条件

如果未来左臂工厂出厂位姿真的精确停在 `[0, 0, ..., 0]`,zero-state guard 会让 L2a 跳过一两帧。但这种情况 mobiman IK 也会同时触发 J2 越界保护,**双重保险**。如果连续 N 帧反馈都精确为 0,就该让 L5 watchdog(`feedback_age_ms`)接管告警了 —— 那是 L5b 的语义,不是 L2a 的。

### C.2 `predict_arm_qpos` Branch 2 的"前 6 关节"语义

#### Plan vs 实施差异

Plan §7.2 草稿曾写过:

```python
if self.use_joint_mode:
    a_arm = np.concatenate([d[key_xyz], d[f"{side}_rpy"], [d.get(f"{side}_gripper", 0.0)]])
    return (cur_q + a_arm[:7] * float(self.action_scale[0])).astype(np.float32)
```

这里把 `[d.get(f"{side}_gripper", 0.0)]` 拼到 6 个 xyz+rpy 之后,**意图凑足 7 个关节**,然后 `a_arm[:7]` 切片。

#### 为什么改

- gripper 不是机械臂第 7 关节(R1 Pro 的第 7 关节是腕旋转 J7,与 G1 夹爪是两个独立机构)。
- 如果 schema 配的是 `no_gripper=True`,`d.get(f"{side}_gripper", 0.0)` 返回 0,看似无害;但语义层面把"夹爪 dim"当成"J7 delta"会让审计 / reason 标签里的 `J7 critical` 变得有歧义。
- M1 阶段动作只控制前 6 关节(肩+肘+腕,不控腕旋转),让 J7 保持不动是更符合实际运行的 bring-up 约定。

#### 实施版

```python
if self.use_joint_mode:
    xyz = d[key_xyz][:3]
    rpy = d[f"{side}_rpy"][:3]
    joint_delta6 = np.concatenate([xyz, rpy])
    new_q = cur_q.copy()
    new_q[:6] = cur_q[:6] + joint_delta6 * float(self.action_scale[0])  # J7 不动
    return new_q.astype(np.float32)
```

未来真正引入 7 维 joint-mode action layout 时,这段需要重写——TODO 已经在 docstring 里写了。

### C.3 L2c chassis 没有 warning 档(只有 overspeed 单档)

#### 设计选择

Plan §7.4.3 草稿对 chassis 写"同样处理":类似 arm 那样 `> 0.9× → scale 0.5`,`> 1.2× → soft_hold`。实施时**刻意去掉了 warning 档**:

```python
if np.any(ratio_c > over):    # 只剩 overspeed (> 1.2×)
    info.safe_action = schema.rewrite_action_chassis(info.safe_action, 0.0)
    info.soft_hold = True
    ...
# 没有 elif np.any(ratio_c > warn): ...
```

#### 理由

1. **底盘 odom 速度噪声远大于关节编码器**:轮子打滑 / IMU drift 会让 `state.chassis_qvel` 在没移动时也跳到 0.05 m/s 的水平;0.9× = 0.27 m/s 阈值在 `chassis_qvel_max=0.30 m/s` 下离噪声不远。
2. **导航策略对"渐变 0.5× 缩放"特别敏感**:RL 学到的 base policy 输入是 odom feedback,如果 supervisor 偷偷把命令乘 0.5,policy 学到的"我命令 1.0 m/s 但 base 实际只走 0.5"会污染回报信号——不如直接 `soft_hold` 一帧让 policy 看到清晰的"这步无效"信号,下一帧重新规划。
3. **与 BRS-ctrl `R1ProInterface._mobile_base_control` 行为一致**:BRS 也是 dead-zone+硬 clip,不做渐变。

代码里有显式注释解释这点。

### C.4 L1 clip 与 gripper scale 的相互作用 (v3 归一化更新)

#### v2 现象与根因(历史背景)

v2 默认 gripper scale `action_scale[2] = 1.0`,而 L2 在 `[0, 100]` "百分比"空间检查;`max_gripper_step_pct = 30`。第一版单测想测 "50%/step delta":

```python
schema = ActionSchema(action_scale=[0.05, 0.10, 1.0])  # default scale
a = np.zeros(7); a[6] = 50.0   # 想测 "50%/step delta"
```

**失败原因**:L1 把 `a[6] = 50.0` clip 到 `1.0`,L2 看到的实际 delta 是 `1.0 * 1.0 = 1 %/step`,远小于 `max_step_pct = 30`。**单测必须把 scale 提到 50** 才能让 `a = 1.0` 在 [0, 100] 空间产生 50%/step 的 delta。

#### v3 改良:归一化空间下不再需要 workaround

把 L2 工作语言从 "[0, 100]" 切换到 "[0, 1]" 之后,**陷阱消失**:

```python
schema = ActionSchema(action_scale=[0.05, 0.10, 1.0])  # default scale
a = np.zeros(7); a[6] = 1.0   # 在 [-1,1] 内,L1 不 clip
# scale_norm = 1.0 (≤ 5, treated as already normalised)
# delta_pos = 1.0 * 1.0 = 1.0  -> 1.0 normalised step (= 100% stroke)
# max_gripper_step = 0.30 -> rate_cap 触发
```

**v3 默认 `action_scale[2] = 1.0` 直接好用**:`a=1.0` 表示"想从当前位置走整个 stroke",`max_gripper_step=0.30` 把它限到 30% stroke/step。**无需任何 scale workaround**。

#### v3 兼容老 YAML / 老单测的启发式

`_check_l2_gripper` 里:

```python
raw_scale = max(float(schema.action_scale[2]), 1e-9)
scale_norm = raw_scale / stroke_span if raw_scale > 5.0 else raw_scale
```

- `action_scale[2] ≤ 5` → 视为**已归一化**,直接用(典型 RLinf 默认 1.0,或现场调到 0.5、2.0 等)。
- `action_scale[2] > 5` → 视为 **mm-flavoured**(BRS-style 50,或 v2 单测里用过的 50),自动除以 `stroke_span`(默认 100 mm)转回归一化 [0, 1]。

这条启发式让 v2 → v3 迁移时**老单测的 `action_scale[2]=50` 不需要改**(自动被识别为 50/100 = 0.5 归一化 step)。

#### 给 maintainer 的注意点(v3 版)

- 生产代码中 `action_scale[2]` 的语义在 v3 改为"夹爪 1 normalised action unit 等于多少 normalised pos delta(也就是多少比例的 stroke 行程)"。
- v3 默认 1.0 表示 "a=1 → +1.0 normalised pos delta = 全 stroke 一步" — 这之所以实用,是因为 `max_gripper_step=0.30` 总是把它压到 30% stroke/step。**rate_cap 在每次 a=±1 都会触发,这是预期行为**(让 policy 学会用更小的 a 平滑控制)。
- 现场调参时**首选改 `max_gripper_step`**(0.10~0.50 范围)而不是改 `action_scale[2]`,这样不影响 policy 学到的语义边界。
- 单测和现场调参都要意识到 L1 → L2 这个 clip 链条:**a 必须先在 [-1, 1] 内,scale_norm 才决定单步真实归一化量**;再经 `_dispatch_action` 与 `gripper_open_stroke_mm` 决定真实 mm 量。

### C.5 conftest stub 的"必须保留真包"边界

#### 现象

第一版 conftest 把 `rlinf.envs.realworld` 也 stub 了(为了让 `from .franka import FrankaEnv` 不爆),结果 `from rlinf.envs.realworld.galaxear.r1_pro_safety import ...` 失败,报 `ModuleNotFoundError: No module named 'rlinf.envs.realworld.galaxear'`。

#### 根因

Stub 把 `__path__ = []`,Python 导入子包时会沿 `parent.__path__` 搜索 — `[]` 让搜索失败,**真实的 `galaxear/` 目录被屏蔽**。

#### 修复

只 stub **失败**的子模块,不动父包:

- ✅ `rlinf.scheduler`(失败 → stub)
- ✅ `rlinf.envs.realworld.franka`(失败 → stub)
- ✅ `rlinf.envs.realworld.xsquare`(失败 → stub)
- ✅ `rlinf.envs.realworld.realworld_env`(失败 → stub)
- ❌ **不要** stub `rlinf` / `rlinf.envs` / `rlinf.envs.realworld`(它们必须用真实 path)
- ❌ **不要** stub `rlinf.envs.realworld.galaxear`(它是测试目标的真包)

`_try_import` 帮忙做这件事:**只在 import 失败时**才安插 stub,GPU server 完整 install 上是 no-op。

### C.6 v2 到 v3 的迁移笔记(Migration Notes)

> v3 文档相对 v2 的所有差异都集中在**夹爪一个语义点**上。其它 22 个关节(7+7+4+3)与所有 L2a/b/c/d 行为完全不变。本节给 v2 → v3 的代码与配置迁移列出明确步骤。

#### C.6.1 字段重命名(SafetyConfig)

| v2 字段 | v3 字段 | 数值范围变化 | 备注 |
|---|---|---|---|
| `gripper_pct_min: float = 0.0` | `gripper_pos_min: float = 0.0` | 不变 | 字段名改;0.0 在两个语义下都是"完全闭合" |
| `gripper_pct_max: float = 100.0` | `gripper_pos_max: float = 1.0` | **100 → 1** | 单位从"百分比"改成"归一化" |
| `max_gripper_step_pct: float = 30.0` | `max_gripper_step: float = 0.30` | **30 → 0.30** | 单位同上 |
| (无) | `gripper_closed_stroke_mm: float = 0.0` | (新增) | mm ↔ norm 端点 |
| (无) | `gripper_open_stroke_mm: float = 100.0` | (新增) | 同上,默认匹配 RLinf `_dispatch_action.clip(0,100)` 现状 |

#### C.6.2 RobotState helper 新增

| 新增 | 用途 |
|---|---|
| `state.get_gripper_pos_norm(side, *, closed_stroke_mm=0.0, open_stroke_mm=100.0)` | 把 mm 反馈 → [0, 1] normalised pos,集中换算逻辑;L2 内部调用它取 `cur_pos` |

#### C.6.3 内部实现差异(L2 supervisor)

```diff
-cur_pct = float(state.get_gripper_pos(side))   # mm 被当成 pct 用
-delta_pct = a_g * scale_g
+cur_mm   = float(state.get_gripper_pos(side))
+cur_pos  = (cur_mm - closed_mm) / stroke_span  # mm → [0,1]
+cur_pos  = float(np.clip(cur_pos, 0.0, 1.0))
+delta_pos = a_g * scale_norm

-if abs(delta_pct) > max_step:
-    new_a = float(np.sign(a_g) * max_step / scale_g)
+if abs(delta_pos) > max_step:
+    new_a = float(np.sign(a_g) * max_step / scale_norm)
```

`scale_norm` 来自 `_check_l2_gripper` 头部的启发式:

```python
raw_scale = max(float(schema.action_scale[2]), 1e-9)
scale_norm = raw_scale / stroke_span if raw_scale > 5.0 else raw_scale
```

#### C.6.4 单测层迁移

v2 单测:

```python
schema = ActionSchema(action_scale=[0.05, 0.10, 50.0])  # gripper scale = 50 (mm-flavoured workaround)
a[6] = 1.0
# delta = 1.0 * 50 = 50% / step (in v2's [0, 100] pct space)
# rate_cap fires: max_step_pct = 10 -> new_a = 0.2
assert info.safe_action[6] == pytest.approx(0.2, abs=1e-5)
```

v3 自动兼容:`scale = 50 > 5.0` 触发 mm 启发式,`scale_norm = 50/100 = 0.5`;`delta_pos = 1.0 * 0.5 = 0.5`,`max_gripper_step = 0.30` → `rate_cap` 触发,`new_a = sign(1) * 0.30 / 0.5 = 0.6`。**注意 expected new_a 从 0.2 改到 0.6**(因为 v2 的 `max_step_pct=10` ≠ v3 的 `max_gripper_step=0.30`)。

或者,v3 推荐风格(纯归一化):

```python
schema = ActionSchema(action_scale=[0.05, 0.10, 1.0])  # default
a[6] = 1.0
# scale_norm = 1.0 (≤5, treated as already normalised)
# delta_pos = 1.0 * 1.0 = 1.0 -> exceeds max_gripper_step=0.30 -> rate_cap
# new_a = sign(1) * 0.30 / 1.0 = 0.30
assert info.safe_action[6] == pytest.approx(0.30, abs=1e-5)
```

#### C.6.5 dispatch / controller 层是否需要改?

**不需要**。RLinf [`r1_pro_env.py:491`](../../../rlinf/envs/realworld/galaxear/r1_pro_env.py#L491) 的 `(a + 1) * 50` 仍然把 `a ∈ [-1, 1]` 映射到 `pct ∈ [0, 100]` 然后送到 SDK(SDK 把它当 mm)——这个语义本来就是 "a=+1 → 100mm = open",与 v3 "a=+1 → pos_norm=1 → open" **完全一致**。

如果将来 `gripper_open_stroke_mm` 改成 90(BRS 风格),`_dispatch_action` 也要相应改:

```python
# Optional v3 dispatch upgrade (out of scope for this migration):
pos_norm = (a + 1.0) * 0.5  # [0, 1]
mm = closed_mm + pos_norm * (open_mm - closed_mm)
self._controller.send_gripper(side, mm).wait()
```

**v3 文档不强制此改动**,因为它会改变 stroke 端点,需要重新实测 policy 行为。建议:把 `_dispatch_action` 的 hardcoded `0/100` 与 SafetyConfig 的 `gripper_*_stroke_mm` 字段在后续 PR 中统一,**v3 仅做 L2 层归一化**。

#### C.6.6 一句话 summary

**v2 用 mm 假装是百分比,v3 用归一化 [0, 1] 并显式钉死 0=closed/1=open;两版的物理行为(送给 SDK 的 mm 数字)在默认配置下完全一致,只是字段命名和单位口径更干净**。

---

> 写到这里,**L2 的"位置 + 速度"约束方案 v3 就完整了**——文档与代码、设计与实施、附录 C 的 6 个 corner case(含 v2/v3 迁移)、单测的 16 个用例,全部对齐到归一化 [0, 1] + RLinf 0=closed/1=open 的 G1 工程语义。下一步是创建 PR、在 [`test_galaxea_r1_pro_controller.md`](test_galaxea_r1_pro_controller.md) §A.3.3 的 REPL 流程里验证 §11.2 的现场表现。完整的实施可以参照 [`r1pro5op47_imp1.md`](r1pro5op47_imp1.md) 的滚动实施模式。
