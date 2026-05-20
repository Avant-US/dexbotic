# Galaxea R1 Pro 安全 / 上机前需确认项

本文档对应设计 [bt/docs/rwRL/r1pro5op47.md](bt/docs/rwRL/r1pro5op47.md) §6.4、§9 与**形态 A**（Controller 与 `hdas_msg` 在 **Orin** 同机，GPU 机只跑训练与 USB 相机可选）。

在把 `safety_cfg` 写入 YAML 前，请尽量从**厂商 / 实机**拿到下列事实，否则 `GalaxeaR1ProSafetySupervisor` 的 L3a/L5 与物理不一致。

## 1. 坐标系与 TCP 包络 (L3a)

- 末端位姿、安全盒 **6D (xyz + rpy)** 约定在 **`torso_link4`** 系（与 [GalaxeaR1ProRobotState](r1_pro_robot_state.py) 中 EE 字段一致）。
- 现场测量桌面与围栏下的**保守工作区**，再填 `right_ee_min` / `right_ee_max`（单臂 M1 亦建议核对左臂占位盒，或冻结左臂避免误入对称默认盒）。

## 2. ROS2 / 状态真值 (L5)

- `/hdas/bms` 中 `capital_pct` 或其它电量字段的**工程含义**与低电阈值（默认 25%）。
- `/controller` 中 **SWD** 与 `GalaxeaR1ProRobotState.controller_signal["swd"]` 的映射；软急停极性对应 `safety_cfg.estop_swd_value_down`。
- `/hdas/feedback_status_arm_*` **错误码表**与 `clear_errors` 可恢复性。
- 各反馈源的 **最大延迟** 与 `feedback_stale_threshold_ms`（默认 200ms；全 ROS2 高抖动时常见 250ms 量级）。

## 3. 部署

- 形态 A：Orin 上已 `source` ROS2 + Galaxea install，`hdas_msg` 可 import；否则 L5 中 BMS / SWD / 状态机不完整。
- 两节点 Ray：`RLINF_NODE_RANK`、`RLINF_COMM_NET_DEVICES`、DDS 域与 `ray_utils/realworld/setup_before_ray_galaxea_r1_pro.sh` 一致。

## 4. 运规

- 硬急停后 **CAN 是否需重启**；操作员到急停距离；`RLINF_SAFETY_ACK=1` 后是否允许 `reset`（见 `GalaxeaR1ProEnv` 的 `safety_block_reset_until_ack`）。

## 5. 配置文件

- 共享默认键： [realworld_galaxea_r1_pro_safety_default.yaml](../../../../examples/embodiment/config/env/realworld_galaxea_r1_pro_safety_default.yaml)（在训练 `override_cfg` 中复制 `safety_cfg` 块或按需合并）。
