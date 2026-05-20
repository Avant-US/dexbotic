# R1 Pro 离线多样轨迹生成实施步骤

> **定位**:本文把“优先做离线自研采样规划器,之后再补 MoveIt2 配置”的方案细化为程序员可逐步执行的实施文档。目标是在当前 R1 Pro 真机 Orin 与 `/home/nvidia/galaxea/install` SDK 的现实条件下,不向真机发布控制指令,离线生成更丰富的 SFT waypoint 轨迹。
>
> **适用机器**:当前 Orin,Ubuntu 22.04.5,ROS2 Humble,Galaxea R1 Pro SDK 安装在 `/home/nvidia/galaxea/install`。
>
> **核心原则**:所有方案都必须只读 SDK 模型与本机状态,不发布 `/motion_target/*` 或 `/motion_control/*`。实现路径不限制为“零新增依赖”:先保留一个不装软件也能跑的基础闭环,同时允许安装与当前 Orin/ROS2 Humble/SDK 兼容的软件来提升多样性、碰撞检查和时间参数化质量。

---

## 1. 当前可用条件与缺口

### 1.1 已确认可用

当前 Orin 上已经具备离线轨迹生成所需的关键素材:

| 类别 | 已有资源 | 用途 |
|---|---|---|
| R1 Pro 整机 URDF | `/home/nvidia/galaxea/install/mobiman/share/mobiman/urdf/R1_PRO/urdf/r1_pro.urdf` | 解析右臂 7 关节链、关节限位、速度上限、collision mesh |
| 右臂浮动 URDF | `/home/nvidia/galaxea/install/teleoperation_ros2/share/teleoperation_ros2/config/urdf/r1_pro_floating_right.urdf` | 可作为轻量右臂模型参考 |
| 右臂 SRDF | `/home/nvidia/galaxea/install/teleoperation_ros2/share/teleoperation_ros2/config/urdf/r1_pro_floating_right_srdf.xml` | `right_arm` 规划组定义:base=`torso_link4`,tip=`right_gripper_link` |
| SDK FK/IK 工具 | `/home/nvidia/galaxea/install/mobiman/lib/mobiman/robot.py` | 基于 `urdf_parser_py` + `PyKDL` 的 FK/IK 参考实现 |
| 运动学/优化库 | `PyKDL`,`pinocchio`,`hpp-fcl`,`toppra`,`trac_ik_lib` 包目录 | 后续用于 FK、碰撞检测、时间参数化 |
| 当前脚本 | `bt/docs/rwRL/scripts/generate_sft_waypoints.py` | 现有线性插值 baseline |

右臂 SRDF 中的规划组为:

```xml
<group name="right_arm">
  <chain base_link="torso_link4" tip_link="right_gripper_link"/>
</group>
```

右臂 7 关节 URDF 机械限位为:

| Joint | lower | upper | velocity |
|---|---:|---:|---:|
| `right_arm_joint1` | -4.4506 | 1.3090 | 7.1209 |
| `right_arm_joint2` | -3.1416 | 0.1745 | 7.1209 |
| `right_arm_joint3` | -2.356196 | 2.356196 | 8.3776 |
| `right_arm_joint4` | -2.0944 | 0.3491 | 8.3776 |
| `right_arm_joint5` | -2.356196 | 2.356196 | 10.4720 |
| `right_arm_joint6` | -1.047198 | 1.047198 | 10.4720 |
| `right_arm_joint7` | -1.5708 | 1.5708 | 10.4720 |

当前 `generate_sft_waypoints.py` 使用了更保守的工作域:

```python
q_min = [-4.35, -3.04, -2.26, -1.99, -2.26, -0.95, -1.47]
q_max = [ 1.21,  0.07,  2.26,  0.25,  2.26,  0.95,  1.47]
```

第一阶段继续沿用这组内缩工作域,不要直接使用 URDF 机械极限。

### 1.2 当前缺口

当前机器**不能直接开箱使用 MoveIt2 方案**,因为缺少:

- `moveit_core`
- `moveit_ros_move_group`
- `moveit_planners_ompl`
- `moveit_planners_chomp`
- `moveit_msgs`
- 完整的 `r1_pro_moveit_config`
- `kinematics.yaml`
- `ompl_planning.yaml`
- `planning_pipeline.yaml`
- `joint_limits.yaml`

apt 源里有候选包。本文不把“避免安装软件”作为目标,而是分三档实施:

- **Profile A:基础离线档**。只依赖 Python + Numpy + SDK 已有模型,先跑通数据闭环。
- **Profile B:增强离线档**。允许安装 `scipy`,`networkx`,`toppra` 等兼容依赖,增加 B-spline、PRM、图搜索、轨迹优化和更好的时间参数化。
- **Profile C:MoveIt2 档**。安装 MoveIt2/OMPL/CHOMP 相关包,构建 `r1_pro_moveit_config`,但仍然只做 fake controller/offline planning scene,不连接真机控制链路。

后续代码优先实现 Profile A+B,因为它们最容易在现有 Orin 上稳定落地;Profile C 作为第二阶段增强,不阻塞第一批高多样性 SFT 轨迹生成。

---

## 2. 安全边界

### 2.1 绝对禁止的操作

在离线轨迹生成阶段,不要执行下面任何动作:

- 不向 `/motion_target/target_joint_state_arm_right` 发布消息。
- 不向 `/motion_control/control_arm_right` 发布消息。
- 不启动新的 `r1_pro_jointTracker_demo_node`。
- 不重启或停止 `/HDAS`、`robot_state_publisher`、相机、底盘、夹爪等真机进程。
- 不执行 `ros2 topic pub` 到任何控制话题。

当前真机控制链路实际存在:

```text
/motion_target/target_joint_state_arm_right
  -> r1_pro_jointTracker_demo_node
  -> /motion_control/control_arm_right
  -> HDAS
  -> real robot
```

所以本文所有脚本默认都是**纯文件输入/输出**。ROS2 只允许用于只读检查,不要用于控制。

### 2.2 推荐隔离方式

如果后续需要启动任何 ROS2 节点做可视化或测试,先使用隔离 domain:

```bash
export ROS_DOMAIN_ID=142
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
```

但第一阶段脚本应设计为不依赖 ROS graph,即使不 source ROS2 也能生成 JSON。

---

## 3. 总体落地路线

第一阶段实现“离线自研采样规划器 + 可选增强依赖”,分 8 个里程碑:

1. **模型抽取**:从 R1 Pro URDF/SRDF 读取右臂链、关节名、限位、mesh 路径,生成固定的模型元数据 JSON。
2. **轨迹表示**:定义 episode/trajectory/waypoint 的 JSON schema,兼容现有 `q_rad` 与 `action_norm`。
3. **配置系统**:把特定起点、终点、工作域、速度/加速度/jerk 阈值、planner 权重写入 YAML,避免散落在代码常量里。
4. **采样规划器**:实现多种离线轨迹生成方法:随机 via-points、Bezier/B-spline、nullspace loop、关节空间 RRT、RRT-Connect、PRM、bridge sampling、扰动-优化 hybrid。
5. **轨迹校验**:检查工作域、速度、加速度、jerk、末端误差、重复度、多样性指标、可选 FK/碰撞距离。
6. **多样性筛选**:对候选池做分桶、聚类、最远点采样,不是“生成多少收多少”。
7. **时间参数化**:基础版用 minimum-jerk/ease-in-out,增强版接 TOPPRA。
8. **数据导出**:输出多 episode JSONL/JSON,保留 metadata、seed、planner 类型、失败原因统计、diversity 统计。

允许安装并评估的第三方能力:

- `scipy`:B-spline、Savitzky-Golay 平滑、优化器。
- `networkx`:PRM 图、k-shortest paths、最远路径筛选。
- `toppra`:完整关节速度/加速度约束时间参数化。
- `pinocchio` + `hpp-fcl`:FK/Jacobian/碰撞距离。
- MoveIt2 + OMPL/CHOMP:在 fake controller/offline planning scene 下生成候选路径。

---

## 4. 建议文件结构

不要继续把所有逻辑塞进 `generate_sft_waypoints.py`。建议保留旧脚本作为 baseline,新增一个小型离线生成包:

```text
bt/docs/rwRL/scripts/
  generate_sft_waypoints.py                 # 保留:线性插值 baseline
  genwaypoint/
    __init__.py
    config.py                               # YAML 配置加载与校验
    model.py                                # 读取/缓存 R1 Pro 模型元数据
    schema.py                               # Episode/Waypoint 数据结构
    normalize.py                            # q <-> action_norm 映射
    kinematics.py                           # 可选 Pinocchio/PyKDL FK 与末端位姿
    collision.py                            # 碰撞检查接口与 Noop/hpp-fcl 实现
    time_profile.py                         # minimum-jerk、ease-in-out、速度采样
    spline_sampler.py                       # via-points + B-spline 轨迹
    rrt_sampler.py                          # 关节空间 RRT 轨迹
    rrt_connect_sampler.py                  # 双向 RRT-Connect
    prm_sampler.py                          # PRM 图采样 + 多路径查询
    perturb_opt_sampler.py                  # 线性/样条轨迹扰动 + 平滑优化
    diversity.py                            # 多样性指标、去重、最远点筛选
    validators.py                           # 限位/速度/加速度/jerk/多样性校验
    export.py                               # JSON/JSONL 输出
    visualize.py                            # 离线绘图,不连接 ROS
    configs/
      r1pro_right_arm.yaml                  # 特定起终点与安全阈值
      planner_mix.yaml                      # planner 权重和参数
  generate_sft_waypoints_offline.py         # CLI 入口
  install_genwaypoint_deps.sh               # 可选增强依赖安装脚本
```

对应测试:

```text
tests/unit_tests/
  test_galaxea_genwaypoint_model.py
  test_galaxea_genwaypoint_normalize.py
  test_galaxea_genwaypoint_validators.py
  test_galaxea_genwaypoint_samplers.py
```

如果暂时不想进入 RLinf 单测目录,也可以先放在:

```text
bt/docs/rwRL/scripts/tests/
```

但最终建议进入 `tests/unit_tests/`,因为这些逻辑会直接影响真机 SFT 数据质量。

---

## 5. 里程碑 0:环境只读核验

### 5.1 核验 SDK 路径

执行:

```bash
test -f /home/nvidia/galaxea/install/mobiman/share/mobiman/urdf/R1_PRO/urdf/r1_pro.urdf
test -f /home/nvidia/galaxea/install/teleoperation_ros2/share/teleoperation_ros2/config/urdf/r1_pro_floating_right_srdf.xml
test -f /home/nvidia/galaxea/install/setup.bash
```

期望:三个命令都无输出且退出码为 0。

### 5.2 核验 Python 依赖

执行:

```bash
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
python3 - <<'PY'
mods = ["numpy", "urdf_parser_py", "PyKDL", "pinocchio"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}: OK {getattr(mod, '__file__', 'built-in')}")
    except Exception as exc:
        print(f"{name}: FAIL {type(exc).__name__}: {exc}")
PY
```

期望:

- `numpy`: OK
- `urdf_parser_py`: OK
- `PyKDL`: OK
- `pinocchio`: OK

如果 `pinocchio` 不可用,第一阶段仍可用 `PyKDL` 做 FK,不阻塞。

### 5.3 只读查看当前真机状态

执行:

```bash
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
timeout 8s ros2 node list
timeout 5s ros2 topic echo /joint_states --once
```

用途仅是确认当前机器确实在真机 ROS graph 中。不要在这个阶段运行任何 `ros2 topic pub`。

---

## 6. 里程碑 1:模型抽取

### 6.1 实现 `model.py`

目标:把 URDF/SRDF 中与右臂规划有关的信息抽出来,形成稳定的 `R1ProRightArmModel`。

必备字段:

```python
joint_names = [
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
]

base_link = "torso_link4"
tip_link = "right_gripper_link"

urdf_q_min = [-4.4506, -3.1416, -2.356196, -2.0944, -2.356196, -1.047198, -1.5708]
urdf_q_max = [ 1.3090,  0.1745,  2.356196,  0.3491,  2.356196,  1.047198,  1.5708]
urdf_qd_max = [7.1209, 7.1209, 8.3776, 8.3776, 10.4720, 10.4720, 10.4720]

work_q_min = [-4.35, -3.04, -2.26, -1.99, -2.26, -0.95, -1.47]
work_q_max = [ 1.21,  0.07,  2.26,  0.25,  2.26,  0.95,  1.47]
```

### 6.2 添加模型抽取测试

测试必须确认:

- 7 个关节名顺序不变。
- `right_arm_joint2` 上限是 `0.1745` 左右,不能误写成正大范围。
- `work_q_min/work_q_max` 严格在 URDF 限位内部。
- base/tip 与 SRDF 一致。

建议测试断言:

```python
def test_right_arm_work_limits_inside_urdf():
    model = load_r1pro_right_arm_model()
    assert model.joint_names == [f"right_arm_joint{i}" for i in range(1, 8)]
    assert model.base_link == "torso_link4"
    assert model.tip_link == "right_gripper_link"
    assert np.all(model.work_q_min >= model.urdf_q_min)
    assert np.all(model.work_q_max <= model.urdf_q_max)
```

运行:

```bash
pytest tests/unit_tests/test_galaxea_genwaypoint_model.py -v
```

---

## 7. 里程碑 2:轨迹数据结构与归一化

### 7.1 兼容现有输出

当前 baseline 输出的 waypoint 包含:

```json
{
  "step": 0,
  "q_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "action_norm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

新格式必须继续保留 `q_rad` 和 `action_norm`,并增加可审计 metadata:

```json
{
  "episode_id": "r1pro_right_arm_rrt_000001",
  "task": "Move the right arm to the target joint configuration.",
  "planner": "joint_rrt",
  "seed": 12345,
  "direction": "forward",
  "waypoints": [
    {
      "step": 0,
      "t": 0.0,
      "q_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "action_norm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "q_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  ],
  "metadata": {
    "robot": "galaxea_r1_pro",
    "arm": "right",
    "base_link": "torso_link4",
    "tip_link": "right_gripper_link",
    "work_q_min": [],
    "work_q_max": [],
    "validation": {
      "max_abs_step_delta": 0.0,
      "max_abs_velocity": 0.0,
      "max_abs_acceleration": 0.0,
      "min_distance_to_work_limit": 0.0
    }
  }
}
```

### 7.2 归一化函数

沿用当前脚本语义:

```python
def q_to_norm(q, q_min, q_max):
    return 2.0 * (q - q_min) / (q_max - q_min) - 1.0

def norm_to_q(a, q_min, q_max):
    return q_min + 0.5 * (a + 1.0) * (q_max - q_min)
```

测试必须覆盖:

- `q_min -> -1`
- `q_max -> 1`
- `(q_min + q_max) / 2 -> 0`
- `norm_to_q(q_to_norm(q))` 近似还原
- 超界 action 需要在 validator 中拒绝或 clip,不要在归一化函数中悄悄吞掉错误

---

## 8. 里程碑 3:先实现 via-points + spline 采样器

这是最优先实现的生成器,因为依赖少、可控、容易测试。

### 8.1 输入参数

CLI 建议:

```bash
python bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py \
  --planner spline \
  --num-episodes 200 \
  --num-waypoints-min 24 \
  --num-waypoints-max 80 \
  --seed 20260509 \
  --output bt/docs/rwRL/scripts/out/r1pro_right_arm_spline.jsonl
```

### 8.2 采样逻辑

每条轨迹:

1. 固定起点 `home_q`。
2. 固定终点 `target_q`,第一版可沿用当前脚本值:

   ```python
   target_q = [0.5, 0.5, 0.0, -1.2, 0.0, 1.5, 0.0]
   ```

3. 随机采样 `num_via` 个中间点,范围为 1 到 4。
4. 中间点不要全空间乱采样,而是在起终点连线附近采样:

   ```python
   q_line = home_q + alpha * (target_q - home_q)
   q_via = q_line + noise
   ```

5. `noise` 使用逐关节高斯或均匀扰动,每个关节尺度建议从保守值开始:

   ```python
   noise_scale = [0.25, 0.12, 0.25, 0.12, 0.25, 0.08, 0.20]
   ```

6. 所有 via-point clip 到 `work_q_min/work_q_max`。
7. 用 cubic spline 或分段三次 Hermite 曲线生成连续轨迹。
8. 对轨迹重新采样到随机长度 `N in [num_waypoints_min, num_waypoints_max]`。
9. 运行 validator,失败则丢弃并重采样。

### 8.3 第一版不做的事

第一版不要做:

- 不做笛卡尔目标 IK。
- 不做真机反馈闭环。
- 不接 MoveIt2。
- 不接 hpp-fcl 精确 mesh 碰撞。

先把关节空间多样性做起来。

---

## 9. 里程碑 4:实现关节空间 RRT 采样器

RRT 用于生成更明显的绕行轨迹,作为 spline 的补充。

### 9.1 状态空间

状态为 7 维关节向量 `q`。

采样范围为 `work_q_min/work_q_max`。不要使用 URDF 机械极限。

### 9.2 RRT 参数

建议默认值:

```python
step_size = 0.18
goal_bias = 0.20
max_iters = 4000
goal_tolerance = 0.08
edge_resolution = 0.03
```

### 9.3 RRT 流程

1. 树根为 `home_q`。
2. 每轮以 `goal_bias` 概率采样 `target_q`,否则随机采样 `q_rand`。
3. 找最近节点 `q_near`。
4. 从 `q_near` 朝 `q_rand` 延伸 `step_size` 得到 `q_new`。
5. 检查 `q_near -> q_new` 的边:
   - 插值点都在工作域内。
   - 每段最大关节变化不超过 `edge_resolution`。
   - 第一版碰撞检查可先用 stub 返回 true,但 validator 必须保留接口。
6. 若 `q_new` 距离 `target_q` 小于 `goal_tolerance`,尝试连接终点。
7. 回溯路径,再用 spline/shortcut smoothing 平滑。
8. 时间参数化并导出。

### 9.4 RRT 验收标准

在 100 个固定 seed 中:

- 成功率不低于 80%。
- 每条轨迹首点等于 `home_q`。
- 每条轨迹末点与 `target_q` 的无穷范数误差小于 `1e-6`。
- 所有关节在 `work_q_min/work_q_max` 内。
- 平滑后相邻 waypoint 最大变化不超过配置阈值。

---

## 10. 里程碑 5:轨迹校验器

校验器是这个方案能否用于真机/VLA 数据的关键。生成器可以随机,校验必须保守。

### 10.1 必备校验

对每条轨迹检查:

| 检查 | 默认阈值 | 失败处理 |
|---|---:|---|
| 起点误差 | `1e-6` | 丢弃 |
| 终点误差 | `1e-6` | 丢弃 |
| 位置工作域 | `work_q_min/max` | 丢弃 |
| 单步最大变化 | `0.12 rad` | 丢弃或重采样 |
| 最大速度 | `0.6 rad/s` 初始保守值 | 丢弃 |
| 最大加速度 | `1.5 rad/s^2` 初始保守值 | 丢弃 |
| 最大 jerk | `6.0 rad/s^3` 初始保守值 | 丢弃 |
| action_norm 范围 | `[-1, 1]` | 丢弃 |
| NaN/Inf | 不允许 | 丢弃 |

这里的速度阈值应明显小于 URDF 速度上限。URDF 是机械能力,数据生成应该使用更像人类示教的保守速度。

### 10.2 多样性校验

批量生成完成后计算:

- 平均轨迹长度。
- 每个关节的路径方差。
- 每条轨迹相对线性 baseline 的平均偏离量。
- 轨迹之间的 DTW 或简化 L2 距离。
- 被过滤的失败原因统计。

建议验收:

```text
num_requested = 200
num_accepted >= 150
mean_deviation_from_linear >= 0.05 rad
duplicate_rate <= 5%
```

duplicate 可用降采样后的轨迹向量距离判断。

---

## 11. 里程碑 6:时间参数化

### 11.1 第一版 minimum-jerk

对归一化时间 `s in [0, 1]` 使用:

```python
tau = 10 * s**3 - 15 * s**4 + 6 * s**5
```

用途:

- 起点速度接近 0。
- 终点速度接近 0。
- 比线性插值更像真实机器人运动。

### 11.2 速度随机化

每条 episode 随机持续时间:

```python
duration = rng.uniform(2.0, 6.0)
env_step_hz = 10
n_waypoints = int(duration * env_step_hz)
```

然后根据 `duration` 计算 `q_vel/q_acc/q_jerk`。

### 11.3 第二阶段接 TOPPRA

当前 SDK 里有 `toppra` 包。第一阶段先不用。后续接入时:

1. 把几何路径表示为 path parameter `s`。
2. 给定关节速度/加速度约束。
3. 用 TOPPRA 求时间最优或保守时间参数化。
4. 输出同样的 waypoint schema。

---

## 12. 碰撞检查策略

### 12.1 第一版:先做“可插拔接口”

先定义接口:

```python
class CollisionChecker:
    def is_state_valid(self, q: np.ndarray) -> bool:
        return True

    def is_edge_valid(self, q0: np.ndarray, q1: np.ndarray) -> bool:
        return True
```

第一版实现 `NoopCollisionChecker`,但所有 sampler 都必须通过这个接口。这样后续接 hpp-fcl 或 MoveIt2 时不用改 sampler。

### 12.2 第二版:简化几何碰撞

在没有精确 mesh collision 之前,可以先用胶囊/球近似:

- 把上臂、前臂、腕部建成 capsule。
- torso 建成 box/capsule。
- 检查手臂 link 之间、手臂与 torso 之间最小距离。

这比直接用 mesh 快,适合批量生成。

### 12.3 第三版:hpp-fcl 精确碰撞

当前 SDK 有 `hpp-fcl` 包,后续可基于 URDF collision mesh 构建 BVH。接入顺序:

1. 解析 URDF collision mesh 文件路径。
2. 为每个 link 建立 collision object。
3. 用 FK 更新各 link transform。
4. 检查 SRDF 未禁用的 link pair。
5. 输出 `min_distance` 到 trajectory metadata。

---

## 13. CLI 设计

建议统一入口:

```bash
python bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py --help
```

参数:

```text
--planner {spline,rrt,mixed}
--num-episodes INT
--seed INT
--home-q JSON_OR_CSV
--target-q JSON_OR_CSV
--num-waypoints-min INT
--num-waypoints-max INT
--duration-min FLOAT
--duration-max FLOAT
--env-step-hz FLOAT
--max-step-delta FLOAT
--max-velocity FLOAT
--max-acceleration FLOAT
--max-jerk FLOAT
--output PATH
--summary-output PATH
--reject-log PATH
```

示例:

```bash
python bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py \
  --planner mixed \
  --num-episodes 500 \
  --seed 20260509 \
  --env-step-hz 10 \
  --duration-min 2.0 \
  --duration-max 6.0 \
  --output bt/docs/rwRL/scripts/out/r1pro_right_arm_mixed_500.jsonl \
  --summary-output bt/docs/rwRL/scripts/out/r1pro_right_arm_mixed_500_summary.json \
  --reject-log bt/docs/rwRL/scripts/out/r1pro_right_arm_mixed_500_rejects.jsonl
```

---

## 14. 开发任务清单

### Task 1:冻结 baseline

- [ ] 运行当前脚本:

  ```bash
  cd /home/nvidia/lg_ws/RL/RLinf
  python bt/docs/rwRL/scripts/generate_sft_waypoints.py
  ```

- [ ] 保存输出 `sft_waypoints.json` 作为 baseline 样本。
- [ ] 记录当前线性轨迹的最大单步变化。

### Task 2:实现模型加载

- [ ] 新建 `bt/docs/rwRL/scripts/genwaypoint/model.py`。
- [ ] 从 URDF 解析 7 个右臂关节。
- [ ] 写入保守工作域。
- [ ] 添加单测确认工作域在 URDF 内。

### Task 3:实现归一化与 schema

- [ ] 新建 `normalize.py`。
- [ ] 新建 `schema.py`。
- [ ] 保持 `q_rad/action_norm` 兼容现有脚本。
- [ ] 单测覆盖归一化往返。

### Task 4:实现 spline sampler

- [ ] 新建 `spline_sampler.py`。
- [ ] 支持随机 via-point 数量。
- [ ] 支持 minimum-jerk 时间曲线。
- [ ] 支持固定 seed 复现。
- [ ] 单测确认同 seed 输出一致,不同 seed 输出不同。

### Task 5:实现 validators

- [ ] 新建 `validators.py`。
- [ ] 实现位置、速度、加速度、jerk、NaN、终点误差检查。
- [ ] 单测覆盖每种失败原因。

### Task 6:实现 RRT sampler

- [ ] 新建 `rrt_sampler.py`。
- [ ] 实现 joint-space RRT。
- [ ] 接入 `CollisionChecker` 接口。
- [ ] 对成功路径做 shortcut smoothing。
- [ ] 单测用固定 seed 验证能从 `home_q` 到 `target_q`。

### Task 7:实现 CLI 与导出

- [ ] 新建 `generate_sft_waypoints_offline.py`。
- [ ] 支持 `spline/rrt/mixed`。
- [ ] 输出 JSONL。
- [ ] 输出 summary JSON。
- [ ] 输出 reject log。

### Task 8:批量生成与验收

- [ ] 生成 50 条 spline 轨迹做 smoke test。
- [ ] 生成 200 条 mixed 轨迹做初验。
- [ ] 检查 summary 中成功率、重复率、最大速度、最大 jerk。
- [ ] 随机抽 10 条轨迹人工审阅 `q_rad` 曲线。

### Task 9:再考虑第三方软件

满足下面条件后再进入第二阶段:

- [ ] mixed 生成 500 条轨迹成功率达到 75% 以上。
- [ ] 没有任何轨迹越过工作域。
- [ ] 重复率低于 5%。
- [ ] 输出 JSONL 能被下游 SFT 数据读取脚本消费。

第二阶段可评估:

- MoveIt2 + OMPL
- MoveIt2 + CHOMP
- TOPPRA
- hpp-fcl
- Pinocchio 批量 FK/Jacobian

---

## 15. 验收命令

### 15.1 单元测试

```bash
pytest tests/unit_tests/test_galaxea_genwaypoint_model.py -v
pytest tests/unit_tests/test_galaxea_genwaypoint_normalize.py -v
pytest tests/unit_tests/test_galaxea_genwaypoint_validators.py -v
pytest tests/unit_tests/test_galaxea_genwaypoint_samplers.py -v
```

### 15.2 生成 smoke test

```bash
python bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py \
  --planner spline \
  --num-episodes 50 \
  --seed 1 \
  --output /tmp/r1pro_spline_50.jsonl \
  --summary-output /tmp/r1pro_spline_50_summary.json \
  --reject-log /tmp/r1pro_spline_50_rejects.jsonl
```

期望:

- `/tmp/r1pro_spline_50.jsonl` 存在。
- summary 中 `accepted >= 40`。
- `max_action_norm <= 1.0`。
- `max_position_violation == 0`。
- `max_abs_terminal_error <= 1e-6`。

### 15.3 mixed 批量生成

```bash
python bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py \
  --planner mixed \
  --num-episodes 500 \
  --seed 20260509 \
  --output bt/docs/rwRL/scripts/out/r1pro_right_arm_mixed_500.jsonl \
  --summary-output bt/docs/rwRL/scripts/out/r1pro_right_arm_mixed_500_summary.json \
  --reject-log bt/docs/rwRL/scripts/out/r1pro_right_arm_mixed_500_rejects.jsonl
```

期望:

- 接受轨迹不少于 375 条。
- 重复率不超过 5%。
- 平均相对线性 baseline 偏离不低于 `0.05 rad`。
- 所有轨迹首末点正确。

---

## 16. 常见失败与处理

| 现象 | 可能原因 | 处理 |
|---|---|---|
| RRT 成功率低 | `step_size` 太小或 `goal_bias` 太低 | `step_size` 调到 0.22,`goal_bias` 调到 0.3 |
| spline 轨迹越界 | via-point 噪声太大 | 降低 `noise_scale`,或对 spline 结果整体重采样后再校验 |
| jerk 过大 | waypoint 太少或路径太曲折 | 增加 duration 或提高 waypoint 数 |
| 轨迹重复率高 | seed 使用错误或噪声太小 | 确认每条 episode 派生独立 seed,适当增加 via-point 数 |
| action_norm 超界 | q_to_norm 使用了错误限位 | 确认使用 `work_q_min/work_q_max`,不是 URDF 极限 |
| 右臂 J2 经常越界 | 误用了左臂镜像范围或宽松范围 | 右臂 J2 必须在 `[-3.04, 0.07]` 工作域内 |

---

## 17. 第二阶段 MoveIt2 路线

第一阶段完成后,如果需要 MoveIt2:

1. 安装候选包:

   ```bash
   sudo apt install \
     ros-humble-moveit \
     ros-humble-moveit-planners-ompl \
     ros-humble-moveit-planners-chomp \
     ros-humble-moveit-configs-utils
   ```

2. 新建 `r1_pro_moveit_config`。
3. 使用现有 `r1_pro.urdf` 与 `r1_pro_floating_right_srdf.xml`。
4. 补齐:

   ```text
   kinematics.yaml
   ompl_planning.yaml
   planning_pipeline.yaml
   joint_limits.yaml
   moveit_controllers.yaml
   ```

5. 第一版只用 fake controller 和离线 planning scene。
6. 仍然不要接 `/motion_target/*`。
7. 只有当离线规划结果通过本文 validator 后,才允许进入单独的真机安全评审。

---

## 18. 最小可交付定义

第一阶段完成的最小可交付物:

- 一个可运行 CLI:`bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py`
- 至少两个 planner:`spline` 与 `rrt`
- 一个保守 validator
- JSONL 轨迹输出
- summary/reject log
- 单元测试
- 一份 500 条 mixed 轨迹样本

完成后,新的轨迹数据应该比当前线性 baseline 具备:

- 更高路径形状多样性。
- 更自然的加减速。
- 可复现的随机性。
- 明确的失败过滤机制。
- 不依赖真机、不影响真机。

---

## 19. 实施优先级建议

优先级从高到低:

1. `model.py` + `normalize.py` + 单测。
2. `validators.py`。
3. `spline_sampler.py`。
4. CLI 与 JSONL 导出。
5. `rrt_sampler.py`。
6. summary/reject log 与多样性指标。
7. 简化碰撞检查。
8. TOPPRA。
9. hpp-fcl。
10. MoveIt2。

不要跳过 validator 直接追求复杂 planner。对 SFT/VLA 来说,低质量随机轨迹比少量高质量轨迹更危险。

---

## 20. 第一阶段代码草案与设计说明

本节给出建议新增或修改的代码。它们不是“伪代码”,而是可以按文件逐个落地的第一版实现草案。设计目标是:

- **纯离线**:不 import `rclpy`,不创建 ROS2 publisher,不触碰真机控制话题。
- **小模块**:模型、归一化、采样、校验、导出分开,便于单测和替换。
- **保守默认值**:所有轨迹先被 validator 拦一遍,不把随机性直接写进数据集。
- **可复现**:所有 sampler 都从 `numpy.random.Generator` 接收 seed。
- **可扩展**:碰撞检查先是接口,后续可替换为 hpp-fcl 或 MoveIt2 planning scene。

### 20.1 新增 `bt/docs/rwRL/scripts/genwaypoint/__init__.py`

```python
"""Offline waypoint generation utilities for Galaxea R1 Pro.

This package is intentionally ROS-free. It reads static model metadata and
generates validated trajectory files without publishing robot commands.
"""

from .model import R1ProRightArmModel, load_r1pro_right_arm_model
from .normalize import norm_to_q, q_to_norm

__all__ = [
    "R1ProRightArmModel",
    "load_r1pro_right_arm_model",
    "norm_to_q",
    "q_to_norm",
]
```

**为什么这样设计**:

- 包入口只暴露稳定 API,避免上层脚本直接依赖内部 sampler 细节。
- 文档字符串明确“ROS-free”,防止后续实现误把真机控制依赖带进离线生成器。
- `__all__` 让外部 import 行为更可控。

### 20.2 新增 `bt/docs/rwRL/scripts/genwaypoint/model.py`

```python
"""Static R1 Pro right-arm model metadata used by offline planners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SDK_ROOT = Path("/home/nvidia/galaxea/install")
R1PRO_URDF = SDK_ROOT / "mobiman/share/mobiman/urdf/R1_PRO/urdf/r1_pro.urdf"
R1PRO_RIGHT_SRDF = (
    SDK_ROOT
    / "teleoperation_ros2/share/teleoperation_ros2/config/urdf"
    / "r1_pro_floating_right_srdf.xml"
)


@dataclass(frozen=True)
class R1ProRightArmModel:
    """Right-arm model limits and frame names.

    The URDF limits are mechanical limits. The work limits are deliberately
    narrower and should be used by data generation.
    """

    robot_name: str
    arm: str
    base_link: str
    tip_link: str
    joint_names: tuple[str, ...]
    urdf_path: Path
    srdf_path: Path
    urdf_q_min: np.ndarray
    urdf_q_max: np.ndarray
    urdf_qd_max: np.ndarray
    work_q_min: np.ndarray
    work_q_max: np.ndarray

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def validate(self) -> None:
        """Validate internal consistency early."""
        if self.dof != 7:
            raise ValueError(f"R1 Pro right arm must have 7 joints, got {self.dof}")
        arrays = [
            self.urdf_q_min,
            self.urdf_q_max,
            self.urdf_qd_max,
            self.work_q_min,
            self.work_q_max,
        ]
        for array in arrays:
            if array.shape != (7,):
                raise ValueError(f"Expected shape (7,), got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError("Model limits contain NaN or Inf")
        if not np.all(self.urdf_q_min < self.urdf_q_max):
            raise ValueError("URDF lower limits must be smaller than upper limits")
        if not np.all(self.work_q_min < self.work_q_max):
            raise ValueError("Work lower limits must be smaller than upper limits")
        if not np.all(self.work_q_min >= self.urdf_q_min):
            raise ValueError("Work lower limits must stay inside URDF limits")
        if not np.all(self.work_q_max <= self.urdf_q_max):
            raise ValueError("Work upper limits must stay inside URDF limits")


def load_r1pro_right_arm_model() -> R1ProRightArmModel:
    """Return the audited R1 Pro right-arm model metadata.

    This function intentionally pins the audited values instead of reparsing
    URDF on every run. A separate test should compare these values with URDF
    when the SDK version changes.
    """
    model = R1ProRightArmModel(
        robot_name="galaxea_r1_pro",
        arm="right",
        base_link="torso_link4",
        tip_link="right_gripper_link",
        joint_names=(
            "right_arm_joint1",
            "right_arm_joint2",
            "right_arm_joint3",
            "right_arm_joint4",
            "right_arm_joint5",
            "right_arm_joint6",
            "right_arm_joint7",
        ),
        urdf_path=R1PRO_URDF,
        srdf_path=R1PRO_RIGHT_SRDF,
        urdf_q_min=np.array(
            [-4.4506, -3.1416, -2.356196, -2.0944, -2.356196, -1.047198, -1.5708],
            dtype=np.float64,
        ),
        urdf_q_max=np.array(
            [1.3090, 0.1745, 2.356196, 0.3491, 2.356196, 1.047198, 1.5708],
            dtype=np.float64,
        ),
        urdf_qd_max=np.array(
            [7.1209, 7.1209, 8.3776, 8.3776, 10.4720, 10.4720, 10.4720],
            dtype=np.float64,
        ),
        work_q_min=np.array(
            [-4.35, -3.04, -2.26, -1.99, -2.26, -0.95, -1.47],
            dtype=np.float64,
        ),
        work_q_max=np.array(
            [1.21, 0.07, 2.26, 0.25, 2.26, 0.95, 1.47],
            dtype=np.float64,
        ),
    )
    model.validate()
    return model
```

**为什么这样设计**:

- `frozen=True` 防止运行中误改模型限位。轨迹生成一旦写入数据集,限位必须可追溯。
- 第一版把审计过的限位显式写死,比“每次运行动态解析 URDF”更可控。后续 SDK 升级时由单测提醒差异。
- 同时保留 `urdf_path/srdf_path`,导出的 metadata 可以追溯到 SDK 文件。
- `work_q_min/work_q_max` 与 URDF 机械限位分离,符合真机安全文档中“工作域应比机械极限内缩”的原则。

### 20.3 新增 `bt/docs/rwRL/scripts/genwaypoint/normalize.py`

```python
"""Joint-space normalization helpers."""

from __future__ import annotations

import numpy as np


def q_to_norm(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    """Map joint radians to [-1, 1] using the configured work limits."""
    q = np.asarray(q, dtype=np.float64)
    q_min = np.asarray(q_min, dtype=np.float64)
    q_max = np.asarray(q_max, dtype=np.float64)
    return (2.0 * (q - q_min) / (q_max - q_min) - 1.0).astype(np.float64)


def norm_to_q(action: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    """Map normalized actions in [-1, 1] back to joint radians."""
    action = np.asarray(action, dtype=np.float64)
    q_min = np.asarray(q_min, dtype=np.float64)
    q_max = np.asarray(q_max, dtype=np.float64)
    return (q_min + 0.5 * (action + 1.0) * (q_max - q_min)).astype(np.float64)
```

**为什么这样设计**:

- 与现有 `generate_sft_waypoints.py` 保持完全相同的归一化语义,避免下游 SFT action 空间突然变化。
- 函数不做 clip。越界属于数据质量问题,应由 validator 显式报告,不能在归一化层悄悄修正。

### 20.4 新增 `bt/docs/rwRL/scripts/genwaypoint/schema.py`

```python
"""Serializable waypoint and episode schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Waypoint:
    step: int
    t: float
    q_rad: list[float]
    action_norm: list[float]
    q_vel: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    task: str
    planner: str
    seed: int
    direction: str
    waypoints: list[Waypoint]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["waypoints"] = [waypoint.to_dict() for waypoint in self.waypoints]
        return data
```

**为什么这样设计**:

- `Waypoint` 保留当前 `q_rad/action_norm`,新增 `t/q_vel`,使轨迹既能被旧逻辑读取,又能用于后续速度/jerk 分析。
- `Episode.metadata` 不设固定强类型,因为 planner 统计、拒绝原因、碰撞距离等字段会逐步扩展。
- 使用 dataclass 让单测能直接比较对象,导出时再转 JSON。

### 20.5 新增 `bt/docs/rwRL/scripts/genwaypoint/time_profile.py`

```python
"""Time profiles and path resampling."""

from __future__ import annotations

import numpy as np


def minimum_jerk(s: np.ndarray) -> np.ndarray:
    """Minimum-jerk scalar profile with zero start/end velocity."""
    s = np.asarray(s, dtype=np.float64)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def resample_polyline(points: np.ndarray, num_waypoints: int) -> np.ndarray:
    """Resample a joint-space polyline with a minimum-jerk progress profile."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected points shape (N, D), got {points.shape}")
    if len(points) < 2:
        raise ValueError("At least two control points are required")
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least 2")

    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total = cumulative[-1]
    if total <= 1e-12:
        return np.repeat(points[:1], num_waypoints, axis=0)

    s = np.linspace(0.0, 1.0, num_waypoints)
    query = minimum_jerk(s) * total
    result = np.empty((num_waypoints, points.shape[1]), dtype=np.float64)
    for dim in range(points.shape[1]):
        result[:, dim] = np.interp(query, cumulative, points[:, dim])
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def finite_difference(path: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return velocity, acceleration, and jerk using stable finite differences."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    path = np.asarray(path, dtype=np.float64)
    velocity = np.gradient(path, dt, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, dt, axis=0, edge_order=1)
    return velocity, acceleration, jerk
```

**为什么这样设计**:

- minimum-jerk 比线性时间进度更接近示教动作:起止速度自然为零。
- `resample_polyline()` 不依赖 SciPy,当前 Orin 上只要有 Numpy 就能跑。
- `finite_difference()` 集中计算速度、加速度、jerk,validator 和 exporter 共用同一套数值定义。

### 20.6 新增 `bt/docs/rwRL/scripts/genwaypoint/validators.py`

```python
"""Trajectory validation for offline generated R1 Pro waypoints."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import R1ProRightArmModel
from .normalize import q_to_norm
from .time_profile import finite_difference


@dataclass(frozen=True)
class ValidationConfig:
    env_step_hz: float = 10.0
    max_step_delta: float = 0.12
    max_velocity: float = 0.6
    max_acceleration: float = 1.5
    max_jerk: float = 6.0
    terminal_tolerance: float = 1e-6


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def validate_trajectory(
    path: np.ndarray,
    home_q: np.ndarray,
    target_q: np.ndarray,
    model: R1ProRightArmModel,
    config: ValidationConfig,
) -> ValidationResult:
    """Validate a generated trajectory before it can enter the dataset."""
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    path = np.asarray(path, dtype=np.float64)
    home_q = np.asarray(home_q, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)

    if path.ndim != 2 or path.shape[1] != model.dof:
        return ValidationResult(False, [f"bad_shape:{path.shape}"], {})
    if len(path) < 2:
        return ValidationResult(False, ["too_few_waypoints"], {})
    if not np.all(np.isfinite(path)):
        return ValidationResult(False, ["nan_or_inf"], {})

    start_error = float(np.max(np.abs(path[0] - home_q)))
    terminal_error = float(np.max(np.abs(path[-1] - target_q)))
    metrics["start_error"] = start_error
    metrics["terminal_error"] = terminal_error
    if start_error > config.terminal_tolerance:
        reasons.append("bad_start")
    if terminal_error > config.terminal_tolerance:
        reasons.append("bad_terminal")

    lower_violation = np.maximum(model.work_q_min - path, 0.0)
    upper_violation = np.maximum(path - model.work_q_max, 0.0)
    position_violation = float(np.max(lower_violation + upper_violation))
    metrics["max_position_violation"] = position_violation
    if position_violation > 1e-9:
        reasons.append("position_limit")

    step_delta = np.abs(np.diff(path, axis=0))
    max_step_delta = float(np.max(step_delta)) if len(step_delta) else 0.0
    metrics["max_step_delta"] = max_step_delta
    if max_step_delta > config.max_step_delta:
        reasons.append("step_delta")

    dt = 1.0 / config.env_step_hz
    velocity, acceleration, jerk = finite_difference(path, dt)
    max_velocity = float(np.max(np.abs(velocity)))
    max_acceleration = float(np.max(np.abs(acceleration)))
    max_jerk = float(np.max(np.abs(jerk)))
    metrics["max_velocity"] = max_velocity
    metrics["max_acceleration"] = max_acceleration
    metrics["max_jerk"] = max_jerk
    if max_velocity > config.max_velocity:
        reasons.append("velocity")
    if max_acceleration > config.max_acceleration:
        reasons.append("acceleration")
    if max_jerk > config.max_jerk:
        reasons.append("jerk")

    action = q_to_norm(path, model.work_q_min, model.work_q_max)
    max_action_abs = float(np.max(np.abs(action)))
    metrics["max_action_abs"] = max_action_abs
    if max_action_abs > 1.0 + 1e-9:
        reasons.append("action_norm")

    margin_low = path - model.work_q_min
    margin_high = model.work_q_max - path
    metrics["min_distance_to_work_limit"] = float(np.min(np.minimum(margin_low, margin_high)))

    return ValidationResult(ok=not reasons, reasons=reasons, metrics=metrics)
```

**为什么这样设计**:

- validator 是数据质量闸门,所有 sampler 输出都必须通过它。
- `ValidationConfig` 中的速度/加速度/jerk 阈值远低于 URDF 机械极限,目标是生成“像示教”的数据,不是压榨机械能力。
- `reasons` 使用固定字符串,便于 reject log 汇总失败原因。
- 不在 validator 里修改轨迹。生成失败就丢弃重采样,避免数据里混入“被强行修补”的轨迹。

### 20.7 新增 `bt/docs/rwRL/scripts/genwaypoint/collision.py`

```python
"""Collision-checker interfaces for offline planners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CollisionCheckResult:
    ok: bool
    min_distance: float | None = None
    reason: str = ""


class CollisionChecker:
    """Base collision-checker interface."""

    def is_state_valid(self, q: np.ndarray) -> CollisionCheckResult:
        raise NotImplementedError

    def is_edge_valid(self, q0: np.ndarray, q1: np.ndarray) -> CollisionCheckResult:
        raise NotImplementedError


class NoopCollisionChecker(CollisionChecker):
    """First-stage collision checker.

    This keeps sampler code collision-aware while postponing expensive mesh
    collision implementation to the hpp-fcl stage.
    """

    def is_state_valid(self, q: np.ndarray) -> CollisionCheckResult:
        return CollisionCheckResult(ok=True)

    def is_edge_valid(self, q0: np.ndarray, q1: np.ndarray) -> CollisionCheckResult:
        return CollisionCheckResult(ok=True)
```

**为什么这样设计**:

- 第一阶段可以不做碰撞,但接口必须先定下来,否则后续接 hpp-fcl/MoveIt2 会改动 sampler 主逻辑。
- 返回 `min_distance` 预留给精确碰撞检测,后续可写入 metadata。

### 20.8 新增 `bt/docs/rwRL/scripts/genwaypoint/spline_sampler.py`

```python
"""Random via-point sampler with smooth resampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import R1ProRightArmModel
from .time_profile import resample_polyline
from .validators import ValidationConfig, ValidationResult, validate_trajectory


@dataclass(frozen=True)
class SplineSamplerConfig:
    num_waypoints_min: int = 24
    num_waypoints_max: int = 80
    via_points_min: int = 1
    via_points_max: int = 4
    max_attempts: int = 200
    noise_scale: tuple[float, ...] = (0.25, 0.12, 0.25, 0.12, 0.25, 0.08, 0.20)


@dataclass
class SampleResult:
    ok: bool
    path: np.ndarray | None
    validation: ValidationResult | None
    attempts: int
    planner: str
    seed: int


def sample_spline_trajectory(
    home_q: np.ndarray,
    target_q: np.ndarray,
    model: R1ProRightArmModel,
    rng: np.random.Generator,
    seed: int,
    sampler_config: SplineSamplerConfig,
    validation_config: ValidationConfig,
) -> SampleResult:
    """Generate one accepted spline-style trajectory."""
    home_q = np.asarray(home_q, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    noise_scale = np.asarray(sampler_config.noise_scale, dtype=np.float64)

    last_validation: ValidationResult | None = None
    for attempt in range(1, sampler_config.max_attempts + 1):
        num_via = int(rng.integers(sampler_config.via_points_min, sampler_config.via_points_max + 1))
        alphas = np.sort(rng.uniform(0.15, 0.85, size=num_via))
        control_points = [home_q]
        for alpha in alphas:
            q_line = home_q + alpha * (target_q - home_q)
            noise = rng.normal(loc=0.0, scale=noise_scale)
            q_via = np.clip(q_line + noise, model.work_q_min, model.work_q_max)
            control_points.append(q_via)
        control_points.append(target_q)

        num_waypoints = int(
            rng.integers(sampler_config.num_waypoints_min, sampler_config.num_waypoints_max + 1)
        )
        path = resample_polyline(np.asarray(control_points), num_waypoints)
        path[0] = home_q
        path[-1] = target_q

        validation = validate_trajectory(path, home_q, target_q, model, validation_config)
        last_validation = validation
        if validation.ok:
            return SampleResult(True, path, validation, attempt, "spline", seed)

    return SampleResult(False, None, last_validation, sampler_config.max_attempts, "spline", seed)
```

**为什么这样设计**:

- via-point 不在全空间乱采样,而是在起终点连线附近扰动,可控且更安全。
- 每条轨迹随机 waypoint 数,让 SFT 数据具备不同执行节奏。
- `max_attempts` 使生成器不会无限循环,失败原因交给 reject log。
- 函数不直接导出 JSON,只返回 path,保持 sampler 与 I/O 解耦。

### 20.9 新增 `bt/docs/rwRL/scripts/genwaypoint/rrt_sampler.py`

```python
"""Joint-space RRT sampler for R1 Pro right-arm trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .collision import CollisionChecker, NoopCollisionChecker
from .model import R1ProRightArmModel
from .time_profile import resample_polyline
from .validators import ValidationConfig, ValidationResult, validate_trajectory


@dataclass(frozen=True)
class RrtSamplerConfig:
    step_size: float = 0.18
    goal_bias: float = 0.20
    max_iters: int = 4000
    goal_tolerance: float = 0.08
    edge_resolution: float = 0.03
    num_waypoints_min: int = 32
    num_waypoints_max: int = 96
    smoothing_rounds: int = 80
    max_attempts: int = 50


@dataclass
class RrtNode:
    q: np.ndarray
    parent: int


@dataclass
class RrtSampleResult:
    ok: bool
    path: np.ndarray | None
    validation: ValidationResult | None
    attempts: int
    planner: str
    seed: int


def _steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
    delta = q_to - q_from
    distance = float(np.linalg.norm(delta))
    if distance <= step_size:
        return q_to.copy()
    return q_from + delta / distance * step_size


def _edge_points(q0: np.ndarray, q1: np.ndarray, resolution: float) -> np.ndarray:
    distance = float(np.linalg.norm(q1 - q0))
    count = max(2, int(np.ceil(distance / resolution)) + 1)
    alpha = np.linspace(0.0, 1.0, count)
    return q0[None, :] + alpha[:, None] * (q1 - q0)[None, :]


def _edge_valid(
    q0: np.ndarray,
    q1: np.ndarray,
    model: R1ProRightArmModel,
    collision_checker: CollisionChecker,
    resolution: float,
) -> bool:
    points = _edge_points(q0, q1, resolution)
    if np.any(points < model.work_q_min) or np.any(points > model.work_q_max):
        return False
    for q in points:
        if not collision_checker.is_state_valid(q).ok:
            return False
    return collision_checker.is_edge_valid(q0, q1).ok


def _reconstruct(nodes: list[RrtNode], goal_index: int) -> np.ndarray:
    items = []
    index = goal_index
    while index >= 0:
        node = nodes[index]
        items.append(node.q)
        index = node.parent
    items.reverse()
    return np.asarray(items, dtype=np.float64)


def _shortcut(
    path: np.ndarray,
    model: R1ProRightArmModel,
    rng: np.random.Generator,
    collision_checker: CollisionChecker,
    config: RrtSamplerConfig,
) -> np.ndarray:
    if len(path) <= 2:
        return path
    points = [p.copy() for p in path]
    for _ in range(config.smoothing_rounds):
        if len(points) <= 2:
            break
        i, j = sorted(rng.choice(len(points), size=2, replace=False).tolist())
        if j <= i + 1:
            continue
        if _edge_valid(points[i], points[j], model, collision_checker, config.edge_resolution):
            points = points[: i + 1] + points[j:]
    return np.asarray(points, dtype=np.float64)


def sample_rrt_trajectory(
    home_q: np.ndarray,
    target_q: np.ndarray,
    model: R1ProRightArmModel,
    rng: np.random.Generator,
    seed: int,
    sampler_config: RrtSamplerConfig,
    validation_config: ValidationConfig,
    collision_checker: CollisionChecker | None = None,
) -> RrtSampleResult:
    """Generate one accepted joint-space RRT trajectory."""
    home_q = np.asarray(home_q, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    collision_checker = collision_checker or NoopCollisionChecker()

    last_validation: ValidationResult | None = None
    for attempt in range(1, sampler_config.max_attempts + 1):
        nodes = [RrtNode(q=home_q.copy(), parent=-1)]
        goal_index: int | None = None

        for _ in range(sampler_config.max_iters):
            if rng.random() < sampler_config.goal_bias:
                q_rand = target_q
            else:
                q_rand = rng.uniform(model.work_q_min, model.work_q_max)

            distances = [float(np.linalg.norm(node.q - q_rand)) for node in nodes]
            near_index = int(np.argmin(distances))
            q_new = _steer(nodes[near_index].q, q_rand, sampler_config.step_size)
            if not _edge_valid(
                nodes[near_index].q,
                q_new,
                model,
                collision_checker,
                sampler_config.edge_resolution,
            ):
                continue

            nodes.append(RrtNode(q=q_new, parent=near_index))
            new_index = len(nodes) - 1

            if np.linalg.norm(q_new - target_q) <= sampler_config.goal_tolerance:
                if _edge_valid(q_new, target_q, model, collision_checker, sampler_config.edge_resolution):
                    nodes.append(RrtNode(q=target_q.copy(), parent=new_index))
                    goal_index = len(nodes) - 1
                    break

        if goal_index is None:
            continue

        coarse_path = _reconstruct(nodes, goal_index)
        coarse_path = _shortcut(coarse_path, model, rng, collision_checker, sampler_config)
        coarse_path[0] = home_q
        coarse_path[-1] = target_q
        num_waypoints = int(
            rng.integers(sampler_config.num_waypoints_min, sampler_config.num_waypoints_max + 1)
        )
        path = resample_polyline(coarse_path, num_waypoints)
        path[0] = home_q
        path[-1] = target_q

        validation = validate_trajectory(path, home_q, target_q, model, validation_config)
        last_validation = validation
        if validation.ok:
            return RrtSampleResult(True, path, validation, attempt, "rrt", seed)

    return RrtSampleResult(False, None, last_validation, sampler_config.max_attempts, "rrt", seed)
```

**为什么这样设计**:

- RRT 只在 7 维关节空间工作,不需要 MoveIt2、IK 或真机状态。
- `CollisionChecker` 作为参数注入,第一阶段用 `NoopCollisionChecker`,后续换成 hpp-fcl 不改 RRT 主体。
- RRT 原始路径通常折线感强,所以加 shortcut smoothing 和 minimum-jerk resampling。
- 终点被显式设置为 `target_q`,避免数值误差污染 SFT 标签。

### 20.10 新增 `bt/docs/rwRL/scripts/genwaypoint/export.py`

```python
"""JSONL export helpers for generated waypoint episodes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .model import R1ProRightArmModel
from .normalize import q_to_norm
from .schema import Episode, Waypoint
from .time_profile import finite_difference
from .validators import ValidationResult


def build_episode(
    path: np.ndarray,
    model: R1ProRightArmModel,
    planner: str,
    seed: int,
    episode_index: int,
    env_step_hz: float,
    validation: ValidationResult,
    task: str,
) -> Episode:
    """Convert a validated joint path into a serializable episode."""
    dt = 1.0 / env_step_hz
    velocity, _, _ = finite_difference(path, dt)
    actions = q_to_norm(path, model.work_q_min, model.work_q_max)
    waypoints = [
        Waypoint(
            step=i,
            t=float(i * dt),
            q_rad=path[i].astype(float).tolist(),
            action_norm=np.clip(actions[i], -1.0, 1.0).astype(float).tolist(),
            q_vel=velocity[i].astype(float).tolist(),
        )
        for i in range(len(path))
    ]
    metadata: dict[str, Any] = {
        "robot": model.robot_name,
        "arm": model.arm,
        "base_link": model.base_link,
        "tip_link": model.tip_link,
        "joint_names": list(model.joint_names),
        "urdf_path": str(model.urdf_path),
        "srdf_path": str(model.srdf_path),
        "work_q_min": model.work_q_min.astype(float).tolist(),
        "work_q_max": model.work_q_max.astype(float).tolist(),
        "env_step_hz": env_step_hz,
        "validation": validation.metrics,
    }
    return Episode(
        episode_id=f"r1pro_right_arm_{planner}_{episode_index:06d}",
        task=task,
        planner=planner,
        seed=seed,
        direction="forward",
        waypoints=waypoints,
        metadata=metadata,
    )


def write_jsonl(path: Path, episodes: Iterable[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")


def write_reject_log(path: Path, rejects: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in rejects:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_summary(path: Path, episodes: list[Episode], rejects: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reason_counter: Counter[str] = Counter()
    for reject in rejects:
        for reason in reject.get("reasons", []):
            reason_counter[reason] += 1

    metrics = [episode.metadata["validation"] for episode in episodes]
    summary = {
        "accepted": len(episodes),
        "rejected": len(rejects),
        "reject_reasons": dict(reason_counter),
        "max_action_norm": max((m.get("max_action_abs", 0.0) for m in metrics), default=0.0),
        "max_position_violation": max(
            (m.get("max_position_violation", 0.0) for m in metrics),
            default=0.0,
        ),
        "max_abs_terminal_error": max((m.get("terminal_error", 0.0) for m in metrics), default=0.0),
        "max_velocity": max((m.get("max_velocity", 0.0) for m in metrics), default=0.0),
        "max_acceleration": max((m.get("max_acceleration", 0.0) for m in metrics), default=0.0),
        "max_jerk": max((m.get("max_jerk", 0.0) for m in metrics), default=0.0),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
```

**为什么这样设计**:

- exporter 是唯一负责 JSON 结构的地方,sampler 不关心文件格式。
- summary 用同一套 validator metrics,避免“生成时一套指标,验收时另一套指标”。
- reject log 使用 JSONL,失败样本多时仍可流式处理。

### 20.11 新增 `bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py`

```python
#!/usr/bin/env python3
"""Generate diverse offline R1 Pro right-arm SFT waypoints.

This script is pure offline computation. It does not connect to ROS and does
not publish robot commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from genwaypoint.export import build_episode, write_jsonl, write_reject_log, write_summary
from genwaypoint.model import load_r1pro_right_arm_model
from genwaypoint.rrt_sampler import RrtSamplerConfig, sample_rrt_trajectory
from genwaypoint.spline_sampler import SplineSamplerConfig, sample_spline_trajectory
from genwaypoint.validators import ValidationConfig


DEFAULT_HOME_Q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
DEFAULT_TARGET_Q = np.array([0.5, 0.5, 0.0, -1.2, 0.0, 1.5, 0.0], dtype=np.float64)
DEFAULT_TASK = "Move the right arm to the target joint configuration."


def _parse_vector(value: str) -> np.ndarray:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    array = np.asarray(items, dtype=np.float64)
    if array.shape != (7,):
        raise argparse.ArgumentTypeError(f"Expected 7 comma-separated floats, got {value}")
    return array


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=["spline", "rrt", "mixed"], default="mixed")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--home-q", type=_parse_vector, default=DEFAULT_HOME_Q)
    parser.add_argument("--target-q", type=_parse_vector, default=DEFAULT_TARGET_Q)
    parser.add_argument("--num-waypoints-min", type=int, default=24)
    parser.add_argument("--num-waypoints-max", type=int, default=80)
    parser.add_argument("--env-step-hz", type=float, default=10.0)
    parser.add_argument("--max-step-delta", type=float, default=0.12)
    parser.add_argument("--max-velocity", type=float, default=0.6)
    parser.add_argument("--max-acceleration", type=float, default=1.5)
    parser.add_argument("--max-jerk", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--reject-log", type=Path, required=True)
    return parser.parse_args()


def _choose_planner(requested: str, rng: np.random.Generator) -> str:
    if requested != "mixed":
        return requested
    return "spline" if rng.random() < 0.65 else "rrt"


def main() -> None:
    args = _parse_args()
    model = load_r1pro_right_arm_model()

    home_q = np.asarray(args.home_q, dtype=np.float64)
    target_q = np.asarray(args.target_q, dtype=np.float64)
    if np.any(home_q < model.work_q_min) or np.any(home_q > model.work_q_max):
        raise ValueError("home_q is outside the conservative work limits")
    if np.any(target_q < model.work_q_min) or np.any(target_q > model.work_q_max):
        raise ValueError("target_q is outside the conservative work limits")

    validation_config = ValidationConfig(
        env_step_hz=args.env_step_hz,
        max_step_delta=args.max_step_delta,
        max_velocity=args.max_velocity,
        max_acceleration=args.max_acceleration,
        max_jerk=args.max_jerk,
    )
    spline_config = SplineSamplerConfig(
        num_waypoints_min=args.num_waypoints_min,
        num_waypoints_max=args.num_waypoints_max,
    )
    rrt_config = RrtSamplerConfig(
        num_waypoints_min=max(args.num_waypoints_min, 32),
        num_waypoints_max=max(args.num_waypoints_max, 96),
    )

    master_rng = np.random.default_rng(args.seed)
    episodes = []
    rejects = []
    episode_index = 0
    total_trials = 0
    max_total_trials = args.num_episodes * 20

    while len(episodes) < args.num_episodes and total_trials < max_total_trials:
        total_trials += 1
        child_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(child_seed)
        planner = _choose_planner(args.planner, master_rng)

        if planner == "spline":
            result = sample_spline_trajectory(
                home_q=home_q,
                target_q=target_q,
                model=model,
                rng=rng,
                seed=child_seed,
                sampler_config=spline_config,
                validation_config=validation_config,
            )
        else:
            result = sample_rrt_trajectory(
                home_q=home_q,
                target_q=target_q,
                model=model,
                rng=rng,
                seed=child_seed,
                sampler_config=rrt_config,
                validation_config=validation_config,
            )

        if result.ok and result.path is not None and result.validation is not None:
            episode = build_episode(
                path=result.path,
                model=model,
                planner=result.planner,
                seed=child_seed,
                episode_index=episode_index,
                env_step_hz=args.env_step_hz,
                validation=result.validation,
                task=DEFAULT_TASK,
            )
            episodes.append(episode)
            episode_index += 1
        else:
            reasons = []
            metrics = {}
            if result.validation is not None:
                reasons = result.validation.reasons
                metrics = result.validation.metrics
            rejects.append(
                {
                    "planner": result.planner,
                    "seed": child_seed,
                    "attempts": result.attempts,
                    "reasons": reasons,
                    "metrics": metrics,
                }
            )

    if len(episodes) < args.num_episodes:
        raise RuntimeError(
            f"Only generated {len(episodes)} accepted episodes after {total_trials} trials"
        )

    write_jsonl(args.output, episodes)
    write_reject_log(args.reject_log, rejects)
    write_summary(args.summary_output, episodes, rejects)
    print(f"Saved {len(episodes)} episodes to {args.output}")
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved {len(rejects)} rejects to {args.reject_log}")


if __name__ == "__main__":
    main()
```

**为什么这样设计**:

- CLI 入口明确写着“不连接 ROS、不发布命令”,降低误用风险。
- `mixed` 默认 65% spline + 35% RRT。spline 成功率高,RRT 多样性强,混合后更稳定。
- master seed 派生 child seed,保证整体可复现,也能追溯每条 episode 的随机来源。
- `max_total_trials` 防止配置过严时无限循环。
- `home_q/target_q` 在生成前先检查工作域,避免生成器把非法任务目标藏进 reject log。

### 20.12 修改 `bt/docs/rwRL/scripts/generate_sft_waypoints.py`

原脚本可以保留为 baseline,只建议做两点轻微修改:

1. 文件头说明它是 baseline,不是多样轨迹生成器。
2. 输出文件名改为 `sft_waypoints_linear_baseline.json`,避免和新生成器输出混淆。

建议修改:

```python
#!/usr/bin/env python3
"""generate_sft_waypoints.py - linear baseline waypoint generator.

This is a pure-computation baseline. It does not connect to the robot. For
diverse offline trajectory generation, use generate_sft_waypoints_offline.py.
"""
```

并把:

```python
output_path = "sft_waypoints.json"
```

改为:

```python
output_path = "sft_waypoints_linear_baseline.json"
```

**为什么这样设计**:

- 保留 baseline 便于比较多样性指标。
- 改名可以防止工程师误把线性 baseline 当成新数据集。
- 不大改旧脚本,减少已验证流程的扰动。

### 20.13 新增测试 `tests/unit_tests/test_galaxea_genwaypoint_model.py`

```python
import numpy as np

from bt.docs.rwRL.scripts.genwaypoint.model import load_r1pro_right_arm_model


def test_right_arm_model_has_audited_joint_order():
    model = load_r1pro_right_arm_model()
    assert model.joint_names == tuple(f"right_arm_joint{i}" for i in range(1, 8))
    assert model.base_link == "torso_link4"
    assert model.tip_link == "right_gripper_link"


def test_work_limits_are_inside_urdf_limits():
    model = load_r1pro_right_arm_model()
    assert np.all(model.work_q_min >= model.urdf_q_min)
    assert np.all(model.work_q_max <= model.urdf_q_max)


def test_right_arm_joint2_uses_r1pro_right_arm_range():
    model = load_r1pro_right_arm_model()
    joint2 = 1
    assert np.isclose(model.urdf_q_min[joint2], -3.1416)
    assert np.isclose(model.urdf_q_max[joint2], 0.1745)
    assert model.work_q_max[joint2] <= 0.1745
```

**为什么这样设计**:

- J2 是最容易因左右臂镜像而写错的关节,必须有专项测试。
- 测试不依赖 ROS2 graph,可以在 CI 或普通开发机上跑。

### 20.14 新增测试 `tests/unit_tests/test_galaxea_genwaypoint_normalize.py`

```python
import numpy as np

from bt.docs.rwRL.scripts.genwaypoint.model import load_r1pro_right_arm_model
from bt.docs.rwRL.scripts.genwaypoint.normalize import norm_to_q, q_to_norm


def test_q_to_norm_maps_limits_to_minus_one_and_one():
    model = load_r1pro_right_arm_model()
    assert np.allclose(q_to_norm(model.work_q_min, model.work_q_min, model.work_q_max), -1.0)
    assert np.allclose(q_to_norm(model.work_q_max, model.work_q_min, model.work_q_max), 1.0)


def test_q_to_norm_maps_midpoint_to_zero:
    model = load_r1pro_right_arm_model()
    midpoint = 0.5 * (model.work_q_min + model.work_q_max)
    assert np.allclose(q_to_norm(midpoint, model.work_q_min, model.work_q_max), 0.0)


def test_norm_roundtrip():
    model = load_r1pro_right_arm_model()
    q = np.array([0.1, -0.2, 0.3, -1.0, 0.4, 0.2, -0.1], dtype=np.float64)
    action = q_to_norm(q, model.work_q_min, model.work_q_max)
    restored = norm_to_q(action, model.work_q_min, model.work_q_max)
    assert np.allclose(restored, q)
```

**为什么这样设计**:

- 归一化是 SFT action 标签的核心语义,必须单独测试。
- 不在测试里允许 clip,确保越界问题会暴露。

### 20.15 新增测试 `tests/unit_tests/test_galaxea_genwaypoint_validators.py`

```python
import numpy as np

from bt.docs.rwRL.scripts.genwaypoint.model import load_r1pro_right_arm_model
from bt.docs.rwRL.scripts.genwaypoint.time_profile import resample_polyline
from bt.docs.rwRL.scripts.genwaypoint.validators import ValidationConfig, validate_trajectory


HOME_Q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
TARGET_Q = np.array([0.5, 0.05, 0.0, -1.2, 0.0, 0.5, 0.0], dtype=np.float64)


def test_validator_accepts_smooth_path():
    model = load_r1pro_right_arm_model()
    path = resample_polyline(np.vstack([HOME_Q, TARGET_Q]), 80)
    result = validate_trajectory(path, HOME_Q, TARGET_Q, model, ValidationConfig())
    assert result.ok, result.reasons


def test_validator_rejects_position_limit_violation():
    model = load_r1pro_right_arm_model()
    path = resample_polyline(np.vstack([HOME_Q, TARGET_Q]), 80)
    path[10, 1] = model.work_q_max[1] + 0.1
    result = validate_trajectory(path, HOME_Q, TARGET_Q, model, ValidationConfig())
    assert not result.ok
    assert "position_limit" in result.reasons


def test_validator_rejects_bad_terminal():
    model = load_r1pro_right_arm_model()
    path = resample_polyline(np.vstack([HOME_Q, TARGET_Q]), 80)
    path[-1, 0] += 0.01
    result = validate_trajectory(path, HOME_Q, TARGET_Q, model, ValidationConfig())
    assert not result.ok
    assert "bad_terminal" in result.reasons
```

**为什么这样设计**:

- validator 的测试以“通过样本 + 单点破坏”的方式组织,失败原因清楚。
- `TARGET_Q` 的 J2 使用 `0.05`,在保守工作域内。这样测试不会被默认目标 J2 超界干扰。

### 20.16 新增测试 `tests/unit_tests/test_galaxea_genwaypoint_samplers.py`

```python
import numpy as np

from bt.docs.rwRL.scripts.genwaypoint.model import load_r1pro_right_arm_model
from bt.docs.rwRL.scripts.genwaypoint.rrt_sampler import RrtSamplerConfig, sample_rrt_trajectory
from bt.docs.rwRL.scripts.genwaypoint.spline_sampler import (
    SplineSamplerConfig,
    sample_spline_trajectory,
)
from bt.docs.rwRL.scripts.genwaypoint.validators import ValidationConfig


HOME_Q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
TARGET_Q = np.array([0.5, 0.05, 0.0, -1.2, 0.0, 0.5, 0.0], dtype=np.float64)


def test_spline_sampler_is_reproducible_with_seed():
    model = load_r1pro_right_arm_model()
    cfg = SplineSamplerConfig(num_waypoints_min=80, num_waypoints_max=80)
    val = ValidationConfig()
    r1 = sample_spline_trajectory(HOME_Q, TARGET_Q, model, np.random.default_rng(7), 7, cfg, val)
    r2 = sample_spline_trajectory(HOME_Q, TARGET_Q, model, np.random.default_rng(7), 7, cfg, val)
    assert r1.ok, r1.validation.reasons if r1.validation else []
    assert r2.ok, r2.validation.reasons if r2.validation else []
    assert np.allclose(r1.path, r2.path)


def test_spline_sampler_changes_with_different_seed():
    model = load_r1pro_right_arm_model()
    cfg = SplineSamplerConfig(num_waypoints_min=80, num_waypoints_max=80)
    val = ValidationConfig()
    r1 = sample_spline_trajectory(HOME_Q, TARGET_Q, model, np.random.default_rng(7), 7, cfg, val)
    r2 = sample_spline_trajectory(HOME_Q, TARGET_Q, model, np.random.default_rng(8), 8, cfg, val)
    assert r1.ok and r2.ok
    assert not np.allclose(r1.path, r2.path)


def test_rrt_sampler_reaches_target_with_fixed_seed():
    model = load_r1pro_right_arm_model()
    cfg = RrtSamplerConfig(num_waypoints_min=96, num_waypoints_max=96, max_attempts=10)
    val = ValidationConfig(max_step_delta=0.2)
    result = sample_rrt_trajectory(
        HOME_Q,
        TARGET_Q,
        model,
        np.random.default_rng(3),
        3,
        cfg,
        val,
    )
    assert result.ok, result.validation.reasons if result.validation else []
    assert np.allclose(result.path[0], HOME_Q)
    assert np.allclose(result.path[-1], TARGET_Q)
```

**为什么这样设计**:

- sampler 测试使用更温和的 `TARGET_Q`,先验证算法性质,不把测试难度绑定到某个激进动作。
- RRT 具有随机性,固定 seed 是最重要的回归保障。
- `max_step_delta=0.2` 给 RRT 测试稍微放宽,避免单测因路径风格过分脆弱。

### 20.17 代码落地顺序

按下面顺序创建文件,每一步都可以独立运行测试:

1. 创建 `genwaypoint/__init__.py`、`model.py`。
2. 创建 `test_galaxea_genwaypoint_model.py`,运行模型测试。
3. 创建 `normalize.py`、`test_galaxea_genwaypoint_normalize.py`。
4. 创建 `time_profile.py`、`validators.py`、`test_galaxea_genwaypoint_validators.py`。
5. 创建 `collision.py`、`spline_sampler.py`。
6. 创建 `rrt_sampler.py`、`test_galaxea_genwaypoint_samplers.py`。
7. 创建 `schema.py`、`export.py`、`generate_sft_waypoints_offline.py`。
8. 轻微修改 `generate_sft_waypoints.py` 的文档字符串和输出文件名。
9. 运行 smoke test:

   ```bash
   python bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py \
     --planner spline \
     --num-episodes 50 \
     --seed 1 \
     --output /tmp/r1pro_spline_50.jsonl \
     --summary-output /tmp/r1pro_spline_50_summary.json \
     --reject-log /tmp/r1pro_spline_50_rejects.jsonl
   ```

10. 再运行 mixed test:

    ```bash
    python bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py \
      --planner mixed \
      --num-episodes 500 \
      --seed 20260509 \
      --output bt/docs/rwRL/scripts/out/r1pro_right_arm_mixed_500.jsonl \
      --summary-output bt/docs/rwRL/scripts/out/r1pro_right_arm_mixed_500_summary.json \
      --reject-log bt/docs/rwRL/scripts/out/r1pro_right_arm_mixed_500_rejects.jsonl
    ```

### 20.18 设计取舍总结

| 设计点 | 选择 | 原因 |
|---|---|---|
| 模型限位 | 审计值硬编码 + 单测守护 | 生成数据需要稳定可复现,SDK 变更应显式暴露 |
| 工作域 | 使用内缩 `work_q_min/max` | 数据生成不是机械极限测试,应保守 |
| sampler 与 export | 分离 | 后续可以把同一条 path 导出为 JSONL、NPZ 或 RLDS |
| 碰撞 | 先接口后实现 | 第一阶段快速闭环,第二阶段无痛接 hpp-fcl |
| 时间参数化 | minimum-jerk | 低依赖、平滑、符合示教直觉 |
| RRT | joint-space RRT | 不依赖 MoveIt2,能显著提高路径多样性 |
| validator | 拒绝而不是修补 | 数据质量问题必须可见,不能静默污染 SFT 标签 |
| 测试目标 | 使用温和 target | 测算法稳定性,不把单测变成极限姿态压力测试 |

---

## 21. 对前版方案的反思与改良结论

前版文档能落地一个安全的离线生成闭环,但从“针对特定起点和终点尽可能生成多样性丰富且安全的轨迹”这个目标看,还存在四个不足:

| 问题 | 前版表现 | 改良方向 |
|---|---|---|
| 多样性来源偏少 | 主要是 spline + RRT | 增加 RRT-Connect、PRM、bridge sampling、扰动优化、nullspace loop、planner mixture |
| 过度保守于“不装软件” | 把 MoveIt2/TOPPRA/hpp-fcl 放到较后位置 | 明确允许安装兼容软件,分 Profile A/B/C 执行 |
| 配置不够工程化 | 起点/终点/阈值散在 CLI 和代码里 | 增加 `configs/r1pro_right_arm.yaml` 与 `planner_mix.yaml` |
| 多样性验收不足 | 只有简单重复率和偏离线性轨迹 | 增加候选池、分桶、最远点采样、planner 配额、路径签名 |

改良后的目标不是“随机生成很多条”,而是“先生成一个大候选池,再用安全 validator 和多样性 selector 选出高质量子集”。推荐默认流程:

```text
固定 start/goal
  -> 每个 planner 生成候选池
  -> validator 过滤不安全轨迹
  -> diversity selector 去重和分桶
  -> 每个 planner 保留配额
  -> 导出 SFT JSONL + summary + reject log + diversity report
```

这样做的原因是:如果只靠单个随机 planner,轨迹很容易集中在相似走廊;如果不做候选池筛选,低质量随机轨迹会污染 SFT。对 VLA 来说,多样性必须和安全性、平滑性一起优化。

---

## 22. 增强依赖安装方案

### 22.1 新增 `bt/docs/rwRL/scripts/install_genwaypoint_deps.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-enhanced}"

source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash

python3 -m pip install --user --upgrade pip

if [[ "${PROFILE}" == "base" ]]; then
  python3 -m pip install --user numpy pyyaml pytest
elif [[ "${PROFILE}" == "enhanced" ]]; then
  python3 -m pip install --user numpy pyyaml pytest scipy networkx matplotlib
  python3 -m pip install --user toppra || true
elif [[ "${PROFILE}" == "moveit" ]]; then
  sudo apt update
  sudo apt install -y \
    ros-humble-moveit \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-planners-chomp \
    ros-humble-moveit-configs-utils \
    ros-humble-srdfdom \
    ros-humble-geometric-shapes
  python3 -m pip install --user numpy pyyaml pytest scipy networkx matplotlib
else
  echo "Unknown profile: ${PROFILE}" >&2
  echo "Usage: $0 {base|enhanced|moveit}" >&2
  exit 2
fi

python3 - <<'PY'
mods = ["numpy", "yaml"]
optional = ["scipy", "networkx", "matplotlib"]
for name in mods + optional:
    try:
        mod = __import__(name)
        print(f"{name}: OK {getattr(mod, '__file__', 'built-in')}")
    except Exception as exc:
        print(f"{name}: SKIP {type(exc).__name__}: {exc}")
PY
```

**为什么这样设计**:

- `base` 用于完全保守环境,只保证文档里的基础代码能跑。
- `enhanced` 是推荐档,能启用 B-spline、PRM、图搜索和可视化。
- `moveit` 是单独 profile,因为会安装较多 ROS 包,应由工程师显式选择。
- `toppra` 使用 `|| true`,因为不同 aarch64 Python 环境下 wheel 状态可能变化;安装失败不阻塞基础生成器。

### 22.2 安装验证命令

```bash
cd /home/nvidia/lg_ws/RL/RLinf
bash bt/docs/rwRL/scripts/install_genwaypoint_deps.sh enhanced
python3 - <<'PY'
import numpy
import scipy
import networkx
print("enhanced deps OK")
PY
```

验收标准:

- `numpy/scipy/networkx` 能导入。
- 不要求 `toppra` 必须成功。
- 不启动任何 ROS2 控制节点。

---

## 23. 配置文件草案

### 23.1 新增 `bt/docs/rwRL/scripts/genwaypoint/configs/r1pro_right_arm.yaml`

```yaml
robot:
  name: galaxea_r1_pro
  arm: right
  urdf_path: /home/nvidia/galaxea/install/mobiman/share/mobiman/urdf/R1_PRO/urdf/r1_pro.urdf
  srdf_path: /home/nvidia/galaxea/install/teleoperation_ros2/share/teleoperation_ros2/config/urdf/r1_pro_floating_right_srdf.xml
  base_link: torso_link4
  tip_link: right_gripper_link
  joint_names:
    - right_arm_joint1
    - right_arm_joint2
    - right_arm_joint3
    - right_arm_joint4
    - right_arm_joint5
    - right_arm_joint6
    - right_arm_joint7

task:
  name: move_right_arm_to_target_joint_configuration
  description: Move the right arm from a fixed home joint configuration to a fixed target joint configuration.
  home_q: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  target_q: [0.5, 0.05, 0.0, -1.2, 0.0, 0.5, 0.0]
  target_tolerance: 1.0e-6

limits:
  urdf_q_min: [-4.4506, -3.1416, -2.356196, -2.0944, -2.356196, -1.047198, -1.5708]
  urdf_q_max: [1.3090, 0.1745, 2.356196, 0.3491, 2.356196, 1.047198, 1.5708]
  urdf_qd_max: [7.1209, 7.1209, 8.3776, 8.3776, 10.4720, 10.4720, 10.4720]
  work_q_min: [-4.35, -3.04, -2.26, -1.99, -2.26, -0.95, -1.47]
  work_q_max: [1.21, 0.07, 2.26, 0.25, 2.26, 0.95, 1.47]

validation:
  env_step_hz: 10.0
  max_step_delta: 0.12
  max_velocity: 0.6
  max_acceleration: 1.5
  max_jerk: 6.0
  min_limit_margin: 0.0
  duplicate_distance: 0.03

output:
  directory: bt/docs/rwRL/scripts/out
  dataset_name: r1pro_right_arm_diverse
```

**为什么这样设计**:

- 起点和终点放入配置,程序员可以针对每个任务复制一份 YAML,而不是改代码。
- 默认 `target_q` 的 J2 使用 `0.05`,保证在当前保守工作域内。若要使用原脚本里的 `[0.5, 0.5, 0.0, -1.2, 0.0, 1.5, 0.0]`,必须先修正 J2/J6 是否处于 `work_q_max` 内,否则 validator 应拒绝。
- URDF 极限和工作域同时写入,便于审计“为什么某条轨迹被拒绝”。

### 23.2 新增 `bt/docs/rwRL/scripts/genwaypoint/configs/planner_mix.yaml`

```yaml
generation:
  requested_episodes: 500
  candidate_multiplier: 8
  seed: 20260509
  selection:
    method: farthest_point
    per_planner_quota:
      spline: 120
      bezier_loop: 80
      perturb_opt: 80
      rrt: 80
      rrt_connect: 80
      prm: 60

planners:
  spline:
    enabled: true
    weight: 0.25
    num_waypoints_min: 32
    num_waypoints_max: 96
    via_points_min: 1
    via_points_max: 5
    noise_scale: [0.30, 0.12, 0.30, 0.14, 0.30, 0.08, 0.22]

  bezier_loop:
    enabled: true
    weight: 0.15
    num_waypoints_min: 40
    num_waypoints_max: 100
    loop_amplitude: [0.25, 0.08, 0.25, 0.08, 0.25, 0.05, 0.18]
    loop_frequency_choices: [1, 2]

  perturb_opt:
    enabled: true
    weight: 0.15
    num_waypoints_min: 40
    num_waypoints_max: 100
    fourier_terms: 3
    amplitude: [0.22, 0.08, 0.22, 0.08, 0.22, 0.05, 0.16]
    smoothing_rounds: 20

  rrt:
    enabled: true
    weight: 0.15
    step_size: 0.18
    goal_bias: 0.25
    max_iters: 5000
    goal_tolerance: 0.08
    edge_resolution: 0.03

  rrt_connect:
    enabled: true
    weight: 0.15
    step_size: 0.20
    max_iters: 3000
    edge_resolution: 0.03

  prm:
    enabled: true
    weight: 0.15
    num_samples: 1200
    k_neighbors: 12
    max_edge_length: 0.65
    num_paths_per_query: 20
    edge_resolution: 0.03
```

**为什么这样设计**:

- 不让单个 planner 主导数据集。对固定起终点任务,多样性来自“不同路径族”而不是纯随机种子。
- `candidate_multiplier=8` 表示先尝试生成约 4000 条候选,再筛选 500 条。候选池越大,最终数据越容易覆盖多个运动风格。
- `per_planner_quota` 避免成功率最高的 spline 把 PRM/RRT 的多样轨迹挤掉。

### 23.3 新增 `bt/docs/rwRL/scripts/genwaypoint/config.py`

```python
"""Configuration loading for offline waypoint generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str
    home_q: np.ndarray
    target_q: np.ndarray
    target_tolerance: float


@dataclass(frozen=True)
class LimitConfig:
    urdf_q_min: np.ndarray
    urdf_q_max: np.ndarray
    urdf_qd_max: np.ndarray
    work_q_min: np.ndarray
    work_q_max: np.ndarray


@dataclass(frozen=True)
class RuntimeConfig:
    task: TaskConfig
    limits: LimitConfig
    raw: dict[str, Any]


def _array7(data: Any, name: str) -> np.ndarray:
    array = np.asarray(data, dtype=np.float64)
    if array.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def load_runtime_config(path: Path) -> RuntimeConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    task_data = data["task"]
    limits_data = data["limits"]
    task = TaskConfig(
        name=str(task_data["name"]),
        description=str(task_data["description"]),
        home_q=_array7(task_data["home_q"], "task.home_q"),
        target_q=_array7(task_data["target_q"], "task.target_q"),
        target_tolerance=float(task_data["target_tolerance"]),
    )
    limits = LimitConfig(
        urdf_q_min=_array7(limits_data["urdf_q_min"], "limits.urdf_q_min"),
        urdf_q_max=_array7(limits_data["urdf_q_max"], "limits.urdf_q_max"),
        urdf_qd_max=_array7(limits_data["urdf_qd_max"], "limits.urdf_qd_max"),
        work_q_min=_array7(limits_data["work_q_min"], "limits.work_q_min"),
        work_q_max=_array7(limits_data["work_q_max"], "limits.work_q_max"),
    )
    if not np.all(limits.urdf_q_min < limits.urdf_q_max):
        raise ValueError("URDF limits are invalid")
    if not np.all(limits.work_q_min < limits.work_q_max):
        raise ValueError("work limits are invalid")
    if not np.all(limits.work_q_min >= limits.urdf_q_min):
        raise ValueError("work_q_min must stay inside urdf_q_min")
    if not np.all(limits.work_q_max <= limits.urdf_q_max):
        raise ValueError("work_q_max must stay inside urdf_q_max")
    if np.any(task.home_q < limits.work_q_min) or np.any(task.home_q > limits.work_q_max):
        raise ValueError("task.home_q is outside work limits")
    if np.any(task.target_q < limits.work_q_min) or np.any(task.target_q > limits.work_q_max):
        raise ValueError("task.target_q is outside work limits")
    return RuntimeConfig(task=task, limits=limits, raw=data)


def load_planner_mix_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data["generation"]["requested_episodes"] <= 0:
        raise ValueError("requested_episodes must be positive")
    enabled = [
        name
        for name, cfg in data["planners"].items()
        if bool(cfg.get("enabled", False)) and float(cfg.get("weight", 0.0)) > 0.0
    ]
    if not enabled:
        raise ValueError("At least one planner must be enabled")
    return data
```

**为什么这样设计**:

- 配置加载阶段就拒绝非法起终点,比生成几千条后才发现目标越界更高效。
- 不把 YAML 直接传遍全系统,而是转换成强约束 dataclass。
- 保留 `raw`,便于 exporter 把完整配置写入 summary。

---

## 24. 更多轨迹生成方法与代码草案

### 24.1 新增 `bt/docs/rwRL/scripts/genwaypoint/diversity.py`

```python
"""Diversity metrics and candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiversityConfig:
    signature_points: int = 16
    duplicate_distance: float = 0.03


def trajectory_signature(path: np.ndarray, signature_points: int = 16) -> np.ndarray:
    """Convert a variable-length trajectory to a fixed-length signature."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        raise ValueError("path must contain at least two waypoints")
    source = np.linspace(0.0, 1.0, len(path))
    target = np.linspace(0.0, 1.0, signature_points)
    out = np.empty((signature_points, path.shape[1]), dtype=np.float64)
    for dim in range(path.shape[1]):
        out[:, dim] = np.interp(target, source, path[:, dim])
    return out.reshape(-1)


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.sqrt(len(a)))


def linear_deviation(path: np.ndarray) -> float:
    """Mean distance from the straight joint-space line with same endpoints."""
    path = np.asarray(path, dtype=np.float64)
    alpha = np.linspace(0.0, 1.0, len(path))[:, None]
    line = path[0][None, :] + alpha * (path[-1] - path[0])[None, :]
    return float(np.mean(np.linalg.norm(path - line, axis=1)))


def farthest_point_select(
    paths: list[np.ndarray],
    count: int,
    config: DiversityConfig,
) -> list[int]:
    """Select diverse paths by greedy farthest-point sampling."""
    if count <= 0 or not paths:
        return []
    signatures = [trajectory_signature(path, config.signature_points) for path in paths]
    deviations = [linear_deviation(path) for path in paths]
    first = int(np.argmax(deviations))
    selected = [first]
    remaining = set(range(len(paths))) - {first}

    while remaining and len(selected) < count:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            min_distance = min(signature_distance(signatures[idx], signatures[j]) for j in selected)
            if min_distance > best_score:
                best_score = min_distance
                best_idx = idx
        if best_idx is None or best_score < config.duplicate_distance:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected
```

**为什么这样设计**:

- 轨迹长度不一致,必须先降采样成固定签名才能比较。
- 先选偏离线性 baseline 最大的轨迹,能避免最终集合仍然像线性插值。
- 贪心最远点采样简单、稳定、无需额外依赖,适合第一版。

### 24.2 新增 `bt/docs/rwRL/scripts/genwaypoint/perturb_opt_sampler.py`

```python
"""Perturb-and-smooth sampler for fixed start/goal trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import R1ProRightArmModel
from .time_profile import minimum_jerk
from .validators import ValidationConfig, ValidationResult, validate_trajectory


@dataclass(frozen=True)
class PerturbOptConfig:
    num_waypoints_min: int = 40
    num_waypoints_max: int = 100
    fourier_terms: int = 3
    amplitude: tuple[float, ...] = (0.22, 0.08, 0.22, 0.08, 0.22, 0.05, 0.16)
    smoothing_rounds: int = 20
    max_attempts: int = 200


@dataclass
class PerturbResult:
    ok: bool
    path: np.ndarray | None
    validation: ValidationResult | None
    attempts: int
    planner: str
    seed: int


def _smooth(path: np.ndarray, rounds: int) -> np.ndarray:
    result = path.copy()
    for _ in range(rounds):
        middle = result[1:-1]
        result[1:-1] = 0.25 * result[:-2] + 0.5 * middle + 0.25 * result[2:]
    return result


def sample_perturb_opt_trajectory(
    home_q: np.ndarray,
    target_q: np.ndarray,
    model: R1ProRightArmModel,
    rng: np.random.Generator,
    seed: int,
    sampler_config: PerturbOptConfig,
    validation_config: ValidationConfig,
) -> PerturbResult:
    """Generate a trajectory by adding zero-endpoint Fourier perturbations."""
    home_q = np.asarray(home_q, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    amplitude = np.asarray(sampler_config.amplitude, dtype=np.float64)
    last_validation: ValidationResult | None = None

    for attempt in range(1, sampler_config.max_attempts + 1):
        num_waypoints = int(
            rng.integers(sampler_config.num_waypoints_min, sampler_config.num_waypoints_max + 1)
        )
        s = np.linspace(0.0, 1.0, num_waypoints)
        tau = minimum_jerk(s)
        path = home_q[None, :] + tau[:, None] * (target_q - home_q)[None, :]
        perturb = np.zeros_like(path)
        for k in range(1, sampler_config.fourier_terms + 1):
            coeff = rng.normal(0.0, amplitude / k)
            perturb += np.sin(np.pi * k * s)[:, None] * coeff[None, :]
        path = path + perturb
        path[0] = home_q
        path[-1] = target_q
        path = np.clip(path, model.work_q_min, model.work_q_max)
        path[0] = home_q
        path[-1] = target_q
        path = _smooth(path, sampler_config.smoothing_rounds)
        path[0] = home_q
        path[-1] = target_q

        validation = validate_trajectory(path, home_q, target_q, model, validation_config)
        last_validation = validation
        if validation.ok:
            return PerturbResult(True, path, validation, attempt, "perturb_opt", seed)

    return PerturbResult(False, None, last_validation, sampler_config.max_attempts, "perturb_opt", seed)
```

**为什么这样设计**:

- Fourier 扰动天然在起终点为 0,不会破坏精确 start/goal。
- 扰动优化类轨迹能产生“同一任务不同风格”的平滑轨迹,比 RRT 更像示教。
- 不依赖 SciPy,但如果安装了 SciPy,后续可把 `_smooth()` 换成 Savitzky-Golay 或 constrained optimization。

### 24.3 新增 `bt/docs/rwRL/scripts/genwaypoint/prm_sampler.py`

```python
"""PRM-style multi-path sampler for fixed start/goal joint configurations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .collision import CollisionChecker, NoopCollisionChecker
from .model import R1ProRightArmModel
from .time_profile import resample_polyline
from .validators import ValidationConfig, ValidationResult, validate_trajectory


@dataclass(frozen=True)
class PrmSamplerConfig:
    num_samples: int = 1200
    k_neighbors: int = 12
    max_edge_length: float = 0.65
    num_waypoints_min: int = 40
    num_waypoints_max: int = 120
    edge_resolution: float = 0.03
    max_attempts: int = 30


@dataclass
class PrmResult:
    ok: bool
    path: np.ndarray | None
    validation: ValidationResult | None
    attempts: int
    planner: str
    seed: int


def _edge_points(q0: np.ndarray, q1: np.ndarray, resolution: float) -> np.ndarray:
    distance = float(np.linalg.norm(q1 - q0))
    count = max(2, int(np.ceil(distance / resolution)) + 1)
    alpha = np.linspace(0.0, 1.0, count)
    return q0[None, :] + alpha[:, None] * (q1 - q0)[None, :]


def _edge_valid(
    q0: np.ndarray,
    q1: np.ndarray,
    model: R1ProRightArmModel,
    collision_checker: CollisionChecker,
    resolution: float,
) -> bool:
    points = _edge_points(q0, q1, resolution)
    if np.any(points < model.work_q_min) or np.any(points > model.work_q_max):
        return False
    for q in points:
        if not collision_checker.is_state_valid(q).ok:
            return False
    return collision_checker.is_edge_valid(q0, q1).ok


def _dijkstra(graph: list[list[tuple[int, float]]], start: int, goal: int) -> list[int] | None:
    import heapq

    queue: list[tuple[float, int]] = [(0.0, start)]
    parent = {start: -1}
    cost = {start: 0.0}
    while queue:
        current_cost, node = heapq.heappop(queue)
        if node == goal:
            break
        if current_cost > cost[node]:
            continue
        for nxt, weight in graph[node]:
            new_cost = current_cost + weight
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                parent[nxt] = node
                heapq.heappush(queue, (new_cost, nxt))
    if goal not in parent:
        return None
    path = []
    node = goal
    while node >= 0:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def sample_prm_trajectory(
    home_q: np.ndarray,
    target_q: np.ndarray,
    model: R1ProRightArmModel,
    rng: np.random.Generator,
    seed: int,
    sampler_config: PrmSamplerConfig,
    validation_config: ValidationConfig,
    collision_checker: CollisionChecker | None = None,
) -> PrmResult:
    """Build a small roadmap and return one validated path."""
    collision_checker = collision_checker or NoopCollisionChecker()
    home_q = np.asarray(home_q, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    last_validation: ValidationResult | None = None

    for attempt in range(1, sampler_config.max_attempts + 1):
        random_samples = rng.uniform(
            model.work_q_min,
            model.work_q_max,
            size=(sampler_config.num_samples, model.dof),
        )
        nodes = np.vstack([home_q, target_q, random_samples])
        graph: list[list[tuple[int, float]]] = [[] for _ in range(len(nodes))]
        for i, q in enumerate(nodes):
            distances = np.linalg.norm(nodes - q[None, :], axis=1)
            neighbor_ids = np.argsort(distances)[1 : sampler_config.k_neighbors + 1]
            for j in neighbor_ids:
                distance = float(distances[j])
                if distance > sampler_config.max_edge_length:
                    continue
                if _edge_valid(q, nodes[j], model, collision_checker, sampler_config.edge_resolution):
                    graph[i].append((int(j), distance))
                    graph[int(j)].append((i, distance))

        index_path = _dijkstra(graph, 0, 1)
        if index_path is None:
            continue
        coarse_path = nodes[index_path]
        num_waypoints = int(
            rng.integers(sampler_config.num_waypoints_min, sampler_config.num_waypoints_max + 1)
        )
        path = resample_polyline(coarse_path, num_waypoints)
        path[0] = home_q
        path[-1] = target_q
        validation = validate_trajectory(path, home_q, target_q, model, validation_config)
        last_validation = validation
        if validation.ok:
            return PrmResult(True, path, validation, attempt, "prm", seed)

    return PrmResult(False, None, last_validation, sampler_config.max_attempts, "prm", seed)
```

**为什么这样设计**:

- PRM 对固定起终点很合适:同一张 roadmap 可以产生多条不同拓扑路径。
- 第一版用内置 Dijkstra,不强依赖 `networkx`;安装 `networkx` 后可扩展为 k-shortest paths。
- PRM 比单树 RRT 更容易覆盖多个关节空间走廊,适合多样性候选池。

### 24.4 新增 `bt/docs/rwRL/scripts/genwaypoint/rrt_connect_sampler.py`

```python
"""Bidirectional RRT-Connect sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .collision import CollisionChecker, NoopCollisionChecker
from .model import R1ProRightArmModel
from .rrt_sampler import _edge_valid, _reconstruct, _steer, RrtNode
from .time_profile import resample_polyline
from .validators import ValidationConfig, ValidationResult, validate_trajectory


@dataclass(frozen=True)
class RrtConnectConfig:
    step_size: float = 0.20
    max_iters: int = 3000
    edge_resolution: float = 0.03
    num_waypoints_min: int = 40
    num_waypoints_max: int = 100
    max_attempts: int = 50


@dataclass
class RrtConnectResult:
    ok: bool
    path: np.ndarray | None
    validation: ValidationResult | None
    attempts: int
    planner: str
    seed: int


def _nearest(nodes: list[RrtNode], q: np.ndarray) -> int:
    distances = [float(np.linalg.norm(node.q - q)) for node in nodes]
    return int(np.argmin(distances))


def _extend(
    nodes: list[RrtNode],
    q_target: np.ndarray,
    model: R1ProRightArmModel,
    collision_checker: CollisionChecker,
    config: RrtConnectConfig,
) -> int | None:
    near = _nearest(nodes, q_target)
    q_new = _steer(nodes[near].q, q_target, config.step_size)
    if not _edge_valid(nodes[near].q, q_new, model, collision_checker, config.edge_resolution):
        return None
    nodes.append(RrtNode(q=q_new, parent=near))
    return len(nodes) - 1


def sample_rrt_connect_trajectory(
    home_q: np.ndarray,
    target_q: np.ndarray,
    model: R1ProRightArmModel,
    rng: np.random.Generator,
    seed: int,
    sampler_config: RrtConnectConfig,
    validation_config: ValidationConfig,
    collision_checker: CollisionChecker | None = None,
) -> RrtConnectResult:
    collision_checker = collision_checker or NoopCollisionChecker()
    home_q = np.asarray(home_q, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    last_validation: ValidationResult | None = None

    for attempt in range(1, sampler_config.max_attempts + 1):
        tree_a = [RrtNode(q=home_q.copy(), parent=-1)]
        tree_b = [RrtNode(q=target_q.copy(), parent=-1)]
        connected: tuple[int, int] | None = None
        swapped = False

        for _ in range(sampler_config.max_iters):
            q_rand = rng.uniform(model.work_q_min, model.work_q_max)
            a_new = _extend(tree_a, q_rand, model, collision_checker, sampler_config)
            if a_new is None:
                tree_a, tree_b = tree_b, tree_a
                swapped = not swapped
                continue
            b_new = _extend(tree_b, tree_a[a_new].q, model, collision_checker, sampler_config)
            if b_new is not None and np.linalg.norm(tree_b[b_new].q - tree_a[a_new].q) < sampler_config.step_size:
                connected = (a_new, b_new)
                break
            tree_a, tree_b = tree_b, tree_a
            swapped = not swapped

        if connected is None:
            continue

        path_a = _reconstruct(tree_a, connected[0])
        path_b = _reconstruct(tree_b, connected[1])
        if swapped:
            coarse_path = np.vstack([path_b, path_a[::-1]])
        else:
            coarse_path = np.vstack([path_a, path_b[::-1]])
        if np.linalg.norm(coarse_path[0] - home_q) > np.linalg.norm(coarse_path[0] - target_q):
            coarse_path = coarse_path[::-1]
        coarse_path[0] = home_q
        coarse_path[-1] = target_q
        num_waypoints = int(
            rng.integers(sampler_config.num_waypoints_min, sampler_config.num_waypoints_max + 1)
        )
        path = resample_polyline(coarse_path, num_waypoints)
        path[0] = home_q
        path[-1] = target_q
        validation = validate_trajectory(path, home_q, target_q, model, validation_config)
        last_validation = validation
        if validation.ok:
            return RrtConnectResult(True, path, validation, attempt, "rrt_connect", seed)

    return RrtConnectResult(False, None, last_validation, sampler_config.max_attempts, "rrt_connect", seed)
```

**为什么这样设计**:

- RRT-Connect 对固定 start/goal 比单向 RRT 更容易成功。
- 双向树常生成和 RRT 不同的路径形态,能增加数据集多样性。
- 仍然走同一个 validator,不会因为 planner 更激进而绕过安全规则。

---

## 25. CLI 与导出流程的改良

### 25.1 CLI 新增参数

把 `generate_sft_waypoints_offline.py` 从“命令行常量驱动”改成“YAML 配置驱动”,同时保留 CLI 覆盖:

```text
--task-config PATH
--planner-config PATH
--profile {base,enhanced,moveit}
--candidate-multiplier INT
--selection {first_valid,farthest_point,quota_farthest}
--visualize-report PATH
```

推荐命令:

```bash
python bt/docs/rwRL/scripts/generate_sft_waypoints_offline.py \
  --task-config bt/docs/rwRL/scripts/genwaypoint/configs/r1pro_right_arm.yaml \
  --planner-config bt/docs/rwRL/scripts/genwaypoint/configs/planner_mix.yaml \
  --profile enhanced \
  --output bt/docs/rwRL/scripts/out/r1pro_right_arm_diverse_500.jsonl \
  --summary-output bt/docs/rwRL/scripts/out/r1pro_right_arm_diverse_500_summary.json \
  --reject-log bt/docs/rwRL/scripts/out/r1pro_right_arm_diverse_500_rejects.jsonl
```

### 25.2 候选池选择逻辑

CLI 主流程应调整为:

```python
candidate_paths_by_planner = {
    "spline": [],
    "bezier_loop": [],
    "perturb_opt": [],
    "rrt": [],
    "rrt_connect": [],
    "prm": [],
}

for planner_name, quota in per_planner_quota.items():
    target_candidates = quota * candidate_multiplier
    while len(candidate_paths_by_planner[planner_name]) < target_candidates:
        result = run_one_planner(planner_name)
        if result.ok:
            candidate_paths_by_planner[planner_name].append(result.path)
        else:
            rejects.append(result)

selected = []
for planner_name, quota in per_planner_quota.items():
    paths = candidate_paths_by_planner[planner_name]
    indices = farthest_point_select(paths, quota, diversity_config)
    selected.extend((planner_name, paths[i]) for i in indices)
```

**为什么这样设计**:

- 先按 planner 生成候选,再按 planner 配额筛选,保证不同方法都能进入最终数据。
- 多样性选择放在 validator 后面,避免为不安全轨迹计算复杂指标。
- 对固定起终点任务,候选池筛选比单次采样更重要。

### 25.3 Summary 增加多样性指标

`summary.json` 应增加:

```json
{
  "accepted": 500,
  "rejected": 912,
  "accepted_by_planner": {
    "spline": 120,
    "bezier_loop": 80,
    "perturb_opt": 80,
    "rrt": 80,
    "rrt_connect": 80,
    "prm": 60
  },
  "diversity": {
    "mean_linear_deviation": 0.084,
    "min_pairwise_signature_distance": 0.031,
    "mean_pairwise_signature_distance": 0.177,
    "duplicate_rate": 0.0
  }
}
```

验收时不要只看 `accepted` 数量,还要看:

- `accepted_by_planner` 是否接近配额。
- `mean_linear_deviation` 是否明显大于线性 baseline。
- `min_pairwise_signature_distance` 是否大于 `duplicate_distance`。
- `reject_reasons` 是否主要来自合理的安全过滤,而不是配置错误。

---

## 26. MoveIt2 配置模板

本节不是第一批代码必须完成的内容,但既然允许安装兼容软件,就应把 MoveIt2 所需配置写清楚。它可以作为 Profile C 的执行清单。

### 26.1 建议目录

```text
bt/docs/rwRL/scripts/r1_pro_moveit_config/
  config/
    r1_pro.urdf
    r1_pro.srdf
    kinematics.yaml
    ompl_planning.yaml
    planning_pipelines.yaml
    joint_limits.yaml
    moveit_controllers.yaml
  launch/
    demo.launch.py
    offline_planning.launch.py
```

`r1_pro.urdf` 可以先复制:

```bash
cp /home/nvidia/galaxea/install/mobiman/share/mobiman/urdf/R1_PRO/urdf/r1_pro.urdf \
  bt/docs/rwRL/scripts/r1_pro_moveit_config/config/r1_pro.urdf
cp /home/nvidia/galaxea/install/teleoperation_ros2/share/teleoperation_ros2/config/urdf/r1_pro_floating_right_srdf.xml \
  bt/docs/rwRL/scripts/r1_pro_moveit_config/config/r1_pro.srdf
```

### 26.2 `kinematics.yaml`

```yaml
right_arm:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.05
  kinematics_solver_attempts: 5
```

### 26.3 `ompl_planning.yaml`

```yaml
planner_configs:
  RRTConnectkConfigDefault:
    type: geometric::RRTConnect
    range: 0.20
  RRTstarkConfigDefault:
    type: geometric::RRTstar
    range: 0.18
    goal_bias: 0.05
    delay_collision_checking: 1
  PRMkConfigDefault:
    type: geometric::PRM
    max_nearest_neighbors: 12

right_arm:
  default_planner_config: RRTConnectkConfigDefault
  planner_configs:
    - RRTConnectkConfigDefault
    - RRTstarkConfigDefault
    - PRMkConfigDefault
  projection_evaluator: joints(right_arm_joint1,right_arm_joint2)
  longest_valid_segment_fraction: 0.005
```

### 26.4 `planning_pipelines.yaml`

```yaml
planning_pipelines:
  pipeline_names:
    - ompl

ompl:
  planning_plugin: ompl_interface/OMPLPlanner
  request_adapters: >-
    default_planner_request_adapters/AddTimeOptimalParameterization
    default_planner_request_adapters/ResolveConstraintFrames
    default_planner_request_adapters/FixWorkspaceBounds
    default_planner_request_adapters/FixStartStateBounds
    default_planner_request_adapters/FixStartStateCollision
    default_planner_request_adapters/FixStartStatePathConstraints
  start_state_max_bounds_error: 0.1
```

### 26.5 `joint_limits.yaml`

```yaml
joint_limits:
  right_arm_joint1:
    has_position_limits: true
    min_position: -4.35
    max_position: 1.21
    has_velocity_limits: true
    max_velocity: 0.6
  right_arm_joint2:
    has_position_limits: true
    min_position: -3.04
    max_position: 0.07
    has_velocity_limits: true
    max_velocity: 0.6
  right_arm_joint3:
    has_position_limits: true
    min_position: -2.26
    max_position: 2.26
    has_velocity_limits: true
    max_velocity: 0.6
  right_arm_joint4:
    has_position_limits: true
    min_position: -1.99
    max_position: 0.25
    has_velocity_limits: true
    max_velocity: 0.6
  right_arm_joint5:
    has_position_limits: true
    min_position: -2.26
    max_position: 2.26
    has_velocity_limits: true
    max_velocity: 0.6
  right_arm_joint6:
    has_position_limits: true
    min_position: -0.95
    max_position: 0.95
    has_velocity_limits: true
    max_velocity: 0.6
  right_arm_joint7:
    has_position_limits: true
    min_position: -1.47
    max_position: 1.47
    has_velocity_limits: true
    max_velocity: 0.6
```

### 26.6 `moveit_controllers.yaml`

```yaml
moveit_simple_controller_manager:
  controller_names:
    - fake_right_arm_controller

  fake_right_arm_controller:
    type: FollowJointTrajectory
    joints:
      - right_arm_joint1
      - right_arm_joint2
      - right_arm_joint3
      - right_arm_joint4
      - right_arm_joint5
      - right_arm_joint6
      - right_arm_joint7
```

**为什么这样设计**:

- `joint_limits.yaml` 使用保守工作域,不是 URDF 机械极限。
- `moveit_controllers.yaml` 只定义 fake controller,避免任何 MoveIt 规划结果自动流向真机。
- MoveIt2 生成的路径仍必须回到本文 validator 中复检,不能直接进入 SFT 数据。

---

## 27. 改良后的实施顺序

程序员按下面顺序操作,最不容易踩坑:

1. 运行 `install_genwaypoint_deps.sh enhanced`,确认 `numpy/scipy/networkx` 可用。
2. 创建 `configs/r1pro_right_arm.yaml`,把当前任务的精确起点和终点写进去。
3. 创建 `configs/planner_mix.yaml`,先使用本文默认 planner 配额。
4. 创建 `config.py`,让 CLI 从 YAML 读取起点、终点和阈值。
5. 创建 `diversity.py`,先跑固定签名和最远点采样单测。
6. 创建 `perturb_opt_sampler.py`,这是除 spline 外最稳的高多样性来源。
7. 创建 `rrt_connect_sampler.py`,提升采样规划成功率。
8. 创建 `prm_sampler.py`,用 roadmap 增加多路径族。
9. 修改 CLI 为候选池模式,每个 planner 先生成配额数倍候选。
10. 用 `quota_farthest` 选择最终 500 条。
11. 检查 summary 中 `accepted_by_planner`、`mean_linear_deviation`、`duplicate_rate`。
12. 随机抽样可视化 20 条轨迹曲线。
13. 通过 validator 后再考虑 TOPPRA/hpp-fcl/MoveIt2。

改良后的最低验收标准:

```text
requested_episodes = 500
accepted = 500
accepted_by_planner 至少覆盖 5 种 planner
duplicate_rate <= 0.03
mean_linear_deviation >= 0.08 rad
max_position_violation = 0
max_abs_terminal_error <= 1e-6
max_velocity <= 0.6 rad/s
max_acceleration <= 1.5 rad/s^2
max_jerk <= 6.0 rad/s^3
```

如果某个特定起终点太接近关节限位,可以降低多样性阈值,但不能放松位置/速度/加速度/jerk 安全阈值。

