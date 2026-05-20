# "overshoot"是什么意思?

Overshoot 是控制系统中的经典概念：当你给关节一个目标角度（比如从 0.0 rad 到 0.5 rad），实际关节不会精确停在 0.5 rad，而是会冲过头——比如先到 0.52 rad，再回摆到 0.49，最终收敛到 0.5。

在 R1 Pro 的语境下，OCS2 MPC (10Hz) 算出一条最优轨迹，MRT (200Hz) 做插值跟踪，最终 ACU 的 FOC (~1kHz) 执行力矩控制。这条三级控制链中，每一级都引入延迟，而 MPC 的代价函数会在"快速到达"和"平滑减速"之间做 tradeoff。如果 MPC 的 constraintPenaltyInitialValue 偏小或阻尼不足，关节就会在目标点附近出现 overshoot。

这就是为什么 SFT 数据必须录真机反馈：如果 observation 用的是"你以为机器人在的位置"（即你发送的 waypoint），而实际关节因为 overshoot 偏离了那个位置，策略学到的 observation-action 映射就是错的。录 HDAS 真实 right_arm_qpos 才能让策略学会"在有 overshoot 的执行动力学下该怎么出动作"。

# `OCS2 MPC`, `MRT`, `ACU`, `FOC`, `constraintPenaltyInitialValue` 分别是什么意思?

按从上层到底层的顺序：

OCS2 MPC — ETH 开源的最优控制求解器（Optimal Control for Switched Systems 2）。MPC = Model Predictive Control（模型预测控制），每隔 100ms (10Hz) 用 Pinocchio 动力学模型向前"预测" 1 秒的最优关节轨迹，然后只执行第一小段，下次再重新规划。好处是能同时处理关节限位、自碰撞等约束。R1 Pro 把左右臂各建一个独立的 MPC 问题求解。

MRT — Model Reference Tracking，OCS2 的内置插值层。MPC 10Hz 出的轨迹是离散的粗点，MRT 在 200Hz 上做样条插值，生成更平滑的关节目标流给下层。可以理解为"把 10Hz 的规划结果上采样到 200Hz"。

ACU — Arm Control Unit，R1 Pro 手臂上的嵌入式 MCU（微控制器）。左臂是 ACUL (CAN 地址 0x15)，右臂是 ACUR (0x16)。它接收 Orin 通过 CAN-FD 发来的关节目标（位置、速度、kp、kd、前馈力矩），在板上执行最底层的电机闭环控制。

FOC — Field-Oriented Control（磁场定向控制），无刷电机的标准驱动算法。ACU 内部以 ~1kHz 频率运行 FOC 环，把"期望力矩/位置"转换为三相电流指令驱动电机。这是整条控制链的最底层，纯硬件实时。

constraintPenaltyInitialValue — OCS2 SLQ-MPC 配置文件（r1_pro_task_left.info）中的一个参数，默认值 20.0。OCS2 用罚函数法处理约束（关节限位、碰撞等），这个值是罚项的初始权重。太小则约束容易被违反（overshoot 冲过限位），太大则求解器过于保守（动作迟缓）。配合 constraintPenaltyIncreaseRate: 2.0，求解器会在迭代中自动加大罚权重直到约束满足。

整条链路的频率金字塔：
```
MPC   10 Hz   — 规划"往哪走"
MRT  200 Hz   — 插值"怎么走过去"
CAN-FD        — 传输关节指令
ACU/FOC ~1kHz — 执行电机力矩闭环
越往下频率越高、越"硬实时"。overshoot 主要产生在 MPC→MRT 这一层，因为 MPC 的 10Hz 规划粒度相对粗，MRT 插值无法完美补偿规划误差。
```

# 解释一下`Pinocchio 动力学模型`

Pinocchio 是一个开源的刚体动力学计算库（C++/Python），专门用来算"多关节机器人的运动学和动力学"。名字来自木偶匹诺曹——关节串联的刚体链条。

它解决什么问题
给定一个机器人（比如 R1 Pro 的 7 关节手臂），你需要频繁回答这类问题：

正运动学：7 个关节角是 [0.5, 0.3, 0.0, -1.2, 0.0, 1.5, 0.0]，手指尖（末端执行器）在空间中的 xyz 坐标是多少？

逆动力学：我想让手臂以某个加速度运动，每个关节电机需要出多大力矩？

雅可比矩阵：关节 3 转动 0.01 rad，末端会往哪个方向移动多少？

这些计算的核心输入是 URDF 文件——它描述了每根连杆的长度、质量、惯量，以及关节的连接方式。Pinocchio 把 URDF 解析成内部的运动树，然后用高效的递推算法（Featherstone 算法）来计算。

用一个简单例子说明
想象一个只有 2 个关节的平面机械臂（像台灯臂）：
```
        关节2 (q2)
  肩 ──── 上臂 ──── 肘 ──── 前臂 ──── 手
关节1 (q1)    L1=0.3m              L2=0.3m
```

正运动学（Forward Kinematics）——已知关节角，求手的位置：
```
手_x = L1·cos(q1) + L2·cos(q1 + q2)
手_y = L1·sin(q1) + L2·sin(q1 + q2)
```

q1=0, q2=0 时手在 (0.6, 0) — 臂完全伸直。q1=90°, q2=-90° 时手在 (0, 0.3) — 上臂竖直、前臂水平。

雅可比矩阵（Jacobian）——关节微小变化如何影响末端：
```
J = [ ∂手_x/∂q1  ∂手_x/∂q2 ]
    [ ∂手_y/∂q1  ∂手_y/∂q2 ]
```
这个 2×2 矩阵告诉你：q1 转一点点，手往哪移；q2 转一点点，手往哪移。MPC 就靠它来规划"想让手往右移 1cm，各关节该怎么动"。

逆动力学（Inverse Dynamics）——已知期望运动，求所需力矩：
```
τ = M(q)·q̈ + C(q, q̇)·q̇ + g(q)
M(q)：惯量矩阵（臂越伸展，转动惯量越大）
C(q, q̇)：科里奥利力和离心力（关节之间的耦合效应）
g(q)：重力项（臂水平伸出时肩关节要承受最大重力力矩）
```

2 关节的情况手算还行，但 R1 Pro 是 7 关节——惯量矩阵是 7×7，里面每个元素都是 7 个关节角的三角函数组合。手算不现实，这就是 Pinocchio 存在的意义。

在 R1 Pro 中的角色
OCS2 MPC 每 100ms 要做一次"未来 1 秒的最优轨迹规划"。规划过程中要反复调用：

正运动学 — 当前关节角下末端在哪？会不会碰到自己（自碰撞检测交给 HPP-FCL）？
雅可比 — 调整关节角后末端怎么变化？（MPC 的梯度下降需要）
动力学 — 这条轨迹需要的力矩在电机能力范围内吗？
每次 MPC 迭代要调这些函数几十次。Pinocchio 用 Featherstone 的 O(n) 递推算法（n = 关节数），比朴素矩阵乘法快得多。再加上 R1 Pro 用 CppADCodeGen 把 Pinocchio 的计算在编译期固化成 .so（auto_generated/r1_robot_left/dynamics_flow_map/），运行时直接调编译好的 C 代码，避免了重复的符号计算开销。

简单说：Pinocchio 是 MPC 的"物理引擎"——MPC 负责决策"往哪走"，Pinocchio 负责回答"这么走物理上可行吗、需要多大力"。

# 上面提到的知识属于什么学科

这些内容主要横跨两个学科：

Robot Dynamics and Kinematics（机器人动力学与运动学）— 研究多关节刚体链的运动（正/逆运动学、雅可比）和力/力矩关系（正/逆动力学、惯量矩阵、重力补偿）。Pinocchio、Featherstone 算法、URDF 建模都属于这个范畴。它是机械工程（Mechanical Engineering）和机器人学（Robotics）的核心分支。

Optimal Control / Model Predictive Control（最优控制 / 模型预测控制）— 研究如何在约束条件下找到最优的控制输入序列。OCS2 的 SLQ-MPC、罚函数法（constraintPenaltyInitialValue）、代价函数设计、Riccati 方程求解都属于这里。它根植于控制理论（Control Theory），属于电气工程（Electrical Engineering）或应用数学的传统领域。

两者的交叉点就是 Robot Control（机器人控制）— 用最优控制的数学工具驱动动力学模型描述的物理机器人。R1 Pro 的整条控制链（MPC → MRT → FOC）就是这个交叉领域的典型工程实现。

如果要找一个统一的学科名称，最接近的是 Robotics（机器人学），它本身就是机械、控制、计算机科学的交叉学科。在大学课程体系中，这些内容通常出现在 Robot Dynamics and Control（机器人动力学与控制）或 Introduction to Robotics: Mechanics and Control（经典教材：Craig 的同名书）这类课程中。