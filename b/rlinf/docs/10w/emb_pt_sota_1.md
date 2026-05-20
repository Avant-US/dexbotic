# 具身智能预训练模型：4B+ 参数、跨本体 Zero-shot 部署方案与 2025-2026 SOTA 综述

> 目标：训练一个 **4B+ 参数** 的具身智能预训练模型，基于 **1 万小时以上** 机器人操作视频训练，能部署到不同机器人本体上 **zero-shot** 执行各种复杂任务。

---

## 1. 问题分析与核心挑战

### 1.1 目标拆解

| 维度 | 要求 | 技术含义 |
|------|------|---------|
| **模型规模** | 4B+ 参数 | 需要大容量 VLM backbone + 动作解码头，典型选择：Qwen2.5-VL-3B/7B, InternVL2-4B, Gemma-2B+DiT |
| **数据规模** | 1 万小时+ | 纯机器人遥操数据极难达到；需混合人类视频（自我中心视角）+ 机器人轨迹 + 仿真数据 |
| **跨本体** | 多种机器人构型 | 动作空间必须抽象/统一——末端执行器相对位姿(EEF delta)或通用动作码本 |
| **Zero-shot** | 无需微调即可部署新本体 | 预训练时本体多样性 + 动作表示的本体无关性是关键 |

### 1.2 核心矛盾

- **数据瓶颈**：当前最大公开机器人数据集 Open X-Embodiment 约 1M 轨迹（~数千小时），远未达到 1 万小时纯机器人数据。解决路径：大规模人类视频预训练 + 少量机器人数据对齐。
- **动作空间异构**：不同机器人的自由度(DoF)、关节构型、控制频率各不相同。解决路径：通用动作表示（相对 EEF delta / 通用动作码本 / 语言化动作）。
- **Zero-shot 泛化**：需要模型在未见过的本体上也能正确输出动作。解决路径：本体无关的预训练 + 本体 prompt/adapter。

---

## 2. 2025-2026 SOTA 模型全景

### 2.1 Vision-Language-Action (VLA) 模型

#### LAP — Language-Action Pre-training (2026.02)
- **论文**: [arXiv:2602.10556](https://arxiv.org/abs/2602.10556)
- **项目**: https://lap-vla.github.io/
- **核心贡献**: **首个在未见本体上实现实质性 zero-shot 迁移的 VLA**
- 将低级机器人动作直接表示为自然语言 token，无需学习动作分词器、无需昂贵标注、无需本体特定架构设计
- LAP-3B 在未见本体上 zero-shot 平均成功率 >50%，比最强基线提升 ~2×；其他开源 VLA 在相同评估下均为 0%
- 从 4B 扩展到 27B 参数时，token 和动作验证损失持续下降，而可比基线饱和或退化
- 真实机器人上，LAP-3B 使用约 2.5× 更少的演示数据即达到同等性能
- **与你的目标高度匹配**：4B+ 参数、跨本体 zero-shot、可扩展

#### SpatialVLA (RSS 2025)
- **论文**: [arXiv:2501.15830](https://arxiv.org/abs/2501.15830)
- **代码**: https://github.com/SpatialVLA/SpatialVLA
- **模型**: [spatialvla-4b-224-pt](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt) (HuggingFace)
- 核心思想：空间理解是机器人操作的关键——引入 Ego3D Position Encoding 注入 3D 信息 + Adaptive Action Grid 离散化空间动作
- 在 1.1M 真实机器人 episodes 上预训练，**恰好 4B 参数**
- 以 3.5B 参数超越 55B 的 RT-2-X（Visual Matching 71.9% vs 60.7%）
- 支持 zero-shot 部署多种任务

#### π0 / π0-FAST / π0.5 (Physical Intelligence)
- **论文**: [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (π0) | [pi0.5 PDF](https://www.pi.website/download/pi05.pdf)
- **代码**: https://github.com/Physical-Intelligence/openpi
- **π0**: 基于 Gemma-2B 的 VLA，使用 **Flow Matching** 生成连续动作（迭代去噪），在 1 万小时+ 多样机器人数据上预训练
- **π0-FAST**: 引入 FAST 动作分词器（基于 DCT 变换将连续动作压缩为离散 token），推理速度提升 5×
- **π0.5**: 加入 web 数据、语义子任务预测的共训练，实现开放世界泛化
- **π0.6** (2025.11): 引入 RL 在线学习提升成功率
- **MEM** (2026.03): 多尺度具身记忆，支持 10 分钟+ 长时任务
- 开源 OpenPI 支持 JAX/PyTorch 双框架训练部署

#### OpenVLA + OFT + FAST (Stanford, 2024-2025)
- **论文**: [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- **代码**: https://github.com/openvla/openvla
- 7B 参数，DINOv2 + SigLIP 双视觉编码器 + Llama-2 LLM backbone
- 在 Open X-Embodiment 数据集上训练，超越 RT-2
- **OFT** (2025.03): 优化微调方案，25-50× 加速推理，支持多视角输入和双臂控制
- **FAST** (2025.01): 动作分词器，推理加速 15×
- 完全开源，是社区最广泛使用的 VLA baseline

#### UniAct — Universal Action Space (CVPR 2025)
- **论文**: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Zheng_Universal_Actions_for_Enhanced_Embodied_Foundation_Models_CVPR_2025_paper.pdf)
- 核心创新：**通用动作空间**——通过 VQ 码本捕获跨不同机器人的通用原子行为
- 0.5B 的 UniAct 达到 14× 更大模型的 SOTA 水平
- 输入通过共享 VLM 提取跨数据源可迁移特征 → 输出通用动作码 → 不同执行头转换为特定本体控制命令

#### RDT-2 — Robotics Diffusion Transformer 2
- **项目**: https://rdt-robotics.github.io/rdt2/
- **首个在未见本体上实现 zero-shot 部署的基础模型**（简单开放词汇任务：抓取、放置、按压、擦拭）
- 使用 Residual VQ 作为动作分词器，在 UMI 数据集上预训练 Qwen2.5-VL-7B-Instruct
- 通过 UMI 硬件提供统一末端执行器，最小化本体差距——无需遥操、无需微调，即插即用

#### SmolVLA (Hugging Face, 2025)
- 仅 450M 参数的紧凑 VLA，完全在 LeRobot 上训练
- 性能与 Octo、OpenVLA、π0 等大模型可比
- 使用 Flow Matching + 异步推理

### 2.2 World Action Model (WAM)

#### DreamZero (2026.02)
- **论文**: [arXiv:2602.15922](https://arxiv.org/abs/2602.15922)
- **14B 参数**，构建于预训练图像→视频扩散骨干之上
- 同时预测动作和视觉未来状态——**World Action Model** 范式
- 泛化能力比 SOTA VLA 提升 >2×
- 支持跨本体迁移：从不同本体的仅视频数据学习，10-20 分钟视频即可适配新本体，提升 >42%
- 14B 模型实现 7Hz 实时闭环控制

### 2.3 双系统架构（大脑-小脑 / System 1-2）

#### NVIDIA GR00T N1 → N1.5 → N1.7
- **论文**: [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) (N1)
- **代码**: https://github.com/NVIDIA/Isaac-GR00T
- **N1** (2025.03): 双系统架构——System 2 (VLM) 进行环境推理和任务规划，System 1 (扩散 Transformer) 生成实时连续运动
- **N1.5** (2025 中): Eagle 2.5 VLM + FLARE 损失（未来潜在表示对齐），1K H100 训练 250K 步，batch size 16384
- **N1.7** (2026.04 Early Access):
  - **3B 参数**，Cosmos-Reason2-2B backbone + 32 层 DiT
  - **EgoScale**: 在 20,854 小时自我中心人类视频上预训练——发现了机器人灵巧操作的首个 Scaling Law
  - 相对 EEF 动作空间，跨人类和机器人本体共享
  - Apache 2.0 商用许可
  - 从 1K→20K 小时数据，任务完成率从 0.30→0.71

| 版本 | VLM Backbone | 动作头 | 人类视频数据 | 许可 |
|------|-------------|--------|------------|------|
| N1 | Eagle VLM | DiT | 有限 | 开源 |
| N1.5 | Eagle 2.5 | DiT + FLARE | 启用 | 开源 |
| N1.7 | Cosmos-Reason2-2B | 32层 DiT | 20K+ 小时 | Apache 2.0 |

#### Gemini Robotics (Google DeepMind, 2025-2026)
- **论文**: [arXiv:2503.20020](https://arxiv.org/abs/2503.20020)
- 基于 Gemini 2.0 构建，新增物理动作输出模态
- **Gemini Robotics**: 通用 VLA，可执行折纸、包装等极复杂多步任务
- **Gemini Robotics-ER**: 增强具身推理（物体检测、轨迹预测、3D 边界框）
- **Gemini Robotics 1.5** (2025.09): 最强 VLA，思考后行动 + 跨本体学习
- **Gemini Robotics On-Device**: 低算力双臂机器人版本
- 闭源，仅对 Agile Robots、Boston Dynamics 等受信测试者开放

#### GEN-0 / GEN-1 (Generalist AI)
- **博客**: [GEN-0](https://generalistai.com/blog/nov-04-2025-GEN-0) | [GEN-1](https://generalistai.com/blog/apr-02-2026-GEN-1)
- GEN-0 预训练数据集：**270,000+ 小时** 真实世界操作数据，每周增长 1 万小时
- GEN-1: >99% 成功率，任务速度比 SOTA 快 ~3×
- 预训练数据不含机器人数据——适配新任务时同时适配新本体和新任务
- 已测试 6DoF、7DoF、16+DoF 半人形机器人
- 无预训练从头训练: 19% → GEN-0 微调: 64% → GEN-1: 99%

### 2.4 中国生态

#### 智源 RoboOS / RoboBrain
- 全球首个跨本体具身大小脑协作框架
- RoboBrain：云端具身大脑，适配机械臂/轮式/双足/四足等构型
- RoboOS 2.0 (2025.06): 全链路响应时延 <3ms，端云通信效率提升 27×，任务规划准确率提升 74%
- 支持 MCP 机制，构建具身智能"应用商店"生态

#### 上海 AI 实验室 InternVLA
- InternVLA N1: 双系统解耦（M1 推理规划 + A1 敏捷执行）
- 60Hz 连续推理效率，跨场景、跨本体零样本泛化

#### 小米 MiMo-Embodied (X-Embodied)
- **论文**: [arXiv:2511.16518](https://arxiv.org/abs/2511.16518)
- 跨本体基础模型技术报告

---

## 3. 关键技术路线深度分析

### 3.1 架构选择：三大范式

```
┌─────────────────────────────────────────────────────────────┐
│                    架构范式对比                               │
├─────────────────┬──────────────┬──────────────┬─────────────┤
│                 │ 单模型 VLA   │ 双系统架构    │  WAM        │
│ 代表            │ OpenVLA, π0  │ GR00T, Helix │ DreamZero   │
│                 │ LAP, SpatialVLA│            │             │
├─────────────────┼──────────────┼──────────────┼─────────────┤
│ 感知+语言+动作  │ 单 forward   │ VLM + DiT    │ 视频扩散    │
│                 │ pass         │ 双模块       │ + 动作解码  │
├─────────────────┼──────────────┼──────────────┼─────────────┤
│ 优点            │ 简洁、低延迟 │ 灵巧、可分离 │ 强泛化、    │
│                 │              │ 训练         │ 仅视频可学  │
├─────────────────┼──────────────┼──────────────┼─────────────┤
│ 缺点            │ 动作精度受限 │ 计算复杂     │ 推理慢      │
├─────────────────┼──────────────┼──────────────┼─────────────┤
│ 4B 可行性       │ ✅ 直接可用  │ ✅ VLM ~2-3B │ ⚠️ 需优化   │
│                 │              │ + DiT ~1B    │             │
├─────────────────┼──────────────┼──────────────┼─────────────┤
│ 跨本体 Zero-shot│ LAP ✅       │ N1.7 ✅      │ DreamZero ✅│
│                 │ SpatialVLA ⚠️│              │ (few-shot)  │
└─────────────────┴──────────────┴──────────────┴─────────────┘
```

### 3.2 动作表示：关键设计决策

| 方案 | 代表 | 机制 | 跨本体友好度 |
|------|------|------|-------------|
| **离散化 bin** | OpenVLA 原版 | 将连续动作离散到 256 bin | ⭐⭐ |
| **FAST 分词** | π0-FAST | DCT 变换 → VQ 离散码 | ⭐⭐⭐ |
| **Flow Matching** | π0, GR00T N1 | 迭代去噪生成连续动作 | ⭐⭐⭐⭐ |
| **扩散 (DDPM)** | Octo, RDT | 扩散过程建模动作分布 | ⭐⭐⭐ |
| **语言化动作** | LAP | 动作直接表示为自然语言 | ⭐⭐⭐⭐⭐ |
| **通用动作码本** | UniAct | VQ 码捕获跨机器人原子行为 | ⭐⭐⭐⭐⭐ |
| **相对 EEF delta** | GR00T N1.7 | 末端执行器增量位姿 | ⭐⭐⭐⭐ |

**推荐**：对于跨本体 zero-shot，**语言化动作 (LAP)** 或 **通用动作码本 (UniAct)** 最为适合，因为它们从根本上消除了本体特定的动作空间耦合。

### 3.3 预训练数据策略

达到 1 万小时的数据构成方案：

```
数据金字塔：

          ┌──────────────┐
          │  目标本体数据  │  ~100-500 小时
          │  (遥操采集)    │  高质量、任务对齐
          ├──────────────┤
          │  多本体机器人   │  ~1,000-3,000 小时
          │  数据 (OXE +   │  建立跨本体表示
          │  DROID + UMI)  │
          ├──────────────┤
          │  自我中心人类   │  ~5,000-20,000 小时
          │  视频 (EgoScale │  学习物理世界动力学
          │  / HumanNet)   │  和操作先验
          ├──────────────┤
          │  互联网视频 +   │  ~100,000+ 小时
          │  VLM 预训练数据 │  视觉-语言对齐
          └──────────────┘
```

**关键发现**（EgoScale, 2026）：
- 1K→20K 小时人类视频预训练，下游任务完成率从 0.30→0.71，呈 **log-linear** 增长
- 人类→机器人迁移在预训练足够多样化时自然涌现（场景、任务、本体多样性 > 绝对数据量）

### 3.4 训练流程：多阶段配方

参考 π0 和 GR00T N1.7 的实践，推荐三阶段训练：

```
阶段 1: VLM 基座预训练
  ├─ 初始化自预训练 VLM (如 Qwen2.5-VL-3B, InternVL2-4B)
  ├─ 在互联网规模图文/视频数据上继续训练视觉-语言对齐
  └─ 输出: 具有强视觉理解和语言能力的基座

阶段 2: 具身预训练 (核心阶段)
  ├─ 混合数据: 人类自我中心视频 + 多本体机器人轨迹
  ├─ 动作表示: 语言化动作 或 通用动作码本
  ├─ 训练目标: 
  │   ├─ 动作预测 (flow matching / autoregressive)
  │   ├─ 未来视觉状态预测 (FLARE / video prediction)
  │   └─ 语言-动作对齐
  └─ 输出: 跨本体通用具身策略

阶段 3: 后训练 (可选，用于特定部署)
  ├─ 针对目标本体的少量微调 (20-100 demonstrations)
  ├─ RL 在线优化 (参考 π0.6)
  └─ 本体适配器 (LoRA / soft prompt)
```

---

## 4. 核心数据集与工具链

### 4.1 机器人操作数据集

| 数据集 | 规模 | 本体数 | 特点 | 链接 |
|--------|------|--------|------|------|
| **Open X-Embodiment** | 1M+ 轨迹 | 22 | 最大开源机器人数据集，60个子集，34个实验室 | [link](https://robotics-transformer-x.github.io/) |
| **DROID** | 76K 轨迹, 350h | 1 (Franka) | 564 场景, 86 任务, 极高场景多样性 | [link](https://droid-dataset.github.io/) |
| **UMI** | — | 多种臂 | 统一末端执行器，实现零样本跨臂部署 | [link](https://rdt-robotics.github.io/rdt2/) |
| **RH20T** | 20K+ 轨迹 | 多种 | 中国团队，真实世界操作 | — |

### 4.2 人类视频数据集

| 数据集 | 规模 | 特点 | 链接 |
|--------|------|------|------|
| **EgoScale** | 20,854 小时 | 自我中心人类视频, 20+任务类别, wrist+hand 动作标注 | NVIDIA GR00T N1.7 |
| **HumanNet** | 1,000,000 小时 | 人类中心视频语料，活动/环境/物体多样性 | [arXiv:2605.06747](https://arxiv.org/abs/2605.06747) |
| **Ego4D / Ego-Exo4D** | 数千小时 | Meta 自我中心视频 | [ego4d](https://ego4d-data.org/) |

### 4.3 训练框架

| 框架 | 提供方 | 特点 | 链接 |
|------|--------|------|------|
| **LeRobot** | Hugging Face | 端到端：数据采集→训练→部署，支持 π0/GR00T/SmolVLA | [GitHub](https://github.com/huggingface/lerobot) |
| **OpenPI** | Physical Intelligence | π0/π0.5 官方训练系统，JAX+PyTorch，10K+小时预训练 | [GitHub](https://github.com/Physical-Intelligence/openpi) |
| **Isaac GR00T** | NVIDIA | GR00T N1.7 训练管线，含 EgoScale 预训练 | [GitHub](https://github.com/NVIDIA/Isaac-GR00T) |

### 4.4 awesome 列表

| 资源 | 链接 |
|------|------|
| awesome-physical-ai | https://github.com/keon/awesome-physical-ai |
| awesome-embodied-vla | https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln |

---

## 5. 推荐方案

### 5.1 方案 A：基于 LAP 路线（最佳 Zero-shot 跨本体）

**核心思路**：参考 LAP 的语言化动作表示，从根本上消除本体耦合。

```
Backbone:     Qwen2.5-VL-7B 或 InternVL2-4B (预训练 VLM)
动作表示:     语言化动作 (将机器人动作编码为自然语言 token)
预训练数据:   OXE 全量 + DROID + 人类自我中心视频 (~10K-20K 小时)
训练目标:     自回归 next-token prediction (动作 = 语言 token)
框架:         基于 OpenPI 或自建，PyTorch
算力估计:     ~256-512× H100, 训练 ~1-2 周
```

**优势**：
- 已验证的 zero-shot 跨本体迁移能力（唯一一个在新本体上 >50% 成功的开源 VLA）
- 无需设计动作分词器或本体特定组件
- 参数高效扩展（4B→27B 持续改善）

**风险**：
- 语言化动作的精度天花板（复杂灵巧操作可能受限）
- 论文刚发布（2026.02），社区验证较少

### 5.2 方案 B：基于 GR00T 双系统架构（最佳灵巧操作）

**核心思路**：参考 NVIDIA GR00T N1.7 的 Action Cascade 架构 + EgoScale 人类视频预训练。

```
System 2:     Cosmos-Reason2-2B 或 Qwen3-VL-3B (VLM 推理)
System 1:     32 层 DiT (扩散 Transformer 动作生成)
动作空间:     相对 EEF delta (跨人类/机器人共享)
预训练:
  Phase 1:    20K+ 小时人类自我中心视频 (EgoScale)
  Phase 2:    OXE + DROID 机器人数据对齐
  Phase 3:    目标本体微调 (20-100 demonstrations)
损失函数:     Flow Matching + FLARE (未来潜在表示对齐)
框架:         Isaac GR00T (官方开源)
算力估计:     ~1K H100, 250K 步, batch size 16384
```

**优势**：
- 完整开源 (Apache 2.0)，工业级工程质量
- EgoScale Scaling Law 已验证（数据越多性能越好）
- 双系统设计灵巧度高，实时性好

**风险**：
- 当前 N1.7 仅 3B（可扩展至 4B+）
- 相对 EEF delta 对关节级控制的表达有局限

### 5.3 方案 C：融合方案（推荐）

**核心思路**：融合 LAP 的本体无关动作表示 + GR00T 的双系统架构 + EgoScale 的数据策略。

```
┌────────────────────────────────────────────────────┐
│                 推荐融合架构                         │
│                                                    │
│  ┌──────────────────────┐                          │
│  │ System 2 (VLM)       │  ~3B params              │
│  │ Qwen2.5-VL-3B /      │  输入: RGB + 语言指令     │
│  │ InternVL2-4B          │  输出: 高层语义动作 token  │
│  └──────────┬───────────┘                          │
│             │ 语义动作 embedding                    │
│  ┌──────────▼───────────┐                          │
│  │ System 1 (Action DiT) │  ~1B params              │
│  │ 扩散 Transformer      │  输入: 语义 token +       │
│  │ Flow Matching         │  本体状态 + 噪声动作      │
│  │                       │  输出: 连续运动命令        │
│  └──────────────────────┘                          │
│                                                    │
│  总参数: ~4B                                        │
│  动作空间: 通用动作码本 (UniAct) 或                   │
│           语言化动作 (LAP) + EEF delta 解码          │
└────────────────────────────────────────────────────┘

预训练数据配方:
  ├─ 阶段 0: VLM 初始化 (预训练权重)
  ├─ 阶段 1: 人类视频预训练 ~10K-20K 小时
  │   ├─ EgoScale / HumanNet 子集
  │   ├─ 训练目标: FLARE + 视频预测
  │   └─ 学习物理世界先验
  ├─ 阶段 2: 机器人数据对齐 ~2K-5K 小时
  │   ├─ OXE + DROID + 自采数据
  │   ├─ 训练目标: Flow Matching 动作生成
  │   └─ 建立人类→机器人迁移
  └─ 阶段 3: 目标任务/本体后训练
      ├─ 20-100 demonstrations
      ├─ 可选 RL 在线优化
      └─ LoRA adapter 本体适配
```

**优势**：
- 综合 2025-2026 最新技术路线
- 4B 总参数（3B VLM + 1B DiT）满足目标
- 数据策略已有 EgoScale Scaling Law 验证
- 可 zero-shot 部署 + 少样本微调提升

---

## 6. 参考文献与资源

### 6.1 核心论文

| 论文 | 年份 | 关键词 | 链接 |
|------|------|--------|------|
| LAP: Language-Action Pre-training | 2026.02 | 语言化动作, zero-shot 跨本体 | [arXiv:2602.10556](https://arxiv.org/abs/2602.10556) |
| DreamZero: World Action Models are Zero-shot Policies | 2026.02 | WAM, 视频扩散, 14B | [arXiv:2602.15922](https://arxiv.org/abs/2602.15922) |
| GR00T N1: Open Foundation Model for Humanoid Robots | 2025.03 | 双系统, NVIDIA | [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) |
| Gemini Robotics: Bringing AI into the Physical World | 2025.03 | Gemini VLA | [arXiv:2503.20020](https://arxiv.org/abs/2503.20020) |
| SpatialVLA: Spatial Representations for VLA | 2025.01 | 4B, 空间编码, RSS 2025 | [arXiv:2501.15830](https://arxiv.org/abs/2501.15830) |
| UniAct: Universal Actions for Embodied Foundation Models | 2025 | 通用动作空间, CVPR 2025 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Zheng_Universal_Actions_for_Enhanced_Embodied_Foundation_Models_CVPR_2025_paper.pdf) |
| π0: VLA Flow Model for General Robot Control | 2024.10 | Flow Matching, 10K+小时 | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
| OpenVLA: Open-Source VLA Model | 2024.06 | 7B, 开源, OXE | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |
| Open X-Embodiment: Robotic Learning Datasets and RT-X | 2023.10 | 1M+ 轨迹, 22 本体 | [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) |
| EgoScale: Scaling Dexterous Manipulation with Human Data | 2026 | 20K+小时, Scaling Law | [Paper Notes](https://davidlxu.github.io/posts/2026/02/egoscale-paper-notes/) |
| HumanNet: Scaling Human-centric Video to 1M Hours | 2025.05 | 100万小时人类视频 | [arXiv:2605.06747](https://arxiv.org/abs/2605.06747) |
| Latent Action Pretraining from Videos | 2025.04 | 无监督动作学习, NVIDIA | [NVIDIA Research](https://research.nvidia.com/publication/2025-04_latent-action-pretraining-videos) |
| Emergence of Human-to-Robot Transfer in VLAs | 2025.12 | 人→机器人迁移涌现 | [arXiv:2512.22414](https://arxiv.org/abs/2512.22414) |
| Scalable VLA Pretraining with Human Activity Videos | 2025.10 | 1M episodes, 26M frames | [arXiv:2510.21571](https://arxiv.org/abs/2510.21571) |
| World Model for Robot Learning: A Comprehensive Survey | 2026.05 | 世界模型综述 | [arXiv:2605.00080](https://arxiv.org/abs/2605.00080) |
| Comprehensive Survey on World Models for Embodied AI | 2025.10 | 世界模型分类框架 | [arXiv:2510.16732](https://arxiv.org/abs/2510.16732) |
| MiMo-Embodied: X-Embodied Foundation Model | 2025.11 | 小米, 跨本体 | [arXiv:2511.16518](https://arxiv.org/abs/2511.16518) |
| Pelican-VL 1.0: Foundation Brain Model | 2025.11 | 具身大脑 | [arXiv:2511.00108](https://arxiv.org/abs/2511.00108) |
| Being-H0.5: Human-centric VLA with Mixture-of-Flow | 2026.01 | 35K+小时, 30 本体 | — |
| X-VLA: Scalable Cross-Embodiment VLA | 2025 | Soft Prompt, 0.9B | [ResearchGate](https://www.researchgate.net/publication/396462440) |

### 6.2 开源项目

| 项目 | 描述 | 链接 |
|------|------|------|
| OpenVLA | 7B 开源 VLA + FAST/OFT | https://github.com/openvla/openvla |
| OpenPI | π0/π0.5 训练部署系统 | https://github.com/Physical-Intelligence/openpi |
| Isaac GR00T | NVIDIA GR00T N1.7 开源 | https://github.com/NVIDIA/Isaac-GR00T |
| LeRobot | HuggingFace 机器人学习框架 | https://github.com/huggingface/lerobot |
| SpatialVLA | 4B 空间增强 VLA | https://github.com/SpatialVLA/SpatialVLA |
| Open X-Embodiment | Google DeepMind 数据集 | https://github.com/google-deepmind/open_x_embodiment |
| awesome-physical-ai | Physical AI 论文列表 | https://github.com/keon/awesome-physical-ai |
| awesome-embodied-vla | VLA/VLN 论文列表 | https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln |

### 6.3 行业博客与报告

| 标题 | 来源 | 链接 |
|------|------|------|
| Top 10 Physical AI Models Powering Robots in 2026 | MarkTechPost | [link](https://www.marktechpost.com/2026/04/28/top-10-physical-ai-models-powering-real-world-robots-in-2026/) |
| GEN-0: Embodied Foundation Models That Scale | Generalist AI | [link](https://generalistai.com/blog/nov-04-2025-GEN-0) |
| GEN-1: Scaling Embodied Foundation Models to Mastery | Generalist AI | [link](https://generalistai.com/blog/apr-02-2026-GEN-1) |
| Foundation Models for Robotics: VLA Guide | Rohit Bandaru | [link](https://rohitbandaru.github.io/blog/Foundation-Models-for-Robotics-VLA/) |
| Open-Weight Foundation Models for Robotics (2025) | RoboCloud Hub | [link](https://robocloud-dashboard.vercel.app/learn/blog/open-weight-robot-models) |
| π0 and π0-FAST: VLA Models for Robot Control | Hugging Face | [link](https://huggingface.co/blog/pi0) |
| GR00T N1.7 Blog | Hugging Face/NVIDIA | [link](https://huggingface.co/blog/nvidia/gr00t-n1-7) |
| 面向具身操作的 VLA 模型综述 | 自动化学报 2026 | [link](https://aas.net.cn/cn/article/doi/10.16383/j.aas.c250394) |
| 基于大模型的具身智能系统综述 | 自动化学报 2025 | [link](https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c240542) |

---

## 7. 总结

2025-2026 年具身智能预训练进入了 **VLA + 大规模人类视频预训练 + 跨本体泛化** 的黄金时期。实现你的目标（4B+参数、1万小时+数据、跨本体 zero-shot）的技术路线已经成熟：

1. **数据不是瓶颈**：EgoScale 证明了 20K 小时人类视频即可显著提升机器人性能，HumanNet 提供了百万小时级语料。"1万小时机器人视频"的目标可通过 ~2K 小时机器人数据 + ~10K 小时人类视频达成。

2. **Zero-shot 跨本体已可行**：LAP 首次证明 VLA 可在未见本体上实现 >50% 成功率；RDT-2 实现了即插即用部署；UniAct 的通用动作空间从架构层面解决了本体耦合。

3. **4B 参数处于甜蜜点**：SpatialVLA (4B) 以 3.5B 超越 55B RT-2-X；GR00T N1.7 (3B) 达到生产级性能；LAP 展示了 4B→27B 的平滑 scaling。4B 是性能和部署成本的良好平衡点。

4. **推荐的技术栈**：融合双系统架构 (VLM + DiT) + 通用动作表示 (LAP 语言化动作 或 UniAct 码本) + EgoScale 人类视频预训练 + OXE/DROID 机器人数据对齐，使用 LeRobot 或 OpenPI 框架构建训练管线。
