# DM0 L_AR 可比方法：2025-2026 离散 Action Token + 自回归 Loss 开源论文综述

> 最后更新: 2026-05-15

## 0. 概述

DM0 论文提出的核心训练范式：
- **离散 Action Token**：将连续机器人动作量化为离散 token（如 256-bin）
- **L_AR (自回归交叉熵 Loss)**：`L_AR(θ) = -E_D [log π_θ(l̂ | o_t, l)]`，标准 next-token prediction CE loss
- **L_total = λ·L_AR + L_FM**：自回归文本 loss 与 flow matching 动作 loss 的组合

本文收集 2024-2026 年使用**类似方法**且**开源代码**的论文，按方法相似度分类整理。

---

## 1. 经典离散 Token + 纯自回归 L_AR

> 最接近 DM0 论文中 L_AR 部分的实现。将连续动作离散化后，直接用标准 next-token prediction CE loss 训练。

### 1.1 OpenVLA

- **论文**: OpenVLA: An Open-Source Vision-Language-Action Model
- **会议**: ICML 2024 → 持续更新至 2025
- **arXiv**: [2406.09246](https://arxiv.org/abs/2406.09246)
- **代码**: [github.com/openvla/openvla](https://github.com/openvla/openvla)
- **方法**:
  - 7B 参数 VLA (Llama 2 + DINOv2 + SigLIP)
  - 每个动作维度独立离散化为 256 bins
  - 复用 Llama tokenizer 中最少使用的 256 个 token
  - 标准 next-token CE loss，仅在 action token 位置计算
- **与 DM0 异同**:
  - 相同：256-bin 离散化 + CE loss
  - 不同：OpenVLA 无 flow matching 分支，纯自回归；无 merged attention 架构
- **规模**: 970K 真实机器人轨迹，64×A100 训练 14 天

### 1.2 π₀-FAST (Physical Intelligence)

- **论文**: FAST: Efficient Action Tokenization for Vision-Language-Action Models
- **时间**: 2025.01
- **arXiv**: [2501.09747](https://arxiv.org/abs/2501.09747)
- **代码**:
  - 官方训练框架: [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
  - FAST tokenizer: [huggingface.co/physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast)
  - LeRobot 集成: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- **方法**:
  - 基于 π₀ 的 3B VLM backbone
  - **FAST tokenizer**: DCT (离散余弦变换) + BPE 压缩 action chunk → 30-60 个 dense token（对比传统方法压缩 7-10x）
  - 标准自回归 CE loss
  - FAST+ 是在 1M 真实轨迹上预训练的通用 action tokenizer
- **与 DM0 异同**:
  - 相同：自回归 CE loss 预测离散 action token
  - 不同：FAST 用 DCT 频域压缩而非简单 bin 量化；π₀-FAST 无 flow matching 分支（原版 π₀ 有 FM）
  - 关键启示：**FAST tokenizer 可直接集成到 DM0 的 L_AR 分支**，替代 256-bin 量化
- **性能**: 匹配 diffusion 版 π₀ 精度，训练速度提升 5x

### 1.3 VQ-VLA (ICCV 2025)

- **论文**: VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers
- **会议**: ICCV 2025
- **arXiv**: [2507.01016](https://arxiv.org/abs/2507.01016)
- **代码**: [github.com/xiaoxiao0406/VQ-VLA](https://github.com/xiaoxiao0406/VQ-VLA)
- **方法**:
  - 卷积 Residual VQ-VAE 作为 action tokenizer
  - 在 Open X-Embodiment + LIBERO + ManiSkill 上预训练 VQ-VAE
  - 冻结 VQ-VAE，用其 token 替换 OpenVLA 的 256-bin 方法
  - LoRA 微调 OpenVLA
- **与 DM0 异同**:
  - 相同：离散 token + CE loss
  - 不同：用学习到的 VQ-VAE 替代简单 bin 量化；强调 action tokenizer 的可扩展性
  - 关键发现：模拟数据与真实数据在 action 轨迹层面 domain gap 极小，可用合成数据扩展 tokenizer
- **性能**: 长 horizon 任务成功率从 23% 提升到 46.25%

### 1.4 UniVLA (ICLR 2026)

- **论文**: Unified Vision-Language-Action Model
- **会议**: ICLR 2026
- **arXiv**: [2506.19850](https://arxiv.org/abs/2506.19850)
- **代码**: [github.com/baaivision/UniVLA](https://github.com/baaivision/UniVLA)
- **方法**:
  - **统一离散化**：视觉(VQ encoder) + 语言 + 动作(DCT/FAST) 全部离散化为 token
  - 单一 8.5B 自回归 Transformer 统一建模
  - 两阶段训练：(1) world model 预训练（不需要 action 标注），(2) action 微调
  - 支持文本监督感知定位、视觉监督世界模型、动作监督策略学习
- **与 DM0 异同**:
  - 相同：离散 action token + 自回归 CE loss
  - 不同：UniVLA 把视觉也离散化了（DM0 用连续 ViT 特征）；无 merged attention / flow matching
  - 关键启示：**UniVLA 验证了 "text-as-everything" 的可行性**——所有模态走统一自回归
- **性能**: LIBERO 95.5%（超过 π₀-FAST 的 85.5%），CALVIN、SimplerEnv-Bridge SOTA

### 1.5 OmniSAT (ICLR 2026)

- **论文**: OmniSAT: Compact Action Token, Faster Auto Regression
- **会议**: ICLR 2026
- **arXiv**: [2510.09667](https://arxiv.org/abs/2510.09667)
- **代码**: 待发布（论文中提及将开源）
- **方法**:
  - B-Spline 编码统一不同 action chunk 长度
  - 多阶段 Residual VQ 量化（position / rotation / gripper 子空间分别量化）
  - 在 DROID 大规模数据集上预训练 tokenizer
- **与 DM0 异同**:
  - 相同：离散 token + 自回归 CE
  - 不同：B-Spline + RVQ 比简单 bin 量化压缩率高很多（6.8-8.1x vs FAST 3.7x）
- **性能**: 压缩率最优，同等训练步数下成功率高于 FAST 和 BEAST

---

## 2. 离散 Token + 离散 Diffusion / Flow Matching

> 训练时仍用 CE loss（与 L_AR 相同），但推理时不做左到右自回归，而是迭代去噪/精修。

### 2.1 Discrete Diffusion VLA (DD-VLA)

- **论文**: Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies
- **时间**: 2025.08
- **arXiv**: [2508.20072](https://arxiv.org/abs/2508.20072)
- **代码**: 已发布（arXiv 中标注 "code available"）
- **方法**:
  - 单一 Transformer VLM backbone（SigLIP + DINOv2 ViTs）
  - Action bin 离散化（类似 OpenVLA 的 256-bin）
  - 训练：标准 CE loss（与 VLM backbone 相同的目标函数）
  - 推理：所有 action token 初始为 [MASK]，迭代预测 → re-mask 低置信度 → 再预测
  - Adaptive Decoding: 先简单后困难的并行解码
  - Secondary Re-Masking: 跨步一致性检查
- **与 DM0 异同**:
  - 相同：离散 bin + CE loss 训练
  - 不同：推理时迭代解码（非左到右自回归）；无 flow matching 分支
  - 关键启示：**训练阶段与 DM0 的 L_AR 完全一致**，仅推理策略不同
- **性能**: LIBERO 96.3%, SimplerEnv-Fractal 71.2%

### 2.2 DIVA (ICLR 2026)

- **论文**: DIVA: Discrete Diffusion Vision-Language-Action Models for Parallelized Action Generation
- **会议**: ICLR 2026
- **OpenReview**: [openreview.net/forum?id=mNya9d1DA2](https://openreview.net/forum?id=mNya9d1DA2)
- **代码**: 待发布
- **方法**:
  - 学习型离散 action tokenization（连续 → 离散 token）
  - Latent-Driven Policy Learning: 联合优化离散 VL 表示和连续 action
  - Selective Group Unmasking (SGU): 组级别解码（而非 token 级）
- **与 DM0 异同**:
  - 相同：离散 token + CE 类 loss
  - 不同：解码策略为组级别离散扩散；有学习型 tokenizer
- **性能**: LIBERO 平均 97.4%（四个子集 98.0/98.8/97.6/95.2）

### 2.3 DFM-VLA

- **论文**: DFM-VLA: Iterative Action Refinement for Robot Manipulation via Discrete Flow Matching
- **时间**: 2026.03
- **arXiv**: [2603.26320](https://arxiv.org/abs/2603.26320)
- **代码**: 论文中提及 "project available"，具体链接待确认
- **方法**:
  - 离散 flow matching：建模 token 级概率速度场
  - 跨迭代动态更新完整 action 序列
  - 两种速度场构建方式：辅助 velocity-head / action-embedding-guided
  - 两阶段解码：迭代精修 + 确定性验证
- **与 DM0 异同**:
  - 相同：离散 token 表示
  - 不同：flow matching 在离散空间做迭代精修（DM0 的 FM 在连续空间）
  - 关键启示：**将 flow matching 思想迁移到离散 token 空间**，解决了自回归的 error accumulation
- **性能**: CALVIN 4.44, LIBERO 95.7%

---

## 3. 双 Loss 架构（L_AR + L_FM / L_Diffusion）

> **最接近 DM0 论文 L_total = λ·L_AR + L_FM 的架构设计。**

### 3.1 HybridVLA

- **论文**: HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model
- **时间**: 2025.03
- **arXiv**: [2503.10631](https://arxiv.org/abs/2503.10631)
- **代码**: [github.com/PKU-HMI-Lab/Hybrid-VLA](https://github.com/PKU-HMI-Lab/Hybrid-VLA)
- **项目页**: [hybrid-vla.github.io](https://hybrid-vla.github.io/)
- **方法**:
  - **单一 LLM 内同时做自回归 + 扩散**
  - 将 diffusion-noised action 编码为连续向量，投影到 LLM embedding 空间
  - Token 序列组织：多模态输入 → diffusion action → autoregressive action，用 marker token 连接
  - 训练配置：`AR_DIFF_LOSS=true`，双 loss 联合优化
  - 推理时 collaborative action ensemble：自适应融合两种预测
- **与 DM0 异同**:
  - **极为相似**：DM0 = merged attention + L_AR + L_FM；HybridVLA = 单 LLM + AR loss + diffusion loss
  - 不同：DM0 用 merged attention 物理分离 VLM 和 Action Expert；HybridVLA 在同一 LLM 内做
  - 不同：HybridVLA 的 diffusion 是 discrete diffusion（DM0 是 continuous flow matching）
  - 关键启示：**验证了双 loss 协作训练的有效性**——两种 loss 互相增强而非干扰
- **性能**: 模拟和真实任务分别超过 SOTA 14% 和 19%
- **训练**: 8×A100, `REPEATED_DIFFUSION_STEPS=4`

---

## 4. Action Tokenizer 方法论

> 不是完整 VLA 模型，而是**动作离散化方法**。可直接用于改进 DM0 的 L_AR 分支中 action token 的质量。

### 4.1 FAST (Frequency-space Action Sequence Tokenization)

- **论文**: FAST: Efficient Action Tokenization for Vision-Language-Action Models
- **时间**: 2025.01 (Physical Intelligence)
- **arXiv**: [2501.09747](https://arxiv.org/abs/2501.09747)
- **代码**: [huggingface.co/physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast)
- **方法**: DCT 变换 → BPE 编码 → 30-60 个 dense token（压缩 ~7x）
- **DM0 适用性**: 可替代 DM0 的 256-bin 量化方案，用 `AutoProcessor.from_pretrained("physical-intelligence/fast")` 三行代码即可接入

### 4.2 FASTer

- **论文**: FASTer: DCT Early Stopping for FAST Inference
- **时间**: 2025
- **代码**: [github.com/uynitsuj/FASTer](https://github.com/uynitsuj/FASTer)
- **方法**: FAST 的推理优化——不等所有 DCT 系数生成完毕就开始执行，推理延迟减半
- **DM0 适用性**: 推理阶段优化，配合 FAST tokenizer 使用

### 4.3 BEAST (B-Spline Encoded Action Sequences)

- **论文**: BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning
- **时间**: 2025
- **arXiv**: [2506.06072](https://arxiv.org/abs/2506.06072)
- **代码**: 待确认
- **方法**: 将 action chunk 压缩为 B-Spline 控制点 → 固定长度表示 → 支持并行解码
- **DM0 适用性**: 固定长度特性适合 DM0 的 chunk_size 设计

---

## 5. Text-as-Action / 文本形式表达动作

> **最接近 DM0 "text-as-everything" 设计哲学的论文。**

### 5.1 Actions as Language (VLM2VLA) — ICLR 2026

- **论文**: Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting
- **会议**: ICLR 2026
- **arXiv**: [2509.22195](https://arxiv.org/abs/2509.22195)
- **项目页**: [vlm2vla.github.io](https://vlm2vla.github.io/)
- **代码**: 项目页标注将开源，具体 GitHub 待发布
- **方法**:
  - **不用离散 action token**，而是将底层动作重新标注为自然语言文本
  - 例如：`"move gripper left 2cm, then close"` 而非 `[128, 64, 200, ...]`
  - 数据重标注管线：连续 action → 子任务描述 + 动作文本 + 运动规划短语
  - 仅用 LoRA 微调 VLM → VLA，避免灾难性遗忘
- **与 DM0 异同**:
  - **最接近 DM0 的 "text-as-everything" 理念**：所有信息（包括动作）都用文本表达
  - 不同：DM0 仍然用数值 token（bin 量化），VLM2VLA 用自然语言
  - 关键发现：文本形式表达动作**保留了 VLM 85%+ 的 VQA 能力**（离散 token 会导致灾难性遗忘）
  - 启示：DM0 的 Embodied Spatial Scaffolding（子任务预测、目标预测等）如果用自然语言而非数值 token，可能更好地保留 VLM 能力

---

## 6. 相关 VLA 模型（不完全匹配但有参考价值）

### 6.1 OpenVLA-OFT

- **论文**: Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
- **时间**: 2025.02
- **arXiv**: [2502.19645](https://arxiv.org/abs/2502.19645)
- **代码**: [github.com/moojink/openvla-oft](https://github.com/moojink/openvla-oft)
- **关键点**: 系统比较了自回归 vs 并行解码、离散 vs 连续 action。结论：并行解码 + 连续 action + L1 loss 效果最好。26x 推理加速。
- **与 DM0 关系**: 提供了离散 vs 连续的实证对比数据

### 6.2 CogACT (Microsoft)

- **论文**: CogACT: A Foundational VLA for Synergizing Cognition and Action
- **时间**: 2025
- **代码**: [github.com/microsoft/CogACT](https://github.com/microsoft/CogACT)
- **关键点**: VLM cognition features → Diffusion Action Transformer 生成连续动作。解耦认知与动作。
- **与 DM0 关系**: 类似的解耦思路，但 CogACT 用 diffusion head 而非 flow matching

### 6.3 InternVLA-A1

- **论文**: InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation
- **时间**: 2026.01
- **代码**: [github.com/InternRobotics/InternVLA-A1](https://github.com/InternRobotics/InternVLA-A1)
- **关键点**: 统一视觉理解、视觉生成和动作生成的框架
- **许可**: CC BY-NC-SA 4.0

### 6.4 StarVLA

- **论文**: StarVLA: A Lego-like Codebase for VLA Model Developing
- **时间**: 2025-2026
- **代码**: [github.com/starVLA/starVLA](https://github.com/starVLA/starVLA)
- **关键点**: 模块化 VLA 框架，支持插拔式 backbone（Qwen3.5）+ 多种 action head（FAST, OFT, FM 等）
- **与 DM0 关系**: 如果 DM0 要对比多种 action head 方案，StarVLA 提供了现成的实验框架

---

## 7. 总结对照表

| 论文 | 年份 | 离散化方式 | Loss 类型 | 解码方式 | 双 Loss | 开源代码 | 与 DM0 相似度 |
|------|------|-----------|----------|---------|--------|---------|-------------|
| **OpenVLA** | 2024 | 256-bin | CE | 自回归 | 否 | [GitHub](https://github.com/openvla/openvla) | ★★★☆☆ |
| **π₀-FAST** | 2025 | DCT+BPE | CE | 自回归 | 否 | [GitHub](https://github.com/Physical-Intelligence/openpi) | ★★★☆☆ |
| **VQ-VLA** | 2025 | RVQ-VAE | CE | 自回归 | 否 | [GitHub](https://github.com/xiaoxiao0406/VQ-VLA) | ★★★☆☆ |
| **UniVLA** | 2025 | VQ+DCT | CE | 自回归 | 否 | [GitHub](https://github.com/baaivision/UniVLA) | ★★★★☆ |
| **OmniSAT** | 2025 | B-Spline+RVQ | CE | 自回归 | 否 | 待发布 | ★★★☆☆ |
| **DD-VLA** | 2025 | 256-bin | CE | 离散扩散 | 否 | 已发布 | ★★★☆☆ |
| **DIVA** | 2025 | 学习型 | CE | 离散扩散 | 否 | 待发布 | ★★★☆☆ |
| **DFM-VLA** | 2026 | 离散 bin | FM(离散) | 迭代精修 | 否 | 待确认 | ★★★☆☆ |
| **HybridVLA** | 2025 | 离散 bin | CE+Diff | 自回归+扩散 | **是** | [GitHub](https://github.com/PKU-HMI-Lab/Hybrid-VLA) | ★★★★★ |
| **VLM2VLA** | 2025 | 自然语言 | CE | 自回归 | 否 | 待发布 | ★★★★☆ |
| **OpenVLA-OFT** | 2025 | 连续 | L1 | 并行 | 否 | [GitHub](https://github.com/moojink/openvla-oft) | ★★☆☆☆ |
| **CogACT** | 2025 | 连续 | CE+Diff | VLM+Diff head | **是** | [GitHub](https://github.com/microsoft/CogACT) | ★★★☆☆ |
| **InternVLA-A1** | 2026 | — | — | — | — | [GitHub](https://github.com/InternRobotics/InternVLA-A1) | ★★☆☆☆ |
| **StarVLA** | 2025 | 多种 | 多种 | 多种 | 可配 | [GitHub](https://github.com/starVLA/starVLA) | ★★☆☆☆ |

### 相似度评分标准
- ★★★★★: 离散 token + 双 Loss (L_AR + L_FM/Diff) + 代码开源
- ★★★★☆: 离散 token + CE + 与 DM0 设计哲学高度一致
- ★★★☆☆: 离散 token + CE，标准实现
- ★★☆☆☆: 相关但方法不同

---

## 8. DM0 实现建议

基于以上论文调研，对 DM0 实现 L_AR 的建议：

1. **Action Tokenizer 选择**: 优先考虑 **FAST tokenizer**（π₀-FAST），压缩率和精度都优于简单 256-bin
2. **双 Loss 参考**: **HybridVLA** 是最接近的参考实现——在单一 LLM 内同时做 AR + Diffusion，与 DM0 的 merged attention + L_AR + L_FM 高度同构
3. **Text-as-Everything**: **VLM2VLA (Actions as Language)** 验证了"用文本表达一切"可以保留 VLM 能力，支持 DM0 论文中 Embodied Spatial Scaffolding 的设计方向
4. **统一离散化**: **UniVLA** 证明了全模态离散自回归的可行性，LIBERO 95.5% 远超 π₀-FAST 85.5%

---

## 附录 A: 资源索引

### 综述与论文列表
- [Awesome Physical AI](https://github.com/keon/awesome-physical-ai) — 最全的 VLA 论文/代码列表
- [Awesome VLA Papers (Action Tokenization)](https://github.com/Psi-Robot/Awesome-VLA-Papers) — 按 action tokenization 视角分类的综述
- [ICLR 2026 VLA Research Overview](https://mbreuss.github.io/blog_post_iclr_26_vla.html) — ICLR 2026 的 164 篇 VLA 投稿分析
- [Large VLM-based VLA List](https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation)

### ICLR 2026 统计
- VLA 投稿从 ICLR 2025 的 9 篇增长到 ICLR 2026 的 **164 篇**
- 离散 diffusion VLA 是 ICLR 2026 最热门的子方向之一
- Action tokenizer 方向有 3+ 篇被接收（BEAST, OmniSAT, UniVLA 等）
