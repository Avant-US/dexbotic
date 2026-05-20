
**π₀.₅ 对比基线**（因榜单未单独列出 π₀.₅ 全量分数，按下表统一口径）：

| 指标来源 | LIBERO | LIBERO-Plus | SimplerEnv | RoboTwin 2.0 | CALVIN | RoboCasa |
|---------|--------|-------------|------------|--------------|--------|----------|
| **π₀.₅** | **97.7%**（Allen AI `vla-eval` 复现） | 未单独公开 | 未单独公开 | 未单独公开 | 未单独公开 | 未单独公开 |
| **π₀（参考）** | 94.2% | 56.1% | 71.0% | 31.35% | 3.92 | 62.5% |

> 说明：π₀.₅ 真机长程能力见 [Physical Intelligence π₀.₅ 报告](https://www.physicalintelligence.company/download/pi05.pdf)（arXiv:2504.16054）；仿真榜对比以 **97.7% LIBERO** 为主锚点，其余榜用 **π₀** 作参考并标注「π₀.₅ 未测」。

**全局 Top 3 边界说明**：SimplerEnv 前三为 **Octo+PLD 96.63 / FASTerVLA 87.9 / X-VLA 86.6**（均为 2025 及以前），故 **2026 年论文均未进 SimplerEnv 全局 Top 3**；下表仍列出 2026 年在该榜的分数及相对 π₀/π₀.₅ 的对比。

---

## 2026 年 VLA / 具身论文：SOTA 或 Top 3 一览（新→旧）

| 时间 | 论文名 | 下载地址 | Benchmark 成绩（SOTA=第1，T2/T3=第2/3） | 是否优于 π₀.₅？好多少？ | 简介（内容 / 创新 / 消融有效点） | 提及 |
|------|--------|----------|----------------------------------------|-------------------------|----------------------------------|------|
| **2026-05-07** | **OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation** | [PDF](https://arxiv.org/pdf/2605.06481) · [abs](https://arxiv.org/abs/2605.06481) | **LIBERO-Plus T2 83.9%**（全局#1 为 RLDX-1 86.7）；LIBERO 97.8%；SimplerEnv 79.3% | LIBERO **+0.1pp**；LIBERO-Plus **显著优于 π₀（+27.8pp）**；SimplerEnv 较 π₀ **+8.3pp**（π₀.₅ 未测） | 将场景分解为 **N+1 对象槽位**（地址向量+内容向量），世界头预测下一帧槽位，流匹配动作头一次前向出 16 步动作；消融：**地址键 cross-slot 注意力** 使 swap-binding 达 0.87， holistic 基线 ≤0.09 | √ |
| **2026-05-07** | **ConsisVLA-4D: Advancing Spatiotemporal Consistency in Efficient 3D-Perception and 4D-Reasoning** | [PDF](https://arxiv.org/pdf/2605.05126) · [abs](https://arxiv.org/abs/2605.05126) | **ManiSkill2 SOTA 94.3%** | π₀.₅ 未测；较 π₀ 无同榜分数 | 强调 **3D 感知 + 4D 时空一致性** 的高效 VLA；消融表明时空一致表征对仿真操作成功率提升最大 | |
| **2026-05-05** | **RLDX-1: A Dexterity-First Foundation Model for Robot Hands**（技术报告） | [PDF](https://arxiv.org/pdf/2605.03269) · [abs](https://arxiv.org/abs/2605.03269) · [HF](https://huggingface.co/collections/RLWRLD/rldx-1) | **综合榜 #1**；**LIBERO-Plus SOTA 86.7%**；LIBERO 97.8%；RoboCasa **70.6%**；RoboCasa365 **32.1%**；SimplerEnv 76.9%；GR-1 Tabletop **58.7%**；RoboTwin 2.0 **T3 87.8%** | LIBERO **+0.1pp**；LIBERO-Plus **远超 π₀（+30.6pp）**；RoboCasa 较 π₀ **+8.1pp**；SimplerEnv 较 π₀ **+5.9pp**；真机报告称较 **π₀.₅ / GR00T N1.6 ~40%** 提升至 **~86.8%**（ALLEX 人形任务） | **MSAT（Multi-Stream Action Transformer）** 融合视觉/语言/动作/触觉/力矩/记忆多流；消融：**运动感知、长期记忆、物理传感流** 为对抗 VLA 失败模式的关键；灵巧操作优先而非仅桌面 pick-place   |  |
| **2026-05-04** | **MolmoAct2: Action Reasoning Models for Real-world Deployment** | [PDF](https://arxiv.org/pdf/2605.02881) · [abs](https://arxiv.org/abs/2605.02881) | **MolmoSpace SOTA 37.7**；**MolmoBot SOTA 20.6**；**RoboEval SOTA 44.3** | 新榜无 π₀.₅ 基线 | Allen AI **动作推理型 VLA**（ARM），强调可部署推理链；消融：**显式动作推理头** 对长程与 sim2real 相关任务提升最大   |  |
| **2026-04-30** | **PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations** | [PDF](https://arxiv.org/pdf/2604.27472) · [abs](https://arxiv.org/abs/2604.27472) | LIBERO **98.4%**（全局约 #4）；**LIBERO-Plus 81.4%**（#4）；LIBERO-Pro **58.8%**；SimplerEnv **77.1%**；论文称多榜 SOTA | LIBERO **+0.7pp**；LIBERO-Plus 较 π₀ **+25.3pp**；SimplerEnv 较 π₀ **+6.1pp** | 用 **目标条件对比 RL** 学统一嵌入，内积近似到达概率；167B token 预训练；消融：**角色感知因果 mask + 对比目标** 优于纯 BC，长程/接触丰富任务增益最大   |  |
| **2026-04-22** | **PokéVLA: Empowering Pocket-Sized VLA with Comprehensive World Knowledge** | [PDF](https://arxiv.org/pdf/2604.20834) · [abs](https://arxiv.org/abs/2604.20834) | **LIBERO-Plus T2 83.5%**；LIBERO 98.2% | LIBERO **+0.5pp**；LIBERO-Plus 较 π₀ **+27.4pp** | 小参数 VLA + **世界知识引导**；消融：知识蒸馏/检索模块对 **LIBERO-Plus 语言与几何扰动** 最有效   |  |
| **2026-04-23** | **LoHo-Manip: Long-Horizon Manipulation via Trace-Conditioned VLA Planning** | [PDF](https://arxiv.org/pdf/2604.21924) · [abs](https://arxiv.org/abs/2604.21924) | **VLABench T1 0.39**（该榜绝对分普遍极低） | 难与 π₀.₅ 直接比 | **轨迹条件规划 + VLA** 分层长程；消融：**trace conditioning** 对长程子任务串联最关键   |  |
| **2026-04-15** | **HAMLET: Switch your VLA into a History-Aware Policy** | [PDF](https://arxiv.org/abs/2510.00695) · [abs](https://arxiv.org/abs/2510.00695) | **RoboCasa T3 65.4%**（#1 Cosmos 67.1） | 较 π₀ RoboCasa **+2.9pp** | 将现有 VLA **改写为历史感知策略**（帧缓存/记忆融合）；消融：**历史窗口 + 轻量记忆融合** 优于单帧   |  |
| **2026-04-13** | **StarVLA-α: Reducing Complexity in Vision-Language-Action Systems** | [PDF](https://arxiv.org/pdf/2604.11757) · [abs](https://arxiv.org/abs/2604.11757) · [Code](https://github.com/starvla/starvla) | **LIBERO SOTA 98.8%**（Specialist）；**RoboTwin 2.0 T2 88.3%**；RoboCasa-GR1 T2 53.8%；RoboChallenge **33.6%**（Generalist） | LIBERO **+1.1pp**；RoboTwin 2.0 较 π₀ **+56.95pp**；论文称 **RoboChallenge 真机较 π₀.₅ +20%**（相对提升） | **极简 VLA**：Qwen3-VL + MLP 动作头，统一数据管线；消融：**去掉复杂专用模块仍达 SOTA**，说明强 VLM backbone + 规范训练配方已足够   |  |
| **2026-04-09** | **HiF-VLA: Hindsight, Insight and Foresight through Motion Representation** | [PDF](https://arxiv.org/abs/2512.09928) · [abs](https://arxiv.org/abs/2512.09928) | CALVIN 4.35；LIBERO 96.4%（未进全局 Top3） | LIBERO 较 π₀.₅ **-1.3pp** | 用 **运动表征** 统一后见/当下/前瞻；消融：**三向运动 token** 对 CALVIN 长链最有效   |  |
| **2026-03-30** | **FocusVLA: Focused Visual Utilization for VLAs** | [PDF](https://arxiv.org/pdf/2603.28740) · [abs](https://arxiv.org/abs/2603.28740) | LIBERO **98.6%**（全局 #4，距 Top3 +0.2pp） | LIBERO **-0.1pp**（基本持平） | **视觉 token 聚焦/剪枝** 提升有效注意力；消融：**任务相关 patch 选择** 降算力且保精度   |  |
| **2026-03-11** | **World2Act: Latent Action Post-Training via Skill-Compositional World Models** | [PDF](https://arxiv.org/pdf/2603.10422) · [abs](https://arxiv.org/abs/2603.10422) | **RoboCasa T2 66.3%**（配合 Cosmos Policy） | 较 π₀ **+3.8pp** | 在 **Cosmos 世界模型** 上做技能组合潜动作后训练；消融：**技能组合世界模型** 优于单任务后训练   |  |
| **2026-03-11** | **FutureVLA: Joint Visuomotor Prediction for VLA** | [PDF](https://arxiv.org/pdf/2603.10712) · [abs](https://arxiv.org/abs/2603.10712) | LIBERO 98.3%；LIBERO-Plus 81.3%；SimplerEnv 75.87%（均未进全局 Top3） | LIBERO **+0.6pp**；LIBERO-Plus 较 π₀ **+25.2pp** | **联合视觉-运动预测**，门控分离静态场景与动态动作；消融：**Joint Visuomotor Gating** + 潜对齐后训练可 plug-in 到多种 VLA   |  |
| **2026-03-10** | **GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware VLAs** | [PDF](https://arxiv.org/pdf/2603.09079) · [abs](https://arxiv.org/abs/2603.09079) | SimplerEnv **80.2%**（2026 年该榜最高之一，但仍低于全局 Top3 门槛 86.6%） | 较 π₀ **+9.2pp** | **3D 高斯空间 token** 编码深度几何；消融：深度感知 token 对 **相机扰动** 鲁棒性提升最大   |  |
| **2026-03-10** | **NS-VLA: Towards Neuro-Symbolic VLAs** | [PDF](https://arxiv.org/pdf/2603.09542) · [abs](https://arxiv.org/abs/2603.09542) | **CALVIN T3 4.72**（#1 Xiaomi 4.75）；LIBERO-Plus 79.4% | CALVIN 优于 π₀（+0.8pp）；LIBERO 69.2% 低于 π₀.₅ | **神经符号推理** 模块处理长链逻辑；消融：**符号规划器 + VLA 执行** 对 CALVIN 多步任务有效   |  |
| **2026-03** | **Fast-WAM**（Physical Intelligence，世界动作模型加速线） | [PDF](https://arxiv.org/pdf/2603.16666) · [abs](https://arxiv.org/abs/2603.16666) | **RoboTwin 2.0 SOTA 91.83%**；LIBERO 97.6% | LIBERO **-0.1pp**；RoboTwin 2.0 较 π₀ **+60.48pp** | π 系 **WAM** 加速与精度折中；消融：**单遍动作头 + 流匹配** 为 RoboTwin 高分关键   |  |
| **2026-03** | **Cosmos Policy + World2Act**（见上行） | 同上 | 见 World2Act / Cosmos | 见上 | 见上 | √ |
| **2026-02** | **Xiaomi-Robotics-0: Open-Sourced VLA with Real-Time Execution** | [PDF](https://arxiv.org/pdf/2602.12684) · [abs](https://arxiv.org/abs/2602.12684) | **CALVIN SOTA 4.75**；**LIBERO T2 98.7%** | LIBERO **+1.0pp**；CALVIN 优于 π₀ **+0.83** | 工业级 **实时 VLA**；消融：**动作分块 + 蒸馏** 保证低延迟下 LIBERO/CALVIN 双高 | |
| **2026-02** | **ABot-M0: VLA Foundation Model with Action Manifold Learning** | [PDF](https://arxiv.org/pdf/2602.11236) · [abs](https://arxiv.org/abs/2602.11236) | **LIBERO T3 98.6%**；**LIBERO-Plus T3 81.6%** | LIBERO **+0.9pp**；LIBERO-Plus 较 π₀ **+25.5pp** | 在 **动作流形** 上建模；消融：**流形约束损失** 减少 OOD 动作，Plus 榜鲁棒性提升明显   |  |
| **2026-02** | **MINT / Mimic Intent, Not Just Trajectories**（MINT-4B） | [PDF](https://arxiv.org/pdf/2602.08602) · [abs](https://arxiv.org/abs/2602.08602) | LIBERO **98.2%**；LIBERO-Plus **80.1%** | LIBERO **+0.5pp**；Plus 较 π₀ **+24.0pp** | 监督 **意图标签** 而非纯轨迹；消融：**意图对齐** 对语言扰动与长尾任务更有效   |  |
| **2026-02** | **VLA-JEPA: Enhancing VLA with Latent World Model** | [PDF](https://arxiv.org/pdf/2602.10098) · [abs](https://arxiv.org/abs/2602.10098) | LIBERO-Plus **79.5%**（2026 年高位，非 Top3） | Plus 较 π₀ **+23.4pp** | **JEPA 式潜世界模型** 辅助动作；消融：**潜预测辅助损失** 提升 Plus   |  |
| **2026-01** | **Cosmos Policy**（NVIDIA） | [PDF](https://arxiv.org/pdf/2601.16163) · [abs](https://arxiv.org/abs/2601.16163) | **RoboCasa SOTA 67.1%**；LIBERO **98.5%**（全局 #4） | LIBERO **+0.8pp**；RoboCasa 较 π₀ **+4.6pp** | 基于 **Cosmos 世界基础模型** 的策略头；消融：**世界模型预训练初始化** 是 RoboCasa 关键 | √ |
| **2026-01** | **Pose-VLA: Universal Pose Pretraining for Generalizable VLAs** | [PDF](https://arxiv.org/pdf/2602.19710) · [abs](https://arxiv.org/abs/2602.19710) | RoboTwin 2.0 **79.5%**（未进 Top3） | 较 π₀ **+48.15pp** | **2D/3D 姿态预训练** 再迁移动作；消融：**姿态预训练阶段** 对双臂泛化最关键 | |

---

## 按 Benchmark 汇总的 2026 年「冠军/领奖台」（便于对照）

| Benchmark | SOTA (#1) | Top2 | Top3 | 备注 | 提及 |
|-----------|-----------|------|------|------|------|
| **LIBERO** | StarVLA-α **98.8** | Xiaomi **98.7** | ABot-M0 **98.6** | 全为 2026   |  |
| **LIBERO-Plus** | **RLDX-1 86.7** | OA-WAM 83.9 | PokéVLA 83.5 | 全为 2026；π₀ 仅 56.1 | √ |
| **LIBERO-Pro** | GLaD **59.47**（2025） | π₀.₅ 评测条目 **62?*** | PRTS 58.8 | *榜内「π₀.₅(LIBERO-Pro)」62 为评测论文条目，非模型本体 | √ |
| **SimplerEnv** | Octo+PLD **96.63**（2025） | FASTerVLA **87.9** | X-VLA **86.6** | **无 2026 进 Top3**；2026 最高约 GST-VLA 80.2 | √ |
| **RoboTwin 2.0** | Fast-WAM **91.83** | StarVLA-α **88.3** | RLDX-1 **87.8** | Top3 全为 2026/π系 2026   |  |
| **RoboCasa** | Cosmos Policy **67.1** | World2Act **66.3** | HAMLET **65.4** | Top3 全为 2026 | √ |
| **CALVIN** | Xiaomi **4.75** | NS-VLA **4.72** | UD-VLA+Fast-dVLA **4.54** | Top3 含多个 2026   |  |
| **ManiSkill2** | ConsisVLA-4D **94.3** | — | — | 2026 SOTA   |  |
| **RoboCasa365** | **RLDX-1 32.1** | π₀ 14.8 | — | 明显领先 π₀ | √ |
| **MolmoSpace / MolmoBot / RoboEval** | MolmoAct2 | — | — | 2026 新榜 | √ |

---

## 重要说明（读表前建议看）

1. **数据来源**：RLDX-1 论文链接为 **arXiv:2605.03269**；π₀.₅ LIBERO **97.7%** 来自 [vla-eval harness](https://arxiv.org/abs/2603.13966) 复现。  
2. **「是否优于 π₀.₅」**：有 π₀.₅ 公开分数的 mainly **LIBERO（97.7）**；其余榜用 **π₀** 作参考并标明「π₀.₅ 未测」。  
3. **未列入但值得跟踪的 2026 强工作**（未达全局 Top3 但分数很高）：**FutureVLA**（LIBERO 98.3）、**PRTS**（LIBERO 98.4）、**Cosmos Policy**（LIBERO 98.5）、**FocusVLA**（98.6）等，距 Top3 仅 **0.2–0.5pp**。  
4. **真机榜**：**RoboChallenge** 上 StarVLA-α Generalist **33.6%** 为榜内高位；**RoboArena** 以分布式真机偏好为主。  
5. **VLA-Arena / CEBench** 为新综合框架（2025.12–2026）。

如需，我可以在此基础上再出一版：**仅保留「严格全局 Top3」的精简表**，或 **按单个 benchmark（例如只做 LIBERO-Plus + SimplerEnv）的逐榜排名 CSV**。

---

## 补充：遗漏 benchmark 中的 2026 SOTA / Top3 论文（新→旧）

> 口径：本节只补充前文未覆盖、且**论文名未在上表出现过**的工作；“2026”同时接受 **arXiv 2026** 与 **arXiv 2025 / 2026 会议录用** 两类，并在论文名中标注。π₀.₅ 在多数导航、家庭、人形控制榜上没有同榜公开分数，因此只在有可比公开基线时做弱比较。

| 论文名 | 论文下载地址 | 在哪个 benchmark 拿到 SOTA 或 Top3 以及分数多少 | 是否比 π₀.₅ 效果好以及好多少 | 论文简介（内容 / 创新 / 消融有效点） | 提及 |
|---|---|---|---|---|---|
| **Beyond Textual Knowledge: Leveraging Multimodal Knowledge Bases for Enhancing Vision-and-Language Navigation**（arXiv 2026-03） | [PDF](https://arxiv.org/pdf/2603.26859) · [abs](https://arxiv.org/abs/2603.26859) | **R2R Top/SOTA 指标**：Test Unseen SR **74**、SPL **63**；Val Unseen SR **75**、SPL **64**。**REVERIE Top/SOTA 指标**：Test Unseen RGS **35.08**、RGSPL **25.23**；Val Unseen RGS **34.82**、RGSPL **24.86**。 | 无同榜 π₀.₅；这是 VLN 离散导航榜，π₀.₅ 主要公开在机器人操作/家庭长程真机任务上，不能直接比较。 | 提出 **BTK**，用 Qwen3-4B 抽取目标短语、Flux-Schnell 生成目标视觉知识库、BLIP-2 构建环境文本知识库，再通过 Goal-Aware / Knowledge Augmentor 注入导航模型。消融显示：Qwen3-4B 目标短语抽取优于 SpaCy；图像知识 + 文本知识联合时 REVERIE Val Unseen SPL/RGSPL 最好。   |  |
| **ELITE: Experiential Learning and Intent-Aware Transfer for Self-improving Embodied Agents**（arXiv 2026-03） | [PDF](https://arxiv.org/pdf/2603.24018) · [abs](https://arxiv.org/abs/2603.24018) | **EmbodiedBench 子榜 SOTA**：EB-ALFRED 在线无监督 Avg **61%**（Qwen2.5-VL-72B，较 base +9pp）；EB-Habitat 在线无监督 Avg **67%**（InternVL3-78B，较 base +5pp）；EB-ALFRED 监督设置 Avg **70.8%**，高于 Claude-3.5-Sonnet 66.4 和 ERA 65.2。 | 无同榜 π₀.₅；EmbodiedBench 是 AI2-THOR / Habitat 离散 embodied-agent 评测。若只参考 π₀.₅ 家庭长程能力，二者任务形态不同，不能直接换算优劣。 | 让 embodied agent 从自己的执行轨迹里提炼经验，并用 **intent-aware retrieval** 按过程意图检索相似策略。消融显示：去掉 Intent-Aware Retrieval 后 EB-ALFRED Avg 从 61 降到 56；去掉 Context Consolidation 降到 55；CoT 检索在长程任务进度上优于 TF-IDF 和随机检索。   |  |
| **P3Nav: End-to-End Perception, Prediction and Planning for Vision-and-Language Navigation**（arXiv 2026-03） | [PDF](https://arxiv.org/pdf/2603.17459) · [abs](https://arxiv.org/abs/2603.17459) | **REVERIE SOTA**：Test Unseen SR **60.06**、SPL **40.57**、RGS **39.75**、RGSPL **26.56**；**R2R-CE SOTA**：Val Unseen SR **62**、SPL **52**；**RxR-CE SOTA**：Val Unseen SR **58.01**、SPL **47.92**、nDTW **64.29**、SDTW **48.04**。 | 无同榜 π₀.₅；这是 VLN/VLN-CE 导航任务，π₀.₅ 无公开同榜结果。 | 将感知、未来 waypoint 预测、未来语义地图预测和规划放入统一端到端网络，避免外部模块信息损失。消融显示：对象解码、地图/waypoint 预测等中间模块都提升 REVERIE、R2R-CE、RxR-CE 的 SR/SPL，说明“先显式理解场景再规划”比纯 planning-only 更稳。   |  |
| **Let's Reward Step-by-Step: Step-Aware Contrastive Alignment for Vision-Language Navigation in Continuous Environments**（arXiv 2026-03） | [PDF](https://arxiv.org/pdf/2603.09740) · [abs](https://arxiv.org/abs/2603.09740) | **VLN-CE SOTA**：R2R-CE Val Unseen SR **60.3**、SPL **55.1**（无额外数据），使用额外数据后 SR **64.7**、SPL **56.9**；RxR-CE Val Unseen SR **60.3**、SPL **49.8**、nDTW **62.1**，使用额外数据后 SR **62.1**、SPL **51.7**、nDTW **66.0**。 | 无同榜 π₀.₅；该榜是 Habitat 连续导航，不是机器人操作策略榜。 | 提出 **SACA**，用 Perception-Grounded Step-Aware Auditor 给失败轨迹分配逐步奖励，并在 mixed/all-failure batch 中分别做 Repair Resampling 与 All-Failure Rescue。消融显示：Soft Score、All-Failure Rescue、Repair Resampling 三者合用时 R2R-CE / RxR-CE 指标最高；相比 StreamVLN，R2R-CE SR +7.5pp、RxR-CE SR +11.7pp。   |  |
| **Scaling Tasks, Not Samples: Mastering Humanoid Control through Multi-Task Model-Based Reinforcement Learning**（EZ-M，arXiv 2026-03） | [PDF](https://arxiv.org/pdf/2603.01452) · [abs](https://arxiv.org/abs/2603.01452) | **HumanoidBench / HumanoidBench-Hard SOTA**：论文称 HumanoidBench-Medium **9 个任务中 7 个达到 SOTA 或超过 success score**，HumanoidBench-Hard **14 个任务中 10 个达到 SOTA**；Hard 例子包括 h1hand-walk **919.960**、h1hand-run **818.397**、h1hand-pole **841.353**、h1hand-reach **5353.093**。 | 无同榜 π₀.₅；π₀.₅ 不是 HumanoidBench 在线 RL 控制基线。只能说 EZ-M 在该仿真人形全身控制榜强于 TD-MPC2、DreamerV3、BRC 等已报告基线。 | 多任务 model-based RL 方法，强调“扩任务数而非单任务样本数”，用共享 world model 和 Gumbel search 提升样本效率。消融/分析显示：Independent Experience Replay 是关键组件；共享 dynamics/reward/value-policy 模块在相关任务间保持更高梯度相似度，解释了跨任务正迁移。   |  |
| **RealMirror: A Comprehensive, Open-Source Vision-Language-Action Platform for Embodied AI**（arXiv 2025 / ICRA 2026） | [PDF](https://arxiv.org/pdf/2509.14687) · [abs](https://arxiv.org/abs/2509.14687) | **RealMirror Humanoid VLA Benchmark Top1**：SmolVLA 平均成功率 **79.75%**，高于 Diffusion Policy **75.15%**、ACT **73.55%**；分任务 Top1 包括 ACT Kitchen Cleanup **100.00%** / Assembly Line Sorting **95.00%**，Diffusion Policy Air Fryer **85.50%**，SmolVLA Cup-to-Cup **68.00%** / Can Stacking **62.00%**。Sim2Real 实验：ACT 在真实 A2 机器人上 zero-shot pick-place **92.86%**、ball transfer **71.43%**。 | 无同榜 π₀.₅；这是人形平台/benchmark 论文，π₀.₅ 未在 RealMirror 上公开。弱参考：π₀.₅ 报告偏移动操作家庭任务，RealMirror 偏双臂/灵巧手人形操作，不可直接比较。 | 更像 **benchmark + 平台论文** 而非单一新模型：提供 VR 遥操作、LeRobot/Isaac Sim 训练推理、5 类人形任务和 1200 条轨迹，并用生成模型 + 3DGS 做高保真 Sim2Real。分析认为 SmolVLA 的语言条件与双系统结构让平均表现最稳；3DGS/高保真场景是 zero-shot Sim2Real 成功的主要因素。   |  |

### 本轮未纳入但已检查的 benchmark

- **GenManip / RH20T**：能查到 RH20T 数据集与 2026 相关操作论文（如 DynamicVLA / SimVLA），但未发现公开、稳定的 2026 RH20T 或 GenManip Top3 论文榜单分数。
- **DROID Eval / RoboArena**：DROID 更多作为数据集或 RoboArena 真机评测底座；公开资料能说明 π₀、GR00T、OpenVLA 等参与，但本轮未找到“不在上表且 2026 独立论文拿到 DROID Eval Top3”的可核验分数。
- **Isaac Lab / Isaac Lab Arena Eval Tasks、GR00T N1/N1.5 套件**：公开资料包含 Nut Pouring、Exhaust Pipe Sorting、GR-1 Tabletop 等任务，但上表已覆盖 GR00T/RLDX/StarVLA 相关条目；未找到额外独立 2026 论文可加入。
- **StaticEmbodiedBench / Embodied4C**：这两者主要是 benchmark 论文，榜上 Top 模型多为 GPT/InternVL/Octo/OpenVLA 等模型或已有工作；未发现符合“2026 新论文 + 不在原文件 + 明确 Top3”的独立 VLA 论文。
- **BEHAVIOR-1K**：可查到 2025 BEHAVIOR Challenge Top3（Robot Learning Collective q_score 0.2599、Comet 0.2514、SimpleAI Robot 0.1591）以及 GR00T N1.6 vs π₀.₅ 的相邻报告，但它们不满足“2026 独立论文且不在原文件”的主表口径。
- **Habitat 3.0、HomeRobot OVMM、ProcTHOR/RoboTHOR、ALFRED/TEACh、VLN-CE/R2R/REVERIE/SOON**：VLN 方向中可核验的 2026 Top/SOTA 已纳入 P3Nav、SACA、BTK、ELITE；其余平台没有找到新的、可交叉核验的 2026 Top3 论文条目。

---

## 补充：`embd_VLA_WAM_sota_chat_0516.md` 与 `VLAWAM_mdl_opti_op47_1.md` 中出现、主表未收录的论文（新→旧）

> **筛选口径**：在两份源文档中被列为 SOTA / Top3 / 综合榜前列，且**论文标题未出现在上文「2026 年 VLA / 具身论文」主表**（含 2025 年末–2026 年 arXiv；WAM / 后训练 / 跨本体路线单独成节）。π₀.₅ 对比基线同前：**LIBERO 97.7%**（vla-eval）；其余榜无 π₀.₅ 公开分数时用 **π₀** 或论文内 **π₀.₅ 对照** 标注。

| 时间 | 论文名 | Benchmark（SOTA / Top3 及分数） | 是否优于 π₀.₅？好多少？ | 简介（内容 / 创新 / 消融有效点） | 论文相关资源 | 提及 |
|------|--------|--------------------------------|-------------------------|--------------------------------|--------------|------|
| **2026-05** | **Learning While Deploying (LWD)** | **长程真机 fleet SOTA 叙事**：16 台双臂、8 任务，generalist policy 平均成功率 **95%**（SFT 76%→95%，**+19pp**）；3–5 分钟任务 **68%→91%（+23pp）** | 无 LIBERO 同榜；长程真机 **显著优于** 纯 SFT（π₀.₅ 未报该设定） | **DIVL + QAM** 的部署期 online RL；消融：fleet on-policy 数据闭环是长程涨点关键 | [PDF](https://arxiv.org/pdf/2605.00416) · [abs](https://arxiv.org/abs/2605.00416) | √ |
| **2026-04-16** | **π₀.₇: A VLA with Generalist Intelligence** | **真机跨本体**：UR5e 双臂叠衣 zero-shot **85.6% task progress / 80% success**（≈ Top-2% 遥操作员 90.9%/80.6%）；14 个未见厨房/卧室指令链优于 π₀.₅/π₀.₆ | **未跑 LIBERO**；真机长程 **明确优于 π₀.₅**（论文 Fig.6/7/12） | **MEM** 历史编码 + 视觉子目标 + context CFG；消融：**metadata / eval-data** 两路去掉后 throughput 一致显著变弱 | [PDF](https://arxiv.org/pdf/2604.15483) · [abs](https://arxiv.org/abs/2604.15483) · [π₀ 系 blog](https://pi.website) | √ |
| **2026-04** | **STARRY: Spatio-Temporal and Action Reasoning for Robotic Manipulation** | **RoboTwin 2.0 SOTA（论文自报）**：Clean **93.82%** / Random **93.30%**（50 双臂任务）；**真机** π₀.₅ **42.5%→70.8%（+28.3pp）** | RoboTwin：较 π₀（31.35%）**+62.5pp**；真机较 **π₀.₅ +28.3pp**（同论文） | **时空-动作联合扩散** + GASAM（depth/EE 几何对齐 token）；消融：联合预测未来 latent+动作是 RoboTwin/真机双涨点核心 | [PDF](https://arxiv.org/pdf/2604.26848) · [abs](https://arxiv.org/abs/2604.26848) | √ |
| **2026-04** | **HiPolicy: Hierarchical Multi-Frequency Action Chunking** | 长程操作（论文报告显著优于单层 chunking；**无统一公开榜 Top3 百分比**） | π₀.₅ 未测 | **多频率分层 action chunk**（慢规划+快执行）；消融：分层频率对齐长程与高频反应 | [PDF](https://arxiv.org/pdf/2604.06067) · [abs](https://arxiv.org/abs/2604.06067) | √ |
| **2026-04** | **STRONG-VLA: A Unified and Scalable VLA with Perturbation-Consistent Learning** | **鲁棒性榜**：28 种扰动下 **π₀ +16.49% / +5.58%**（seen/unseen）；OpenVLA-OFT **+14.48%** | 在扰动套件上 **优于标准 π₀.₅ 评测**（非 LIBERO 原始榜） | **两阶段**：课程扰动学习 + 干净数据再对齐；消融：Stage I+II 联合对跨架构有效 | [PDF](https://arxiv.org/pdf/2604.10055) · [abs](https://arxiv.org/abs/2604.10055) | √ |
| **2026-04** | **ReconVLA: Reconstructive Vision-Language-Action Models** | OOD 检测（**非成功率榜**）；部署安全向 | 与 π₀.₅ 成功率不可直接比 | **Conformal prediction** 给校准不确定性；消融：实时 OOD 触发回退/求助人类 | [PDF](https://arxiv.org/pdf/2604.16677) · [abs](https://arxiv.org/abs/2604.16677) | √ |
| **2026-04** | **HY-Embodied-0.5** | **22 个 embodied benchmark** 与 Gemini 3.0 Pro **持平**（论文叙事）；**2B 边端 + 32B 云端 MoT** | π₀.₅ 未在同套 22 榜逐榜公开 | **Mixture-of-Transformers** 统一边云；消融：MoT 分规模部署有效 | [PDF](https://arxiv.org/pdf/2604.07430) · [abs](https://arxiv.org/abs/2604.07430) | √ |
| **2026-04** | **VLA Foundry: A Unified Framework for Training VLAs** | 框架论文（**非单一模型 SOTA 榜**）；统一复现 LIBERO / CALVIN / SimplerEnv 等 | — | **VLM→action expert** 端到端训练管线；消融：共享 pipeline 降低复现方差 | [PDF](https://arxiv.org/pdf/2604.19728) · [abs](https://arxiv.org/abs/2604.19728) · [GitHub](https://github.com/allenai/vla-evaluation-harness)（同类生态） | √ |
| **2026-03** | **GigaWorld-Policy: Action-Centered World-Action Model** | **RoboTwin 2.0**：较 **π₀.₅ +95%**（如 Place Fan 0.25→0.94）；**RoboCasa 官榜 T2 20.7%**；推理 **9× 加速**、success **+7%** vs Motus | RoboCasa 较 π₀（62.5% 不同指标）需看官榜 Overall；RoboTwin **远超 π₀.₅** | **因果 WAM** + 统一 Transformer 处理 obs/state/action；消融：**不强制每步生成视频** 仍保持 SOTA 且更快 | [PDF](https://arxiv.org/pdf/2603.17240) · [abs](https://arxiv.org/abs/2603.17240) · [项目](https://giga-world-0.github.io/) | √ |
| **2026-03** | **Ψ₀ (Psi-Zero): Learning Robust Visuomotor Policies via World Model Distillation** | **跨本体**：800h 人类视频 + 30h 真机 **较 10× 数据基线 +40%**；**80 条轨迹**可学长程新技能 | 无 LIBERO 统一榜；整体成功率 **+40%** 相对大数据 VLA 基线（含 π₀.₅ 系） | **世界模型蒸馏** + 两阶段（先人后机）；消融：人类视频预训是数据效率关键 | [PDF](https://arxiv.org/pdf/2603.12263) · [abs](https://arxiv.org/abs/2603.12263) · [Site](https://psi-lab.ai/Psi0) | √ |
| **2026-03** | **WoVR: World Models as Reliable Simulators for Post-Training VLAs** | **LIBERO**：39.95%→**69.2%（+29.3pp）**；**真机** 61.7%→**91.7%（+30.0pp）** | LIBERO 仍 **低于 π₀.₅ 97.7%**（后训练起点不同）；**后训练增益** 对 π₀.₅ 系具参考价值 | 用 WM 当 **可靠 sim** 做 VLA 后训练；消融：WM 质量决定 post-train 上限 | [PDF](https://arxiv.org/pdf/2602.13977) · [abs](https://arxiv.org/abs/2602.13977) | √ |
| **2026-03** | **VLA-OPD: On-Policy Distillation for VLAs** | **LIBERO + RoboTwin 2.0**：采样效率 **>> 在线 RL**、**>> SFT**（论文 Table 1）；RoboTwin 报告 **84.6%** 等条目 | 作为 **π₀.₅/OFT 后训练** 可叠加；LIBERO 绝对值需看起点 checkpoint | **Reverse-KL on-policy 蒸馏**；消融：比 PPO/纯 SFT 更抗遗忘 | [PDF](https://arxiv.org/pdf/2603.26666) · [abs](https://arxiv.org/abs/2603.26666) | √ |
| **2026-03** | **SmoothVLA** | **LIBERO** 平滑度 **+13.8%**（物理一致性 reward） | 需看绝对成功率是否超 97.7%（论文主打 jerk 指标） | **GRPO + jerk 连续 reward**；消融：物理平滑 reward 减抖动 | [PDF](https://arxiv.org/pdf/2603.13925) · [abs](https://arxiv.org/abs/2603.13925) | √ |
| **2026-02** | **DreamZero: World Action Models are Zero-Shot Policies** | **RoboArena #1（团队自报）**；**MolmoSpaces #1**；AgiBot G1 seen **82%** vs π₀.₅ **27%**（**≈3×**）；unseen task progress **62.2% vs 27.4%（2.3×）** | **真机泛化明确优于 π₀.₅**；LIBERO 未评 | **Wan2.1-I2V-14B WAM** + 联合 video-action flow；消融：**KV cache 真观测替换、DreamZero-Flash 1-step、CFG 并行** 实现 38× 加速与 7Hz 闭环 | [PDF](https://arxiv.org/pdf/2602.15922) · [abs](https://arxiv.org/abs/2602.15922) · [Code](https://github.com/dreamzero0/dreamzero) · [Site](https://dreamzero0.github.io/) | √ |
| **2026-02** | **World-VLA-Loop** | **真机闭环**：较基线 **+36.7%**（VLA 失败 rollout 反馈精炼 WM） | π₀.₅ 未同设定；**闭环增益 +36.7%** | VLA↔WM **闭环共改进**；消融：失败轨迹回流 WM 是涨点关键 | [PDF](https://arxiv.org/pdf/2602.06508) · [abs](https://arxiv.org/abs/2602.06508) | √ |
| **2026-02** | **VLAW: Vision-Language-Action World Model** | **绝对成功率 +39.2%**；合成数据 **+11.6%** | 相对提升大；绝对 LIBERO 需对照表 | **迭代共改进** WM+VLA；消融：合成+真实混合迭代 | [PDF](https://arxiv.org/pdf/2602.12063) · [abs](https://arxiv.org/abs/2602.12063) | √ |
| **2026-02** | **VLANeXt: Recipes for Building Strong VLA Models** | **LIBERO-Long T2 94.6%**；**LIBERO 平均 T1 97.4%**（VLANeXt Table 2）；LIBERO+ 鲁棒套件 SOTA 叙事 | LIBERO 平均 **-0.3pp** vs π₀.₅；Long 榜 **Top2** | **12 条设计 recipe** + 2.5B 小模型；消融：recipe 组合 > 单点结构创新 | [PDF](https://arxiv.org/pdf/2602.18532) · [abs](https://arxiv.org/abs/2602.18532) | √ |
| **2026-02** | **SimVLA: A Simple VLA Baseline** | **LIBERO T3 98.6%**（0.5B）；训练 VRAM **9.3GB** | LIBERO **+0.9pp** vs π₀.₅（论文表 98.6 vs 96.9） | **极简 VLA**（轻量 backbone+动作头）；消融：简单配方+高效训练可达近 SOTA | [PDF](https://arxiv.org/pdf/2602.18224) · [abs](https://arxiv.org/abs/2602.18224) · [Code](https://github.com/LUOyk1999/SimVLA) · [HF](https://huggingface.co/YuankaiLuo/SimVLA-LIBERO)   |  |
| **2026-02** | **Green-VLA** | **5 阶段 curriculum**（VLM→RL）；3k h 双臂（论文叙事 SOTA 路线） | π₀.₅ 未逐榜 | **质量对齐 + 动作统一 + RL 精炼** 三阶段；消融：curriculum 后期 RL 必要 | [PDF](https://arxiv.org/pdf/2602.00919) · [abs](https://arxiv.org/abs/2602.00919) | √ |
| **2026-02** | **GeneralVLA** | **零真机数据** affordance+3D agent 零样本（**无 LIBERO 标准 Top3 分数**） | 不同设定 | **3D affordance + 控制策略**；消融：3D agent 冷启动有效 | [PDF](https://arxiv.org/pdf/2602.04315) · [abs](https://arxiv.org/abs/2602.04315) | √ |
| **2026-02** | **LifeLong-RFT** | **LIBERO**：较 SFT **+22%** 平均成功率（**20% 数据**） | 后训练路线；绝对分需看起点是否超 97.7% | **Chunk 级 on-policy RL + MDPR**；消融：防灾难性遗忘 | [PDF](https://arxiv.org/pdf/2602.10503) · [abs](https://arxiv.org/abs/2602.10503) | √ |
| **2026-02** | **QuantVLA** | 部署向：**70% 显存、1.22× 加速**，成功率不降 | 工程优化，非 LIBERO 涨点 | **后训练量化 PTQ**；消融：量化后仍保精度 | [PDF](https://arxiv.org/pdf/2602.20309) · [abs](https://arxiv.org/abs/2602.20309) | √ |
| **2026-02** | **CycleVLA** | **LIBERO** 回溯+MBR 显著提升（Table V/VI，无单句 +X%） | 测试时增强，可与 π₀.₅ 叠加 | **子任务回溯 + MBR decoding**；消融：failure predictor+回溯 | [PDF](https://arxiv.org/pdf/2601.02295) · [abs](https://arxiv.org/abs/2601.02295) | √ |
| **2026-02** | **Mask World Model (MWM)** | **LIBERO / RLBench** 全面优于 RGB 世界模型（论文 claim SOTA） | π₀.₅ 未测 WM 榜 | **语义 mask 预测** 替 RGB；消融：mask 监督减外观噪声 | [PDF](https://arxiv.org/pdf/2604.19683) · [abs](https://arxiv.org/abs/2604.19683) | √ |
| **2026-01** | **Being-H0.5** | 论文自报 **LIBERO 98.9%**、**RoboCasa 53.9%**（**未提交官榜**；官榜 Overall ~22%） | LIBERO 自报 **+1.2pp** vs π₀.₅；RoboCasa 口径与官榜不可直接比 | **UniHand-2.0（35kh/30 本体）+ MoF**；消融：**MoF + 统一动作空间** 跨本体生效 | [PDF](https://arxiv.org/pdf/2601.12993) · [abs](https://arxiv.org/abs/2601.12993) · [Code](https://github.com/BeingBeyond/Being-H) · [HF](https://huggingface.co/BeingBeyond/Being-H05-2B) · [Site](https://research.beingbeyond.com/being-h05) | √ |
| **2026-01** | **LingBot-VLA / A Pragmatic VLA Foundation Model** | **20k h 双臂**；scaling **3k→20k h 持续上涨**（无饱和） | π₀.₅ 未同数据尺度 | 工程化 **VLA 预训练实践** + LingBot-VA 因果 WM；消融：数据规模仍主导 | [PDF](https://arxiv.org/pdf/2601.18692) · [abs](https://arxiv.org/abs/2601.18692) | √ |
| **2026-01** | **SOP: Scalable Online Post-Training** | **Fleet online RL**（小时级涨点；**无单一 LIBERO%**） | 部署后训练，非 zero-shot 榜 | **HG-DAgger + RECAP** 流式 on-policy；消融：fleet 规模数据闭环 | [PDF](https://arxiv.org/pdf/2601.03044) · [abs](https://arxiv.org/abs/2601.03044) | √ |
| **2026-01** | **TT-VLA** | **推理时 RL**（task-progress dense reward） | π₀.₅ 未测 | **On-the-fly** 适应；消融：进度 reward 优于稀疏成功 | [PDF](https://arxiv.org/pdf/2601.06748) · [abs](https://arxiv.org/abs/2601.06748) | √ |
| **2026-01** | **Genie Sim 3.0**（智元） | **第三方对照**：Instruction **π₀.₅ 0.67** > GR00T-N1.6 **0.40** > π₀ **0.28**（三项均第一） | **仿真-真机一致性榜 π₀.₅ 三项 SOTA** | LLM 驱动 sim + VLM 评测；**R²=0.94** sim-real | [PDF](https://arxiv.org/pdf/2601.02078) · [abs](https://arxiv.org/abs/2601.02078) · [AgiBot](https://agibot.com/article/231/detail/55.html) | √ |
| **2026-01** | **Helix 02**（Figure AI，技术报告/blog） | **真机长程**：4 分钟 61 步洗碗 **无干预**；**非标准仿真榜** | π₀.₅ 未同任务 | **S0(1kHz 平衡)+S1(200Hz 视触)+S2(语义)** 三层；消融：神经网络替 **10.9 万行 C++** 平衡控制 | [Blog](https://figure.ai/news/helix-02) | √ |
| **2025-12** | **GR00T N1.6**（NVIDIA） | **RoboCasa 官榜 SOTA #1 Overall 21.9%**；Genie Sim 三项低于 π₀.₅ | RoboCasa 官榜 **无 π₀.₅ 条目**；低于主表 RLDX-1 的 70.6%（不同指标） | **Cosmos-Reason-2B + 32 层 DiT**；state-relative action chunk | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) · [HF GR00T-N1.7-LIBERO](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | √ |
| **2025-12** | **OXE-AugE** | **跨本体**：未见 robot×gripper **+24%~+45%**（4 真机任务） | 数据/增广论文；非单模型 LIBERO 分 | **机器人形态增广** 扩 OXE→4.4M；消融：增广组合 > naive mixing | [PDF](https://arxiv.org/pdf/2512.13100) · [abs](https://arxiv.org/abs/2512.13100) | √ |
| **2025-11** | **π\*₀.₆ / Recap** | 真机：**throughput 2×**、**failure rate ≈½**（咖啡/装箱等 **≥90%**） | 相对 **π₀.₅** 后训练强化（同 PI 系） | **Offline RL + on-robot RL + HG-DAgger**；消融：Recap **>> AWR/PPO** | [PDF](https://arxiv.org/pdf/2511.14759) · [abs](https://arxiv.org/abs/2511.14759) · [openpi](https://github.com/physical-intelligence/openpi) | √ |
| **2025-10** | **X-VLA** | LIBERO **98.1%**、SimplerEnv **86.6%**、CALVIN **4.43**、RoboTwin 2.0 **54.5%** | LIBERO **+0.4pp**；SimplerEnv **较 π₀ +15.6pp** | **Soft-prompt embodiment id**；消融：naive 混合退化，soft prompt 恢复 | [PDF](https://arxiv.org/pdf/2510.10274) · [abs](https://arxiv.org/abs/2510.10274) | √ |
| **2025-10** | **CoLA-World** | Latent action + WM **协同训练**（LIBERO/长程叙事） | π₀.₅ 未测 | **LAM+WM 共进化** 防表征坍塌；消融：warm-up 防坍塌 | [PDF](https://arxiv.org/pdf/2510.26433) · [abs](https://arxiv.org/abs/2510.26433) | √ |
| **2025-09** | **FLOWER** | **LIBERO-Long SOTA #1 94.9%**（VLANeXt Table 2）；Spatial **96.9%** | Long 榜领先；与 π₀.₅ **分项互有高低** | **高效 VLA flow policy**；消融：flow head 在 Long 套件最强 | [PDF](https://arxiv.org/pdf/2509.04996) · [abs](https://arxiv.org/abs/2509.04996) | √ |

---

## 补充：用户追加论文（2026，新→旧）

> 以下为用户指定收录、**主表与上文「源文档补漏」节均未出现**的工作；「提及」列规则不变（检索 `embd_VLA_WAM_sota_chat_0516.md` 与 `VLAWAM_mdl_opti_op47_1.md`）。

| 时间 | 论文名 | Benchmark（SOTA / Top3 及分数） | 是否优于 π₀.₅？好多少？ | 简介（内容 / 创新 / 消融有效点） | 论文相关资源 | 提及 |
|------|--------|--------------------------------|-------------------------|--------------------------------|--------------|------|
| **2026-03-31** | **From Human Skill to Robotic Mastery**（Psi-R2 / Psi-W0 技术博客，灵初） | **无统一公开仿真榜 Top3**；叙事：**95,472h 人类 + 5,417h 真机** 预训练 Psi-R2（Wan2.2 WAM），**<100 条**真机轨迹可微调长程精细任务 | 无 LIBERO 同榜；与 π₀.₅ **无直接百分比对照**（博客强调人类数据规模化 + WM 评估闭环） | **Psi-R2**：视频-动作联合 WAM；**Psi-W0**：动作条件世界模型做策略评估与数据质检；**Raw Data In** 极简人类/真机对齐 | [Blog](https://cypypccpy.github.io/tech-blog.github.io/) | |
| **2026-03-10** | **TiPToP: A Modular Open-Vocabulary Planning System for Robotic Manipulation** | **28 任务 / 165+ trial（仿真+真机）**：总成功率 **59.4%** vs **π₀.₅-DROID 33.3%**；任务进度 **74.6%** vs **52.4%**；语义/干扰物子集 **62.4%** vs **25.9%**；多步子集 **57.5%** vs **15%**（**零机器人演示数据**） | **DROID 设定真机+仿真明确优于 π₀.₅-DROID**（成功率约 **+26pp**，进度 **+22pp**） | **TAMP（cuTAMP）+ 现成 VFM 感知**（Gemini-ER、SAM-2、M2T2、FoundationStereo）；模块化可独立换感知/规划；**<1h** 可部署标准 DROID | [Site](https://tiptop-robot.github.io/) · [PDF](https://arxiv.org/pdf/2603.09971) · [abs](https://arxiv.org/abs/2603.09971) · [Docs](https://tiptop-robot.readthedocs.io/en/latest/) | |
| **2026-03** | **MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation** | **DROID 真机 40 任务×3 trial**：Overall **79.2%**（MolmoBot F=2）vs **π₀.₅ 39.2%**；**DROID 仿真** 7 任务平均 **64.1%** vs π₀.₅ **10.0%**（均 **zero-shot、无真机微调**）；**MolmoBot-Data** 1.7M 轨迹 / 94k+ 场景 | 真机 pick-place **+40pp**（**≈2×** π₀.₅）；LIBERO 未作主评测 | **纯仿真 MolmoBot-Engine**（MolmoSpaces）+ **Molmo2 VLM + flow 动作头**；另含 MolmoBot-Pi0 / SPOC 变体；[Ai2 博客](https://allenai.org/blog/molmobot-robot-manipulation) 称 **sim-only 可匹敌/超越** 大规模真机 VLA | [Site](https://allenai.github.io/MolmoBot/) · [PDF](https://arxiv.org/pdf/2603.16861) · [abs](https://arxiv.org/abs/2603.16861) · [Code](https://github.com/allenai/MolmoBot) · [MolmoSpaces](https://github.com/allenai/molmospaces) · [HF Models](https://huggingface.co/collections/allenai/molmobot-models) · [HF Data](https://huggingface.co/datasets/allenai/MolmoBot-Data) · [Demo nb](https://github.com/allenai/MolmoBot/blob/main/MolmoBot/demo_policy.ipynb) | √ |
| **2026-02** | **LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer** | **零样本跨本体真机**：未见机器人平均成功率 **>50%**（**≈2×** 最强 prior VLA，其余开源 VLA 约 **0%**）；LIBERO 微调 **数据效率 ~2.5×**（论文叙事） | 零样本跨本体 **显著优于 π₀.₅-DROID / π₀.₅-replicated**（官网定性对比）；LIBERO 绝对分需看微调表 | **语言-动作预训练**：动作用自然语言监督 VLM，**无 action tokenizer**；**LAP-3B**；与 VQA 共训练 | [Site](https://lap-vla.github.io/) · [PDF](https://arxiv.org/pdf/2602.10556) · [abs](https://arxiv.org/abs/2602.10556) · [Code](https://github.com/lihzha/lap) · [HF](https://huggingface.co/collections/lihzha/lap) | |

### 源文档高频提及、但不宜列入「论文 Top3 表」的条目

| 名称 | 原因 |
|------|------|
| **π₀.₆** | 多为 model card / 技术报告，**未在 LIBERO-Long 等统一榜公开可与主表对齐的 Top3 分数**（π₀.₇ 论文亦未跑 LIBERO）。 |
| **GR00T N2 / Cosmos Predict 2.5** | 偏 **WAM/视频 WM 基座**；操作成功率见 Cosmos Policy / GigaWorld / DreamZero 子论文。 |
| **dWorldEval / PolaRiS / RobotArena ∞** | **评测工具或框架**，非「模型在榜单位列 Top3」。 |
| **Rethinking VLA Scaling、F-ACIL、Scanford** | 主要为 **数据混合 / 采集方法论**，无单一模型 SOTA 分数。 |
| **InternVLA-M1、WALL-OSS** | 源文档仅作对照提及，**缺可与主表对齐的独立 Top3 分数**；**MolmoBot** 已见上文「用户追加」节。 |

### 与主表关系说明

- **已覆盖**：OA-WAM、RLDX-1、StarVLA-α、PRTS、FutureVLA、GST-VLA、Cosmos Policy、Xiaomi-Robotics-0、ABot-M0、MINT、VLA-JEPA、BTK/ELITE/P3Nav/SACA 等 **不再重复**。
- **主表 MolmoAct2** 与 **MolmoBot**（纯仿真 79.2% DROID 真机）为 **不同论文**，勿混淆。
- **建议优先补读（源文档 Top3 权重最高）**：**DreamZero**、**STARRY**、**GigaWorld-Policy**、**Being-H0.5**、**VLANeXt**、**X-VLA**、**LWD**、**π₀.₇**、**WoVR**、**SimVLA**。
- **用户追加节**：**TiPToP**（模块化 TAMP vs π₀.₅-DROID）、**LAP**（跨本体零样本）、**MolmoBot**（sim-only 真机 SOTA 叙事）、**Psi-R2/W0 博客**（人类数据规模化）。
