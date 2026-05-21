# VLA / 具身智能 SOTA 论文综合列表 v2

> 整理口径：基于 [vla_sota_ls.md](vla_sota_ls.md) 中提到的所有论文，按提出（创建/提交）时间从新到旧排序，每篇论文一个小节。
> Benchmark 归类基于 [vla_benchmark_3.md](vla_benchmark_3.md) 主表清单。
> 「提及」列沿用源文件 √ 标记。

## 字段说明

- **提出日期**：论文 arXiv 首次提交日期或正式发布时间。
- **SOTA**：在 [vla_benchmark_3.md](vla_benchmark_3.md) 主表 benchmark 中获得过 SOTA / Top2 / Top3 成绩的列出榜名、分数、排名。
- **Benchmark**：在主表 benchmark 中跑过但未进 Top3 的列出榜名、分数。
- **Other Benchmark**：主表未列出的 benchmark / 数据集（如 MolmoSpace、MolmoBot、AgiBot G1、HumanoidBench-Hard、EB-ALFRED、EB-Habitat 等）。
- **比 Pi0.5 好**：在论文自身或第三方文章中与 π0.5 直接对比的内容（对比对象、对比方式、好多少）；未对比则留空。
- **简介**：内容大概、创新点、消融实验或在其他文章/实践中被验证有效或负面的方法。
- **相关资料**：论文下载地址、项目网站、代码或 GitHub 地址、模型权重、数据、Notebook、W&B、博客等。
- **数据规模**：训练数据规模（样本/轨迹、时长、本体与任务等）。
- **模型大小**：参数量、骨干与动作头、数值格式等。
- **算力**：GPU/TPU、训练时长与算力预算等。
- **世界模型用法**：WM/WAM/视频生成等在模型中的用法、消融验证、正负影响与量化提升；未使用则写「未使用」。
- **提及**：原 [vla_sota_ls.md](vla_sota_ls.md) 同一论文「提及」列为 √ 的打 √，否则留空。

## π0.5 对比基线

| 指标来源 | LIBERO | LIBERO-Plus | SimplerEnv | RoboTwin 2.0 | CALVIN | RoboCasa |
|---------|--------|-------------|------------|--------------|--------|----------|
| **π0.5** | **97.7%**（Allen AI `vla-eval` 复现） | 未单独公开 | 未单独公开 | 未单独公开 | 未单独公开 | 未单独公开 |
| **π0（参考）** | 94.2% | 56.1% | 71.0% | 31.35% | 3.92 | 62.5% |

> 真机长程能力以 [Physical Intelligence π0.5 报告](https://www.physicalintelligence.company/download/pi05.pdf)（arXiv:2504.16054）为基线。

π0.5 vs π0 提升幅度总结
根据 Physical Intelligence 官方技术报告和相关 benchmark 论文，π0.5 相比 π0 的提升如下：

1. 真实机器人任务（零样本/少样本）
任务进度（task progress）：π0.5 约 60-65% vs π0 约 30-35%，提升约 2 倍
这是 π0.5 最核心的改进——通过加入语言推理和 web 数据联合训练，大幅提升了对自然语言指令的泛化
2. LIBERO 标准四套件（fine-tuned）
子集	π0.5	π0	差距
libero-spatial	98.4%	97.6%	+0.8pp
libero-object	99.2%	99.2%	0
libero-goal	97.6%	92.0%	+5.6pp
libero-10	93.0%	82.0%	+11.0pp
avg	96.85%	92.7%	+4.15pp
最大差距在 libero-10（长程多步任务）和 libero-goal（目标泛化）上。

3. LIBERO-PRO 扰动鲁棒性（来自 arXiv 2510.03827）
扰动类型	π0.5	π0
Position perturbation	0.08–0.38	0.00（全部崩溃）
Environment perturbation (libero-object)	0.73	0.29
Environment perturbation (libero-10)	0.46	0.27
π0 在位置扰动下完全失效（所有子集得分 0），π0.5 则保持一定鲁棒性。这说明 π0.5 的 VLM 推理能力带来了更好的空间泛化。

4. 关于 survey 中 π0.5 LIBERO 基准值的说明
survey（vla_sota_ls_2.md）采用的 π0.5 LIBERO 基准是 97.7%（来自 Allen AI vla-eval 排行榜）
官方技术报告的 fine-tuned 数值是 96.85%（四套件平均）
很多论文引用的是 96.9%（四舍五入值）
三个数值差异的原因可能是评估协议不同（seed、rollout 数、checkpoint 选择等），目前 survey 保持 97.7% 不变，因为它与 vla-eval 排行榜一致

---

## 2026-05-07 · OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation

- **提出日期**：2026-05-07（arXiv 2605.06481）
- **SOTA**：**LIBERO-Plus T2 83.9%**（全局 #1 为 RLDX-1 86.7%）
- **Benchmark**：LIBERO 97.8%（未进全局 Top3）；SimplerEnv 79.3%（未进全局 Top3）
- **Other Benchmark**：causal slot-intervention 测试 swap-binding cosine 0.87（holistic baseline ≤ 0.09）
- **比 Pi0.5 好**：LIBERO **+0.1pp**（97.8 vs 97.7）；LIBERO-Plus 较 π0 **+27.8pp**；SimplerEnv 较 π0 **+8.3pp**
- **简介**：将场景分解为 **N+1 对象槽位**（地址向量 addr + 内容向量），世界头预测下一帧槽位状态，flow-matching 动作头一次前向出 16 步动作。消融：**地址键 cross-slot 注意力 + 残差流地址槽位逐层重置** 是 swap-binding 高分关键。
- **相关资料**：[PDF](https://arxiv.org/pdf/2605.06481) · [abs](https://arxiv.org/abs/2605.06481) · [HTML](https://arxiv.org/html/2605.06481v1) · [scirate](https://scirate.com/arxiv/2605.06481) · [Cool Papers](https://papers.cool/arxiv/2605.06481)
- **数据规模**：LIBERO / LIBERO-Plus / SimplerEnv 标准示范与槽位缓存预计算
- **模型大小**：Chameleon-7B 主干（冻结）+ ~127M 可训练（80M LoRA + 47M 预测头）；flow-matching 一次出 16 步动作
- **算力**：Stage I：384×A100-80GB，~18 天（~166k A100·h）；Stage II LoRA：8×A100，3–4 天；全参微调备选 7–10 天
- **世界模型用法**：**对象可寻址 WAM**：Chameleon-7B 主干 + **world head 回归下一帧 per-slot 状态**（N+1 对象槽位），与 flow-matching 动作头联合训练；非显式 RGB/视频扩散。消融：仅 **addr 键 cross-slot 注意力 + 残差流地址槽位重置** 显著提升 LIBERO-Plus 几何轴 swap-binding（cosine 0.87 vs holistic ≤0.09），LIBERO 分布内几乎不变（**正面、OOD 专用**）。
- **提及**：√

## 2026-05-07 · ConsisVLA-4D: Advancing Spatiotemporal Consistency in Efficient 3D-Perception and 4D-Reasoning for Robotic Manipulation

- **提出日期**：2026-05-07（arXiv 2605.05126）
- **SOTA**：**ManiSkill2 SOTA 94.3%**（论文自报）
- **Benchmark**：LIBERO 较 OpenVLA **+21.6%**（2.3× 加速）
- **Other Benchmark**：真机平台较 OpenVLA **+41.5%**（2.4× 加速）；约 1/8 视觉输入下保持精度
- **比 Pi0.5 好**：（无同榜公开对比）
- **简介**：CV-Aligner（跨视图对象语义一致）+ CO-Fuser（跨对象空间几何一致）+ CS-Thinker（跨场景时空一致）。消融：时空一致表征对仿真操作成功率提升最大；同时显著降低视觉 token 数。
- **相关资料**：[PDF](https://arxiv.org/pdf/2605.05126) · [abs](https://arxiv.org/abs/2605.05126) · [HTML](https://arxiv.org/html/2605.05126v1) · [GitHub](https://github.com/JiuTian-VL/ConsisVLA-4D)
- **数据规模**：LIBERO 四套件 + ManiSkill2 + RoboTwin 2.0 + 真机 AgileX/Galaxea 任务（每类 45–60 demo）
- **模型大小**：在基线 VLA 上约 +2B 参数（主要 VGGT）
- **算力**：训练 4× NVIDIA A800；真机推理单卡 RTX 5090
- **世界模型用法**：未使用。
- **提及**：

## 2026-05-05 · RLDX-1: A Dexterity-First Foundation Model for Robot Hands

- **提出日期**：2026-05-05（arXiv 2605.03269）
- **SOTA**：**综合榜 #1**；**LIBERO-Plus SOTA 86.7%**；**RoboCasa365 SOTA 32.1%**（π0 仅 14.8）；**RoboTwin 2.0 T3 87.8%**
- **Benchmark**：LIBERO 97.8%；RoboCasa 70.6%；SimplerEnv 76.9%（SIMPLER Google-VM 81.5%）；GR-1 Tabletop 58.7%
- **Other Benchmark**：ALLEX 人形真机长程任务 ≈ 86.8%（较 π0.5 / GR00T N1.6 提升约 40pp）
- **比 Pi0.5 好**：LIBERO **+0.1pp**；LIBERO-Plus 较 π0 **+30.6pp**；RoboCasa 较 π0 **+8.1pp**；SimplerEnv 较 π0 **+5.9pp**；真机较 π0.5 / GR00T N1.6 **~+40pp**
- **简介**：**Multi-Stream Action Transformer (MSAT)** 显式融合 Cognition (Qwen3-VL-8B 视觉问答) / Physics（触觉、力矩）/ Motion（视频时序）/ Memory（长期历史）多流；6.9B base / 8B 完全版；16 步 flow-matching 动作头；43.7ms/step on RTX 5090。消融：**运动感知、长期记忆、物理传感流** 为对抗 VLA 失败模式的关键；灵巧操作优先而非仅桌面 pick-place。
- **相关资料**：[PDF](https://arxiv.org/pdf/2605.03269) · [abs](https://arxiv.org/abs/2605.03269) · [Site](https://www.rlwrld.ai/en/rldx-1) · [GitHub Docs](https://github.com/RLWRLD/RLDX-1) · [HF](https://huggingface.co/RLWRLD/RLDX-1-PT) · [HF Collection](https://huggingface.co/collections/RLWRLD/rldx-1) · [alphaXiv](https://www.alphaxiv.org/audio/2605.03269)
- **数据规模**：ALLEX 人形（72K 合成 + 真机遥操作）+ FR3（DROID 92K + 真机）；多流触觉/力矩/记忆
- **模型大小**：MSAT 灵巧手基础模型（VLM 顶 4 层可训）
- **算力**：预训练 100K steps、global batch 8192；64× NVIDIA H200，约 195 小时
- **世界模型用法**：策略为 MSAT 多流 VLA；**Motion 流**消费视频时序特征，**非** 预测未来帧的 WM/WAM。I2V 仅出现在**数据合成**管线（若使用）。**未使用** 世界模型/ WAM 作为策略核心。
- **提及**：

## 2026-05-04 · MolmoAct2: Action Reasoning Models for Real-world Deployment

- **提出日期**：2026-05-04（arXiv 2605.02881）
- **SOTA**：**RoboEval SOTA 44.3**（vla_benchmark_3 主表的 RoboEval 泛称下首个 SOTA 报告）；**MolmoSpace SOTA 37.7**；**MolmoBot SOTA 20.6**（均为 Allen AI 内部新榜）
- **Benchmark**：—
- **Other Benchmark**：**MolmoSpace SOTA 37.7**；**MolmoBot SOTA 20.6**（均为 Allen AI 内部新榜）；处理多种真机任务无需 per-task 微调；37× 推理加速 vs MolmoAct v1；MolmoAct 2-Bimanual YAM 数据集 720 小时
- **比 Pi0.5 好**：（新榜无 π0.5 公开同条目）
- **简介**：Allen AI **动作推理型 VLA (ARM)**，强调可部署的显式动作推理链；基于 Molmo 2-ER（embodied-reasoning VLM）训练在 ~3M embodied-reasoning samples。消融：**显式动作推理头** 对长程与 sim2real 相关任务提升最大；3D 深度感知 token 是核心组件之一。
- **相关资料**：[PDF](https://arxiv.org/pdf/2605.02881) · [abs](https://arxiv.org/abs/2605.02881) · [HTML](https://arxiv.org/html/2605.02881v1) · [Blog](https://allenai.org/blog/molmoact2) · [GitHub](https://github.com/allenai/MolmoAct) · [HF Models](https://huggingface.co/allenai/MolmoAct-7B-O-0812)
- **数据规模**：Molmo2-ER 预训练 3.3M 样本；MolmoAct2-BimanualYAM 720h（34.5k demo）；DROID/SO-100 过滤子集
- **模型大小**：Molmo2-ER VLM + flow 动作专家（离散 token VLM 嫁接）
- **算力**：Pretrain：64×H100，~5760 GPU·h；Finetune：64×H100 ~2304h；DROID：32×H100 ~1152h
- **世界模型用法**：未使用。
- **提及**：

## 2026-05 · Learning While Deploying (LWD): Fleet-Scale Reinforcement Learning for Generalist Robot Policies

- **提出日期**：2026-05（arXiv 2605.00416）
- **SOTA**：—（真机 fleet 长程实验，无传统仿真榜 Top3）
- **Benchmark**：—
- **Other Benchmark**：16 台双臂 / 8 任务 fleet 平均成功率 **76% (SFT) → 95% (LWD)**；3-5 分钟长程任务 **68% → 91%（+23pp）**
- **比 Pi0.5 好**：LWD 显著优于 π0.5 系 SFT 基线（同设定无 π0.5 RL 公开对照）
- **简介**：**DIVL（Distributional Implicit Value Learning）+ QAM（Q-learning via Adjoint Matching）** 的部署期 online RL 框架；fleet on-policy 经验聚合到共享 replay buffer 持续改进策略。消融：fleet on-policy 数据闭环是长程任务涨点关键；DIVL 对稀疏奖励分布的健壮值估计为核心。
- **相关资料**：[PDF](https://arxiv.org/pdf/2605.00416) · [abs](https://arxiv.org/abs/2605.00416) · [HTML](https://arxiv.org/html/2605.00416v1) · [AGIBOT Finch Research](https://finch.agibot.com/research/lwd) · [Cool Papers](https://papers.cool/arxiv/2605.00416)
- **数据规模**：离线 buffer：演示 + 历史 rollout + play 数据（按任务累计小时，见论文 Table IV）；16 台双臂 fleet、8 任务
- **模型大小**：基于 π 系 generalist policy（action chunk H=30）
- **算力**：部署期 online RL（DIVL+QAM）；非固定离线预训练 GPU 表
- **世界模型用法**：未使用。
- **提及**：√

## 2026-04-30 · PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations

- **提出日期**：2026-04-30（arXiv 2604.27472）
- **SOTA**：论文称 **多榜 SOTA**（LIBERO / LIBERO-Plus / LIBERO-Pro / SimplerEnv 综合）
- **Benchmark**：LIBERO **98.4%**（全局 #4，距 Top3 +0.2pp）；LIBERO-Plus 81.4%（#4）；LIBERO-Pro 58.8%（与 GLaD 59.47 接近）；SimplerEnv 77.1%
- **Other Benchmark**：14 个真实世界复杂任务套件
- **比 Pi0.5 好**：LIBERO **+0.7pp**；LIBERO-Plus 较 π0 **+25.3pp**；SimplerEnv 较 π0 **+6.1pp**
- **简介**：用 **目标条件对比 RL** 学统一嵌入，内积近似 log 折扣目标占用率（到达概率）；167B token 多任务/具身推理预训练；4B 参数 PRTS-4B。消融：**角色感知因果 mask + 对比目标** 优于纯 BC，长程/接触丰富/zero-shot 新指令任务增益最大。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.27472) · [abs](https://arxiv.org/abs/2604.27472) · [HTML](https://arxiv.org/html/2604.27472) · [GitHub](https://github.com/TeleHuman/PRTS) · [HF Collection](https://huggingface.co/collections/TeleEmbodied/prts) · [HF Data](https://huggingface.co/datasets/TeleEmbodied/PRTS-PostTrain-Data)
- **数据规模**：预训练 **167B tokens** 多样化操作与具身推理数据
- **模型大小**：**PRTS-4B**（Qwen3-VL-4B-Instruct + flow-matching action expert）
- **算力**：**64× H100，约 1 周**（CuTe-FlashAttention + sequence packing）
- **世界模型用法**：未使用。
- **提及**：

## 2026-04-26 · STARRY: Spatio-Temporal Action-Centric World Modeling for Robotic Manipulation

- **提出日期**：2026-04-26（arXiv 2604.26848）
- **SOTA**：**RoboTwin 2.0**：Clean **93.82%** / Random **93.30%**（50 双臂任务，自报 SOTA）
- **Benchmark**：—
- **Other Benchmark**：真机平均成功率 **70.8%**（vs π0.5 42.5%）
- **比 Pi0.5 好**：RoboTwin 较 π0 **+62.5pp**；真机较 **π0.5 +28.3pp**（同论文）
- **简介**：**时空-动作联合扩散** + GASAM（Geometry-Aware Selective Attention Modulation，将预测深度/末端几何转 token-aligned 权重）。消融：联合预测未来 spatial-temporal latent + 动作是 RoboTwin / 真机双涨点核心；几何感知模块对接触丰富任务最有效。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.26848) · [abs](https://arxiv.org/abs/2604.26848) · [HTML](https://arxiv.org/html/2604.26848) · [Cool Papers](https://papers.cool/arxiv/2604.26848)
- **数据规模**：RoboTwin 2.0 50 双臂任务 + 真机实验
- **模型大小**：时空-动作联合扩散 + GASAM（具体骨干见论文）
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：**三阶段时空世界建模**：Stage1 预训练 **ST World Model**（Wan 视频扩散初始化）**+ Understanding Expert**（Qwen-VL 初始化）；Stage2 引入 **Action Expert + Geometry Expert**；Stage3 **时空-动作联合扩散** 与 GASAM 几何调制共训（分支独立扩散步）。消融（Table 4）：**联合预测未来 spatial-temporal latent + 动作** 是核心（Full ST+GASAM **93.30%** vs Action-Only **63.42%** on RoboTwin randomized）；真机 **70.8% vs π0.5 42.5%，+28.3pp**。**正面**。
- **提及**：√

## 2026-04-23 · LoHo-Manip: Long-Horizon Manipulation via Trace-Conditioned VLA Planning

- **提出日期**：2026-04-23（arXiv 2604.21924）
- **SOTA**：**VLABench T1 0.39**（该榜绝对分普遍极低，T1 仍可视为榜首梯队）
- **Benchmark**：—
- **Other Benchmark**：embodied planning / long-horizon reasoning / trajectory prediction / 端到端 manipulation 在仿真和真机 Franka 上的综合实验
- **比 Pi0.5 好**：（无同榜直接对比）
- **简介**：UCSD × NVIDIA。**Task Manager (VLM)**（预测剩余子任务序列 + 视觉 trace 2D keypoint trajectory）+ **Executor (VLA)**（条件化 trace 完成短程控制）。receding-horizon 重新预测，每步 trace 更新。消融：**trace conditioning** 对长程子任务串联最关键；隐式闭环行为是长程鲁棒性的来源。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.21924) · [abs](https://arxiv.org/abs/2604.21924) · [HTML](https://arxiv.org/html/2604.21924v1) · [Site](https://www.liuisabella.com/LoHoManip/)
- **数据规模**：VLABench 长程 + 真机 Franka
- **模型大小**：trace-conditioned 分层 VLA 规划
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：未使用。
- **提及**：

## 2026-04-22 · PokéVLA: Empowering Pocket-Sized VLA with Comprehensive World Knowledge Guidance

- **提出日期**：2026-04-22（arXiv 2604.20834）
- **SOTA**：**LIBERO-Plus T2 83.5%**（全局 #3，仅次于 RLDX-1 86.7 和 OA-WAM 83.9）
- **Benchmark**：LIBERO 98.2%
- **Other Benchmark**：1.22B 参数；推理速度比同类 18×↓ 模型快 12×；2.4M multimodal 预训练样本（spatial grounding / affordance / embodied reasoning）
- **比 Pi0.5 好**：LIBERO **+0.5pp**；LIBERO-Plus 较 π0 **+27.4pp**
- **简介**：**两阶段训练**：先预训练紧凑 VLM（PokeVLM），再注入 manipulation 表征（多视图目标语义学习 + 几何对齐 + 动作专家）。消融：知识蒸馏 / 检索模块对 **LIBERO-Plus 语言与几何扰动** 最有效。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.20834) · [abs](https://arxiv.org/abs/2604.20834) · [HTML](https://arxiv.org/html/2604.20834) · [scirate](https://scirate.com/arxiv/2604.20834)
- **数据规模**：2.4M multimodal 预训练样本（spatial grounding / affordance 等）
- **模型大小**：**1.22B**（Qwen2.5-0.5B VLM + DINO-SigLIP 双视觉编码器 + VLA-Adapter）小参数 VLA + 世界知识引导
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：未使用。
- **提及**：

## 2026-04-19 · VLA Foundry: A Unified Framework for Training VLAs

- **提出日期**：2026-04-19（arXiv 2604.19728）
- **SOTA**：—（框架论文，统一复现 LIBERO / CALVIN / SimplerEnv 等）
- **Benchmark**：—
- **Other Benchmark**：支持 LLM → VLM → VLA 端到端训练；text / image-caption / robotics 多模态；FSDP2 + WebDataset 流式；多 GPU 与 AWS SageMaker
- **比 Pi0.5 好**：（无成功率对比）
- **简介**：Toyota Research Institute 出品。统一 LLM / VLM / VLA 训练管线，可从头训练或 bootstrap from pretrained。消融：共享 pipeline 降低复现方差，使各 benchmark 结果可信对比。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.19728) · [abs](https://arxiv.org/abs/2604.19728) · [HTML](https://arxiv.org/html/2604.19728v1) · [Site](https://tri-ml.github.io/vla_foundry/) · [GitHub](https://github.com/TRI-ML/vla_foundry) · [Notebook](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/training_llm_vlm_vla.ipynb)
- **数据规模**：统一 LIBERO / CALVIN / SimplerEnv 等评测数据管线
- **模型大小**：框架论文（VLM→action expert 端到端）
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-04-19 · Mask World Model (MWM): Predicting What Matters for Robust Robot Policy Learning

- **提出日期**：2026-04-19（arXiv 2604.19683）
- **SOTA**：—（论文 claim LIBERO / RLBench 全面优于 RGB 世界模型基线）
- **Benchmark**：LIBERO 98.3%；RLBench 平均成功率 68.3%（GE-ACT RGB 基线 30.8%，**翻倍以上**）
- **Other Benchmark**：背景变化 / 光照偏移 / 物体颜色变化 鲁棒性测试；纯 RGB 输入即可推理（无需外部分割模型）
- **比 Pi0.5 好**：LIBERO **+0.6pp**（98.3 vs 97.7）
- **简介**：**用语义 mask 替代 RGB 像素** 作为视频世界模型预测目标，形成「几何信息瓶颈」聚焦物理动力学/接触关系。两阶段训练：DiT mask 预测 + diffusion policy。消融：mask 监督减外观噪声；token pruning 鲁棒性显著优于 RGB WM。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.19683) · [abs](https://arxiv.org/abs/2604.19683) · [HTML](https://arxiv.org/html/2604.19683v2) · [scirate](https://scirate.com/arxiv/2604.19683) · [Cool Papers](https://papers.cool/arxiv/2604.19683)
- **数据规模**：LIBERO / RLBench 世界模型训练
- **模型大小**：28 层 DiT 2048 维（mask 预测）+ 28 层 action expert 512 维
- **算力**：见论文 Table 6（Implementation Details）
- **世界模型用法**：**视频 WM（mask 目标）**：用语义 **mask 潜变量** 替代 RGB 像素作 DiT 预测目标，两阶段（mask WM 预训练 + diffusion policy）。消融：相对 RGB WM 基线，RLBench 平均成功率 **68.3% vs 30.8%（约 +37.5pp）**；LIBERO **98.3%（+0.6pp vs π0.5）**；外观扰动鲁棒性显著更好。**正面**。
- **提及**：√

## 2026-04-16 · π0.7: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities

- **提出日期**：2026-04-16（arXiv 2604.15483）
- **SOTA**：—（不跑 LIBERO 等仿真榜，主打真机长程）
- **Benchmark**：—
- **Other Benchmark**：UR5e 双臂叠衣 zero-shot **task progress 85.6% / success 80%**（≈ Top-2% 遥操作员）；14 个未见厨房/卧室指令链显著优于 π0.5 / π0.6；espresso、box assembly、shirt folding 单任务匹配/超越 RL 专项模型
- **比 Pi0.5 好**：未跑 LIBERO；真机长程 **明确优于 π0.5**（论文 Fig.6/7/12）；diverse context conditioning（语言+元数据+子目标图像）支持中途自然语言修正
- **简介**：Physical Intelligence。**MEM 历史编码 + 视觉子目标 + context CFG**；可使用人类 demos / 失败 autonomous data / 非机器人数据。消融：**metadata / eval-data** 两路任意去掉 throughput 显著下降；可在推理时通过新语言指令调整行为。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.15483) · [abs](https://arxiv.org/abs/2604.15483) · [HTML](https://arxiv.org/html/2604.15483) · [PI Blog](https://www.pi.website/blog) · [PI 系 blog index](https://pi.website/) · [openpi GitHub](https://github.com/physical-intelligence/openpi) · [PI download pi07.pdf](https://www.pi.website/download/pi07.pdf)
- **数据规模**：UR5e 双臂叠衣 + 14 个未见厨房/卧室长程真机任务
- **模型大小**：π0.7（MEM + 视觉子目标 + context CFG）
- **算力**：真机长程优于 π0.5/π0.6；未跑 LIBERO
- **世界模型用法**：**BAGEL-14B 图像世界模型独立生成 subgoal 图像**，作为条件输入给 VLA 策略（SuSIE 范式）；VLA 主干为 **Gemma3 4B** + 860M flow-matching action expert（总约 5B），VLA 使用 block-causal masking（非 block-bidirectional）。BAGEL WM 为独立模型，**非与 VLA 共训**。**非** Wan/Cosmos 视频 WAM，而是图像级目标预测。真机长程显著优于 π0.5/π0.6（论文 Fig.6/7/12）。**正面**（独立 WM 生成子目标条件化 VLA）。
- **提及**：√

## 2026-04-16 · ReconVLA: An Uncertainty-Guided and Failure-Aware VLA Framework

- **提出日期**：2026-04-16（arXiv 2604.16677）
- **SOTA**：—（OOD 检测 / 部署安全向，非成功率榜）
- **Benchmark**：—
- **Other Benchmark**：Conformal prediction 给出 token 级校准不确定性；state-space outlier detection
- **比 Pi0.5 好**：（与 π0.5 成功率口径不可直接比）
- **简介**：**Conformal prediction** 给 VLA 动作 token 校准不确定性，状态空间异常检测识别不安全状态；可 plug-and-play 到已训 VLA 上无需重训。消融：实时 OOD 触发回退/求助人类对 fail-safe 部署关键。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.16677) · [abs](https://arxiv.org/abs/2604.16677) · [AAAI-26 Reconstructive 同名](https://ojs.aaai.org/index.php/AAAI/article/view/38921) · [GitHub OpenHelix](https://github.com/OpenHelix-Team/ReconVLA)
- **数据规模**：部署安全 / OOD 检测向
- **模型大小**：Conformal prediction 不确定性框架
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-04-15 · HAMLET: Switch your VLA into a History-Aware Policy

- **提出日期**：2026-04-15（arXiv 2510.00695）
- **SOTA**：**RoboCasa T2 66.4%**（300 demos；#1 Cosmos 67.1 / #3 World2Act 66.3）
- **Benchmark**：—
- **Other Benchmark**：可 plug-in 到任意 VLA backbone；轻量历史记忆模块
- **比 Pi0.5 好**：RoboCasa 较 π0 **+3.9pp**（300 demos：66.4% vs 62.5%）
- **简介**：将现有 VLA **改写为历史感知策略**（frame cache + 记忆融合），不改主干、可即插即用。消融：**历史窗口 + 轻量记忆融合** 优于单帧 baseline；plug-in 后多个 VLA 提升一致。
- **相关资料**：[PDF](https://arxiv.org/pdf/2510.00695) · [abs](https://arxiv.org/abs/2510.00695)
- **数据规模**：plug-in 历史记忆（不改主干训练数据）
- **模型大小**：在 π0 等上约 **2.72B–2.86B** 总参（+轻量记忆模块）
- **算力**：训练 **4× / 2× A100**（见论文实验设置）
- **世界模型用法**：未使用。
- **提及**：

## 2026-04-13 · StarVLA-α: Reducing Complexity in Vision-Language-Action Systems

- **提出日期**：2026-04-13（arXiv 2604.11757）
- **SOTA**：**LIBERO SOTA 98.8%**（Specialist）；**RoboTwin 2.0 T2 88.3%**
- **Benchmark**：RoboCasa-GR1 T2 53.8%；RoboChallenge **33.6%**（Generalist 真机）
- **Other Benchmark**：unified multi-benchmark 训练（LIBERO / SimplerEnv / RoboTwin / RoboCasa）；Lego-style 模块化 backbone-action-head
- **比 Pi0.5 好**：LIBERO **+1.1pp**；RoboTwin 2.0 较 π0 **+22.3pp~+29.9pp**（clean\*/random\* 设定）；论文称 **RoboChallenge 真机较 π0.5 +20.9pp**（33.6% vs 12.7%，绝对提升）
- **简介**：**极简 VLA**：Qwen3-VL backbone + MLP 动作头，统一数据管线。消融：**去掉复杂专用模块仍达 SOTA**，说明强 VLM backbone + 规范训练配方已足够；无需 benchmark-specific 工程。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.11757) · [abs](https://arxiv.org/abs/2604.11757) · [HTML](https://arxiv.org/html/2604.11757v1) · [GitHub](https://github.com/starvla/starvla)
- **数据规模**：LIBERO 8×A100；SimplerEnv/RoboCasa 16×A100；RoboTwin 48×A100；联合训练 64×A100
- **模型大小**：Qwen3-VL-4B + MLP 动作头（Specialist / Generalist）
- **算力**：最多 100k steps；LIBERO 8×A100，联合 64×A100（Table 8）
- **世界模型用法**：未使用。
- **提及**：

## 2026-04-10 · STRONG-VLA: Decoupled Robustness Learning for VLAs under Multimodal Perturbations

- **提出日期**：2026-04-10（arXiv 2604.10055）
- **SOTA**：—（鲁棒性专项榜，非传统成功率 SOTA）
- **Benchmark**：LIBERO（多 VLA 架构上一致提升）
- **Other Benchmark**：**STRONG-VLA 28 种扰动 benchmark**（12 类文本 + 16 类视觉）：π0 +16.49% / +5.58%（seen/unseen）；OpenVLA-OFT +14.48% / +13.81%；OpenVLA +12.60% / +7.77%；AIRBOT 真机验证
- **比 Pi0.5 好**：扰动套件上 **优于标准 π0.5 评测**（基线非 LIBERO 原始榜）
- **简介**：**两阶段**：Stage I 课程扰动学习 + Stage II 干净数据再对齐。消融：Stage I+II 联合对跨架构有效；优于"静态扰动训练"基线。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.10055) · [abs](https://arxiv.org/abs/2604.10055) · [HTML](https://arxiv.org/html/2604.10055v1) · [catalyzex](https://www.catalyzex.com/paper/strong-vla-decoupled-robustness-learning-for)
- **数据规模**：28 种扰动课程 + 干净数据再对齐
- **模型大小**：可 plug-in OpenVLA-7B / OpenVLA-OFT / π0
- **算力**：两阶段 LoRA / 全参微调（未统一 GPU 表）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-04-09 · HiF-VLA: Hindsight, Insight and Foresight through Motion Representation

- **提出日期**：2026-04-09（arXiv 2512.09928）
- **SOTA**：—
- **Benchmark**：CALVIN 4.35；LIBERO **98.0%** 平均（Long 96.4%）（未进全局 Top3）
- **Other Benchmark**：三向运动表征；vision-only motion tokens
- **比 Pi0.5 好**：LIBERO 平均 **+0.3pp**（98.0% vs 97.7%）；CALVIN 较 π0 **+0.43**
- **简介**：用 **运动表征** 统一后见（hindsight）/ 当下（insight）/ 前瞻（foresight）三视角动作建模；以可见 motion tokens 替代隐式动作意图。消融：**三向运动 token** 对 CALVIN 长链最有效；hindsight 单一视角对短任务收益少。
- **相关资料**：[PDF](https://arxiv.org/abs/2512.09928) · [abs](https://arxiv.org/abs/2512.09928)
- **数据规模**：OpenVLA 初始化 + OXE 预训练权重
- **模型大小**：Prismatic-7B VLM backbone
- **算力**：**8× A100**，global batch 64；LIBERO 150k / CALVIN 80k steps
- **世界模型用法**：**运动空间前瞻预测（Foresight）**：预测未来运动向量（H.264 宏块位移），论文自称 "motion-centric world model"；辅助训练损失，推理时可选跳过。消融（Table 3）：Foresight 单独 **+1.2pp** SR（91.0→92.2%），叠加 Hindsight 达 93.2%；推理仅 1.13× 延迟。**正面**（轻量运动级 WM，非像素/视频生成）。
- **提及**：

## 2026-04-07 · HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents

- **提出日期**：2026-04-07（arXiv 2604.07430）
- **SOTA**：—（22 个 embodied benchmark 与 Gemini 3.0 Pro 持平，但非单榜 SOTA）
- **Benchmark**：—
- **Other Benchmark**：MoT-2B 在 16 benchmarks 上超过同等大小 SOTA；32B 与 Gemini 3.0 Pro comparable；>100M embodied / spatial 数据，>200B tokens；on-policy 蒸馏从 32B → 2B
- **比 Pi0.5 好**：（22 榜套件无 π0.5 逐榜公开）
- **简介**：**Mixture-of-Transformers (MoT)** 边云协同：2B 边端 (4B 总参，2.2B 激活) + 32B 云端；latent token 模态特化计算。消融：MoT 分规模部署有效；自我演化 on-policy 蒸馏对 2B 紧凑模型保留 32B 能力关键。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.07430) · [abs](https://arxiv.org/abs/2604.07430) · [HTML](https://arxiv.org/html/2604.07430v1) · [GitHub](https://github.com/Tencent-Hunyuan/HY-Embodied) · [emergentmind](https://www.emergentmind.com/papers/2604.07430)
- **数据规模**：>100M embodied/spatial，>200B tokens
- **模型大小**：MoT-2B（4B 总参，2B 激活）+ MoE-A32B（407B 总参，32B 激活）
- **算力**：32B→2B on-policy 蒸馏；GPU 细节见正文
- **世界模型用法**：未使用。
- **提及**：√

## 2026-04-06 · HiPolicy: Hierarchical Multi-Frequency Action Chunking for Policy Learning

- **提出日期**：2026-04-06（arXiv 2604.06067）
- **SOTA**：—（论文报告显著优于单层 chunking，但无统一公开榜 Top3 百分比）
- **Benchmark**：RoboTwin 高精度任务相对提升 **+105%**；简单任务 +34%；总体 37-41% → 59-60%
- **Other Benchmark**：可 plug-in 到 2D / 3D 生成式 policy
- **比 Pi0.5 好**：（无同榜直接对比）
- **简介**：**多频率分层 action chunk**：高频 branch（短链精控）+ 低频 branch（长链规划）+ **action-entropy guided adaptive execution**（低熵执行高频 / 高熵执行低频）。消融：分层频率对齐长程与高频反应；entropy gating 是动态切换关键。
- **相关资料**：[PDF](https://arxiv.org/pdf/2604.06067) · [abs](https://arxiv.org/abs/2604.06067) · [Site](https://hipolicy.github.io/) · [GitHub](https://github.com/HiPolicy/HiPolicy)
- **数据规模**：RoboTwin / Robomimic 仿真 + 真机
- **模型大小**：分层多频率 action chunk（plug-in policy）
- **算力**：batch 128，AdamW（见 Table D.1）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-03-31 · From Human Skill to Robotic Mastery (Psi-R2 / Psi-W0 技术博客，灵初智能)

- **提出日期**：2026-03-31（技术博客）
- **SOTA**：—（无统一公开仿真榜 Top3）
- **Benchmark**：—
- **Other Benchmark**：Psi-R2 预训练 95,472h 人类 + 5,417h 真机；Psi-W0 作 action-conditioned WM 做策略评估；<100 条真机轨迹微调长程精细任务；首批 1,000 小时多模态人手数据集开源（总计 100,000h）
- **比 Pi0.5 好**：（博客强调人类数据规模化 + WM 评估闭环，无与 π0.5 直接百分比对照）
- **简介**：**Psi-R2**：视频-动作联合 WAM（基于 Wan2.2）；**Psi-W0**：动作条件世界模型用于策略评估与数据质检；"Raw Data In" 极简人类/真机对齐管线；3D 轨迹精度通过定制外骨骼手套达到亚毫米。
- **相关资料**：[Blog](https://cypypccpy.github.io/tech-blog.github.io/) · [Psi-Lab Site](https://psi-lab.ai/Psi0) · [chinaz coverage](https://www.chinaz.com/tags/908189.shtml)
- **数据规模**：Psi-R2：95,472h 人类 + 5,417h 真机；开源 1,000h 人手（目标 100k h）
- **模型大小**：Psi-R2（Wan2.2 WAM）+ Psi-W0
- **算力**：技术博客，无正式训练表
- **世界模型用法**：**视频-动作联合 WAM（Psi-R2，Wan2.2-IT2V-5B-480P）** + **动作条件 WM（Psi-W0）**：R2 联合预测未来视频帧与动作；W0 **替代传统仿真器做 RL 微调**（rollout + RL 飞轮），兼策略评估与数据质检。核心创新：**95K+ h 人类数据 + 5.4K h 机器人数据**（首个万小时级预训练）；W0 集成**触觉模态**做预测目标（mask 训练策略）。推理优化至 **<100ms**（DiT 缓存 + 量化）。博客称 WM 闭环为长程精细任务关键；**无与 π0.5 同设定成功率表**。**正面**（工程叙事）。
- **提及**：

## 2026-03-30 · FocusVLA: Focused Visual Utilization for VLAs

- **提出日期**：2026-03-30（arXiv 2603.28740）
- **SOTA**：—（LIBERO 98.7% 全局 #4，距 Top3 ≈ +0.1pp）
- **Benchmark**：LIBERO 98.7%（0.5B 参数；多权重平均）；OpenVLA-OFT 7B 仅 97.1%
- **Other Benchmark**：训练收敛 1.5× 加速（LIBERO-Spatial 5×）；attention map 显示集中于接触/操作目标
- **比 Pi0.5 好**：LIBERO **+1.0pp**（98.7 vs 97.7）
- **简介**：**Modality Cascaded Attention**（消除"绕过视觉"的结构捷径）+ **Focus Attention**（patch-level pruning + channel-level gating）。核心发现：VLA 性能瓶颈在 **如何使用视觉信息**，而非视觉表征本身。消融：换 backbone（DINOv2+SigLIP / VLM / VGGT 3D）后利用率修复显著提升 → 结构问题非感知问题。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.28740) · [abs](https://arxiv.org/abs/2603.28740) · [HTML](https://arxiv.org/html/2603.28740v1) · [tektonian VLA Arena](https://tektonian.com/vla-benchmark)
- **数据规模**：LIBERO + RoboTwin 评测数据
- **模型大小**：Qwen2.5-0.5B VLM；可训练 **342.7M**（VLM 103.6M + Policy 239.1M）
- **算力**：LIBERO：**4× A100** bs=64；RoboTwin：**8× A100** bs=64
- **世界模型用法**：未使用。
- **提及**：

## 2026-03-26 · Beyond Textual Knowledge (BTK): Leveraging Multimodal Knowledge Bases for Enhancing VLN

- **提出日期**：2026-03-26（arXiv 2603.26859）
- **SOTA**：**R2R Test Unseen SR 74 / SPL 63**（top 报告）；**REVERIE Test Unseen RGS 35.08 / RGSPL 25.23**
- **Benchmark**：R2R Val Unseen SR 75 / SPL 64；REVERIE Val Unseen RGS 34.82 / RGSPL 24.86
- **Other Benchmark**：R2R_GP / REVERIE_GP 图像知识库（Flux-Schnell 生成）；BLIP-2 文本知识库
- **比 Pi0.5 好**：（VLN 离散导航榜，π0.5 主要在机器人操作/真机长程，不可直接比较）
- **简介**：**Qwen3-4B** 抽取目标短语（优于 SpaCy）+ **Flux-Schnell** 生成目标视觉知识库 + **BLIP-2** 构建环境文本知识库；Goal-Aware Augmentor / Knowledge Augmentor 注入。消融：Qwen3-4B > SpaCy 抽取；图像+文本联合时 REVERIE Val Unseen SPL/RGSPL 最佳。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.26859) · [abs](https://arxiv.org/abs/2603.26859) · [HTML](https://arxiv.org/html/2603.26859v1) · [GitHub](https://github.com/yds3/IPM_BTK/)
- **数据规模**：R2R 7,189 轨迹；REVERIE 21,702 指令；知识库 R2R ~93K / REVERIE ~50K 图像
- **模型大小**：DUET + Qwen3-4B 短语抽取 + Flux 知识库
- **算力**：知识库构建：A40（R2R 文本 120h；图像 60h×2 / 63h）；微调单卡 A40
- **世界模型用法**：未使用。
- **提及**：

## 2026-03-26 · VLA-OPD: Bridging Offline SFT and Online RL for VLAs via On-Policy Distillation

- **提出日期**：2026-03-26（arXiv 2603.26666）
- **SOTA**：—（作为 π0.5/OFT 后训练叠加层）
- **Benchmark**：LIBERO + RoboTwin 2.0（采样效率 >> 在线 RL，>> SFT；论文 Table 1）；RoboTwin VLA-OPD (Distill) avg **71.1%**
- **Other Benchmark**：—
- **比 Pi0.5 好**：可叠加在 π0.5 / OFT 后训练上；LIBERO 绝对值需看起点 checkpoint
- **简介**：**Reverse-KL on-policy 蒸馏**（区别于 Forward-KL mode-covering / Hard CE entropy collapse）；用专家教师对学生自生成轨迹做 dense token-level 监督。消融：比 PPO / 纯 SFT 更抗灾难性遗忘，组合 RL 少样本 + SFT 快收敛优势。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.26666) · [abs](https://arxiv.org/abs/2603.26666) · [HF Paper](https://huggingface.co/papers/2603.26666) · [scirate](https://scirate.com/arxiv/2603.26666)
- **数据规模**：LIBERO + RoboTwin 后训练（on-policy 蒸馏）
- **模型大小**：可叠加 π0.5 / OFT checkpoint
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-03-24 · ELITE: Experiential Learning and Intent-Aware Transfer for Self-improving Embodied Agents

- **提出日期**：2026-03-24（arXiv 2603.24018）
- **SOTA**：**EmbodiedBench 子榜**：EB-ALFRED 在线无监督 Avg **61%**（Qwen2.5-VL-72B，较 base +9pp）；EB-ALFRED 监督设置 **70.8%**（高于 Claude-3.5-Sonnet 66.4 / ERA 65.2）
- **Benchmark**：EB-Habitat 在线无监督 Avg **67%**（InternVL3-78B，+5pp）
- **Other Benchmark**：CoT 检索在长程任务进度上优于 TF-IDF / 随机检索
- **比 Pi0.5 好**：（EmbodiedBench 是离散 embodied-agent 评测，与 π0.5 真机长程任务形态不同，不可直接换算）
- **简介**：让 embodied agent 从执行轨迹中提炼经验，**intent-aware retrieval** 按过程意图检索相似策略；self-reflective knowledge construction 维护可演化策略池。消融：去掉 Intent-Aware Retrieval 后 EB-ALFRED Avg 61 → 56；去掉 Context Consolidation → 55。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.24018) · [abs](https://arxiv.org/abs/2603.24018) · [HTML](https://arxiv.org/html/2603.24018v1) · [Gist.Science](https://gist.science/paper/2603.24018) · [EmbodiedBench](https://embodiedbench.github.io/)
- **数据规模**：EmbodiedBench（EB-ALFRED / EB-Habitat）；无从头训练 VLA
- **模型大小**：基于 Qwen2.5-VL-72B 或 InternVL3-78B（推理+经验池）
- **算力**：无大规模预训练算力（在线/监督设置）
- **世界模型用法**：未使用。
- **提及**：

## 2026-03-17 · P3Nav: End-to-End Perception, Prediction and Planning for VLN

- **提出日期**：2026-03-17（arXiv 2603.17459）
- **SOTA**：**REVERIE SOTA**：Test Unseen SR 60.06 / SPL 40.57 / RGS 39.75 / RGSPL 26.56；**R2R-CE SOTA**：Val Unseen SR 62 / SPL 52；**RxR-CE SOTA**：Val Unseen SR 58.01 / SPL 47.92 / nDTW 64.29 / SDTW 48.04
- **Benchmark**：—
- **Other Benchmark**：—
- **比 Pi0.5 好**：（VLN/VLN-CE 榜无 π0.5 公开）
- **简介**：HKUST(GZ)。统一端到端框架：感知（对象+地图级互补线索）→ 预测（waypoint + 语义地图）→ 规划。消融：对象解码 / 地图 / waypoint 预测等中间模块都提升 REVERIE / R2R-CE / RxR-CE，"先显式理解场景再规划"优于 planning-only。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.17459) · [abs](https://arxiv.org/abs/2603.17459) · [HTML](https://arxiv.org/html/2603.17459v1)
- **数据规模**：R2R / REVERIE / RxR-CE / R2R-CE 导航数据
- **模型大小**：端到端 P3Nav（感知+预测+规划）
- **算力**：预训练 200k iter，**4× RTX 4090** bs=12；微调 50k iter
- **世界模型用法**：未使用。
- **提及**：

## 2026-03-17 · GigaWorld-Policy: An Efficient Action-Centered World–Action Model

- **提出日期**：2026-03-17（arXiv 2603.17240）
- **SOTA**：**RoboTwin 2.0** 较 π0.5 **+95%**（如 Place Fan 0.25 → 0.94）
- **Benchmark**：—
- **Other Benchmark**：推理 9× 加速 vs Motus；success +7%；单次推理仅 0.36s；35% 任务成功率提升
- **比 Pi0.5 好**：RoboTwin 2.0 远超 π0.5（**+95%** 相对提升）
- **简介**：**因果 WAM**：动作预测 + action-conditioned 视频生成共训；causal mask 防 future-video token 影响 action token，使推理时可跳过显式视频生成。消融：**不强制每步生成视频** 仍保 SOTA 且更快；轻量 GigaWorld-0.5 backbone 是预训练核心。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.17240) · [abs](https://arxiv.org/abs/2603.17240) · [HTML](https://arxiv.org/html/2603.17240) · [Site](https://gigaai-research.github.io/GigaWorld-Policy/) · [HF Paper](https://huggingface.co/papers/2603.17240)
- **数据规模**：GigaWorld 预训练 + RoboTwin 评测
- **模型大小**：**Wan 2.2 5B** diffusion Transformer 动作中心 WAM
- **算力**：推理 latency 在 A100 上评测（见论文 Tab.3）
- **世界模型用法**：**因果 WAM（Wan2.2-5B）**：动作预测与 **action-conditioned 视频 共训**；因果 mask 使推理可 **跳过显式视频生成**（Fast-WAM 类思路）。消融：不强制每步想象仍竞争力强；RoboTwin 均值 **0.86 vs π0.5 ~0.43（约 2× 提升）**；推理 **9× 加速**（360ms vs Motus 3231ms）。Motus 略高（0.87-0.89）但慢 9×。**正面**。
- **提及**：√

## 2026-03-16 · MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation

- **提出日期**：2026-03-16（arXiv 2603.16861）
- **SOTA**：—（自建 MolmoBot Humanoid VLA Benchmark）
- **Benchmark**：—
- **Other Benchmark**：**真机 4 settings** Overall **79.2%**（vs π0.5 39.2%）；**DROID 仿真** 7 任务平均 **64.1%**（vs π0.5 10.0%）；MolmoBot-Data 1.7M 轨迹 / 94k+ 场景；MolmoBot-Engine 程序化数据生成
- **比 Pi0.5 好**：真机 pick-place **+40pp**（≈2× π0.5）；纯仿真训练 **匹敌/超越** 大规模真机 VLA
- **简介**：**纯仿真 MolmoBot-Engine**（MolmoSpaces 程序化生成 232K 环境，48K 可操作物体，8 任务类型）+ **Molmo2 VLM + flow 动作头**；含 MolmoBot-Pi0 / SPOC 变体；MolmoBot-DROID 仅需腕+外视摄像头。消融：3DGS / 高保真场景是 zero-shot Sim2Real 成功的主要因素。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.16861) · [abs](https://arxiv.org/abs/2603.16861) · [Site](https://allenai.github.io/MolmoBot/) · [Code](https://github.com/allenai/MolmoBot) · [MolmoSpaces](https://github.com/allenai/molmospaces) · [HF Models](https://huggingface.co/collections/allenai/molmobot-models) · [HF Data](https://huggingface.co/datasets/allenai/MolmoBot-Data) · [Demo nb](https://github.com/allenai/MolmoBot/blob/main/MolmoBot/demo_policy.ipynb) · [Blog](https://allenai.org/blog/molmobot-robot-manipulation)
- **数据规模**：**MolmoBot-Data 1.7M 轨迹 / 94k+ 场景**（纯仿真）
- **模型大小**：Molmo2 VLM + flow 头（MolmoBot-Pi0 / SPOC 变体）
- **算力**：仿真数据引擎为主；无大规模真机预训练 GPU 公开表
- **世界模型用法**：未使用。
- **提及**：√

## 2026-03-16 · Fast-WAM: Do World Action Models Need Test-time Future Imagination?

- **提出日期**：2026-03-16（arXiv 2603.16666）
- **SOTA**：**RoboTwin 2.0 SOTA 91.83%**（论文报告 89.5% 在另一统计口径下）；LIBERO 97.6%
- **Benchmark**：—
- **Other Benchmark**：190 ms 单 GPU 延迟（4× 加速 vs imagine-then-execute WAM）；真实毛巾折叠任务无具身预训练即可达成
- **比 Pi0.5 好**：LIBERO **+0.7pp**（97.6% vs π0.5 96.9%）；RoboTwin 2.0 较 π0 **+29.6pp**（91.8% vs 62.2%）
- **简介**：Wan2.2-5B 视频 DiT backbone + 1B action expert；shared-attention MoT 结构化注意力 mask 解耦 video-cotraining 与 action 生成。**关键发现**：训练时视频 co-training 比测试时显式 future imagination 更重要。消融：单遍动作头 + 流匹配是 RoboTwin 高分关键。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.16666) · [abs](https://arxiv.org/abs/2603.16666) · [HTML](https://arxiv.org/html/2603.16666) · [Site](https://yuantianyuan01.github.io/FastWAM) · [GitHub](https://github.com/yuantianyuan01/FastWAM) · [HF Paper](https://huggingface.co/papers/2603.16666)
- **数据规模**：LIBERO + RoboTwin 2.0 + 真机 Galaxea 60h 毛巾折叠
- **模型大小**：Wan2.2-5B + **1B action expert，总计 6B**
- **算力**：训练细节见正文；推理 **单卡 RTX 5090D 32GB**
- **世界模型用法**：**Wan2.2-5B 视频 DiT 作 world modeling backbone**，MoT 共享注意力 + 1B action expert；**训练时 video co-training**，推理 **移除 future video 分支**（不做 test-time imagination）。核心结论：**训练共训 > 测试时想象**；RoboTwin **91.8%**（无 embodied pretraining 最强，LingBot-VA 预训练版 92.2% 略高），推理 **190ms（RTX 5090D）、约 4× 加速** vs imagine-then-execute 变体。**正面**（共训有效、显式想象非必需）。
- **提及**：

## 2026-03-13 · SmoothVLA: Aligning VLAs with Physical Constraints via Intrinsic Smoothness Optimization

- **提出日期**：2026-03-13（arXiv 2603.13925）
- **SOTA**：—（平滑度专项指标）
- **Benchmark**：LIBERO 平滑度较标准 RL **+13.8%**（jerk-based reward）
- **Other Benchmark**：—
- **比 Pi0.5 好**：（绝对成功率视起点 checkpoint，非主指标）
- **简介**：**GRPO + jerk 连续 reward**（physics-informed hybrid reward = 稀疏二值任务奖励 + 轨迹 jerk 内在密集项）。消融：物理平滑 reward 减抖动；不依赖外部 reward engineering 是工程优势。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.13925) · [abs](https://arxiv.org/abs/2603.13925) · [scirate](https://scirate.com/arxiv/2603.13925) · [OpenReview PDF](https://openreview.net/pdf/01fe44202488e15cf8ca2dce6425ca09e58bbdd5.pdf)
- **数据规模**：LIBERO 监督 + 偏好学习
- **模型大小**：OpenVLA + LoRA
- **算力**：同硬件环境对比实验（未单独列 GPU 数）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-03-12 · Ψ0 (Psi-Zero): An Open Foundation Model Towards Universal Humanoid Loco-Manipulation

- **提出日期**：2026-03-12（arXiv 2603.12263）
- **SOTA**：—（人形 loco-manipulation，新方向）
- **Benchmark**：—
- **Other Benchmark**：800h 人类视频 + 30h 真机 → **较 10× 数据基线 +40%**；**80 条轨迹** 可学长程新技能；EgoDex 829h + Humanoid Everyday 31h（260 任务）
- **比 Pi0.5 好**：整体成功率较大数据 VLA 基线（含 π0.5 系）**+40%** 相对提升
- **简介**：USC Physical Superintelligence Lab。**世界模型蒸馏** + 两阶段（人类视频自回归预训练 → 高质量人形数据 flow expert post-train）；Qwen3-VL-2B + 500M MMDiT + RL 跟踪控制器。消融：**人类视频预训** 是数据效率关键；Stage 解耦解决人-机运动学差异。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.12263) · [abs](https://arxiv.org/abs/2603.12263) · [Site](https://psi-lab.ai/Psi0) · [GitHub](https://github.com/physical-superintelligence-lab/Psi0) · [HF Paper](https://huggingface.co/papers/2603.12263)
- **数据规模**：EgoDex 829h 人类 + Humanoid Everyday 31h（260 任务）；预训练 800h 人类 + 30h 真机
- **模型大小**：Qwen3-VL-2B + **~500M** MMDiT action expert
- **算力**：预训练 **64×A100×10 天** bs=1024；后训练 **32×A100×30h** bs=2048
- **世界模型用法**：人类视频阶段为 **next-action 自回归预训练**（论文定位为学习 "generalizable visual-action representations"），**非像素/视频生成 WM**，论文未使用 "world model" 或 "知识蒸馏" 术语；后训练为 ~500M flow MM-DiT action expert。消融：人类视频预训是数据效率关键；约 **829h EgoDex + 31h Humanoid Everyday** 较大数据 VLA **+40%** 相对提升（vs GR00T-N1.6）。**正面**（表征学习，非经典 WAM）。
- **提及**：√

## 2026-03-11 · World2Act: Latent Action Post-Training via Skill-Compositional World Models

- **提出日期**：2026-03-11（arXiv 2603.10422）
- **SOTA**：**RoboCasa T2 66.3%**（搭配 Cosmos Policy）
- **Benchmark**：LIBERO 平均 **98.1%**（基线 97.0）
- **Other Benchmark**：真机 +6.7%（pick & place / pick bowl / close drawer）；GR00T-N1.6-ft 提升 +2.5%
- **比 Pi0.5 好**：RoboCasa 较 π0 **+3.8pp**（66.3 vs 62.5）
- **简介**：在 **Cosmos 世界模型** 上做技能组合潜动作后训练；contrastive matching 把 VLA 动作直接对齐 WM 的 video-dynamics latent，**避开像素监督的伪影 / 幻觉**。LLM-based skill decomposition 切分高阶指令到低阶 prompt 应对变长 horizon；产出 RoboCasa-Skill / LIBERO-Skill 数据。消融：**技能组合世界模型** 优于单任务后训练；contrastive latent 对齐优于 RGB 监督。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.10422) · [abs](https://arxiv.org/abs/2603.10422) · [HTML](https://arxiv.org/html/2603.10422) · [Site](https://wm2act.github.io/) · [HF Paper](https://huggingface.co/papers/2603.10422)
- **数据规模**：RoboCasa-Skill + LIBERO-Skill（Cosmos 技能组合）
- **模型大小**：Cosmos-Predict2 等视频 WM 骨干（LoRA rank 8–128 / 2B full FT）
- **算力**：WM 微调 **8× AMD MI210（48GB）**
- **世界模型用法**：**Cosmos 技能组合 WM 作后训练骨干**：在 Cosmos-Predict2 上技能分解 + **contrastive 将 VLA 动作对齐 WM video-dynamics latent**（避免 RGB 幻觉）。消融：技能组合 WM > 单任务后训练；latent 对齐 > RGB 监督；LIBERO **98.1%（+1.1pp）**；真机 **+6.7%**；RoboCasa T2 **66.3%**。**正面**。
- **提及**：

## 2026-03-11 · FutureVLA: Joint Visuomotor Prediction for VLA

- **提出日期**：2026-03-11（arXiv 2603.10712）
- **SOTA**：—（未进全局 Top3）
- **Benchmark**：LIBERO 98.3%；LIBERO-Plus 81.3%；SimplerEnv 75.87%
- **Other Benchmark**：SimplerEnv +11.4%（相对基线）；真机操作 +21.7%
- **比 Pi0.5 好**：LIBERO **+0.6pp**；LIBERO-Plus 较 π0 **+25.2pp**
- **简介**：**联合视觉-运动预测**：Joint Visuomotor Predictive Architecture，**Joint Visuomotor Gating** 结构性分离视觉状态保留与时序动作建模；**latent embeddings alignment** 后训练可 plug-in 到多种 VLA。消融：Joint Visuomotor Gating + 潜对齐后训练对多 VLA 通用。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.10712) · [abs](https://arxiv.org/abs/2603.10712) · [HTML](https://arxiv.org/html/2603.10712) · [GitHub](https://github.com/xuxiaoxxxx/FutureVLA)
- **数据规模**：LIBERO / LIBERO-Plus / SimplerEnv + 真机
- **模型大小**：**4B** 级联合视觉-运动架构
- **算力**：**4× A100**（见实验设置）
- **世界模型用法**：**联合潜空间视觉-运动预测（非像素视频 WM）**：WAN 2.2 3D-VAE 编码视频 clip 为时序 token + Joint Visuomotor Gating（门控交叉注意力）；可 plug-in 多种 VLA（OFT/GR00T 风格）。视觉流重建首帧潜表征（静态锚点），非预测未来帧。消融：gating + 潜对齐后训练通用有效；LIBERO **98.3% vs π0 94.2%（+4.1pp）**（论文未比 π0.5）；SimplerEnv **+11.4%**；真机 **+21.7%**。**正面**（类动力学辅助，非 WAM 命名）。
- **提及**：

## 2026-03-10 · TiPToP: A Modular Open-Vocabulary Planning System for Robotic Manipulation

- **提出日期**：2026-03-10（arXiv 2603.09971）
- **SOTA**：—（DROID 真机 + 仿真自建 28 任务套件）
- **Benchmark**：—
- **Other Benchmark**：**28 任务 / 165+ trial（仿真+真机）总成功率 59.4%**（vs π0.5-DROID 33.3%）；任务进度 74.6%（vs 52.4%）；语义/干扰物子集 62.4%（vs 25.9%）；多步子集 57.5%（vs 15%）；**零机器人演示数据**
- **比 Pi0.5 好**：DROID 设定真机+仿真显著优于 π0.5-DROID（成功率 +26pp，进度 +22pp）
- **简介**：**TAMP（cuTAMP）+ 现成 VFM 感知**（Gemini-ER、SAM-2、M2T2、FoundationStereo）；模块化 Perception / Planning / Execution。**< 1h 部署到标准 DROID**；可独立换感知/规划。消融：模块化设计 → 跨任务/环境/本体 compositional 泛化无需 task-specific 数据。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.09971) · [abs](https://arxiv.org/abs/2603.09971) · [HTML](https://arxiv.org/html/2603.09971) · [Site](https://tiptop-robot.github.io/) · [Code](https://github.com/tiptop-robot/tiptop) · [Docs](https://tiptop-robot.readthedocs.io/en/latest/)
- **数据规模**：**零机器人演示**；28 任务 DROID 设定
- **模型大小**：模块化 TAMP + VFM（非端到端单模型参数量）
- **算力**：规划超时 30–60s；无统一训练 GPU
- **世界模型用法**：未使用。
- **提及**：

## 2026-03-10 · GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware VLAs

- **提出日期**：2026-03-10（arXiv 2603.09079）
- **SOTA**：—（SimplerEnv 80.2% 为 2026 高位但低于全局 Top3 86.6%）
- **Benchmark**：SimplerEnv **80.2%**（基线 +5.4%）；LIBERO 96.4%（基线 +2.0%）
- **Other Benchmark**：300M 参数 flow-matching action expert + MoE FFN
- **比 Pi0.5 好**：SimplerEnv 较 π0 **+9.2pp**；LIBERO 较 π0.5 **-1.3pp**
- **简介**：**Gaussian Spatial Tokenizer (GST)**：将冻结的 dense depth + semantic patch 转为 128 个各向异性 3D Gaussian primitives（μ / log σ / α）；**Depth-Aware Chain-of-Thought (DA-CoT)**：监督 4 类中间空间思维（3D grounding / grasp affordance / 距离 / SE(3) waypoint）。消融：深度感知 token 对相机扰动鲁棒性提升最大；DA-CoT 监督集中于精度敏感任务。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.09079) · [abs](https://arxiv.org/abs/2603.09079) · [HF Paper](https://huggingface.co/papers/2603.09079) · [3DGS Papers index](https://github.com/Awesome3DGS/3D-Gaussian-Splatting-Papers/blob/main/abs/2603.09079.md)
- **数据规模**：LIBERO / SimplerEnv
- **模型大小**：**300M** flow-matching action expert + MoE FFN
- **算力**：**8×A100-80GB**（Stage 1 训练）
- **世界模型用法**：未使用。
- **提及**：

## 2026-03-10 · NS-VLA: Towards Neuro-Symbolic VLAs

- **提出日期**：2026-03-10（arXiv 2603.09542）
- **SOTA**：**CALVIN T3 4.72**（#1 Xiaomi 4.75）
- **Benchmark**：LIBERO Full 98.6% (vs prev best 97.3)；LIBERO 1-shot 69.1%；LIBERO-Plus 79.4%；CALVIN 5-Task 91.2%（vs 80.0）
- **Other Benchmark**：2B 参数；primitive vocabulary（pick / place_in / place_on / close 等）
- **比 Pi0.5 好**：CALVIN 优于 π0 (+0.8pp 相对 4.75 SOTA)；LIBERO 1-shot 69.1% 低于 π0.5 LIBERO 97.7%（不同设定）
- **简介**：**Symbolic Encoder + Symbolic Solver + Online RL (GRPO with primitive-segmented rewards)**。one-shot 训练、数据扰动、零样本泛化、高数据效率。消融：**符号规划器 + VLA 执行** 对 CALVIN 多步任务有效；Solver 视觉 token 稀疏化是数据高效关键。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.09542) · [abs](https://arxiv.org/abs/2603.09542) · [HTML](https://arxiv.org/html/2603.09542) · [GitHub](https://github.com/Zuzuzzy/NS-VLA) · [HF Data](https://huggingface.co/datasets/zuzuzzy/NS-VLA-Dataset) · [alphaXiv](https://www.alphaxiv.org/overview/2603.09542)
- **数据规模**：LIBERO 全量 / 1-shot；primitive vocabulary
- **模型大小**：**2B**；Symbolic Encoder + Solver + GRPO
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：未使用。
- **提及**：

## 2026-03-09 · SACA: Step-Aware Contrastive Alignment for VLN-CE

- **提出日期**：2026-03-09（arXiv 2603.09740）
- **SOTA**：**VLN-CE SOTA**：R2R-CE Val Unseen SR **60.3** / SPL **55.1**（无额外数据）；R2R-CE +额外数据 SR **64.7** / SPL **56.9**；RxR-CE Val Unseen SR **60.3** / SPL **49.8** / nDTW **62.1**；RxR-CE +额外数据 SR 62.1 / SPL 51.7 / nDTW 66.0
- **Benchmark**：—
- **Other Benchmark**：相对 StreamVLN R2R-CE SR +7.5pp / RxR-CE SR +11.7pp
- **比 Pi0.5 好**：（VLN-CE 是 Habitat 连续导航榜，π0.5 不在该榜公开）
- **简介**：**Perception-Grounded Step-Aware Auditor** 给失败轨迹分配逐步奖励（识别 divergence point，将失败轨迹拆 valid prefix + failure point）；**Scenario-Conditioned Group Construction** 动态路由 mixed/all-failure batch 到 Repair Resampling / All-Failure Rescue。消融：Soft Score、All-Failure Rescue、Repair Resampling 三者合用最佳。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.09740) · [abs](https://arxiv.org/abs/2603.09740) · [alphaXiv](https://www.alphaxiv.org/overview/2603.09740v1) · [bytez](https://bytez.com/docs/arxiv/2603.09740/paper)
- **数据规模**：VLN-CE R2R / RxR
- **模型大小**：Video-LLM 初始化（如 LLaVA-Video-8B）+ SFT/RFT
- **算力**：SFT **8× A6000 ~36h**；RFT **~24h/epoch**
- **世界模型用法**：未使用。
- **提及**：

## 2026-03-01 · EZ-M: Scaling Tasks, Not Samples - Mastering Humanoid Control through Multi-Task Model-Based RL

- **提出日期**：2026-03-01（arXiv 2603.01452）
- **SOTA**：**HumanoidBench-Hard SOTA**：14 任务中 10 个达 SOTA（含 h1hand-walk 919.96 / run 818.40 / pole 841.35 / reach 5353.09）
- **Benchmark**：HumanoidBench-Medium：9 任务中 7 达 SOTA 或超 success score
- **Other Benchmark**：仅 1M env steps、16M 参数（vs BRC 1B），训练 ~4× 快；task diversity 作为 dynamics regularizer
- **比 Pi0.5 好**：（π0.5 不是 HumanoidBench 在线 RL 控制基线）
- **简介**：**EfficientZero-Multitask**：扩任务数而非单任务样本数；shared world model + Gumbel search；task-sharing 架构 + 学习任务嵌入 + path consistency + temporal consistency + Independent Experience Replay。消融：Independent Experience Replay 关键；shared dynamics/reward/value 在相关任务间梯度相似度更高（解释跨任务正迁移）。
- **相关资料**：[PDF](https://arxiv.org/pdf/2603.01452) · [abs](https://arxiv.org/abs/2603.01452) · [HTML](https://arxiv.org/html/2603.01452) · [Site](https://yewr.github.io/ez_m/) · [GitHub](https://github.com/liushaohuai5/EfficientZero_Multitask) · [HumanoidBench](https://humanoid-bench.github.io/)
- **数据规模**：HumanoidBench 多任务在线 RL（扩任务非扩样本）
- **模型大小**：**16M** 参数 world model（vs BRC 1B）
- **算力**：Medium 套件约 **10h×2×A40**（vs BRC 40h 单卡 A40）
- **世界模型用法**：**模型-based RL 共享世界模型（16M）**：多任务 HumanoidBench 用 compact **dynamics/reward/value world model** + Gumbel search（非视频生成）。消融：Independent Experience Replay 关键；Hard 榜 **10/14 任务 SOTA**；样本效率优于 BRC 1B model-free baseline（BRC 非 WM）。**正面**（控制用 WM，非 WAM）。
- **提及**：

## 2026-02-XX · WoVR: World Models as Reliable Simulators for Post-Training VLAs

- **提出日期**：2026-02（arXiv 2602.13977）
- **SOTA**：—
- **Benchmark**：LIBERO **39.95% → 69.2%（+29.3pp）** 后训练增益
- **Other Benchmark**：真机 61.7% → 91.7%（+30.0pp）
- **比 Pi0.5 好**：LIBERO 仍 **低于 π0.5 97.7%**（后训练起点不同）；**后训练增益** 对 π0.5 系具参考价值
- **简介**：用 WM 当 **可靠 sim** 做 VLA 后训练：在 WM 仿真环境中 RL fine-tune VLA。消融：**WM 质量决定 post-train 上限**；低保真 WM 反而误导。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.13977) · [abs](https://arxiv.org/abs/2602.13977)
- **数据规模**：LIBERO / 真机后训练（WM 作 sim）
- **模型大小**：WM 后训练 VLA（起点依赖 base checkpoint）
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：**WM 作可靠仿真器后训练 VLA**：在 WM 环境中 on-policy/RL 微调已有 VLA。消融：**WM 质量决定上限**，低保真 WM 反而误导。LIBERO **39.95%→69.2%（+29.3pp）**；真机 **61.7%→91.7%（+30.0pp）**。**正面**（后训练增益大，绝对 LIBERO 仍低于 π0.5 97.7%）。
- **提及**：√

## 2026-02-XX · Xiaomi-Robotics-0: Open-Sourced VLA with Real-Time Execution

- **提出日期**：2026-02（arXiv 2602.12684）
- **SOTA**：**CALVIN SOTA 4.75**（#1）；**LIBERO T2 98.7%**（#2）
- **Benchmark**：LIBERO-Plus 高位但具体值需看 paper；RoboTwin 2.0 / SimplerEnv 多榜共训
- **Other Benchmark**：Xiaomi 真机长程；3.6B 参数；动作分块 + 蒸馏低延迟
- **比 Pi0.5 好**：LIBERO **+1.0pp**（98.7 vs 97.7）；CALVIN 较 π0 **+0.83**（远超）
- **简介**：工业级 **实时 VLA**：动作分块 + 异步执行 + Λ-shape 注意力 mask 维持低延迟下的双榜（LIBERO + CALVIN）双高；Xiaomi 自研机器人平台主线模型。消融：Λ-shape mask 强制模型关注视觉/语言条件而非过度依赖动作前缀；CALVIN 长链上后训练优于 single-pass。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.12684) · [abs](https://arxiv.org/abs/2602.12684)
- **数据规模**：预训练 40k steps bs=32768；后训练 Lego 40k / Towel 80k steps bs=2048
- **模型大小**：**4.7B**；Qwen3-VL-4B-Instruct VLM + 16 层 DiT action expert（MoT 架构）
- **算力**：DeepSpeed ZeRO-2；GPU 型号论文正文未在摘要列出
- **世界模型用法**：未使用。
- **提及**：

## 2026-02-XX · ABot-M0: VLA Foundation Model with Action Manifold Learning

- **提出日期**：2026-02（arXiv 2602.11236）
- **SOTA**：**LIBERO T3 98.6%**（#3）
- **Benchmark**：LIBERO-Plus **80.5%**；LIBERO-Spatial / LIBERO-Object / LIBERO-Goal 子项均高位
- **Other Benchmark**：动作流形约束损失；OOD 鲁棒性提升
- **比 Pi0.5 好**：LIBERO **+0.9pp**；LIBERO-Plus 较 π0 **+25.5pp**
- **简介**：在 **动作流形** 上建模动作分布；流形约束损失减少 OOD 动作分布漂移。消融：**流形约束损失** 是 LIBERO-Plus 鲁棒性主要来源；ablate 后扰动子集下降明显。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.11236) · [abs](https://arxiv.org/abs/2602.11236)
- **数据规模**：UniACT-dataset：**6M+ 轨迹、~9,500 小时**，6 个开源数据集整合
- **模型大小**：Qwen3-VL-4B + **0.16B** DiT（AML）
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：未使用。
- **提及**：

## 2026-02-XX · MINT: Mimic Intent, Not Just Trajectories (MINT-4B)

- **提出日期**：2026-02（arXiv 2602.08602）
- **SOTA**：—
- **Benchmark**：LIBERO **98.3%**；LIBERO-Plus **80.1%**
- **Other Benchmark**：4B 参数；意图标签监督 vs 纯轨迹监督
- **比 Pi0.5 好**：LIBERO **+0.5pp**；LIBERO-Plus 较 π0 **+24.0pp**
- **简介**：监督 **意图标签**（intent labels）而非纯轨迹；强化对语言扰动与长尾任务的语义鲁棒。消融：**意图对齐** 对语言扰动与长尾任务更有效；纯轨迹模仿在 LIBERO-Plus 上明显劣化。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.08602) · [abs](https://arxiv.org/abs/2602.08602)
- **数据规模**：多 embodiment 不一致数据共训（Bridge 等）
- **模型大小**：**MINT-4B**（PaliGemma-2.6B VLM + 300M action expert + 意图监督）
- **算力**：**4× NVIDIA H200**
- **世界模型用法**：未使用。
- **提及**：

## 2026-02-XX · VLA-JEPA: Enhancing VLA with Latent World Model

- **提出日期**：2026-02（arXiv 2602.10098）
- **SOTA**：—
- **Benchmark**：LIBERO-Plus **79.5%**（2026 年高位但非 Top3）
- **Other Benchmark**：JEPA 风格潜世界模型；预训练辅助损失
- **比 Pi0.5 好**：LIBERO-Plus 较 π0 **+25.9pp**（79.5% vs 53.6%）
- **简介**：**JEPA 式潜世界模型** 辅助 VLA 动作预测；不重建像素而预测 latent embedding。消融：**潜预测辅助损失** 提升 LIBERO-Plus 鲁棒性，对视角扰动尤为有效。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.10098) · [abs](https://arxiv.org/abs/2602.10098)
- **数据规模**：预训练 Something-Something-v2（220K 视频）+ DROID 76K；微调 LIBERO ~2K demo
- **模型大小**：Qwen3-VL-2B backbone
- **算力**：**8× A100**
- **世界模型用法**：**JEPA 式潜世界模型辅助损失**：预训练阶段预测未来 **latent embedding**（冻结 V-JEPA2 编码器为目标，不重建像素），微调 VLA 动作头。消融：潜预测辅助损失提升 **LIBERO-Plus 至 79.5%**（π0 53.6%，**+25.9pp**）；相机扰动 63.3% vs π0 13.8%，机器人扰动 67.1% vs 6.0%。**正面**。
- **提及**：

## 2026-02-XX · DreamZero: World Action Models are Zero-Shot Policies

- **提出日期**：2026-02（arXiv 2602.15922）
- **SOTA**：**RoboArena #1（团队自报）**；**MolmoSpaces #1**
- **Benchmark**：—
- **Other Benchmark**：AgiBot G1 seen task progress **62.2%** vs best pretrained VLA **27.4%**（>2×）；from-scratch unseen **39.5%**；38× 推理加速；7Hz 闭环
- **比 Pi0.5 好**：真机泛化明确优于 best pretrained VLA（seen 62.2% vs 27.4%，>2×）
- **简介**：**Wan2.1-I2V-14B WAM** + 联合 video-action flow；零样本策略可直接应用 WAM 做开放任务。消融：**KV cache 真观测替换、DreamZero-Flash 1-step、CFG 并行** 实现 38× 加速；联合训练优于分离训练。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.15922) · [abs](https://arxiv.org/abs/2602.15922) · [Site](https://dreamzero0.github.io/) · [Code](https://github.com/dreamzero0/dreamzero)
- **数据规模**：~500h AgiBot G1（7.2K ep, 22 环境）+ DROID；强调多样数据
- **模型大小**：**14B**（Wan2.1-I2V-14B + video-action flow）
- **算力**：匹配 batch/steps；推理 **H100/GB200** 优化至 38×（Flash）
- **世界模型用法**：**Wan2.1-I2V-14B WAM**：**联合预测 video + action** flow；可零样本作策略。消融：联合 vs 分离未做数值 ablation（仅设计论证）；DreamZero-Flash（KV 替换、1-step、CFG 并行）**38× 推理加速**；AgiBot G1 seen **62.2% vs best pretrained VLA 27.4%（>2×）**。**正面**。
- **提及**：√

## 2026-02-XX · World-VLA-Loop: Closed-Loop World Models for VLAs

- **提出日期**：2026-02（arXiv 2602.06508）
- **SOTA**：—
- **Benchmark**：—
- **Other Benchmark**：真机闭环较基线 **+36.7%**；VLA 失败 rollout 反馈精炼 WM
- **比 Pi0.5 好**：闭环增益 +36.7%（不同设定，π0.5 未同设定）
- **简介**：**VLA↔WM 闭环共改进**：每步动作经 world model 反推 → 失败轨迹回流 WM 训练。消融：失败轨迹回流 WM 是涨点关键；闭环 reweight 优于 open-loop。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.06508) · [abs](https://arxiv.org/abs/2602.06508)
- **数据规模**：ManiSkill SANS + 下游 <100 成功 轨迹
- **模型大小**：Cosmos Predict 2 WM + VLA 闭环
- **算力**：WM 生成 24 帧约 **7s/批（H100）**；GRPO ~50 步收敛
- **世界模型用法**：**VLA↔WM 闭环共进化**（Cosmos Predict2 WM）：动作 rollout 经 WM 反推，**失败/近成功轨迹回流训 WM**（SANS 数据集）。消融（Table 4）：近成功回流数据是关键（移除降 25-30pp）；真机 2 轮迭代 **13.3%→36.7%→50.0%（总 +36.7pp）**。**正面**。
- **提及**：√

## 2026-02-XX · VLAW: Vision-Language-Action World Model

- **提出日期**：2026-02（arXiv 2602.12063）
- **SOTA**：—
- **Benchmark**：—
- **Other Benchmark**：绝对成功率 +39.2%；合成数据 +11.6%
- **比 Pi0.5 好**：相对提升大；绝对 LIBERO 需对照表
- **简介**：**迭代共改进** WM+VLA：交替训 WM 与 VLA，使用合成数据 + 真实数据混合。消融：合成 + 真实混合迭代优于单一来源；同步迭代 > 交替单边训练。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.12063) · [abs](https://arxiv.org/abs/2602.12063)
- **数据规模**：WM+VLA 迭代共训（合成+真实混合）
- **模型大小**：迭代 WM+VLA 联合
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：**迭代共改进 WM+VLA**（Ctrl-World action-conditioned WM + π0.5 VLA）：迭代循环（收集真实 rollout → 微调 WM → 生成合成数据 → 训练 VLA），合成+真实混合。消融：混合数据（真实+合成）> 仅真实（Filtered BC）；WM 加入在线 rollout 数据显著降低误判（FP: 11→1）；绝对成功率 **+39.2%** vs base，合成数据贡献 **+11.6%** vs Filtered BC。**正面**。
- **提及**：√

## 2026-02-XX · VLANeXt: Recipes for Building Strong VLA Models

- **提出日期**：2026-02（arXiv 2602.18532）
- **SOTA**：**LIBERO-Long T2 94.6%**（#2）
- **Benchmark**：LIBERO 平均 T1 97.4%；LIBERO+ 鲁棒套件 SOTA 叙事
- **Other Benchmark**：12 条设计 recipe；2.5B 小模型
- **比 Pi0.5 好**：LIBERO 平均 **-0.3pp** vs π0.5；Long 榜 **Top2**
- **简介**：**12 条设计 recipe** + 2.5B 小模型，通过 recipe 组合而非单点结构创新建强 VLA。消融：**recipe 组合 > 单点结构创新**；任意单条 recipe 去掉 LIBERO-Long 显著下降。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.18532) · [abs](https://arxiv.org/abs/2602.18532)
- **数据规模**：多 benchmark 混合
- **模型大小**：**2.5B** + 12 条 recipe
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：recipe 消融曾验证 **预测未来观测有益**，但训练时间 **≈3×**，**最终模型明确排除 world modeling**（采用 Qwen3-VL + 轻量 policy head）。属于 **负向设计决策**（成本-收益）；LIBERO-Long **94.6% Top2** 在无 WM 下达成。**未在终稿使用 WM**。
- **提及**：√

## 2026-02-XX · SimVLA: A Simple VLA Baseline

- **提出日期**：2026-02（arXiv 2602.18224）
- **SOTA**：**LIBERO T3 98.6%**（#3，0.5B 参数）
- **Benchmark**：—
- **Other Benchmark**：训练 VRAM 9.3GB；轻量 backbone + 动作头
- **比 Pi0.5 好**：LIBERO **+0.9pp** vs π0.5（98.6 vs 97.7）
- **简介**：**极简 VLA**（轻量 backbone + 动作头 0.5B 参数）作 baseline；强配方 + 高效训练即可接近 SOTA。消融：简单配方 + 高效训练可达近 SOTA；说明 **复杂结构非必需**。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.18224) · [abs](https://arxiv.org/abs/2602.18224) · [Code](https://github.com/LUOyk1999/SimVLA) · [HF](https://huggingface.co/YuankaiLuo/SimVLA-LIBERO)
- **数据规模**：LIBERO 等
- **模型大小**：**0.5B** 极简 VLA
- **算力**：训练 **9.3GB VRAM**
- **世界模型用法**：未使用。
- **提及**：

## 2026-02-XX · Green-VLA: 5-Stage Curriculum to Strong VLA

- **提出日期**：2026-02（arXiv 2602.00919）
- **SOTA**：—（5 阶段 curriculum 路线 SOTA 叙事）
- **Benchmark**：3k h 双臂数据规模
- **Other Benchmark**：VLM → RL 五阶段
- **比 Pi0.5 好**：（无逐榜 π0.5 直接对比）
- **简介**：**质量对齐 + 动作统一 + RL 精炼** 三阶段（实际 5 阶段 curriculum）；VLM → RL 渐进式训练。消融：**curriculum 后期 RL 必要**；缺失任一阶段下游显著降。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.00919) · [abs](https://arxiv.org/abs/2602.00919)
- **数据规模**：**3k h** 双臂数据（5 阶段 curriculum）
- **模型大小**：VLM→RL 五阶段路线
- **算力**：后期 RL 必要（未列 GPU 数）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-02-XX · GeneralVLA: 3D Affordance + Control Strategy

- **提出日期**：2026-02（arXiv 2602.04315）
- **SOTA**：—（**零真机数据** affordance + 3D agent 零样本）
- **Benchmark**：—
- **Other Benchmark**：3D affordance + 控制策略
- **比 Pi0.5 好**：（不同设定，无 LIBERO 标准 Top3 分数）
- **简介**：**3D affordance + 控制策略**：先用 3D affordance 模型识别可操作区域，再调通用控制策略。消融：3D agent 冷启动有效；零真机数据下仍可工作。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.04315) · [abs](https://arxiv.org/abs/2602.04315)
- **数据规模**：零真机数据 affordance 冷启动
- **模型大小**：3D affordance + 控制策略
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-02-XX · LifeLong-RFT: Lifelong Reinforcement Fine-Tuning

- **提出日期**：2026-02（arXiv 2602.10503）
- **SOTA**：—
- **Benchmark**：LIBERO 较 SFT **+22%** 平均成功率（**20% 数据**）
- **Other Benchmark**：—
- **比 Pi0.5 好**：（后训练路线，绝对分需看起点是否超 97.7%）
- **简介**：**Chunk 级 on-policy RL + MDPR**（Mixed Data Policy Replay）；防灾难性遗忘的连续学习。消融：**防灾难性遗忘** 是核心，仅 20% 数据下仍提升 +22%。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.10503) · [abs](https://arxiv.org/abs/2602.10503)
- **数据规模**：LIBERO 多任务持续学习（20% 数据 +22% SR）
- **模型大小**：NORA-Long base（Fast+ tokenizer）
- **算力**：**8× NVIDIA H20**
- **世界模型用法**：未使用。
- **提及**：√

## 2026-02-XX · QuantVLA: Post-Training Quantization for VLA

- **提出日期**：2026-02（arXiv 2602.20309）
- **SOTA**：—（部署向）
- **Benchmark**：—
- **Other Benchmark**：70% 显存节省，成功率不降
- **比 Pi0.5 好**：（工程优化，非 LIBERO 涨点）
- **简介**：**后训练量化 PTQ**：选择性 mixed-precision 量化保持成功率不降。消融：**量化后仍保精度**；仅量化 vision/text 部分而保 action expert 是关键。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.20309) · [abs](https://arxiv.org/abs/2602.20309)
- **数据规模**：校准 buffer 无标签
- **模型大小**：W4A8 量化 π0.5 / GR00T N1.5
- **算力**：实验在 **NVIDIA A100**
- **世界模型用法**：未使用。
- **提及**：√

## 2026-01-XX · CycleVLA: Backtracking + MBR Decoding for VLA

- **提出日期**：2026-01（arXiv 2601.02295）
- **SOTA**：—
- **Benchmark**：LIBERO 回溯 + MBR 显著提升（论文 Table V/VI，无单句 +X%）
- **Other Benchmark**：测试时增强，可与 π0.5 叠加
- **比 Pi0.5 好**：（可叠加，未直接百分比）
- **简介**：**子任务回溯 + MBR decoding**：测试时通过 failure predictor 触发 backtracking；Minimum Bayes Risk decoding 提升采样质量。消融：**failure predictor + 回溯** 是核心；MBR 比 greedy / top-k 在长程任务更稳。
- **相关资料**：[PDF](https://arxiv.org/pdf/2601.02295) · [abs](https://arxiv.org/abs/2601.02295)
- **数据规模**：LIBERO 测试时增强
- **模型大小**：OpenVLA + diffusion action expert
- **算力**：训练 **4× A100 40GB**；评测 **1× A10**
- **世界模型用法**：未使用。
- **提及**：√

## 2026-02-XX · LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer

- **提出日期**：2026-02（arXiv 2602.10556）
- **SOTA**：—（零样本跨本体真机）
- **Benchmark**：—
- **Other Benchmark**：零样本未见机器人平均成功率 **>50%**（≈2× 最强 prior VLA，其他开源 VLA ≈ 0%）；LIBERO 微调数据效率 ≈2.5×
- **比 Pi0.5 好**：零样本跨本体 **显著优于 π0.5-DROID / π0.5-replicated**（官网定性对比）
- **简介**：**语言-动作预训练**：动作以自然语言监督 VLM，**无 action tokenizer**；与 VQA 共训提升语义对齐。LAP-3B 模型。消融：**无 action tokenizer + 语言监督** 是跨本体零样本核心；VQA 共训提升数据效率 ≈2.5×。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.10556) · [abs](https://arxiv.org/abs/2602.10556) · [Site](https://lap-vla.github.io/) · [Code](https://github.com/lihzha/lap) · [HF](https://huggingface.co/collections/lihzha/lap)
- **数据规模**：OXE + MolmoAct；shuffle buffer **16M** samples
- **模型大小**：**LAP-3B**（PaliGemma-3B）
- **算力**：**64× TPU v6e**，15k steps，约 **10 小时**
- **世界模型用法**：未使用。
- **提及**：

## 2026-01-XX · Cosmos Policy (NVIDIA)

- **提出日期**：2026-01（arXiv 2601.16163）
- **SOTA**：**RoboCasa SOTA 67.1%**（#1）；LIBERO 98.5%（全局 #4）
- **Benchmark**：—
- **Other Benchmark**：基于 NVIDIA Cosmos 世界基础模型的策略头
- **比 Pi0.5 好**：LIBERO **+0.8pp**；RoboCasa 较 π0 **+4.6pp**
- **简介**：基于 **Cosmos 世界基础模型** 的策略头；通过视频预训练 + latent dynamics 提供 manipulation 先验。消融：**世界模型预训练初始化** 是 RoboCasa 关键；从零训显著降。
- **相关资料**：[PDF](https://arxiv.org/pdf/2601.16163) · [abs](https://arxiv.org/abs/2601.16163) · [NVIDIA Cosmos](https://research.nvidia.com/labs/dir/cosmos/) · [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2) · [HF Models](https://huggingface.co/collections/nvidia/cosmos-predict2)
- **数据规模**：RoboCasa 185 demo 等；Cosmos Predict2 初始化
- **模型大小**：Cosmos Predict2 **2B** 全参微调
- **算力**：40K/45K/50K steps：**64/32/8× H100** 各 **48h**
- **世界模型用法**：**Cosmos-Predict2-2B 视频基础模型作策略初始化**：单阶段后训练，在潜扩散中联合 **action / future state / value**；非从零 VLA。消融（Table 4，LIBERO）：**WM 预训练初始化** 关键，从零训降 3.9pp（98.5→94.6%），真机 ALOHA 降 18.7pp；LIBERO **98.5%（+1.1pp vs CogVLA 97.4%）**；RoboCasa **67.1% SOTA（+4.6pp vs π0，仅需 50 demos vs π0 300）**。**正面**。
- **提及**：√

## 2026-01-XX · Pose-VLA: Universal Pose Pretraining for Generalizable VLAs

- **提出日期**：2026-01（arXiv 2602.19710）
- **SOTA**：—
- **Benchmark**：RoboTwin 2.0 **79.5%**（未进 Top3）
- **Other Benchmark**：2D / 3D 姿态预训
- **比 Pi0.5 好**：RoboTwin 2.0 Hard 较 π0 **+14.0pp**（79.1% vs 65.12%）
- **简介**：**2D / 3D 姿态预训练** 再迁移动作策略；普适 pose 表征作 VLA 中介。消融：**姿态预训练阶段** 对双臂泛化最关键；缺失则 RoboTwin 2.0 显著降。
- **相关资料**：[PDF](https://arxiv.org/pdf/2602.19710) · [abs](https://arxiv.org/abs/2602.19710)
- **数据规模**：3D grounding + 轨迹估计预训练任务
- **模型大小**：VLM + 姿态预训练阶段
- **算力**：各阶段 **16× H20×2 天**
- **世界模型用法**：未使用。
- **提及**：

## 2026-01-XX · Being-H0.5

- **提出日期**：2026-01（arXiv 2601.12993）
- **SOTA**：—（论文自报 LIBERO 98.9%、RoboCasa 53.9%；未提交官榜，官榜 Overall ≈ 22%）
- **Benchmark**：LIBERO 自报 98.9%；RoboCasa 自报 53.9%（口径与官榜不可直接比）
- **Other Benchmark**：UniHand-2.0（35kh / 30 本体）+ MoF
- **比 Pi0.5 好**：LIBERO 自报 +1.2pp vs π0.5；RoboCasa 口径与官榜不可直接比
- **简介**：**UniHand-2.0（35kh / 30 本体）+ MoF**（Mixture-of-Foundations）；统一动作空间。消融：**MoF + 统一动作空间** 跨本体生效；缺一则跨平台精度显著降。
- **相关资料**：[PDF](https://arxiv.org/pdf/2601.12993) · [abs](https://arxiv.org/abs/2601.12993) · [Code](https://github.com/BeingBeyond/Being-H) · [HF](https://huggingface.co/BeingBeyond/Being-H05-2B) · [Site](https://research.beingbeyond.com/being-h05)
- **数据规模**：UniHand-2.0：**400M 样本 / 35k h / 30 本体；122B tokens**
- **模型大小**：Being-H0.5-2B（MoF）
- **算力**：预训练约 **1000 GPU·hour**（recipe 将公开）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-01-XX · LingBot-VLA / A Pragmatic VLA Foundation Model

- **提出日期**：2026-01（arXiv 2601.18692）
- **SOTA**：—
- **Benchmark**：—
- **Other Benchmark**：20kh 双臂数据；scaling 3kh → 20kh 持续上涨（无饱和）
- **比 Pi0.5 好**：（数据尺度不同，无 π0.5 同条件对比）
- **简介**：工程化 **VLA 预训练实践** + LingBot-VA 因果 WM；强调数据规模化与单一稳定配方。消融：**数据规模仍主导**；3kh → 20kh 几乎线性涨。
- **相关资料**：[PDF](https://arxiv.org/pdf/2601.18692) · [abs](https://arxiv.org/abs/2601.18692)
- **数据规模**：**3k→20k h** 双臂（scaling 未饱和）
- **模型大小**：**3B**（Qwen2.5-VL + MoT action expert）
- **算力**：**256 GPU** 规模预训练（论文叙事）
- **世界模型用法**：未使用。本文 LingBot-VLA 为纯 VLA 架构（Qwen2.5-VL + MoT action expert + Flow Matching），无 WM 组件。数据 scale 3k→20k 小时近乎单调涨（Fig 5），但属数据规模消融，非 WM 相关。配套 LingBot-Depth 提供深度感知（非 WM）。注："LingBot-VA" 为另一独立模型/论文，非本文内容。
- **提及**：√

## 2026-01-XX · SOP: Scalable Online Post-Training

- **提出日期**：2026-01（arXiv 2601.03044）
- **SOTA**：—（Fleet online RL 路线，无单 LIBERO%）
- **Benchmark**：—
- **Other Benchmark**：小时级在线涨点；fleet 规模数据闭环
- **比 Pi0.5 好**：（部署后训练，非 zero-shot 榜可比）
- **简介**：**HG-DAgger + RECAP** 流式 on-policy 后训练；适合大规模 fleet 持续部署。消融：**fleet 规模数据闭环** 是涨点关键；单点机器人在线 RL 收益小。
- **相关资料**：[PDF](https://arxiv.org/pdf/2601.03044) · [abs](https://arxiv.org/abs/2601.03044)
- **数据规模**：π0.5 微调 **~160h** 多任务真机（Grocery/Laundry/Box）
- **模型大小**：π θ0 基座 + RECAP 后训练
- **算力**：**8× H100** 云端 learner
- **世界模型用法**：未使用。
- **提及**：√

## 2026-01-XX · TT-VLA: Test-Time RL with Task-Progress Reward

- **提出日期**：2026-01（arXiv 2601.06748）
- **SOTA**：—
- **Benchmark**：—
- **Other Benchmark**：推理时 RL（task-progress dense reward）
- **比 Pi0.5 好**：（π0.5 未测）
- **简介**：**On-the-fly** 推理时 RL：用 task-progress dense reward 实时调整策略；零额外训练数据。消融：**进度 reward 优于稀疏成功**；推理时 RL 主要受益视觉编码器层 adaptation。
- **相关资料**：[PDF](https://arxiv.org/pdf/2601.06748) · [abs](https://arxiv.org/abs/2601.06748)
- **数据规模**：仿真 80 trials/任务；真机 10 trials
- **模型大小**：LoRA rank {16,32} 于 VLA
- **算力**：—（论文/本地材料未明确给出）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-01-XX · Genie Sim 3.0（智元 / AgiBot）

- **提出日期**：2026-01（arXiv 2601.02078）
- **SOTA**：第三方对照中 Instruction **π0.5 0.67** > GR00T-N1.6 0.40 > π0 0.28（三项均第一）
- **Benchmark**：—
- **Other Benchmark**：仿真-真机一致性 R²=0.94
- **比 Pi0.5 好**：仿真-真机一致性榜上 π0.5 三项 SOTA（**论文测试方而非提案方**）
- **简介**：智元 / AgiBot 第三代 Sim 3.0：LLM 驱动 sim + VLM 评测；仿真-真机一致性 R² 0.94。消融：**LLM 驱动场景生成 + VLM 评测** 提供高一致性的 sim2real 评测路径。
- **相关资料**：[PDF](https://arxiv.org/pdf/2601.02078) · [abs](https://arxiv.org/abs/2601.02078) · [AgiBot](https://agibot.com/article/231/detail/55.html)
- **数据规模**：**10,000+ h** 仿真、**100,000+** 场景、200 任务
- **模型大小**：LLM 驱动 sim + VLM 评测平台
- **算力**：sim-real 一致性评测（非单一模型训练表）
- **世界模型用法**：未使用。
- **提及**：√

## 2026-01-XX · Helix 02（Figure AI 技术报告 / Blog）

- **提出日期**：2026-01（Figure AI 博客）
- **SOTA**：—（真机长程，非标准仿真榜）
- **Benchmark**：—
- **Other Benchmark**：4 分钟 / 61 步洗碗 **无干预**；S0(1kHz 平衡) + S1(200Hz 视触) + S2(语义) 三层
- **比 Pi0.5 好**：（不同任务设定，π0.5 未同任务）
- **简介**：Figure AI Helix 02：**S0/S1/S2 三层** 控制：1kHz 神经平衡 + 200Hz 视触 + 语义 LLM；端到端 VLA + 闭环视觉。消融：**神经网络替 10.9 万行 C++ 平衡控制**，长程稳定性显著提升。
- **相关资料**：[Helix 02 Blog](https://figure.ai/news/helix-02) · [Figure AI](https://www.figure.ai/) · [Figure 02 Spec](https://www.figure.ai/figure-02)
- **数据规模**：Figure 真机长程（如洗碗 61 步）
- **模型大小**：S0/S1/S2 三层控制栈
- **算力**：技术博客
- **世界模型用法**：未使用。
- **提及**：√

## 2025-12-XX · GR00T N1.6（NVIDIA）

- **提出日期**：2025-12（NVIDIA 博客 / 技术报告）
- **SOTA**：**RoboCasa 官榜 SOTA #1 Overall 21.9%**
- **Benchmark**：—
- **Other Benchmark**：Genie Sim 三项指标低于 π0.5；Cosmos-Reason-2B + 32 层 DiT；state-relative action chunk
- **比 Pi0.5 好**：RoboCasa 官榜 **无 π0.5 条目**；低于主表 RLDX-1 70.6%（不同指标）
- **简介**：GR00T N1.6 升级：**Cosmos-Reason-2B + 32 层 DiT**；state-relative action chunk 解决长程稳定性；面向 humanoid 通用 generalist 模型。消融：state-relative action chunk 优于 absolute；Cosmos-Reason backbone 比纯 VLM backbone 在 RoboCasa 上更稳。
- **相关资料**：[Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T) · [HF GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) · [HF GR00T-N1.6-BEHAVIOR1k](https://huggingface.co/nvidia/GR00T-N1.6-BEHAVIOR1k) · [GEAR 研究博客](https://research.nvidia.com/labs/gear/gr00t-n1_6/) · [NVIDIA 开发者博客（sim-to-real）](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow/) · [NVIDIA GR00T 项目页](https://research.nvidia.com/labs/gear/gr00t/)
- **数据规模**：在 **GR00T N1.5 数据混合** 之上，预训练新增 **数千小时（thousands of hours）** 遥操作数据：双臂 **YAM**、**AGIBot Genie1**、**BEHAVIOR** 套件上仿真 **Galaxea R1 Pro**、**Unitree G1** 全身 loco-manipulation 等（GEAR 博客附预训练数据权重分布图）。开发者博客进一步列出：仿真含 **BEHAVIOR、RoboCasa、GR-1 定制环境**；真机含 **GR-1 (Fourier)、G1 (Unitree)、双臂 YAM、Agibot、DROID** 及多本体跨形态混合（Figure 1 各数据集占比）。任务部署时多在 **小规模任务数据集** 上 post-train；另提供 **BEHAVIOR-1K** 后训练变体权重。
- **模型大小**：开源 **`GR00T-N1.6-3B`**。骨干 **Cosmos-2B / Cosmos-Reason-2B** VLM（原生宽高比图像编码）；**32 层 DiT** diffusion transformer（N1.5 为 16 层，约 2×）；去掉 N1.5 的 4 层 post-VLM adapter，预训练 **解冻 VLM 顶 4 层**；默认 **state-relative action chunk**（相对动作空间，非绝对关节/EEF）。结合 **Cosmos Reason** 世界模型做 VLA 推理与任务分解。
- **算力**：两篇 HTML 博客 **未给出 GPU 型号、卡数或墙钟训练时长**。GEAR 博客披露训练配方：**预训练 300K steps、global batch size 16,384**；下游机器人实验 **post-train 通常 10K–30K steps、global batch size ≤1K**。整机栈还包含 Isaac Lab **全身 RL**（GR00T-WholeBodyControl）与 COMPASS **大规模合成导航** 微调（各自算力未单独列表）。
- **世界模型用法**：**Cosmos-Reason-2B VLM 变体作骨干**（博客称为 world model，用于任务分解/规划），**32 层 DiT** 动作头（N1.5 为 16 层，约 2×），配合 Isaac Lab 全身 RL 与 COMPASS 导航。技术文档显示 **state-relative action 优于 absolute**。无量化消融对比 Cosmos-Reason vs 其他 VLM backbone（博客无正式 ablation 表）。**正面**（推理/planning 用 WM，非视频-动作联合训练主路径）。
- **提及**：√

## 2025-12-XX · OXE-AugE: Augmenting OXE with Embodiment Aug

- **提出日期**：2025-12（arXiv 2512.13100）
- **SOTA**：—
- **Benchmark**：—
- **Other Benchmark**：跨本体 4 真机任务未见 robot×gripper 场景 +24%~+45%；OXE 扩到 4.4M
- **比 Pi0.5 好**：（数据增广论文，非单模型 LIBERO 分对照）
- **简介**：**机器人形态增广**（embodiment augmentation）：扩 OXE 到 4.4M episodes；通过形态变换增广提升未见 robot × gripper 组合。消融：**增广组合 > naive mixing**；多形态混合显著超均匀混合。
- **相关资料**：[PDF](https://arxiv.org/pdf/2512.13100) · [abs](https://arxiv.org/abs/2512.13100)
- **数据规模**：OXE 扩至 **4.4M trajectories / 9 本体**
- **模型大小**：扩散策略 ResNet18 + 1D CNN；OpenVLA/π0 微调
- **算力**：仿真 250k steps；真机微调 20–25k steps（未统一 GPU 表）
- **世界模型用法**：策略主干**未使用** WM/WAM。数据管线用 **仿真渲染 + 视频修复** 做 embodiment 增广（SAM2 分割源机器人 → E2FGVI 修复背景 → MuJoCo 渲染目标机器人合成），**非视频生成模型**（论文对比扩散增广 RoVi-Aug 反降 27-30%）。OXE→4.4M 轨迹；真机未见 robot×gripper **+24%~+45%**。**正面**（仅数据侧仿真增广，非 WM/视频生成）。
- **提及**：√

## 2025-11-XX · π*0.6 / Recap

- **提出日期**：2025-11（arXiv 2511.14759）
- **SOTA**：—
- **Benchmark**：—
- **Other Benchmark**：真机 throughput 2×、failure rate ≈½；咖啡 / 装箱等任务 ≥ 90%
- **比 Pi0.5 好**：相对 **π0.5** 后训练强化（同 PI 系，明确优于 π0.5 真机）
- **简介**：**Offline RL + on-robot RL + HG-DAgger**（Recap 模块）。消融：**Recap >> AWR / PPO**；on-robot RL 是真机长程关键。
- **相关资料**：[PDF](https://arxiv.org/pdf/2511.14759) · [abs](https://arxiv.org/abs/2511.14759) · [openpi](https://github.com/physical-intelligence/openpi)
- **数据规模**：真机咖啡/装箱等长程任务
- **模型大小**：π0.6 + Recap（openpi）
- **算力**：offline + on-robot RL（Physical Intelligence 系）
- **世界模型用法**：未使用。
- **提及**：√

## 2025-10-XX · X-VLA: Soft-Prompt Cross-Embodiment VLA

- **提出日期**：2025-10（arXiv 2510.10274）
- **SOTA**：—
- **Benchmark**：LIBERO 98.1%；Simpler VM 80.4% / VA 75.7% / WidowX 95.8%；CALVIN 4.43；RoboTwin 2.0 54.5%
- **Other Benchmark**：—
- **比 Pi0.5 好**：LIBERO **+0.4pp**；Simpler 各子项较 π0 VM **+21.6pp** / VA **+18.9pp** / WidowX **+68.0pp**
- **简介**：**Soft-prompt embodiment id**：用可学 soft prompt 编码本体差异；30+ 机器人形态共训。消融：**naive 混合退化，soft prompt 恢复**；soft prompt 对跨本体涌现核心。
- **相关资料**：[PDF](https://arxiv.org/pdf/2510.10274) · [abs](https://arxiv.org/abs/2510.10274) · [Site](https://x-vla.github.io/)
- **数据规模**：跨 30+ 本体异构混合数据
- **模型大小**：X-VLA-0.9B（Florence-Large VLM + 24 层 Transformer + soft-prompt）
- **算力**：预实验 **8× A100**，bs=256，200K iter
- **世界模型用法**：未使用。
- **提及**：√

## 2025-10-XX · CoLA-World: Co-evolution of Latent Action + World Model

- **提出日期**：2025-10（arXiv 2510.26433）
- **SOTA**：—（LIBERO / 长程叙事）
- **Benchmark**：—
- **Other Benchmark**：Latent action + WM 协同训练
- **比 Pi0.5 好**：（π0.5 未测）
- **简介**：**LAM + WM 共进化**：latent action model 与 world model 联合训练；warm-up 防表征坍塌。消融：**warm-up 防坍塌**；纯交替训练易坍塌到 trivial latent。
- **相关资料**：[PDF](https://arxiv.org/pdf/2510.26433) · [abs](https://arxiv.org/abs/2510.26433)
- **数据规模**：LAM+WM 联合（OpenSora ~1.2B + 74M 新增）
- **模型大小**：IDM 0.12B + OpenSora ~0.93B 可学习
- **算力**：2-stage LAM30K+WM30K = 60K steps；联合 Warm8K+E2E52K = 60K steps（论文 Tables 1-4，无 Table 8）
- **世界模型用法**：**LAM + 视频 WM 共进化**：**OpenSora v1.2（~1.2B）** 预训练视频 WM + 从零 LAM（IDM + VQ 量化器）；**warm-up 8K 步后端到端联合训练**（无 warm-up 直接联合导致码本坍塌，Fig 2-3）。消融：**warm-up 必需**；**共进化（evolving WM+LAM）> 冻结组件**（Fig 4）；CoLA-World（Warm8K+E2E52K）> 2-stage（LAM30K+WM30K），总步数均 60K。**正面**（表征共进化）。
- **提及**：√

## 2025-09-XX · FLOWER: Efficient VLA Flow Policy

- **提出日期**：2025-09（arXiv 2509.04996）
- **SOTA**：**LIBERO-Long SOTA #1 93.4%**（FLOWER Table 1）；LIBERO-Spatial 97.1%
- **Benchmark**：LIBERO 多子集
- **Other Benchmark**：—
- **比 Pi0.5 好**：LIBERO-Long 领先；与 π0.5 **分项互有高低**
- **简介**：**高效 VLA flow policy**：flow-matching 替代 diffusion 做 action head。消融：**flow head 在 LIBERO-Long 套件最强**；少 NFE 推理已达 diffusion 多步性能。
- **相关资料**：[PDF](https://arxiv.org/pdf/2509.04996) · [abs](https://arxiv.org/abs/2509.04996) · [Site](https://flower-vla.github.io/)
- **数据规模**：异构 manipulation 混合；360k steps/48h 预训练混合
- **模型大小**：**950M** flow policy
- **算力**：预训练 **200 H100·h**（4×H100×48h）；对比 OpenVLA 21500h
- **世界模型用法**：未使用。
- **提及**：√

## 2025-09-XX · RealMirror: Comprehensive Open-Source VLA Platform for Embodied AI

- **提出日期**：2025-09（arXiv 2509.14687）；ICRA 2026 录用
- **SOTA**：**RealMirror Humanoid VLA Benchmark Top1**：SmolVLA Avg 79.75%（vs Diffusion Policy 75.15% / ACT 73.55%）
- **Benchmark**：—
- **Other Benchmark**：分任务 Top1：ACT Kitchen Cleanup 100% / Assembly Line Sorting 95%；Diffusion Policy Air Fryer 85.5%；SmolVLA Cup-to-Cup 68% / Can Stacking 62%；Sim2Real：ACT zero-shot pick-place 92.86% / ball transfer 71.43%
- **比 Pi0.5 好**：（人形平台 / benchmark 论文，π0.5 未在 RealMirror 上公开）
- **简介**：**benchmark + 平台论文**：VR 遥操作 + LeRobot/Isaac Sim 训练推理 + 5 类人形任务 + 1200 条轨迹；生成模型 + 3DGS 高保真 Sim2Real。消融：SmolVLA 语言条件 + 双系统结构平均最稳；**3DGS / 高保真场景** 是 zero-shot Sim2Real 主因。
- **相关资料**：[PDF](https://arxiv.org/pdf/2509.14687) · [abs](https://arxiv.org/abs/2509.14687)
- **数据规模**：**1200** 人形轨迹；5 类任务
- **模型大小**：平台+benchmark（SmolVLA 等对比）
- **算力**：VR+Isaac Sim 管线
- **世界模型用法**：未使用。
- **提及**：√

---

## 附录 A · Benchmark 汇总反查表（2026 冠军 / Top3 一览）

> 沿用 [vla_sota_ls.md](vla_sota_ls.md) 第 47–58 行口径。

| Benchmark | SOTA (#1) | Top2 | Top3 | 备注 |
|-----------|-----------|------|------|------|
| **LIBERO** | StarVLA-α **98.8%** | Xiaomi-Robotics-0 **98.7%** | ABot-M0 **98.6%** | 全为 2026 |
| **LIBERO-Plus** | RLDX-1 **86.7%** | OA-WAM **83.9%** | PokéVLA **83.5%** | 全为 2026；π0 仅 56.1% |
| **LIBERO-Pro** | GLaD **59.47%**（2025） | π0.5（评测条目）**62%\*** | PRTS **58.8%** | \*评测论文条目，非模型本体 |
| **SimplerEnv** | Octo+PLD **96.63%**（2025） | FASTerVLA **87.9%** | — | X-VLA 报 Simpler-WidowX 95.8% / VM 80.4% / VA 75.7%（无单一聚合分）；2026 最高 GST-VLA 80.2% |
| **RoboTwin 2.0** | Fast-WAM **91.83%** | StarVLA-α **88.3%** | RLDX-1 **87.8%** | Top3 全为 2026 / π 系 2026 |
| **RoboCasa** | Cosmos Policy **67.1%** | HAMLET **66.4%**（300 demos） | World2Act **66.3%** | Top3 全为 2026 |
| **CALVIN** | Xiaomi-Robotics-0 **4.75** | NS-VLA **4.72** | UD-VLA + Fast-dVLA **4.54** | Top3 含多个 2026 |
| **ManiSkill2** | ConsisVLA-4D **94.3%** | — | — | 2026 SOTA |
| **RoboCasa365** | RLDX-1 **32.1%** | π0 **14.8%** | — | 明显领先 π0 |
| **MolmoSpace / MolmoBot / RoboEval** | MolmoAct2 / MolmoBot 系 | — | — | 2026 新榜 |
| **HumanoidBench-Hard** | EZ-M（10/14 SOTA） | — | — | 2026 新榜 |
| **EmbodiedBench EB-ALFRED** | ELITE **70.8%** | Claude-3.5 66.4 | ERA 65.2 | 2026 新榜 |
| **R2R / R2R-CE** | P3Nav / SACA / BTK | — | — | VLN 路线 |
| **REVERIE** | P3Nav SR 60.06 / RGS 39.75 | BTK | — | VLN 路线 |
| **RxR-CE** | SACA SR 60.3 | P3Nav | — | VLN 路线 |
| **RealMirror Humanoid** | SmolVLA **79.75%** | Diffusion Policy 75.15% | ACT 73.55% | 2025/ICRA2026 平台 |
| **DROID 真机（自建对照）** | TiPToP **59.4%** / MolmoBot **79.2%** | — | — | vs π0.5-DROID 33.3% / 39.2% |
| **AgiBot G1（DreamZero 对比）** | DreamZero seen 62.2% / unseen 39.5% | best pretrained VLA 27.4% | — | DreamZero 自报 |

> 真机长程 / Sim2Real 设定下普遍报告 **较 π0.5 显著提升**（参见 π0.7、STARRY、TiPToP、MolmoB0T、DreamZero 等小节）。

---

## 附录 B · 论文统计

- 主索引共 **68 篇 unique 论文**（去重：源文件第 35 行 `Cosmos Policy + World2Act` 与第 30 行 World2Act 合并、与第 40 行 Cosmos Policy 合并）。
- 「提及」列 √ 共 **38 篇**（覆盖：源文件第 19、26（待考量）、35、40、50（部分）、52–58 子表头、104–139 补漏节、147–152 用户追加节中显式 √ 的条目）。
- 时间分布：
  - 2026-05：5 篇（OA-WAM / ConsisVLA-4D / RLDX-1 / MolmoAct2 / LWD）
  - 2026-04：14 篇（PRTS / STARRY / LoHo-Manip / PokéVLA / VLA Foundry / MWM / π0.7 / ReconVLA / HAMLET / StarVLA-α / STRONG-VLA / HiF-VLA / HY-Embodied-0.5 / HiPolicy）
  - 2026-03：18 篇（Psi-R2 博客 / FocusVLA / BTK / VLA-OPD / ELITE / P3Nav / GigaWorld-Policy / MolmoB0T / Fast-WAM / SmoothVLA / Ψ0 / World2Act / FutureVLA / TiPToP / GST-VLA / NS-VLA / SACA / EZ-M）
  - 2026-02：15 篇（WoVR / Xiaomi-Robotics-0 / ABot-M0 / MINT / VLA-JEPA / DreamZero / World-VLA-Loop / VLAW / VLANeXt / SimVLA / Green-VLA / GeneralVLA / LifeLong-RFT / QuantVLA / LAP）
  - 2026-01：9 篇（Cosmos Policy / Pose-VLA / Being-H0.5 / LingBot-VLA / SOP / TT-VLA / Genie Sim 3.0 / Helix 02 / CycleVLA）
  - 2025-12：2 篇（GR00T N1.6 / OXE-AugE）
  - 2025-11：1 篇（π*0.6 / Recap）
  - 2025-10：2 篇（X-VLA / CoLA-World）
  - 2025-09：2 篇（FLOWER / RealMirror）
- 合计 **68 篇**。

---

## 附录 C · 来源与口径说明（与 [vla_sota_ls.md](vla_sota_ls.md) 同口径）

1. **数据来源**：RLDX-1 论文链接 **arXiv:2605.03269**；π0.5 LIBERO **97.7%** 来自 [vla-eval harness](https://arxiv.org/abs/2603.13966) 复现。
2. **「是否优于 π0.5」**：有 π0.5 公开分数的主要为 **LIBERO（97.7%）**；其余榜用 **π0** 作参考并标明「π0.5 未测」。
3. **未列入但值得跟踪的高分 2026 工作**（未达全局 Top3 但分数很高）：FutureVLA（LIBERO 98.3%）、PRTS（LIBERO 98.4%）、Cosmos Policy（LIBERO 98.5%）、FocusVLA（98.6%）等距 Top3 仅 **0.2–0.5pp**。
4. **真机榜**：RoboChallenge 上 StarVLA-α Generalist **33.6%** 为榜内高位；RoboArena 以分布式真机偏好为主。
5. **新综合框架**：VLA-Arena / CEBench（2025.12–2026）。
6. **MolmoAct2 与 MolmoB0T 为不同论文**，勿混淆（前者 Allen AI ARM 推理型；后者 Allen AI 纯仿真训练 + 真机 zero-shot 79.2%）。

