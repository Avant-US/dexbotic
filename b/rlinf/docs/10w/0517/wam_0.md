# VLA / VLM / WAM 如何使用世界模型、WAM 与视频生成技术

> 综合梳理 2017–2026 年经过验证的主流用法，按世界模型所扮演的**角色**分类，覆盖 30+ 个系统。
> 每段内容末尾以 `[编号]` 标注出处，完整引用列表见文末「参考文献」。

---

## 一、训练时表征学习（Video Co-training）

视频生成任务作为辅助训练目标，强迫模型学到物理先验（运动、接触、遮挡等动力学信息）。**推理时不生成视频**，视频生成在训练阶段起"物理正则化"作用。

### 1.1 Fast-WAM

Wan2.2-5B 视频 DiT backbone + 1B action expert；shared-attention MoT 结构化注意力 mask 解耦 video-cotraining 与 action 生成。**关键发现**：训练时视频 co-training 比测试时显式 future imagination 更重要。单遍动作头 + 流匹配是 RoboTwin 高分关键。RoboTwin 2.0 SOTA 91.83%；LIBERO 97.6%；190 ms 单 GPU 延迟（4× 加速 vs imagine-then-execute WAM）。[1]

### 1.2 GigaWorld-Policy

**因果 WAM**：动作预测 + action-conditioned 视频生成共训；causal mask 防 future-video token 影响 action token，使推理时可跳过显式视频生成。**不强制每步生成视频** 仍保 SOTA 且更快。RoboTwin 2.0 较 π0.5 +95%（如 Place Fan 0.25 → 0.94）；推理 9× 加速 vs Motus。轻量 GigaWorld-0.5 backbone 是预训练核心。[2]

### 1.3 GR-2（ByteDance）

在 38M 视频片段 / 50B token 上预训练视频生成模型，再联合微调视频生成与动作预测。真机 100+ 任务平均成功率 97.7%。网络规模视频预训练提供强物理先验，使少量真机数据即可迁移。[3]

### 1.4 LingBot-VA（蚂蚁集团）

视频帧与动作命令交织为单一序列，自回归预训练 1.4T token。LIBERO long-sequence 98.5%，RoboTwin 2.0 双臂 90%+。在光照/噪声/布局扰动下鲁棒率 74.2%。但**相机视角和初始状态变化仍是软肋**——视频先验对几何配置变化无能为力。[4]

### 1.5 DreamZero

**Wan2.1-I2V-14B WAM** + 联合 video-action flow；零样本策略可直接应用 WAM 做开放任务。AgiBot G1 真机 seen 82% vs π0.5 27%（≈3×）；unseen task progress 62.2% vs 27.4%（2.3×）。KV cache 真观测替换 + DreamZero-Flash 1-step + CFG 并行实现 38× 加速；**联合训练优于分离训练**。[5]

### 1.6 STARRY

**时空-动作联合扩散** + GASAM（Geometry-Aware Selective Attention Modulation，将预测深度/末端几何转 token-aligned 权重）。联合预测未来 spatial-temporal latent + 动作是 RoboTwin / 真机双涨点核心；几何感知模块对接触丰富任务最有效。RoboTwin 2.0 Clean 93.82% / Random 93.30%（50 双臂任务，自报 SOTA）；真机平均成功率 70.8%（vs π0.5 42.5%，+28.3pp）。[6]

### 1.7 Psi-R2 / Psi-W0（灵初智能）

**Psi-R2**：视频-动作联合 WAM（基于 Wan2.2）。预训练 95,472h 人类 + 5,417h 真机数据。**Psi-W0**：动作条件世界模型用于策略评估与数据质检。"Raw Data In" 极简人类/真机对齐管线。3D 轨迹精度通过定制外骨骼手套达到亚毫米。人类数据规模化 + WM 评估闭环是核心理念。[7]

### 小结

这是目前**验证最充分、涨点最稳**的用法。核心洞见：视频生成在训练时起"物理正则化"作用，迫使表征编码运动、接触、遮挡等动力学信息。Fast-WAM [1] 和 GigaWorld-Policy [2] 独立证明：**推理时不需要真的去想象未来**。"先想象未来再行动"这个直觉性框架不如"用视频预训练来学好的表征"有效。

---

## 二、预训练骨干初始化

用已训好的视频世界模型的权重 / 表征来初始化策略网络——"站在 WM 的肩膀上"。不做联合模型，而是把 WM 当成一种预训练手段。

### 2.1 Cosmos Policy（NVIDIA）

基于 Cosmos-Predict2 世界基础模型全参微调为策略头；通过视频预训练 + latent dynamics 提供 manipulation 先验。RoboCasa SOTA 67.1%（#1）；LIBERO 98.5%。仅需同类方法 1/6 的演示数据。消融：**世界模型预训练初始化是 RoboCasa 关键；从零训显著降**（断崖式下降）。模型基规划（model-based planning）在困难真机任务上额外提升 12.5%。[8]

### 2.2 World2Act

在 **Cosmos 世界模型** 上做技能组合潜动作后训练；contrastive matching 把 VLA 动作直接对齐 WM 的 video-dynamics latent，**避开像素监督的伪影 / 幻觉**。LLM-based skill decomposition 切分高阶指令到低阶 prompt 应对变长 horizon。RoboCasa T2 66.3%（搭配 Cosmos Policy）；LIBERO 平均 98.1%。消融：**技能组合世界模型优于单任务后训练；contrastive latent 对齐优于 RGB 监督**。[9]

### 2.3 Ψ0（Psi-Zero）

**世界模型蒸馏** + 两阶段（人类视频自回归预训练 → 高质量人形数据 flow expert post-train）；Qwen3-VL-2B + 500M MMDiT + RL 跟踪控制器。800h 人类视频 + 30h 真机 → 较 10× 数据基线 +40%。80 条轨迹即可学长程新技能。消融：**人类视频预训是数据效率关键**；Stage 解耦解决人-机运动学差异。[10]

### 2.4 GR00T N1.6（NVIDIA）

GR00T N1.6 升级：**Cosmos-Reason-2B + 32 层 DiT**；state-relative action chunk 解决长程稳定性。面向 humanoid 通用 generalist 模型。RoboCasa 官榜 SOTA #1 Overall 21.9%。结合 Cosmos Reason 世界模型做 VLA 推理与任务分解。消融：state-relative action chunk 优于 absolute；Cosmos-Reason backbone 比纯 VLM backbone 在 RoboCasa 上更稳。[11]

### 小结

**最实用的路线之一**。WM 在海量视频上学到的时空表征，迁移到机器人上非常高效。Cosmos Policy [8] 的消融直接说明"WM 预训练初始化"是涨分关键，去掉后断崖式下降。World2Act [9] 进一步说明**不要在像素空间对齐**（会引入幻觉），应在 latent 空间用对比学习对齐。

---

## 三、学到的仿真器——在 WM 想象中训 RL

把世界模型当成"可微的仿真环境"，在里面跑 RL 来改进已有的 VLA，再部署到真机。

### 3.1 DayDreamer

Dreamer/RSSM 世界模型在真机上在线学习，在想象的 rollout 中训练 actor-critic。异步 actor-learner 架构。验证了 4 种真实机器人：Unitree A1 四足机器人 **1 小时从零学会走路**；UR5/XArm 从相机图像和稀疏奖励学习 pick-and-place（3.1 物体/分钟）；Sphero 导航 2 小时完成。**同一套超参数跨所有实验**，样本效率远超 model-free 基线（SAC、Rainbow、DrQv2）。局限：光照突变时初始失败，但数小时内自适应。[12]

### 3.2 DreamerV3

RSSM 世界模型 + actor-critic 全想象训练，单一超参数集跨 150+ 任务。从简单连续控制到 Minecraft（从零收集钻石）均不需调参。与专门方法竞争或胜出。Nature 2025 发表。[13]

### 3.3 UniSim（ICLR 2024 杰出论文）

从视频学来的**通用交互式仿真器**，条件于动作 / 文本。RL agent 在 UniSim 内训练，零样本部署到真实机器人。RL 在 UniSim 内训练显著提升 VLA 策略性能。[14]

### 3.4 WoVR

用 WM 当**可靠 sim** 做 VLA 后训练：在 WM 仿真环境中 RL fine-tune VLA。LIBERO 39.95% → 69.2%（+29.3pp）后训练增益；真机 61.7% → 91.7%（+30.0pp）。消融：**WM 质量决定 post-train 上限**；低保真 WM 反而误导——这是该路线的硬约束。[15]

### 3.5 World-VLA-Loop

**VLA↔WM 闭环共改进**：每步动作经 world model 反推 → 失败轨迹回流 WM 训练。真机闭环较基线 +36.7%。消融：**失败轨迹回流 WM 是涨点关键**；单向"WM→VLA"不够，闭环 reweight 优于 open-loop。[16]

### 3.6 VLAW

**迭代共改进** WM+VLA：交替训 WM 与 VLA，使用合成数据 + 真实数据混合。绝对成功率 +39.2%。消融：合成 + 真实混合迭代优于单一来源；同步迭代 > 交替单边训练。[17]

### 3.7 EZ-M（EfficientZero-Multitask）

扩任务数而非单任务样本数；shared world model + Gumbel search；task-sharing 架构 + 学习任务嵌入。HumanoidBench-Hard 14 任务中 10 个达 SOTA。仅 1M env steps、**16M 参数**（vs BRC 1B），训练 ~4× 快。Independent Experience Replay 是关键；shared dynamics/reward/value 在相关任务间梯度相似度更高（解释跨任务正迁移）。[18]

### 3.8 TD-MPC2（ICLR 2024 Spotlight）

无解码器 latent 世界模型 + planning。单一超参数集跨 104 任务（DMControl、MetaWorld、ManiSkill2、MyoSuite）。规模化到 317M 参数，跨多种具身形态。DMControl 全部 39 个任务 >90%。超越 SAC 和 DreamerV3 的效率和性能。[19]

### 3.9 Robotic World Model / RWM

双自回归世界模型做策略优化（MBPO-PPO）。自监督训练。在想象 rollout 中交替真实数据收集、模型更新和长 horizon PPO。零样本迁移到 ANYmal D 和 Unitree G1 硬件做速度跟踪。[20]

### 小结

这条路线的天花板是 **WM 保真度**。DayDreamer [12] 证明了即使在真机上在线学 WM 也可行（1 小时学走路！），但 WoVR [15] 警告低质量 WM 会误导策略。World-VLA-Loop [16] 和 VLAW [17] 的闭环是解法——让 VLA 的失败经验反过来改进 WM。

---

## 四、视频规划器——先生成视频，再提取动作（Predict-then-Act）

WM 生成未来执行的视频计划，再用逆动力学或光流从视频中解码出动作序列。

### 4.1 Visual Foresight（Finn & Levine, 2017）——奠基之作

从无标签数据学习动作条件视频预测模型；通过 MPC 优化动作，使预测帧与目标帧距离最小。真机推拿新物体，无需标定、3D 模型或深度传感器。开创了视频预测用于操作规划的路线。局限：仅限推拿，性能有较大提升空间。[21]

### 4.2 UniPi（Google）

1.7B 参数视频扩散模型从文本生成轨迹视频 → 逆动力学模型提取 7-DoF 动作。仿真多任务操作。泛化到未见语言提示组合；非机器人数据预训练提升 FID/FVD/CLIPScore。**但 256+ TPU pods，推理极慢**。compounding error：视频生成误差 × 动作提取误差双重累积。[22]

### 4.3 AVDC（ICLR 2024）

视频预测 + 密集光流 → RGBD 重建 3D 运动，**无需动作标签**。真机 Franka（~20 demo 微调）；MetaWorld 仿真。90% 零样本跨具身迁移（Visual Pusher）；31.3% iTHOR 导航。局限：遮挡时丢失跟踪；累积光流估计误差；RGB 无法获取力信息。[23]

### 4.4 RoboDreamer

组合式视频生成：将语言解析为动词/介词短语 → 分别条件化不同扩散模型 → 组合。RT-X 数据集评估。组合分解使泛化到**新指令组合**。[24]

### 4.5 Dream4manip / "Say, Dream, and Act"

帧率无关的视频想象用于长 horizon 操作。Say（鲁棒视频世界模型）→ Dream（长度无关未来想象）→ Act（想象轨迹作为动作模型的 in-context 示例）。长 horizon 操作基准 SOTA。[25]

### 小结

这是最直觉的用法，但 **compounding error 是致命问题**——视频生成误差 × 逆动力学误差双重累积 [22][23]。后续工作（如 SuSIE [26]，见第五节）换了个思路——只生成一帧目标图像而非整段视频——反而效果更好，说明**生成越少、误差越小**。

---

## 五、目标 / 子目标图像生成（Subgoal Synthesis）

WM 不生成完整视频，而只生成"目标状态"的图像，再由目标条件策略去执行。

### 5.1 SuSIE

微调 InstructPix2Pix 生成目标子图像（当前观测 + 语言 → 目标图像） → 下游 diffusion policy 以目标条件执行。真机 Bridge 设定。**击败 RT-2-X**（55B 参数、20× 训练数据）。增加 Something-Something 人类视频数据（220K 片段）进一步提升零样本泛化。[26]

### 5.2 Dreamitate（CoRL 2024）

视频扩散模型从人类演示生成机器人执行视频（弥合具身差距）→ 3D 跟踪提取动作。真机测试 8 个未见形状（训练 26 个字母形状）。数据减少时仍稳定（Diffusion Policy 退化）。对未见物体/环境泛化显著更好。[27]

### 小结

生成一帧目标图 [26] 比生成整段视频 [22] 效果更好，核心原因是减少了 compounding error 的暴露面。Dreamitate [27] 进一步说明视频扩散可以用来弥合人-机具身差距。

---

## 六、潜在空间世界模型——不生成像素

不做真正的视频生成，只在 latent 空间加预测未来状态的辅助 loss 或做 MPC。这是 **性价比最高** 的路线。

### 6.1 V-JEPA 2-AC（Meta）

V-JEPA 2 自监督视频预训练 → 300M transformer action-conditioned 世界模型后训练 → latent 空间 MPC 规划。真机 Franka：reach 100%，操作 60-80%。仅 62 小时无标签交互数据（DROID）。零样本迁移到新环境。**16 秒/动作 vs Cosmos Policy 的 4 分钟**，推理效率高一个数量级。[28]

### 6.2 VLA-JEPA

**JEPA 式潜世界模型**辅助 VLA 动作预测；不重建像素而预测 latent embedding。LIBERO-Plus 79.5%。消融：**潜预测辅助损失**提升 LIBERO-Plus 鲁棒性，对**视角扰动**尤为有效。[29]

### 6.3 MWM（Mask World Model）

**用语义 mask 替代 RGB 像素**作为视频世界模型预测目标，形成「几何信息瓶颈」聚焦物理动力学/接触关系。两阶段训练：DiT mask 预测 + diffusion policy。LIBERO 98.3%；RLBench 平均成功率 68.3%（GE-ACT RGB 基线 30.8%，**翻倍以上**）。消融：**mask 监督减外观噪声；token pruning 鲁棒性显著优于 RGB WM**。[30]

### 6.4 DINO-WM

冻结 DINOv2 patch 特征作为观测空间 → ViT 动力学预测器在离线数据上训练 → 测试时 MPC 规划（latent 空间目标到达）。仿真 Reacher/Push-T：SR 0.82 vs DreamerV3 0.76 vs IRIS 0.06。关键消融：**DINOv2 空间 patch embedding 是核心；换成全局 token（R3M、ResNet18、CLS token）显著退化**。[31]

### 6.5 Seer / PIDM（ICLR 2025 Oral）

Predictive Inverse Dynamics Model：单一 Transformer 联合预测未来视觉状态（条件视觉前瞻）+ 推断到达该状态所需的动作（逆动力学），端到端训练。DROID 76K 轨迹预训练。CALVIN ABC-D SOTA 4.28-4.30；LIBERO-LONG +13%；CALVIN ABC-D +22%；真机 +43%。**端到端单模型 >> 两阶段解耦（先预测再逆推）**。清晰的 scaling 行为——更大模型更好。[32]

### 6.6 FutureVLA

**联合视觉-运动预测**：Joint Visuomotor Gating 结构性分离视觉状态保留与时序动作建模；**latent embeddings alignment** 后训练可 plug-in 到多种 VLA。LIBERO 98.3%；LIBERO-Plus 81.3%；SimplerEnv 75.87%。消融：Joint Visuomotor Gating + 潜对齐后训练对多 VLA 通用。[33]

### 6.7 CoLA-World

**LAM + WM 共进化**：latent action model 与 world model 联合训练。消融：**warm-up 防表征坍塌是必须的**；纯交替训练易坍塌到 trivial latent。[34]

### 6.8 LeWorldModel

极简 JEPA 世界模型：只有两个 loss（next-embedding prediction + Gaussian regularizer）。15M 参数，单 GPU 训练。2D/3D 控制任务。规划速度 **48× 快于基础模型 WM**，同时保持竞争力。[35]

### 小结

Latent WM 是**性价比最高**的路线。V-JEPA 2-AC [28] 用 62 小时无标签数据达到 60-80% 真机操作成功率，而且推理速度是 Cosmos Policy 的 15 倍。MWM [30] 的发现最有启发性：预测目标应是**几何/语义信息**而非 RGB 像素，形成信息瓶颈迫使模型聚焦物理动力学。Seer [32] 证明预测+逆动力学端到端单模型优于两阶段解耦。五篇独立工作（World2Act [9]、VLA-JEPA [29]、MWM [30]、DINO-WM [31]、Seer [32]）共同指向：**latent 空间 >> 像素空间**。

---

## 七、世界模型提供奖励信号

用 WM 的视频预测给 RL 提供 dense reward，替代人工设计奖励函数。

### 7.1 VIPER（NeurIPS 2023）

在专家视频上训练自回归 transformer（VideoGPT），用观测轨迹在该模型下的 log-likelihood 作为 DreamerV3 的 reward。15 个 DMC 任务（接近专家水平）、7 个 Atari 任务、6 个 RLBench 任务。RLBench 上 **VIPER 超过用 ground-truth sparse reward 的任务 oracle**——因为 VIPER 提供 dense reward。跨具身迁移在 Franka 上验证。关键发现：**必须 4× 降采样视频，否则模型偏好静止轨迹**（高帧率下静止帧获得高 likelihood）。局限：需要领域内专家视频数据；可能偏好最小化模型不确定性的状态。[36]

### 7.2 Diffusion Reward

条件视频扩散模型提供 reward，类似 VIPER 但用扩散模型替代 VideoGPT。在未见真机和仿真轨迹上泛化更强。[37]

### 小结

反直觉的发现——VIPER [36] 提供的 dense reward **比任务 oracle 的 sparse reward 还好**，因为密集信号引导更有效。但有陷阱：高帧率时模型偏好静止轨迹，必须降采样。

---

## 八、数据增广

用生成模型创造新的训练场景/轨迹，扩充有限的真机数据。**最无风险**的用法——不改模型架构，不引入 compounding error，纯粹扩充数据。

### 8.1 ROSIE（Google）

开放词汇分割定位增广区域 → Imagen Editor 文本引导修补（换物体/背景/干扰物）→ 增广数据训练 RT-1 策略。真机验证。泛化到未见物体和干扰物/背景显著提升。无需深度或 3D 模型。[38]

### 8.2 GenAug

深度引导扩散模型改变背景、物体纹理、类别，添加干扰物。真机验证。能将行为迁移到新场景。局限：需要深度和人工指定 mask（不如 ROSIE 自动）。[39]

### 8.3 DreMa（ICLR 2025）

物体级 Gaussian Splatting 重建 3D 场景 → 嵌入物理引擎（PyBullet）→ 重排物体并重放动作 → 渲染新演示。真机 Franka one-shot 学习。单任务 +9.1%、多任务 +13.1% vs PerAct；超过不变增广技术 5.0%。**需使用 2DGS 而非标准 3DGS**，因为 3DGS 在物体边缘有深度渲染伪影。[40]

### 8.4 GR00T-Dreams（NVIDIA）

用 Cosmos 世界基础模型生成合成轨迹数据。GR00T N1.5 人形策略开发：**36 小时**生成数据 vs 约 3 个月人工采集。[41]

### 8.5 MolmoBot-Engine

**纯仿真 MolmoBot-Engine**：MolmoSpaces 程序化生成 100,000+ 环境、30,000+ 资产、8 任务类型 → 1.7M 轨迹训练。DROID 真机 40 任务 Overall 79.2%（vs π0.5 39.2%）。3DGS / 高保真场景是 zero-shot Sim2Real 成功的主要因素。**纯仿真训练匹敌甚至超越大规模真机 VLA**。[42]

### 8.6 RoboSplat（RSS 2025）

3DGS 做 6 类真机泛化增广（物体/背景/光照/视角/干扰物/布局）。真机验证多种泛化场景。[43]

### 小结

数据增广是**最无风险**的用法。GR00T-Dreams [41] 把 3 个月缩到 36 小时特别有说服力。MolmoBot [42] 证明足够高保真的合成数据甚至可以完全替代真机数据。

---

## 九、3D 场景表示做世界模型（NeRF / 3DGS）

用显式 3D 表示做"几何世界模型"，支持任意视角渲染、碰撞推理和物理仿真。在**深度传感器失败的场景**（透明、反光、镜面物体）有不可替代的价值。

### 9.1 Dex-NeRF

NeRF 渲染透明物体的高质量几何 → 抓取规划。真机 ABB YuMi。**透明物体抓取 90-100%**（传统深度传感器在这些物体上完全失败）。[44]

### 9.2 SPARTN（CVPR 2023）

NeRF 合成新视角训练数据 → 6-DoF 抓取策略。真机 Franka。绝对抓取成功率 +22.5%（89% vs 30% baseline）。有限演示即可工作。[45]

### 9.3 GWM（ICCV 2025）

3DGS 做 action-conditioned 未来状态预测 + model-based RL。真机 Franka pick-and-place 65% 成功率（vs Diffusion Policy 35%）。Meta-World 收敛速度 ~2× 快于 iVideoGPT。消融：高斯表示 vs 图像 WM → 4% → 18-24%（RoboCasa）。[46]

### 9.4 PEGS（CoRL 2024 Oral）

双高斯-粒子表示创建物理可纠正数字孪生。物理移动视觉，视觉误差生成纠正力回物理——闭环。真机 Franka 30Hz 实时运行。策略永远作用于仿真机器人；真机跟随。演化为 "real-is-sim" 数字孪生（60Hz）。[47]

### 9.5 iVideoGPT（NeurIPS 2024）

自回归 transformer 将视觉观测、动作和奖励统一为 token 序列，产出交互式下一帧预测。预训练在 1.5M 机器人+人类操作轨迹（Open X-Embodiment）。首次成功将 MBPO 应用于视觉连续控制——匹配或超过 DreamerV3（Meta-World）。压缩 tokenization 减少视频 token 16×。预训练在数据稀缺下（100-1000 轨迹）收益最显著。仿真验证。[48]

### 9.6 ParticleFormer + MPPI

Transformer 3D 点云世界模型预测多材料、多物体动力学 → MPPI（Model Predictive Path Integral）做轨迹优化。真机 UFACTORY xArm-6，3 个真机任务 + 6 个仿真任务。一致超越 GNN baseline。同时处理刚性、可变形和柔性材料。目前限于逐场景训练。[49]

### 小结

NeRF/3DGS 在深度传感器失败的场景有不可替代的价值 [44][45]。PEGS [47] 的"real-is-sim"思路很有启发：不做 sim2real，而是让仿真跟踪真实。3DGS 的优势是实时渲染 + 可组合 + 显式几何。

---

## 十、策略评估与质量控制

WM 不直接出动作，而是评估策略质量或检测失败。

### 10.1 IRASim（ByteDance, ICCV 2025）

细粒度视频世界模型，逐帧动作条件化的扩散 transformer。生成分辨率 288×512、150+ 帧视频。与 MuJoCo 真值的 **Pearson r = 0.99**。Push-T IoU 从 0.637 提升到 0.961（model-based planning）。[50]

### 10.2 Psi-W0（灵初智能）

动作条件世界模型用于策略评估与数据质检。配合 Psi-R2 [7] 形成训练闭环：WM 评估策略质量 → 筛选高质量数据 → 改进策略。[7]

### 10.3 Interactive World Simulator

一致性模型做图像解码和 latent 动力学。单卡 RTX 4090 仿真 10+ 分钟、15 FPS。在世界模型生成数据上训练的策略，与等量真实数据训练的策略表现可比。仿真和真实性能之间有强相关性。[51]

---

## 十一、安全约束与可靠性

WM 用于检测不安全状态、约束策略行为。

### 11.1 SafeVLA（NeurIPS 2025 Spotlight）

将 VLA 安全对齐形式化为约束马尔可夫决策过程（CMDP）。两类约束：Object Safety（不损坏无关物体）和 Robot Safety（不损坏硬件）。安全性 +83.58%，任务性能 +3.85%。不安全行为上界降到 SOTA 的 1/35。仿真中对齐的策略成功迁移到物理机器人。Safety-CHORES benchmark 是首个 VLA 安全评估基准。[52]

### 11.2 ReconVLA

**Conformal prediction** 给 VLA 动作 token 校准不确定性；状态空间异常检测识别不安全状态。可 plug-and-play 到已训 VLA 上无需重训。消融：实时 OOD 触发回退/求助人类对 fail-safe 部署关键。[53]

---

## 十二、语言 / 常识作为隐式世界模型

VLM 编码的世界知识本身就是一种"隐式世界模型"。

### 12.1 CoPa（IROS 2024）

VLM 作为隐式世界模型编码常识物理知识。VLM 在部件级推理空间约束 → 分解操作为任务导向抓取（抓哪个部件）+ 任务感知运动规划（满足什么空间约束）。VLM 生成几何约束而非前向动力学预测。真机 Franka 10 个任务 63% 成功率，显著超 VoxPoser baseline。零样本，无需额外训练。[54]

### 12.2 Affordance 预测——RT-Affordance & A0

**RT-Affordance**（Google, CoRL 2024 Workshop）：层次模型先从语言+图像预测 affordance 计划（任务关键阶段的机器人姿态），再以 affordance 条件化策略。真机 69% 整体成功率 vs 语言条件策略 15%、RT-2 仅 3%（超抓取任务）。Affordance 对新设定鲁棒，桥接异构数据源（web + 机器人轨迹 + 领域内 affordance 图像）。**A0**（ICCV 2025）：预测物体级接触点和接触后轨迹。[55][56]

---

## 十三、环境工厂——生成完整交互式世界

WM 作为无限环境生成器，解决数据/环境瓶颈。

### 13.1 Genie 2 / 3（DeepMind）

从单张图像生成动作可控的交互式 3D 世界。模拟物理、物体交互、光照、NPC 行为和物体持久性。训练在视频数据上。Genie 2 生成一致世界 ~10-20 秒（360p）；后续 Genie 3（2025 年 8 月）扩展到 ~1 分钟。SIMA 2 agent 在 Genie 生成的环境中自我改进，"substantially closing the gap with human performance"。未在物理机器人上验证。与具体任务的 WM 不同，这里的 WM 是**整个环境的生成器**。[57]

---

## 十四、部件级运动先验

### 14.1 Puppet Master（ICCV 2025）

交互式视频扩散模型：从单张图像 + 稀疏拖拽轨迹生成部件级运动视频。微调 Stable Video Diffusion。在合成数据（Objaverse-Animation-HQ）训练，零样本泛化到真实图像。超越 DragAnything、MotionCtrl、DragNUWA。all-to-first attention 机制提升生成质量。理解铰链/可变形物体的部件运动（门开关、抽屉滑动、肢体运动）。尚未集成到机器人控制管线中。[58]

---

## 十五、OA-WAM——对象可寻址世界动作模型

### 15.1 OA-WAM

将场景分解为 **N+1 对象槽位**（地址向量 addr + 内容向量）。世界头预测下一帧槽位状态，flow-matching 动作头一次前向出 16 步动作。LIBERO 97.8%；LIBERO-Plus T2 83.9%（全局 #2）；SimplerEnv 79.3%。消融：**地址键 cross-slot 注意力 + 残差流地址槽位逐层重置**是 swap-binding 高分关键（cosine 0.87 vs holistic baseline ≤ 0.09）。[59]

---

## 十六、跨论文一致性结论——什么真的起作用？

### 确定有效 ✓

1. **训练时 co-train >> 推理时 imagination**（Fast-WAM [1]、GigaWorld [2] 独立验证）
   - 视频生成的价值在于训练时的物理正则化，不在推理时的想象

2. **Latent 空间 >> 像素空间**（World2Act [9]、VLA-JEPA [29]、MWM [30]、DINO-WM [31]、Seer [32] 五篇独立验证）
   - 像素预测引入外观噪声/幻觉/伪影，latent/语义空间更鲁棒

3. **WM 预训练初始化**大幅提升数据效率（Cosmos Policy [8]、Ψ0 [10]、GR-2 [3]）
   - 去掉初始化断崖式下降；demo 需求降到 1/6

4. **高保真合成数据可以完全替代真机数据**（MolmoBot [42] 79.2% vs π0.5 39.2%）

5. **闭环 >> 单向**（World-VLA-Loop [16]、VLAW [17]）
   - WM 和 VLA 应互相改进，失败经验回流 WM

6. **生成越少越好**（SuSIE [26] 只生成一帧目标图 >> UniPi [22] 生成整段视频）
   - 减少 compounding error

7. **端到端 >> 两阶段**（Seer [32] 端到端 >> 先预测再逆推；DreamZero [5] 联合 >> 分离）

8. **Dense reward 可以超过 sparse oracle reward**（VIPER [36]）

### 已知陷阱 ✗

1. **低保真 WM 反而误导**（WoVR [15] 明确警告）
2. **视频 WM 在长 horizon 上空间漂移、物体消失**（幻觉问题）
3. **视角 / 初始状态变化仍是软肋**（LingBot-VA [4]，多篇验证光照鲁棒但视角脆弱）
4. **RGB 视频做 reward 时偏好静止轨迹**（VIPER [36] 需 4× 降采样）
5. **两阶段管线（先视频后逆动力学）compounding error 严重**（UniPi [22]、AVDC [23]）
6. **计算成本从数据侧转移到算力侧**——不是免费午餐（UniPi 256+ TPU，DreamZero 14B）
7. **Latent 共进化需要 warm-up 防坍塌**（CoLA-World [34]）

### 性价比排序

| 排名 | 方法 | 理由 |
|------|------|------|
| 1 | **数据增广** [38][40][41][42] | 零架构改动，纯粹多数据，风险最低 |
| 2 | **WM 预训练初始化** [8][3][10] | 一次性收益，demo 需求大幅降低 |
| 3 | **训练时 co-training** [1][2][5] | 涨点稳，推理无额外开销 |
| 4 | **Latent WM 辅助 loss** [29][30][32][33] | 轻量，可 plug-in，鲁棒性提升明确 |
| 5 | **WM 当仿真器跑 RL** [12][14][15] | 潜力大但受 WM 保真度制约 |
| 6 | **视频规划器** [22][23] | 直觉好但 compounding error + 算力开销重 |

---

## 参考文献

- [1] Fast-WAM: Do World Action Models Need Test-time Future Imagination? arXiv:2603.16666, 2026.
- [2] GigaWorld-Policy: An Efficient Action-Centered World–Action Model. arXiv:2603.17240, 2026.
- [3] GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation. https://gr2-manipulation.github.io/, ByteDance, 2024.
- [4] LingBot-VA / A Pragmatic VLA Foundation Model. arXiv:2601.18692, 2026. 鲁棒性数据引自 arXiv:2603.22078.
- [5] DreamZero: World Action Models are Zero-Shot Policies. arXiv:2602.15922, 2026.
- [6] STARRY: Spatio-Temporal Action-Centric World Modeling for Robotic Manipulation. arXiv:2604.26848, 2026.
- [7] From Human Skill to Robotic Mastery (Psi-R2 / Psi-W0 技术博客). 灵初智能, 2026-03-31.
- [8] Cosmos Policy. NVIDIA. arXiv:2601.16163, 2026.
- [9] World2Act: Latent Action Post-Training via Skill-Compositional World Models. arXiv:2603.10422, 2026.
- [10] Ψ0 (Psi-Zero): An Open Foundation Model Towards Universal Humanoid Loco-Manipulation. arXiv:2603.12263, 2026.
- [11] GR00T N1.6. NVIDIA. GitHub: NVIDIA/Isaac-GR00T; HF: nvidia/GR00T-N1.6-3B.
- [12] DayDreamer: World Models for Physical Robot Learning. Wu, Hafner, Abbeel et al. CoRL 2022. arXiv:2206.14176.
- [13] DreamerV3: Mastering Diverse Domains through World Models. Hafner et al. Nature 2025.
- [14] UniSim: Learning Interactive Real-World Simulators. Yang et al. ICLR 2024 Outstanding Paper. https://universal-simulator.github.io/unisim/
- [15] WoVR: World Models as Reliable Simulators for Post-Training VLAs. arXiv:2602.13977, 2026.
- [16] World-VLA-Loop: Closed-Loop World Models for VLAs. arXiv:2602.06508, 2026.
- [17] VLAW: Vision-Language-Action World Model. arXiv:2602.12063, 2026.
- [18] EZ-M: Scaling Tasks, Not Samples — Mastering Humanoid Control through Multi-Task Model-Based RL. arXiv:2603.01452, 2026.
- [19] TD-MPC2: Scalable, Robust World Models for Continuous Control. Hansen et al. ICLR 2024 Spotlight. https://www.tdmpc2.com/
- [20] Robotic World Model (RWM). https://sites.google.com/view/roboticworldmodel, 2025.
- [21] Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control. Finn & Levine. ICRA 2017. arXiv:1610.00696.
- [22] UniPi: Learning Universal Policies via Text-Guided Video Generation. Google. 2023. https://universal-policy.github.io/unipi/
- [23] AVDC: Action-conditioned Video Diffusion for Control. Ko et al. ICLR 2024. https://flow-diffusion.github.io/
- [24] RoboDreamer: Learning Compositional World Models for Robot Imagination. 2024. https://robovideo.github.io/
- [25] Dream4manip / "Say, Dream, and Act". arXiv:2602.10717, 2025.
- [26] SuSIE: Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models. Black et al. 2023. https://rail-berkeley.github.io/susie/
- [27] Dreamitate: Real-Robot-Guided Imagination for Contact-Rich Manipulation. Columbia. CoRL 2024. https://dreamitate.cs.columbia.edu/
- [28] V-JEPA 2-AC: Self-Supervised Video Models Enable Efficient Robotic Manipulation. Meta. arXiv:2506.09985, 2025.
- [29] VLA-JEPA: Enhancing VLA with Latent World Model. arXiv:2602.10098, 2026.
- [30] Mask World Model (MWM): Predicting What Matters for Robust Robot Policy Learning. arXiv:2604.19683, 2026.
- [31] DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning. Zhou, LeCun, Pinto. 2024. https://dino-wm.github.io/
- [32] Seer: Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation. ICLR 2025 Oral. arXiv:2412.15109.
- [33] FutureVLA: Joint Visuomotor Prediction for VLA. arXiv:2603.10712, 2026.
- [34] CoLA-World: Co-evolution of Latent Action + World Model. arXiv:2510.26433, 2025.
- [35] LeWorldModel: Learning End-to-End JEPA World Models for Efficient Visual Planning. arXiv:2603.19312, 2025.
- [36] VIPER: Video Prediction Models as Rewards for Reinforcement Learning. Escontrela, Hafner, Abbeel et al. NeurIPS 2023. https://alescontrela.github.io/viper/
- [37] Diffusion Reward: Learning Rewards via Conditional Video Diffusion. arXiv:2312.14134, 2023.
- [38] ROSIE: Scaling Robot Learning with Semantically Imagined Experience. Google. 2023. arXiv:2302.11550.
- [39] GenAug: Data Augmentation for Finetuning Text Generators. (Robotic context) arXiv:2409.00951, 2023.
- [40] DreMa: Dream to Manipulate — Compositional World Models Empowering Robot Imitation Learning with Imagination. ICLR 2025. arXiv:2412.14957.
- [41] GR00T-Dreams: Enhance Robot Learning with Synthetic Trajectory Data Generated by World Foundation Models. NVIDIA. https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/
- [42] MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation. arXiv:2603.16861, 2026.
- [43] RoboSplat. RSS 2025. GitHub: InternRobotics/RoboSplat.
- [44] Dex-NeRF: Using a Neural Radiance Field to Grasp Transparent Objects. https://sites.google.com/view/dex-nerf
- [45] SPARTN: NeRF in the Palm of Your Hand — Corrective Augmentation for Robotic Grasping. CVPR 2023. https://bland.website/spartn/
- [46] GWM: Gaussian World Model for Streaming 3D Occupancy Prediction. (Robotic variant) ICCV 2025. arXiv:2508.17600.
- [47] PEGS: Physically Embodied Gaussian Splatting — A Realtime Correctable World Model for Robotics. CoRL 2024 Oral. arXiv:2406.10788. https://embodied-gaussians.github.io/
- [48] iVideoGPT: Interactive VideoGPTs are Scalable World Models. NeurIPS 2024. arXiv:2405.15223.
- [49] ParticleFormer: Transformer-based 3D Particle World Model for Multi-Material Manipulation. PMLR, 2025. arXiv:2506.23126.
- [50] IRASim: Learning Interactive Real-Robot Action Simulators. ByteDance. ICCV 2025. https://gen-irasim.github.io/
- [51] Interactive World Simulator. arXiv:2603.08546, 2026.
- [52] SafeVLA: Towards Safety Alignment of Vision-Language-Action Models via Safe Reinforcement Learning. NeurIPS 2025 Spotlight. arXiv:2503.03480. https://safevla.github.io/
- [53] ReconVLA: An Uncertainty-Guided and Failure-Aware VLA Framework. arXiv:2604.16677, 2026.
- [54] CoPa: General Robotic Manipulation through Spatial Constraints of Parts. IROS 2024. arXiv:2403.08248. https://copa-2024.github.io/
- [55] RT-Affordance: Affordances are the Right Abstraction for Robot Policy. Google. CoRL 2024 Workshop. arXiv:2411.02704.
- [56] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation. ICCV 2025.
- [57] Genie 2 / 3. Google DeepMind. Blog: https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/
- [58] Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics. Oxford VGG. ICCV 2025. arXiv:2408.04631. https://vgg-puppetmaster.github.io/
- [59] OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation. arXiv:2605.06481, 2026.
