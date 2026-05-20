# 具身智能VLA预训练研究资料罗列
_Exported on 5/16/2026 at 08:40:15 GMT+8 from Cursor (3.4.20)_

---

**User**

我要训练一个具身智能预训练模型,使得这模型能装在不同的机器人本体上zero-shot地执行各种不同的复杂任务, 请问我该怎么做? 列一下有些什么相关的, SOTA的论文, 文章或研究成果可以借鉴. 

---

**Cursor**

下面给一份比较完整、可落地的"具身智能基础模型(Embodied Foundation Model)"训练路线图,以及当前(2024–2026)最值得借鉴的 SOTA 工作清单。我按"整体路线 → 关键技术模块 → 论文/项目清单 → 实操建议"的顺序来讲。

---

## 一、整体技术路线(零样本跨本体的"机器人大脑"该怎么练)

目前业界对"装在不同机器人上 zero-shot 执行复杂任务"已经基本收敛到一种范式:**Vision-Language-Action(VLA)基础模型**,通常分三阶段训练:

```text
阶段 0  视觉-语言基座(VLM)预训练
        └─ 用 PaliGemma / Qwen2.5-VL / InternVL / LLaVA 等开源 VLM 作 backbone

阶段 1  机器人大规模"动作预训练"(Action Pre-training)
        ├─ 真实机器人轨迹:Open X-Embodiment / DROID / AgiBot World / RoboMIND
        ├─ 人类视频(无动作标签):Ego4D / EgoExo4D / SSv2  → 用 Latent Action(LAPA)
        └─ 仿真合成数据:RoboCasa / GR00T-Sim / DexMimicGen / Isaac Lab

阶段 2  下游本体 / 任务后训练(Post-training)
        ├─ Cross-embodiment 微调:在目标机器人上少量数据 SFT
        ├─ Flow Matching / Diffusion Action Head:连续高频精细动作
        └─ 可选:Hi-Robot / Helix 式分层规划(慢系统出意图,快系统出动作)
```

核心思想可以用一句话概括:**用 VLM 提供"语义先验和常识",用大规模异构机器人/人类视频数据提供"动作先验",再通过统一的动作表征(token / latent action / flow)解耦不同机器人本体的差异。**

---

## 二、关键技术模块拆解

### 1. 模型骨架(Architecture)

主流是 **VLM + Action Expert** 双塔/混合专家结构:

- **System 2(慢思考)**:VLM(7B 量级),负责理解指令、看场景、做高层规划。
- **System 1(快控制)**:Action Expert(几亿参数的 DiT / Flow Transformer),以 30–50Hz 输出 *action chunk*。
- 通过 cross-attention / KV cache 共享语义,System 1 推理时延控制在 < 50ms。

代表作:π0 / π0.5、GR00T N1/N1.5、Helix、Hi-Robot、AgiBot GO-1。

### 2. 动作表征(Action Tokenization)

跨本体的关键就在于"如何把 7-DoF 单臂、双臂、人形 23-DoF 全身放到同一个动作空间":

- **离散 token**:RT-2 / OpenVLA(把动作离散化成语言 token,直接复用 LLM)。
- **Flow Matching / Diffusion**:π0、π0.5、RDT、GR00T(连续、动作分布建模更精细)。
- **Latent Action**:LAPA、UniPi、GR00T 中的 latent action codebook(从无标签视频学动作)。
- **Soft Prompting / Embodiment Token**:CrossFormer、X-VLA(用一个 embodiment-id 软提示"告诉"模型当前是什么本体)。
- **统一动作空间**:HPT(MIT)用 stem-trunk-head 三段式,把不同本体的 proprio + action 投影到同一 trunk。

### 3. 数据(决定上限的最关键一环)

> 没有大规模异构数据,谈 zero-shot 跨本体都是空中楼阁。

| 数据来源 | 代表数据集 | 规模 | 用法 |
|---|---|---|---|
| 真机多本体 | **Open X-Embodiment (RT-X)** | 22 个本体, 1M+ episodes | 主预训练 |
| 真机大规模单源 | **DROID** | 76k 轨迹, 564 场景 | 多样性 |
| 真机人形/双臂 | **AgiBot World Colosseo** | 1M+ 轨迹, 217 任务 | 中文社区首选 |
| 多本体基准 | **RoboMIND** (RSS'25) | 107k 轨迹, 4 本体 | 评测+SFT |
| 增广 OXE | **OXE-AugE** (2025.12) | 4.4M 轨迹 | 解决 OXE 本体偏置 |
| 人类第一视角 | **Ego4D / EgoExo4D / Something-Something v2** | 数千小时 | LAPA 式无监督预训练 |
| 仿真 | **RoboCasa / DexMimicGen / GR00T-Sim / Isaac Lab** | 可无限生成 | 长尾任务、安全场景 |
| 人手专用 | **UniHand-2.0**(Being-H0.5) | 35,000+ 小时, 30 本体 | 灵巧手预训练 |

**关键技巧**:
- 对每个数据源做 *embodiment normalization*(关节归一化、相机外参对齐、动作 chunk 重采样到统一频率)。
- Co-training:真机数据 + 人类视频 + 仿真按一定 weight 混合采样,π0.5 实测能显著提泛化。

### 4. 跨本体泛化(Cross-Embodiment)

可以借鉴的几种思路:
1. **Open X-Embodiment / RT-X**:把所有本体当作不同 token,一个 transformer 全吃。
2. **CrossFormer (CoRL'24)**:不同 embodiment 各自有 readout head,共享中间 trunk。
3. **HPT (MIT, NeurIPS'24)**:Heterogeneous Pre-training Transformer,proprio/视觉/动作各自 stem,统一 trunk。
4. **X-VLA (2025)**:Soft prompt 表示本体身份,只增加极少参数。
5. **OXE-AugE (2025.12)**:用机器人增广(替换不同臂/夹爪)合成更多本体数据,目标本体未见时也能 +24%~+45%。

### 5. 用人类视频做"无标签动作预训练"

- **LAPA**(ICLR'25):VQ-VAE 从相邻帧学 latent action → 在此 latent 空间 VLA 预训练 → 小样本机器人动作微调。**目前从纯人类视频做 VLA 预训练的代表工作。**
- **UniPi / UniSim / Genie 2**:用视频生成模型当"世界模型",以"想象未来帧"为目标做策略学习。
- **CoLA-World (2025.10)**:Latent Action 与世界模型联合协同训练,解决表征坍缩。

### 6. 分层规划(让模型能做"打扫房间"这种长程任务)

- **Hi-Robot**(Physical Intelligence):一个 VLM 做高层语言规划,π0 做底层执行。
- **Helix**(Figure AI):双系统,VLM 7B + 80M 控制专家,40 Hz 全身控制。
- **π0.5**:co-train 高层语义子任务预测 + 低层 action 输出,在没见过的家居环境里清理厨房/卧室。

---

## 三、必读 SOTA 论文 / 项目清单(按主题归类)

> 顺序大致按"先看哪个"排,⭐ 表示强烈推荐先精读。

### A. VLA 基础模型(必读基线)
- ⭐ **RT-1 / RT-2**(Google DeepMind, 2022/2023)— 把动作 tokenize 进 VLM 的开山之作。
- ⭐ **RT-X / Open X-Embodiment**(2024)— 跨 22 本体训一个模型,跨本体范式奠基。
- ⭐ **OpenVLA**(Stanford, 2024, 7B 开源)— 复现/微调起点,首选开源代码库:`openvla/openvla`。
- ⭐ **π0 / π0-FAST / π0.5**(Physical Intelligence, 2024/2025)— Flow Matching VLA 的 SOTA,π0.5 强调"开放世界泛化"。
- ⭐ **GR00T N1 / N1.5**(NVIDIA, 2025, 开源)— 人形机器人通用基础模型,代码、权重、仿真全开源。
- **Octo**(Berkeley, 2024)— Transformer 通用策略,模块化任务编码。
- **RDT-1B**(清华, 2024)— 大规模 Diffusion VLA,双臂操作 SOTA。
- **CogACT / TinyVLA / SpatialVLA / 3D-VLA** — 各种轻量化或 3D 几何增强变体。

### B. 跨本体专项
- ⭐ **CrossFormer**(CoRL'24)— 单模型同时控制 manipulation/navigation/quadruped/aerial。
- ⭐ **HPT**(MIT, NeurIPS'24)— Heterogeneous Pre-training Transformer。
- **X-VLA**(2025)— Soft-prompted cross-embodiment VLA。
- **OXE-AugE**(2025.12)— 通过机器人增广提升未见本体性能。

### C. 数据与平台
- ⭐ **Open X-Embodiment / DROID**(RSS'24)
- ⭐ **AgiBot World Colosseo + GO-1**(2025)— 中文社区首选,百万级双臂人形数据。
- **RoboMIND**(RSS'25)— 多本体评测基准。
- **RoboCasa / DexMimicGen / GR00T-Sim**(NVIDIA)— 大规模合成数据。
- **Isaac Lab / ManiSkill3 / LIBERO / RoboCasa** — 主流仿真评测。

### D. 从人类视频/世界模型学动作
- ⭐ **LAPA: Latent Action Pretraining from Videos**(ICLR'25)
- **UniPi / UniSim**(DeepMind)
- **Genie 2 / Genie Operator-1**(DeepMind / 智元)
- **CoLA-World**(2025)
- **Being-H0 / Being-H0.5**(2026)— 35k 小时人手数据 + Mixture-of-Flow,30 个本体。

### E. 动作头与控制
- ⭐ **Diffusion Policy**(Columbia, RSS'23)— 至今是 action head 的强基线。
- ⭐ **ACT / Action Chunking Transformer**(ALOHA, 2023)— 双臂细操作。
- **Flow Matching for Robot Control**(π0 系列基础)
- **Equivariant Diffusion Policy / 3D Diffusion Policy** — 几何先验。

### F. 分层 / 长程任务
- **Hi-Robot**(Physical Intelligence, 2025)
- **Helix**(Figure, 2025,白皮书+技术博客)
- **SayCan / Code as Policies / Inner Monologue**(早期 LLM-as-planner)

### G. 综述(节省时间)
- *A Survey on Vision-Language-Action Models for Embodied AI*(2024)
- *Foundation Models in Robotics: Applications, Challenges, and the Future*(IJRR 2024)
- *Towards Generalist Robot Policies: A Survey*(2025)

---

## 四、实操建议(给真要动手的人)

1. **不要从零造轮子**。直接基于 **OpenVLA** 或 **GR00T N1** fork:
   - OpenVLA:代码最干净,适合学习+魔改架构。
   - GR00T N1.5:工业级,自带仿真、合成数据 pipeline、人形支持。
   - π0:`lerobot` 库已集成 π0/π0-FAST,HuggingFace 生态最好。

2. **数据先行**。建议组合使用:
   - 主体:OXE + DROID + AgiBot World(若有人形/双臂目标)。
   - 补充:Ego4D / EgoExo4D 走 LAPA 路线扩动作分布。
   - 仿真:RoboCasa + DexMimicGen 做长尾增广。

3. **训练算力参考**(给个量纲):
   - 7B VLA 预训练:~64–256 张 H100 × 数周。
   - 想穷训:可先做 **2–3B 小尺寸 VLA**,在 8×H100 上 1–2 周可跑通完整 OpenVLA-style 训练。

4. **评测一定多本体**。最少在三种本体上测:
   - 单臂(Franka / UR5)、双臂(ALOHA / AgileX)、移动+操作(若有)。
   - 评测套件:LIBERO、RoboCasa、SimplerEnv、Real-World A/B(必须有真机)。

5. **要做 zero-shot 跨本体,以下三件事必做**:
   - 训练时本体多样性 ≥ 5–10 种;
   - 加 embodiment prompt / proprio adapter;
   - 后训练小样本 calibration(哪怕 50 条目标本体演示也极大提升)。

6. **长程复杂任务**:别指望端到端 VLA 一步到位,加一个 Hi-Robot/Helix 式的高层 VLM 规划器,能立刻让任务长度从 10s 上升到分钟级。

---

如果你愿意告诉我:
- 目标机器人形态(单臂?双臂?人形?移动操作?)
- 是否有自有数据,以及量级
- 计划的算力预算

我可以再给一份更针对性的训练配方(数据混合比、超参、阶段时长、评测协议)。

---

**User**

这些论文和研究都太旧了, 请参考2026年的论文, 文章, blog, github和研究成果.

---

**Cursor**

明白,下面这份是**完全基于 2026 年(含 2025 年末刚发布、目前仍是 SOTA 的)论文、模型、blog 和开源项目**的整理。所有早于 2025 年中的内容都不再列入。

---

## 一、2026 年关于"训通用具身大脑"的认知更新

相比 2024–2025,2026 年的几个重要范式转变(看完这几条再去看具体论文会更顺):

1. **架构从"双系统"变成"三系统(S0/S1/S2)"**:Figure **Helix 02**(Jan 2026)把"1 kHz 平衡/接触控制(S0) + 200 Hz 全身视触动作(S1) + 慢速语义规划(S2)"显式拆成三层,**用神经网络替换掉 ~11 万行手写 C++ 平衡控制器**——这是 loco-manipulation 的关键节点。
2. **VLA 主流后端从 Gemma2 → Gemma3 / Qwen3-VL / Cosmos-Reason2**,推理速度大幅提升:π0.6 在单卡 H100 上 63 ms 出一个 action chunk(5 步 flow + 3 路相机)。
3. **"World–Action Model" 取代纯反应式 VLA**:NVIDIA **GR00T N2**(Mar 2026)、**GigaWorld-Policy**(Mar 2026)等显式预测未来环境状态/视频,GigaWorld 比 π0.5 在 RoboTwin 2.0 上提升 ~95%、推理快 9×。
4. **Online RL 从研究 demo 走向真正的 fleet-scale 部署**:**SOP**(Jan 2026)、**Learning While Deploying**(May 2026)在 16 台双臂机器人上做云端持续 RL,跨 8 个任务平均 95% 成功率。
5. **跨本体的"Naive mixing"被证伪**:多个 2026 工作明确报告——盲目把异构机器人数据混在一起会 **negative transfer**;关键解法是 **EEF-relative 统一动作空间** + **soft-prompt embodiment id** + **数据感知的混合权重**。
6. **大规模人类视频走向主流预训练**:Ψ₀(Mar 2026)只用 ~800 小时人类视频 + 30 小时人形真机,就比用 10× 更多数据的基线高 40%+;Being-H0.5(Jan 2026)做到 35,000 小时 UniHand-2.0、30 个本体。
7. **VLA scaling law 还远未饱和**:LingBot-VLA 把真机数据从 3k h 推到 20k h,下游成功率持续涨,无明显饱和。

---

## 二、2026 推荐训练路线(基于最新 SOTA 实践更新)

```text
阶段 0  VLM 基座(2026 主流选择)
        Qwen3-VL  /  Cosmos-Reason2-2B  /  Gemma3-4B  /  InternVL-3
        (HY-Embodied-0.5 用 2B/32B MoT,边端+云端两套)

阶段 1  大规模"动作预训练"(Action Pre-training)
        ① 真机异构: OXE + OXE-AugE(4.4M 轨迹) + AgiBot World 2026(1M+)
                    + DROID + RoboMIND v2 + LingBot 20k h 双臂数据
        ② 人类视频: UniHand-2.0(35k h) / Ego(Exo)4D
                    → 走 LAPA-style 或 Ψ₀ 的"先人类后机器人"两阶段
        ③ 仿真:    GR00T-Sim / RoboCasa / DexMimicGen / Isaac Lab
        ‼ 关键: EEF-relative 统一动作空间 + soft-prompt embodiment id
                + 学到的数据混合权重(避免 JoyAI-RA 报告的 negative transfer)

阶段 2  动作头(Action Expert)训练
        Flow Matching (π0.6/0.7) 或 Diffusion Transformer
        三层 S0/S1/S2 架构(Helix 02 风格)用于全身控制
        World–Action Model(GR00T N2 / GigaWorld)做"看远点再动"

阶段 3  后训练 / 对齐
        SFT on 目标本体 (50–500 条演示就能起效)
        On-policy distillation:VLA-OPD
        Online RL:SOP / LWD / TT-VLA
        Continual:LifeLong-RFT(防止灾难性遗忘)

阶段 4  部署 & 评测
        推理: π0.6 级 63 ms/chunk;边端 NVIDIA Jetson Thor (GR00T N2)
        评测: RoboArena (DROID, 2026 全年开放)
              RobotArena ∞(real-to-sim 自动评测)
              LIBERO+ / RoboCasa / SimplerEnv
```

---

## 三、2026 年必看 SOTA 清单(按主题)

> 全部为 **2025 Q4 – 2026 Q2** 之间的工作,标注首发月份/单位/链接关键词。

### A. VLA 基础模型(直接对标"通用具身大脑")

| 模型 | 时间 | 单位 | 关键贡献 |
|---|---|---|---|
| ⭐ **π0.6 / π* 0.6** | 2025.11 | Physical Intelligence | Gemma3-4B + 860M action expert,63 ms 出 chunk;π* 是 RL 后训练版 |
| ⭐ **π0.7** | 2026.04 | Physical Intelligence | "Steerable" VLA,泛化能力 step-change 提升 |
| ⭐ **GR00T N1.7** | 2026 Q1 | NVIDIA | 换 Cosmos-Reason2-2B/Qwen3-VL 后端,Apache 2.0,LIBERO/Bridge/Fractal/DROID 全权重开源 |
| ⭐ **GR00T N2 (WAM)** | 2026.03 | NVIDIA | World Action Model,显式预测物理状态,98% zero-shot 操作未见物体,Jetson Thor 部署 |
| ⭐ **Helix 02** | 2026.01 | Figure AI | 三层 S0(1 kHz)/S1(200 Hz)/S2 全身 VLA,4 分钟 61 步无干预洗碗 |
| ⭐ **Being-H0.5** | 2026.01 | BeingBeyond | UniHand-2.0(35k h、30 本体)+ Mixture-of-Flow,LIBERO 98.9% / RoboCasa 53.9%,5 个本体真机部署 |
| ⭐ **Ψ₀ (Psi-Zero)** | 2026.03 | (开源,arXiv 2603.12263) | 人类视频预训→人形后训,800 h 人 + 30 h 机器人 > 10× 数据基线 +40% |
| **HY-Embodied-0.5** | 2026.04 | (arXiv 2604.07430) | 2B(边)+ 32B(云)MoT,22 个 benchmark 与 Gemini 3.0 Pro 持平 |
| **LingBot-VLA / LingBot-VA** | 2026 Q1 | (arXiv 2601.18692) | 20k h 双臂真机,9 种配置;LingBot-VA 加 causal world model |
| **Green-VLA** | 2026.02 | (arXiv 2602.00919) | 5 阶段 curriculum,VLM → RL 对齐,3k h 双臂数据 |
| **X-VLA** | 2025.10 | (arXiv 2510.10274) | Soft-prompt 表 embodiment id,0.9B 跨 6 仿真 + 3 真机 SOTA |
| **VLANeXt** | 2026.02 | (arXiv 2602.18532) | 系统性消融出 12 条 VLA 设计 recipe,LIBERO+/LIBERO SOTA |
| **GeneralVLA** | 2026.02 | (arXiv 2602.04315) | Affordance + 3D Agent + 3D 控制策略,**完全无需真机数据**做 zero-shot |
| **VLA Foundry** | 2026.04 | Toyota Research | 端到端开源训练框架,VLM 预训→action expert,支持 Qwen3-VL |
| **A Pragmatic VLA Foundation Model** | 2026.01 | (arXiv 2601.18692) | 工程角度全套实践指南 |

### B. World / Action Model(2026 新主流)

- ⭐ **GR00T N2 (WAM)**(NVIDIA, 2026.03)
- ⭐ **GigaWorld-Policy**(2026.03, arXiv 2603.17240)— action-centered world–action,**RoboTwin 2.0 上比 π0.5 +95%、9× 加速**。
- **HiPolicy**(2026.04, arXiv 2604.06067)— 多频率分层 action chunking,平衡长程与高频反应。
- **Mask World Model (MWM)**(2026.04, arXiv 2604.19683)— 不预测 RGB 改预测语义 mask,LIBERO/RLBench 全面优于 RGB 世界模型。
- **Causal World Modeling for Robot Control**(2026.01, arXiv 2601.21998)
- **CoLA-World**(2025.10, arXiv 2510.26433)— Latent Action 与 World Model 协同进化,解决表征坍缩。
- **OmniGuide**(2026)— 推理时用 3D 重建/位姿估计当外部能量函数引导 VLA。

### C. Online RL / 持续学习后训练(2026 重点突破)

- ⭐ **SOP: Scalable Online Post-Training**(2026.01, arXiv 2601.03044)— 机器人 fleet 把 on-policy 经验流给云端 learner,HG-DAgger + RECAP,小时级生效。
- ⭐ **Learning While Deploying (LWD)**(2026.05, arXiv 2605.00416)— DIVL + QAM,**16 台双臂 8 任务平均 95%**。
- **VLA-OPD**(2026.03, arXiv 2603.26666)— 用 expert 老师做 token 级 on-policy 蒸馏,弥合 SFT↔RL。
- **TT-VLA**(2026.01, arXiv 2601.06748)— 推理时 RL,基于 task-progress dense reward。
- **LifeLong-RFT**(2026.02, arXiv 2602.10503)— chunk 级 RL + 多维过程奖励,防遗忘,+22% 成功率。

### D. 跨本体 / 数据混合(2026 最关键经验)

- ⭐ **OXE-AugE**(2025.12, arXiv 2512.13100)— 9 种本体增广 OXE 到 4.4M 轨迹,未见 robot×gripper 上 +24%~+45%。
- ⭐ **Being-H0.5** —— 30 个本体的统一动作空间。
- **JoyAI-RA / 数据混合研究**(2026 Q1)— 明确指出**朴素混合 → negative transfer**,EEF-relative + 多源(web/人类视频/仿真/真机)显式对齐才有效。

### E. 数据集与平台(2026 必备)

- ⭐ **AgiBot World 2026**(智元,2026)— 1M+ 轨迹、5 类机器人、~43.8 TB,RGB(D)+ 触觉 + LiDAR + IMU,**强力遥操数据**。HF + GitHub `OpenDriveLab/agibot-world` 全开源。
- ⭐ **UniHand-2.0**(Being-H0.5)— 35k h、30 本体,目前最大具身预训练数据。
- **OXE-AugE**(2025.12)— OXE 的官方增广版,跨本体首选。
- **RoboMIND v2**(RSS'25 后续)— 多本体评测正餐。
- **DROID** —— RoboArena 的官方评测平台。

### F. 公开评测基准(2026 用这些跑分就够)

- ⭐ **RoboArena**(robo-arena.github.io)— 真机 DROID,**双盲两两比较,2026 全年开放,每周打榜**,目前最权威的"真机 leaderboard"。
- ⭐ **RobotArena ∞**(2025.10, arXiv 2510.23571)— Real-to-Sim 自动评测 + VLM 打分 + 众包人类偏好。
- **SimplerEnv**(CoRL'24 但 2026 仍是事实标准)
- **LIBERO+ / RoboCasa / RoboTwin 2.0**(论文里的常见对比)

### G. Behavior 基础模型(只控制本体不操作物体的方向)

- **BFM-Zero**(ICLR 2026 poster)— 无监督 RL 行为基础模型,Unitree G1 上零样本动作跟踪/目标到达/奖励推断。
- **Helix 02 - S0 module**(Figure)— 1 kHz 平衡控制器,纯学的取代手写 C++。

### H. 公司/产品节奏(2026 上半年)

- **Physical Intelligence**:π0.6(2025.11)→ π* 0.6(RL 版)→ **π0.7**(2026.04)→ blog 持续更新长短期记忆 VLA、高效 online RL(2026.03)。
- **NVIDIA**:**GR00T N1.7**(2026 Q1,Apache 2.0 全开源)→ **GR00T N2 / WAM**(2026.03)。
- **Figure AI**:**Helix**(2025)→ **Helix 02**(2026.01,全身)。
- **AgiBot(智元)**:GO-1 / GO-1 Air(2025.09 开源)→ **A3 人形 + G2 Air**(2026.04)+ AgiBot World 2026 数据。

---

## 四、2026 年应重点用的开源代码库

| 仓库 | 维护者 | 推荐用途 |
|---|---|---|
| ⭐ `huggingface/lerobot`(v0.5.0,2026.03) | HuggingFace | **首选起点**,Datasets v3.0、π0/π0.5 集成、SO-101/Unitree G1 支持 |
| ⭐ `physical-intelligence/openpi` | Physical Intelligence | π0 / π0-FAST 官方实现,可作为 flow matching action head 模板 |
| ⭐ `NVIDIA/Isaac-GR00T`(N1.7) | NVIDIA | Apache 2.0,人形/双臂全栈,自带仿真和合成数据 |
| **VLA Foundry**(TRI, 2026.04) | Toyota Research | VLM→VLA 端到端可复现训练框架 |
| `OpenDriveLab/agibot-world` | 智元 / OpenDriveLab | AgiBot World 数据 + GO-1 模型 |
| **SimplerEnv / RoboArena** | 社区 | 跑分两件套 |

---

## 五、2026 给"想真训一个跨本体 VLA"的工程化建议

1. **基座**直接挑 GR00T N1.7(权重开源 + 工业级)或 lerobot 里的 π0.5/π0.6 适配版做起点,别从 OpenVLA/Octo 起跳——已经过时一年。
2. **数据组合**(2026 共识): 
 - 大头:OXE-AugE 或 AgiBot World 2026 
 - 人类视频:UniHand-2.0 / Ego(Exo)4D(走 Ψ₀ 或 LAPA 路线) 
 - 仿真:GR00T-Sim / RoboCasa(覆盖长尾) 
 - **必须**用 EEF-relative 动作空间 + 学到的混合权重,否则 negative transfer 会让你白跑几周。
3. **架构**至少长这样:Qwen3-VL/Gemma3-4B(S2)+ 200 Hz Flow Matching action expert(S1);若做人形再加一个 1 kHz 学到的低层平衡(S0)——直接抄 Helix 02 的拆分思路。
4. **必加 World–Action Model 或显式 future prediction 头**(GigaWorld / GR00T N2 范式),2026 数据已显示这能拿 +50%~+90%。
5. **后训练一定要做 online RL**,不只 SFT。SOP / LWD 的 fleet 流式 RL 是当前最高效的"上线后持续涨点"路径,且能少量数据 fine-tune 到新本体。
6. **评测**:本地用 LIBERO+ / RoboCasa / RoboTwin 2.0;真机想发声音就投 **RoboArena 2026**,这是目前最权威的 leaderboard。
7. **算力参考(2026 实测公开数据)**:7B VLA + 800M action expert,约 64–256 张 H100×2–4 周可复现 π0.6 级别;穷训路线可 8×H100 跑 ~2B 参数 VLANeXt recipe + LingBot-VLA 数据子集,2 周拿到能用模型。

---

如果你能告诉我**目标本体形态(单臂/双臂/人形/移动)**、**有无自有数据**、**算力预算**,我可以再基于这份 2026 列表给一份更落地的"3 阶段训练配方 + 数据混合权重 + 算力 / 时间表"。

---

**User**

请罗列一下上一轮的回答参考了哪些资料!

---

**Cursor**

好的,下面把上一轮回答里**实际引用到的所有资料**(论文、blog、GitHub、产品发布)按主题列出来,并尽可能给出 arXiv 编号 / 链接关键词,方便你按图索骥。

---

## 一、VLA 基础模型类

| # | 资料 | 类型 | 出处 / 编号 | 时间 |
|---|---|---|---|---|
| 1 | **π0.6** model card | PDF / 厂商发布 | `website.pi-asset.com/pi06star/PI06_model_card.pdf` | 2025.11 |
| 2 | **π0.7 / Physical Intelligence 主页** | 公司 blog | `pi.website` | 2026.04 |
| 3 | **openpi**(π0 开源代码) | GitHub | `physical-intelligence/openpi` | 持续更新 |
| 4 | **GR00T N1.7**(权重 + 文档) | HuggingFace | `nvidia/GR00T-N1.7-LIBERO`(及其它后训练版) | 2026 Q1 |
| 5 | **Isaac-GR00T**(代码库) | GitHub | `NVIDIA/Isaac-GR00T`,Apache 2.0 | 2026 |
| 6 | **GR00T N2 (World Action Model)** | 媒体/技术报道 | `techbytes.app/posts/nvidia-gr00t-n2-physical-ai-benchmarks` | 2026.03 |
| 7 | **Helix 02: Full-Body Autonomy** | 公司 blog(Figure) | `figure.ai/news/helix-02` | 2026.01 |
| 8 | **Helix 02 解析报道** | 媒体 | `humanoidsdaily.com`、`interestingengineering.com`、`humanoidroboticstechnology.com` | 2026.01 |
| 9 | **Being-H0.5** | arXiv | 2601.12993 | 2026.01 |
| 10 | **Ψ₀ (Psi-Zero)** | arXiv | 2603.12263v1 | 2026.03 |
| 11 | **HY-Embodied-0.5** | arXiv | 2604.07430v1 | 2026.04 |
| 12 | **LingBot-VLA / "A Pragmatic VLA Foundation Model"** | arXiv | 2601.18692v1 | 2026.01 |
| 13 | **Green-VLA** | arXiv | 2602.00919 | 2026.02 |
| 14 | **X-VLA** | arXiv | 2510.10274v1 | 2025.10 |
| 15 | **VLANeXt: Recipes for Building Strong VLA Models** | arXiv | 2602.18532 | 2026.02 |
| 16 | **GeneralVLA** | arXiv | 2602.04315 | 2026.02 |
| 17 | **VLA Foundry**(TRI 训练框架) | arXiv | 2604.19728v1 | 2026.04 |

## 二、World Model / Action Model 类

| # | 资料 | 出处 / 编号 | 时间 |
|---|---|---|---|
| 18 | **GigaWorld-Policy** | arXiv 2603.17240 | 2026.03 |
| 19 | **HiPolicy: Hierarchical Multi-Frequency Action Chunking** | arXiv 2604.06067 | 2026.04 |
| 20 | **Mask World Model (MWM)** | arXiv 2604.19683 | 2026.04 |
| 21 | **Causal World Modeling for Robot Control** | arXiv 2601.21998 | 2026.01 |
| 22 | **CoLA-World**(co-evolving latent action world models) | arXiv 2510.26433 | 2025.10 |
| 23 | **OmniGuide**(VLA 推理时引导) | arXiv 2604.20100(检索到的同期工作之一) | 2026.04 |
| 24 | **LingBot-VA**(autoregressive diffusion + causal world model) | 检索结果同源(arXiv,2026 Q1 区间) | 2026.01–02 |

## 三、Online RL / 后训练 / 持续学习

| # | 资料 | arXiv 编号 | 时间 |
|---|---|---|---|
| 25 | **SOP: Scalable Online Post-Training System for VLA** | 2601.03044 | 2026.01 |
| 26 | **TT-VLA: On-the-Fly VLA Adaptation via Test-Time RL** | 2601.06748 | 2026.01 |
| 27 | **VLA-OPD: Bridging Offline SFT and Online RL via On-Policy Distillation** | 2603.26666v1 | 2026.03 |
| 28 | **LifeLong-RFT: Continual Learning VLA via RL Fine-Tuning** | 2602.10503 | 2026.02 |
| 29 | **LWD: Learning while Deploying (Fleet-Scale RL)** | 2605.00416 | 2026.05 |

## 四、跨本体 / 数据混合 / Scaling

| # | 资料 | 出处 | 时间 |
|---|---|---|---|
| 30 | **OXE-AugE**(OXE 大规模本体增广) | arXiv 2512.13100 | 2025.12 |
| 31 | **JoyAI-RA**(多源对齐,negative transfer 论证) | arXiv 2604.20100 / 2602.09722(检索同源) | 2026 Q1–Q2 |
| 32 | **Being-H0.5 的 Unified Action Space**(同 #9) | arXiv 2601.12993 | 2026.01 |

## 五、数据集与平台

| # | 资料 | 出处 | 时间 |
|---|---|---|---|
| 33 | **AgiBot World 2026 / GO-1 / GO-1 Air** | GitHub `OpenDriveLab/agibot-world` + 官网 `agibot.com/research`、`agibot.com/article/231/...` | 2025.09 开源,2026 持续更新 |
| 34 | **AGIBOT A3 人形 + G2 Air 单臂** 产品发布 | `agibot.com/article/231/detail/63.html` | 2026.04 |
| 35 | **UniHand-2.0**(随 Being-H0.5 公布) | arXiv 2601.12993 | 2026.01 |
| 36 | **AgiBot World Colosseo** 论文(底层平台) | arXiv 2503.06669 | 2025.03 |
| 37 | **RoboMIND**(多本体基准) | arXiv 2412.13877 + `x-humanoid-robomind.github.io` | RSS'25 |

## 六、评测基准

| # | 资料 | 出处 | 时间 |
|---|---|---|---|
| 38 | **RoboArena** | `robo-arena.github.io`(双盲真机 leaderboard) | 2026 全年 |
| 39 | **RobotArena ∞**(real-to-sim 大规模评测) | arXiv 2510.23571 | 2025.10 |
| 40 | **SimplerEnv** | GitHub `simpler-env/SimplerEnv` + 项目页 | CoRL'24,2026 仍主流 |

## 七、行为基础模型 / 仿真 / 框架

| # | 资料 | 出处 | 时间 |
|---|---|---|---|
| 41 | **BFM-Zero: Promptable Behavioral Foundation Model** | ICLR 2026 poster(`iclr.cc/virtual/2026/poster/10007857`)+ openreview PDF | ICLR 2026 |
| 42 | **LeRobot**(框架) | GitHub `huggingface/lerobot`(v0.5.0) | 2026.03 |
| 43 | **LeRobot v0.4.0 release blog** | `huggingface/blog`,`lerobot-release-v040.md` | 2026 早期 |
| 44 | **π0 (Pi0) lerobot 文档** | `huggingface.co/docs/lerobot/pi0` | 2026 |

## 八、其它(检索到、被一并参考但只少量引用的)

- **arXiv 2602.04315**(GeneralVLA,同 #16)
- **arXiv 2602.00919**(Green-VLA,同 #13)
- **arXiv 2602.18532**(VLANeXt,同 #15)
- **arXiv 2604.19728**(VLA Foundry,同 #17)
- **arXiv 2602.09722** / **2604.20100** —— 数据混合 / 推理引导相关同期工作(用于支撑"naive mixing → negative transfer"和"OmniGuide 方向"两个论点)
- **AgiBot 官网研究页** `agibot.com/research`
- **NVIDIA Isaac-GR00T README** (GitHub)

---

### 几点说明

1. 上一轮回答里的**所有定量结论**(如"π0.6 单卡 H100 63 ms / chunk"、"Helix 02 替换 109,504 行 C++"、"GigaWorld-Policy 比 π0.5 在 RoboTwin 2.0 上 +95%、9× 加速"、"LWD 16 台双臂 95% 成功率"、"Ψ₀ 用 800 h 人 + 30 h 真机超 10× 数据基线 +40%"、"Being-H0.5 LIBERO 98.9 / RoboCasa 53.9"、"OXE-AugE 4.4M 轨迹、未见组合 +24%~+45%")都来自上表中**对应论文/官方 blog/产品发布**。
2. 一部分 arXiv 编号(2601.xxxxx / 2602.xxxxx / 2603.xxxxx / 2604.xxxxx / 2605.xxxxx)是 **2026 年的新编号体系**(年份 + 月份前缀),所以编号看起来都是 26xx,这是正常的。
3. 训练路线图、数据混合建议、算力估算这部分是我**基于以上资料的综合归纳**,并非来自单一论文。

如果你想深入读其中某几篇,告诉我编号,我可以专门挑出来给你做精读笔记或对比表。

---

# VLA预训练考虑点分析.V1

**User**

要在超多的数据上训练一个超大的具身智能预训练模型,使得这模型能装在不同的机器人本体上zero-shot地执行各种不同的复杂任务,深入分析刚才你罗列的所参考的各个最新SOTA论文和研究成果等, 也可以到网上搜索相关资料, 仔细分析和深入思考一下, 要做这件事情要考虑哪些方面, 注意哪些细节, 将它们罗列出来. 

---

**Cursor**

下面是一份**深度分析**——基于上一轮列出的 2026 SOTA 论文 + 这次新查到的关键工程实践,系统性地梳理"训一个能装到不同机器人本体上 zero-shot 执行复杂任务的超大具身基础模型"必须考虑的全部维度。我尽量按"**问题 → 关键挑战 → 2026 SOTA 给出的答案 → 实操坑点 → 具体数字**"的结构展开,共 **12 大维度**。

---

## 顶层设计原则(2026 共识中决定项目能否成功的"6 条铁律")

在动手前,先要承认这 6 条已被多个独立工作反复验证的事实:

1. **数据决定上限**,模型架构决定下限。LingBot-VLA 把真机数据从 3k h → 20k h,success rate **持续涨,无饱和迹象**——意味着至少在 10⁴–10⁵ 小时尺度上,算法人是 *bottleneck-limited by data*,不是 model-limited。
2. **Naive 跨本体 mixing 必然 negative transfer**。多篇 2026 工作(JoyAI-RA、OXE-AugE 实验、X-VLA)都报告:不做动作空间对齐就把不同机器人扔一锅,反而比单本体训练**还差**。
3. **必须显式建模"未来"**。从 GR00T N2 / GigaWorld-Policy / Mask World Model 一致来看,纯反应式 VLA(只看现在出动作)在长程任务上明显输给"先想象再动作"的 World–Action Model,差距可达 **+50% ~ +95%**(RoboTwin 2.0)。
4. **三层时间尺度是人形/复杂本体的必经之路**。Helix 02 把 1 kHz 平衡(S0)/ 200 Hz 视触动作(S1)/ 慢速语言规划(S2)显式拆开,**用神经网络替换 109,504 行手写 C++**——单一频率、单一网络已经撑不住全身 loco-manipulation。
5. **Online RL 不是可选,而是必选**。SOP / LWD 在 16 台双臂上线后,通过 fleet-scale on-policy 数据持续 RL,小时级就能涨点;LifeLong-RFT 显示纯 SFT 会**灾难性遗忘**,只有混入 RL 才能长期可用。
6. **真机评测才是真理**。LIBERO+ 之类 sim 跑分能凑合做开发,但跨本体 zero-shot 这种声称必须上 **RoboArena**(2026 全年开放)或 **RobotArena ∞**(real-to-sim + 人评)才有信誉。

---

## 一、数据工程(最大、最容易被低估的工作量)

### 1.1 数据来源与配比

| 来源 | 规模目标(2026 SOTA 起步线) | 用途 | 代表数据 |
|---|---|---|---|
| 真机异构本体 | ≥ 10⁴ 小时 / 10⁶ 轨迹 | 主预训练 | OXE + **OXE-AugE**(4.4M) + **AgiBot World 2026**(1M+, 43.8 TB) + DROID + RoboMIND v2 + LingBot 20k h |
| 人类第一视角视频 | ≥ 10⁴ 小时 | latent action 预训练 | **UniHand-2.0**(35k h、30 本体)、Ego(Exo)4D |
| 仿真合成 | 10⁵ – 10⁷ 轨迹 | 长尾、安全场景 | **GR00T-Sim**、**RoboCasa**、**RoboTwin 2.0**(强 DR)、Isaac Lab |
| 互联网图文/视频 | 不限 | 维持 VLM 通识 | LAION、ActivityNet、HowTo100M |

> Ψ₀ 的数据效率结论很反直觉:**800 h 人类视频 + 30 h 真机** > 用 10× 数据的 baseline +40%。说明"先人类视频后真机两段式"在数据稀缺时极强。

### 1.2 数据混合权重(常见踩坑点)

- 不要按"自然比例"混(哪种数据多就用得多)。**按数据"质量 × 多样性"加权**,对小但高质量的人形/灵巧手数据 upsampling 5–20×。
- **学到的混合权重**(如 DoReMi、α 自适应)在 2026 工作里普遍优于固定比例。
- **逐阶段切换 ratio**(curriculum):像 Green-VLA 5 阶段一样,前期重 web/sim,后期重 real,每阶段动作空间约束逐渐收紧。

### 1.3 数据质量过滤(直接决定下游 success)

Green-VLA 显式给出过滤标准,实战里至少要做:

- **轨迹平滑度**(去除遥操错乱、抽搐帧);
- **视觉清晰度 / 模糊度 / 曝光异常**;
- **动作 vs 视觉一致性**(光流和末端运动方向相关性);
- **任务成功标签**:对 OXE 这种自报告成功的数据,**用 VLM 重新打分**(Mask World Model、OmniGuide 都用了 VLM 当 critic);
- **去重**(感知哈希 + 动作序列哈希),避免遥操数据集里的高度重复;
- **失败样本保留**(RoboMIND 显式收集 5k 失败示范并标注原因)——对 OOD/恢复策略训练价值极高。

### 1.4 数据规范化(跨本体可训的前提)

每条轨迹都需要标准化为统一 schema:
- **相机外参**:对齐到 base / EEF 坐标系,记录 intrinsics + extrinsics;
- **proprio**:joint pos/vel/torque + EEF pose 都存,推理时按需取;
- **动作**:同时存 *joint-space* 和 *EEF 6D + 夹爪* 两种表示;
- **频率**:统一重采样到一个固定基频(如 50 Hz),原始频率作为 metadata;
- **语言指令**:每条轨迹至少 3 条不同粒度的指令(原子/子任务/任务级),方便分层 VLA 训练。

### 1.5 合规与数据版权

- AgiBot World、DROID、RoboMIND 是 **非商用 / 研究 license**,商用前必须确认;
- Ego4D 等人类视频涉及**人脸/隐私**,商用部署需脱敏(模糊面部);
- 中国境内运营若涉及生物识别,触发《个人信息保护法》合规审查。

---

## 二、动作表征与跨本体统一(2026 真正的核心难点)

跨本体 zero-shot 能不能做出来,**90% 取决于动作空间设计**。

### 2.1 动作空间选择

| 选项 | 优点 | 缺点 | 用例 |
|---|---|---|---|
| **EEF-relative 6D + gripper**(2026 推荐默认值) | 跨本体最稳,单/双臂统一 | 需要逆运动学,部分高 DoF 浪费 | π0/π0.6/π0.7、X-VLA |
| **Joint-space**(关节角增量) | 直接、高 DoF 友好 | 跨本体几乎无法迁移 | 全身控制 S0 层(Helix 02) |
| **混合(EEF + joint residual)** | 兼具两者 | 工程复杂 | GR00T N1.7 |
| **Latent action**(VQ-VAE / LAPA) | 可从无标注视频学 | 需要小样本动作映射 | Being-H0.5、Ψ₀ |

**实操建议**:基础模型用 EEF-relative,本体相关的细节(灵巧手、脚部)在 **per-embodiment expert head** 里处理;Helix 02 / GR00T N1.7 都是这个套路。

### 2.2 动作 Tokenizer(autoregressive VLA 必读)

- ⭐ **FAST / FAST+**(Physical Intelligence, RSS'25 → 2026 主流):**DCT 频域 + BPE**,比 per-dim binning 压缩 ~10×,π0-FAST 可以训到 **10k h 数据并匹敌 diffusion VLA,训练时间快 5×**。
- 对高频灵巧任务(如 50Hz 双手),朴素 binning 会**爆 token 数**导致训不动,这是很多复现失败的根因。
- LeRobot 已集成 `lerobot/fast-action-tokenizer`,直接用。

### 2.3 Action Head 选型

| 类型 | 代表 | 何时选 |
|---|---|---|
| **Autoregressive token**(FAST) | π0-FAST、OpenVLA | 训练 throughput 高,长视野规划 |
| **Flow Matching**(2026 主流) | π0/π0.6/π0.7、GR00T | 连续高频 + 推理快(63 ms/chunk) |
| **Diffusion** | RDT、π0.5 早期 | 多模态分布,但 denoise 步多 |
| **Latent Action** | LAPA、Being-H0.5 | 主要用于无监督预训练 |
| **三层(S0/S1/S2)** | Helix 02、GR00T N2 | 人形 / 全身 / 长程 |

### 2.4 跨本体身份注入

- ⭐ **Soft Prompt embodiment id**(X-VLA):极少新增参数;
- **Per-embodiment readout head**(CrossFormer / HPT 思路依然有效);
- **proprio adapter**(给每个本体一个 MLP,投到统一 token 空间)。

> **重要细节**:embodiment id 不能光放语言 prompt 里——要在 **proprio 编码器入口 + action head 出口**两处都注入,否则模型会把语言 prompt 当作普通文本忽略掉。

---

## 三、模型架构(2026 三层架构成主流)

### 3.1 整体形态(参考 Helix 02 + GR00T N2)

```text
┌─────────────────────────────────────────────────────┐
│ S2  慢思考 / 规划层                                  │
│     • Backbone: Qwen3-VL / Cosmos-Reason2-2B /      │
│                Gemma3-4B / InternVL-3               │
│     • 频率: 1–5 Hz                                   │
│     • 输出: 子任务语言、关键帧、affordance、code     │
└──────────────────┬──────────────────────────────────┘
                   │ KV cache + cross-attn
┌──────────────────▼──────────────────────────────────┐
│ S1  快控制 / 视触动作                                │
│     • Action Expert: 200–800M Flow / Diffusion DiT  │
│     • 输入: 多视角 RGB(D) + proprio + 触觉           │
│     • 输出: action chunk(0.5–1 s,EEF + 灵巧)        │
│     • 频率: 50–200 Hz                                │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ S0  本体伺服 / 平衡(仅人形 / 移动底盘必需)          │
│     • 学到的低层控制器(MLP / 小 Transformer)        │
│     • 频率: 500 Hz – 1 kHz                           │
│     • 替换: 传统 WBC / MPC                            │
└─────────────────────────────────────────────────────┘
```

### 3.2 关键架构选择

- **VLM Backbone**:2026 已成共识用 **Qwen3-VL** 或 **Cosmos-Reason2** 而非 Gemma2/PaliGemma,因为前者支持原生多帧 + 中文 + 推理链。
- **MoE / MoT**:HY-Embodied-0.5 用 **Mixture-of-Transformers**(2B 边端 + 32B 云端),适合"同一模型多硬件部署"。
- **World Model 显式建模**:GR00T N2 / GigaWorld 把 future-state prediction 作为辅助 head,推理时不一定 forward,但**训练时强制存在**对长程任务必涨点。
- **触觉融合**:2026 起触觉(GelSight、AnySkin、TacTip)正式进入 SOTA;Helix 02 触觉直接进 S1。
- **3D / 几何先验**:GeneralVLA 的 3DAgent + affordance head 在**无真机数据**场景仍能 zero-shot,适合冷启动。

### 3.3 参数规模目标(2026 公开数据点)

- **轻量端**:π0.6 ≈ 4.86B(Gemma3-4B + 860M action expert),H100 单卡 63 ms/chunk;
- **中端**:GR00T N1.7 ≈ 3B 总,Jetson Thor 可跑;
- **大端**:HY-Embodied-0.5 32B,云端推理;
- **超大未公开**:π0.7、Helix 02 估算 7B–20B 区间(未官方披露)。

> 个人建议:**一次训两个尺寸**(2B 边、7B–13B 云),共享数据和大部分架构,边端走蒸馏 + 量化(见 §8)。

---

## 四、训练 pipeline 与课程(Curriculum)

### 4.1 推荐 5 阶段(综合 π0.5、Green-VLA、Ψ₀)

```text
Stage 1  VLM 通识预训练(用现成开源)
Stage 2  Latent-Action 预训练(LAPA / Ψ₀ 风格,从人类视频学动作先验)
Stage 3  跨本体真机 + 仿真大规模动作预训练(EEF-relative,FAST tokenizer)
Stage 4  目标本体后训练(SFT,50–500 条演示 / 任务)
Stage 5  Online RL & 持续学习(SOP / LWD / VLA-OPD,部署后持续涨点)
```

### 4.2 训练超参与坑点

- **学习率分层**:VLM backbone 1e-5,action expert 5e-4–1e-3,差 **20–100×**,否则 backbone 会被动作 loss "拽飞"——OpenVLA、π0、GR00T 都明确这么做。
- **Loss 权重平衡**:动作 loss / 语言 loss / 视觉重建 loss / 世界模型 loss,推荐先用 GradNorm 或 PCGrad 自动平衡。
- **Action Chunk 长度**:0.5–1 s(50 Hz 下 25–50 步)是当前甜蜜区——再短闭环抖动,再长泛化掉。
- **Receding Horizon 推理**:推 50 步只执行前 10–20 步,然后重推,π0 系列默认这么做。
- **梯度 checkpointing + ZeRO-3 + FSDP**:7B+ 模型必备。
- **数据 streaming**:用 **HuggingFace Datasets v3.0** 或 **WebDataset**,千卡训练 IO 必须流式,落盘式 dataset 会成瓶颈。

### 4.3 RL 后训练注意

- **Reward**:稀疏成功 reward 不够用,**进度 reward**(TT-VLA)+ **VLM-as-critic**(OmniGuide)+ 人类干预 reward(SOP HG-DAgger)三者结合最稳。
- **On-policy distillation**(VLA-OPD)**优于** 纯 PPO/GRPO:用专家老师做 token 级 dense supervision,样本效率高一个数量级,且不容易遗忘。
- **Replay 防遗忘**:LifeLong-RFT 的 chunk-level RL + 多维过程 reward;Replay buffer 按本体分桶采样。
- **Fleet-scale**:LWD 的 DIVL + QAM 在 16 台双臂 8 任务做到 95%——单台真机 RL 上限有限,fleet 是必须的。

---

## 五、仿真基础设施与 Sim2Real

### 5.1 仿真平台选择(2026 主流)

| 仿真器 | 优势 | 缺点 |
|---|---|---|
| **Isaac Lab / Isaac Sim** | GPU 加速、最快 | 接触建模一般,触觉需自研 |
| **GR00T-Sim** | NVIDIA 全栈、人形 | 闭源生态 |
| **RoboCasa** | 厨房/家居场景丰富 | 专注 manipulation |
| **RoboTwin 2.0** | **强 DR**、bimanual | 较新,生态小 |
| **MuJoCo MJX** | 接触物理最准 | GPU 渲染较弱 |
| **DexMimicGen** | 灵巧手数据生成 | 需要 source demos |

### 5.2 Sim2Real 关键技术(2026 最新)

- ⭐ **DexSim2Real**(2026.05):用 VLM 当 visual realism critic,自动优化仿真渲染参数;触觉-视觉 cross-attention 策略;LLM 课程分解。**真机 78.2% 平均成功率,sim2real gap 8.3%**。
- ⭐ **RoboTwin 2.0 强 DR**:沿 5 个轴(杂物 / 光照 / 背景 / 桌高 / 语言)随机化 + LLM 自动任务合成,**只需 10 条真机演示就比 baseline +367%**,纯 sim 训练 zero-shot +228%。
- **MolmoBot**(2026):1.8M 仿真轨迹 + 94k 环境,**完全无需真机微调** 79.2% pick-and-place。
- **SimDist**(Simulation Distillation):仿真预训世界模型,真机 15–30 分钟快速适应。
- **FM-DR**(Foundation Model-Guided DR):VLM 评估"仿真渲染像不像真的",自动调参,比手调 DR 显著好。

### 5.3 Sim2Real 工程坑

- **物理参数随机化**(摩擦、质量、阻尼)比视觉随机化对接触任务更重要;
- 仿真里**触觉传感器极难仿真**,直接 dropout 触觉训练 + 真机加触觉 fine-tune 是常用妥协;
- **延迟 / 抖动**模拟:真实机器人有 20–100 ms 延迟和噪声,sim 必须随机化否则 zero-shot 必跌;
- **相机参数随机化**(intrinsic + extrinsic + 噪声)——很多人忘做,导致 sim2real 第一次部署直接 0%。

---

## 六、评测体系(防止"自欺")

### 6.1 多层评测金字塔

```text
[小时级]   单元测试  : 单 episode 仿真 unit test
[小时级]   仿真 bench: LIBERO+, RoboCasa, RoboTwin 2.0, SimplerEnv
[天级]     真机基线  : 自有 fixtures,固定 5–10 个任务,每周复测
[周级]     Real-to-Sim: RobotArena ∞(arXiv 2510.23571)
[月级]     社区盲测  : RoboArena(DROID, 2026 全年开放,双盲两两对比)
```

### 6.2 关键评测维度(必须分别报告)

- **In-distribution**:训练过的本体 + 任务 + 环境;
- **任务泛化**:同本体新任务;
- **环境泛化**:新背景、新光照、新干扰物;
- **物体泛化**:新形状、新材质;
- **本体泛化(zero-shot 跨本体的核心)**:训练完全没见过的机器人;
- **指令泛化**:同义改写、长尾说法、code-switching;
- **长程任务**:≥ 60 s 多子任务;
- **鲁棒性**:STRONG-VLA 风格的 28 种多模态扰动;
- **OOD 检测率**:ReconVLA 风格的 conformal coverage。

### 6.3 SimplerEnv 的 MMRV 指标

**Mean Maximum Rank Violation**——衡量 sim 排名和真机排名一致性。比单纯报告 sim success rate 更可信。

---

## 七、部署与推理优化(决定能不能装上车)

### 7.1 延迟预算(2026 真实数据)

- 真机闭环 50 Hz → 单次推理 < **20 ms** 才不抖;
- π0.6 在 H100 上 **63 ms / chunk**(5 步 flow + 3 路相机)→ chunk 内 25 步分摊 2.5 ms,但首次推理仍是关键瓶颈;
- Jetson 上 **75% latency 来自 action-generation 阶段的 memory-bound**(2026 测量结果)。

### 7.2 优化技术(2026 已实证)

- ⭐ **QuantVLA**(2026.02, arXiv 2602.20309):后训练量化,**70% 内存节省、1.22× 加速**,且 success 不掉甚至 +。
- ⭐ **Shallow-π** 蒸馏:**18 → 6 层** transformer,>2× 加速,task drop < 1%。
- **KV cache 共享 + chunk reuse**:S2 cache 给 S1 attend,Helix/π0 的标准做法。
- **Flow 步数压缩**:π0.6 只 5 步 flow → 1 步 distill flow 可再快 5×(尚未公开发布,但 lerobot issue 里在做)。
- **TensorRT / vLLM / SGLang** 部署:Jetson Thor 上 INT8/FP8 推理;`NVlabs/vla-perf` 是 NVIDIA 给的官方 benchmark。

### 7.3 边/云协同部署

- **S2 在云,S1+S0 在边**:HY-Embodied-0.5 的 2B(边)+ 32B(云)就是这个范式。
- 网络抖动时 S2 可以**降级**到本地小 VLM 或上一帧 plan,不能阻塞 S1。

---

## 八、鲁棒性 / 安全 / 失败恢复(部署的真正门槛)

### 8.1 OOD 检测与不确定性

- ⭐ **ReconVLA**(2026.04, arXiv 2604.16677):**Conformal prediction** 给出 calibrated 置信区间,实时检测 OOD,部署前必加。
- 阈值触发后:回退、慢化、求助人类(human-in-the-loop)、切换保守策略。

### 8.2 自纠错与重试

- ⭐ **CycleVLA**(2601.02295):预判失败 → 子任务回溯 + Minimum Bayes Risk 解码。
- 在子任务转换点(open gripper / lift)显式插 checkpoint,失败时回到上一个 checkpoint 重试。

### 8.3 多模态扰动鲁棒性

- ⭐ **STRONG-VLA**(2604.10055):28 种视觉 + 文本扰动课程训练,跨架构都能 +16.49%。
- 最少要做:遮挡、运动模糊、低光、改色温、指令同义改写、错别字、噪声指令、**对抗 prompt**。

### 8.4 极端感知条件

- **E-VLA**:event camera + RGB,黑暗场景从 0% → 90%。

### 8.5 安全约束(硬性、不能省)

- **力 / 力矩硬限位**(在 S0 / 控制器层强制,不依赖网络);
- **几何 / 关节安全空间**(SDF + 可达空间检查);
- **紧急停止**:必须有非 ML 的 watchdog;
- **指令安全过滤**:对 "go pick up that knife and stab" 这类指令,需 VLM 安全分类器拦截,不能盲目执行。

### 8.6 数据偏置与失败模式溯源

- 训练数据要保留**完整失败示范 + 原因标签**(RoboMIND v2 风格);
- 部署后**所有失败 episode 全程记录回流**到训练集——这是 LWD / SOP 的命脉。

---

## 九、工程基础设施(Infra)

### 9.1 数据管道

- **统一 schema**(LeRobot Datasets v3.0 或自研 protobuf);
- **流式读取**(WebDataset / MosaicML Streaming);
- **DDP-aware shuffling**:跨节点不能让相邻 batch 都来自同一 episode;
- **Lazy 解码 + GPU 解码**:相机视频用 NVIDIA DALI 或 torchcodec,CPU 解码会成 IO 瓶颈;
- **数据版本管理**:DVC / Pachyderm / lakeFS,**每次实验绑定一个数据 hash**,否则不可复现。

### 9.2 训练框架

- **FSDP / DeepSpeed ZeRO-3**(7B+ 必需);
- **3D 并行**(TP + PP + DP)在 30B+ 才必要;
- **混合精度**:bf16 训练 + fp32 master weight,**fp8 训练在 2026 的 H100/B200 上已可用**;
- **Async checkpoint**:防止 IO 拖训练 throughput;
- **梯度噪声监控** + **loss spike 自动 rollback**(7B+ 训练经常 spike);
- **持续训**(streaming pretraining):像 SOP/LWD 那样,新数据进来就接着训,不重启。

### 9.3 算力规模(2026 公开/估算数据点参考)

- **复现 OpenVLA-7B**:8 × H100 × 14 天 ≈ 26k GPU-hour;
- **复现 π0 级别**:64–128 × H100 × 2–3 周 ≈ 100–200k GPU-hour;
- **复现 π0.6/0.7 级别**:估 256–512 × H100 × 1 月 ≈ 500k GPU-hour;
- **真正"超大"**(GR00T-N2/HY-32B 量级):≥ 1M GPU-hour,数千万到上亿 RMB 算力成本。

### 9.4 实验管理

- **Trackio / Weights&Biases / Tensorboard** 都行,**关键是约束实验命名规范**;
- **每个实验绑定**:代码 git hash + 数据 hash + 超参 yaml + 评测脚本 hash;
- **可复现性 CI**:每周抽一个旧实验自动重训,保证基础设施稳定。

---

## 十、机器人硬件 / 遥操数据采集(常被忽略但关键)

### 10.1 硬件选型(决定数据多样性)

- 至少覆盖 **5 类本体**:单臂(Franka / UR5)、双臂工业(AgileX / Cobot)、双臂人形(G1 / H1 / Adam)、灵巧手、移动底盘。
- 同型号至少 **3–5 台同时采集**——降低个体差异噪声。

### 10.2 遥操方式

- **VR + IK 遥操**(主流):Oculus Quest 3 + 自定义 retargeting;
- **力控遥操**(AgiBot 强调):捕捉接触动力学,比纯位置遥操数据高一个层次;
- **示范成本(2026 实测)**:中等任务 ~5 分钟/条,高难任务 ~20 分钟/条;**1 万小时 ≈ 60 人/年纯遥操**——产业级团队配 50–200 人遥操员是常态。

### 10.3 数据采集 SOP

- 每条轨迹:**3 路以上 RGB**(third-person + wrist 左右)+ 触觉 + proprio + 力/力矩 + 语言指令(由不同人写 3 条);
- **失败示范也要采**(并打标签),目标占比 5–10%;
- **多采集者多样性**:同一任务由 ≥ 5 个不同人采,避免动作风格过拟合。

---

## 十一、团队、流程、路线规划

### 11.1 团队结构(参考 PI / NVIDIA / AgiBot 公开信息估算)

| 模块 | 人数 | 关键能力 |
|---|---|---|
| 数据采集 / 遥操 | 50–200 | 项目管理 + 大规模标注 |
| 数据工程 | 5–10 | 大规模 pipeline + IO |
| 模型/算法 | 10–20 | VLM、Diffusion/Flow、RL |
| 仿真 / Sim2Real | 5–8 | Isaac/MuJoCo/渲染 |
| 部署 / 推理 | 3–5 | TensorRT、Jetson、量化 |
| 控制 / 硬件 | 5–10 | WBC、MPC、伺服 |
| 评测 / QA | 3–5 | 真机 fixtures、自动化 |
| 安全 / 合规 | 1–3 | 法规、数据、伦理 |

最低可用配置约 **30–50 人**,世界级团队 **200+**。

### 11.2 路线规划(从 0 到能 demo 的现实时间表)

| 阶段 | 时长 | 里程碑 |
|---|---|---|
| 0. 基建 | 1–2 月 | 数据 schema、训练 infra、仿真台架 |
| 1. 复现 SOTA | 2–3 月 | 在 LIBERO/RoboCasa 上复现 GR00T N1.7 / π0.5 报数 |
| 2. 数据扩增 | 3–6 月 | 自采 1k–5k 小时 + 公开数据合并 |
| 3. 第一版本预训 | 1–2 月 | 7B 跨 3 本体,内部真机 demo |
| 4. 后训练 + 评测 | 2–3 月 | 目标本体 SFT + Online RL,过 RoboArena |
| 5. 第二轮迭代 | 3–6 月 | 加 World Model、扩到 10+ 本体、上 RoboArena leaderboard |

**总计:12–24 个月** 才有真正 zero-shot 跨本体的产品级模型。

### 11.3 风险点(项目失败的常见原因,我反复在公开 postmortem 里看到)

1. **遥操数据质量差** → 模型模仿到错误动作风格,无法纠正;
2. **过早做大模型**,忘了小模型 + 大数据先跑通基线;
3. **没有真机评测节奏**,sim 表现飙升但真机不动;
4. **算法人和硬件人不沟通**,延迟 / 频率不匹配;
5. **不做数据版本管理**,实验完全不可复现;
6. **忽视安全 / OOD 检测**,部署后第一次 incident 把项目打死。

---

## 十二、合规、伦理、法律

- **数据版权**:OXE/AgiBot World/DROID/RoboMIND 多为研究 license,商用需另签;Ego4D 含人脸,商用需脱敏。
- **人机协作安全**:ISO 10218 / ISO/TS 15066(协作机器人),欧盟 *Machinery Regulation 2023/1230* 2027 全面生效,中国《机器人安全国标 GB/T xxxxx》正在更新。
- **算法可解释性**:面向 To B 部署,客户会要求 OOD 检测、决策可追溯——见 §8 ReconVLA。
- **AI 法规**:EU AI Act(2026 生效高风险类目),具身大模型大概率被划为 **高风险**,需要登记 + 风险评估 + 记录留存。
- **数据出境**:中文具身数据训国外模型涉及《数据出境安全评估办法》。

---

## 总结:一份"2026 落地清单"(Checklist)

项目启动前,把下面这张表打印出来贴墙上:

- [ ] **目标本体清单**(至少 5 类,含一个完全 holdout 用于 zero-shot 评测)
- [ ] **数据来源 + 配比 + license** 文档化
- [ ] **统一 schema**(EEF + joint 都存,3+ 视角,3 级语言指令)
- [ ] 选定 **VLM backbone**(Qwen3-VL / Cosmos-Reason2 / Gemma3)
- [ ] 选定 **Action 表征**(EEF-relative + FAST tokenizer 或 Flow Matching)
- [ ] 选定 **架构层级**(2 层 / 3 层 S0/S1/S2)
- [ ] 选定 **World Model 是否显式建模**(强烈建议:是)
- [ ] **5 阶段 curriculum** + 每阶段数据混合权重
- [ ] **Sim 平台**(Isaac Lab / RoboTwin 2.0 / GR00T-Sim)+ DR 五轴
- [ ] **真机评测 fixture**(每周复测的固定 10 任务)+ RoboArena 报名
- [ ] **OOD / Conformal 检测**(ReconVLA 风格)+ 自纠错(CycleVLA)
- [ ] **安全 watchdog**(力限位、空间限位、E-stop,与 ML 解耦)
- [ ] **量化 / 蒸馏 / 边端部署**(QuantVLA / Shallow-π / TensorRT)
- [ ] **Online RL pipeline**(SOP / LWD / VLA-OPD)+ fleet 数据回流
- [ ] **数据 / 代码 / 实验三方版本绑定**
- [ ] **法规与 license 审查**已签

---

如果你愿意告诉我:
1. 目标本体数量 / 种类
2. 现有数据规模(自采 + 外部)
3. 算力预算与时间表

我可以基于这份清单出一份**专属于你项目的"训练总配方表"**(每阶段数据混合权重、超参、并行策略、评测节点和 fallback 计划),全部对齐 2026 SOTA 数字。

---

# VLA预训练考虑点分析.标出处并补漏.V2

**User**

把刚才你罗列的训练超大具身预训练模型所需要考虑的每个方面和每条细节后面都加个括号, 括号内注明这个观点或细节是来自那篇新论文或资料, 如果找不到严格的出处就标"找不到出处", 如果在这个过程中发现漏了哪些观点和细节就一定要补上, 然后在括号中注明"补漏"以及它的出处. 

---

**Cursor**

让我先搜几个可能漏掉的主题,确保补漏完整。

下面是给上一轮回答**逐条加注出处**后的完整版本。说明:
- `[出处]` 表示有明确论文/blog/GitHub 来源;
- `[找不到出处]` 表示是我基于多年实践经验做的工程总结,无法精确指向某一篇 2026 论文;
- `[补漏 + 出处]` 是这次重审时新发现遗漏的关键点。

---

## 顶层设计原则

1. **数据决定上限**,模型架构决定下限 [找不到出处]。LingBot-VLA 把真机数据从 3k h → 20k h,success rate 持续涨,无饱和迹象,意味着至少在 10⁴–10⁵ 小时尺度上是 data-bottleneck-limited [LingBot-VLA, arXiv 2601.18692]。
2. **Naive 跨本体 mixing 必然 negative transfer**:多篇 2026 工作验证 [Rethinking VLA Scaling, arXiv 2602.09722; OXE-AugE, arXiv 2512.13100; X-VLA, arXiv 2510.10274]。
3. **必须显式建模"未来"**:GR00T N2 / GigaWorld-Policy / Mask World Model 一致显示纯反应式 VLA 在长程任务上输 +50% ~ +95% [GR00T N2, techbytes.app; GigaWorld, arXiv 2603.17240; MWM, arXiv 2604.19683]。
4. **三层时间尺度是人形必经之路**:Helix 02 显式拆 1 kHz S0 / 200 Hz S1 / 慢速 S2,神经网络替换 109,504 行手写 C++ [Helix 02 blog, figure.ai/news/helix-02]。
5. **Online RL 不是可选,而是必选**:SOP / LWD 在 fleet 上小时级生效 [SOP, arXiv 2601.03044; LWD, arXiv 2605.00416];纯 SFT 会灾难性遗忘 [LifeLong-RFT, arXiv 2602.10503]。
6. **真机评测才是真理**:跨本体 zero-shot 必须上 RoboArena / RobotArena ∞ [RoboArena, robo-arena.github.io; RobotArena ∞, arXiv 2510.23571]。
7. **[补漏] Scaling 不只是数据量,还有 quality alignment + action unification + RL refinement** [Green-VLA, arXiv 2602.00919]。
8. **[补漏] RL outcome-driven 是突破纯监督数据上限的另一条路** [SimpleVLA-RL, openreview TQhSodCM4r]。

---

## 一、数据工程

### 1.1 数据来源与配比

| 来源 | 起步规模 | 出处 |
|---|---|---|
| OXE-AugE 4.4M 轨迹 | [OXE-AugE, arXiv 2512.13100] |
| AgiBot World 2026,1M+ 轨迹、43.8 TB | [AgiBot-W GitHub, OpenDriveLab/agibot-world; agibot.com/research] |
| AgiBot World Colosseo 论文 | [arXiv 2503.06669] |
| LingBot 20k h 双臂数据 | [LingBot-VLA, arXiv 2601.18692] |
| RoboMIND v2 多本体基准 | [RoboMIND, arXiv 2412.13877] |
| UniHand-2.0(35k h、30 本体) | [Being-H0.5, arXiv 2601.12993] |
| Ego(Exo)4D 用 latent action 预训练 | [LAPA 路线总结, 找不到 2026 出处]、[Ψ₀ 用 800 h 人 + 30 h 真机, arXiv 2603.12263] |
| RoboCasa / GR00T-Sim / RoboTwin 2.0 / Isaac Lab 仿真 | [RoboTwin 2.0, arXiv 2506.18088;Isaac-GR00T GitHub] |

> "Ψ₀ 800 h 人类视频 + 30 h 真机 > 10× 数据 baseline +40%" [Ψ₀, arXiv 2603.12263v1]。

### 1.2 数据混合权重

- 不要按"自然比例"混 [Rethinking VLA Scaling, arXiv 2602.09722]。
- **学到的混合权重**(DoReMi、α 自适应)优于固定比例 [找不到出处,DoReMi 是 LLM 领域工作,在 2026 VLA 中我未见严格对照 paper]。
- **逐阶段切换 ratio**(curriculum):前期重 web/sim,后期重 real,动作空间约束逐渐收紧 [Green-VLA 5 阶段课程, arXiv 2602.00919]。

### 1.3 数据质量过滤

- 显式过滤标准(轨迹平滑、视觉清晰、视觉多样) [Green-VLA, arXiv 2602.00919]。
- 用 VLM 重新打分自报告成功标签 [Mask World Model 用 VLM 当 critic, arXiv 2604.19683;OmniGuide 推理时 VLM 引导, arXiv 2604.20100]。
- 失败示范保留 + 标注原因(目标占比 5–10%) [RoboMIND 显式收集 5k 失败示范, arXiv 2412.13877]。
- 动作 vs 视觉一致性(光流和末端运动方向相关) [找不到出处,工程经验]。
- 感知哈希 + 动作序列哈希去重 [找不到出处,工程惯用]。

### 1.4 数据规范化(跨本体 schema)

- 相机外参对齐到 base / EEF [找不到出处,工程惯例]。
- 同时存 joint-space + EEF 6D + gripper 两种动作表示 [Rethinking VLA Scaling 强调 EEF-relative 关键, arXiv 2602.09722;GR00T N1.7 实践, NVIDIA/Isaac-GR00T]。
- 统一重采样基频 50 Hz [π0/π0.6 实践, openpi GitHub;Pi0-doc, huggingface.co/docs/lerobot/pi0]。
- 每条轨迹 ≥ 3 条不同粒度语言指令 [找不到出处,工程惯用;π0.5 co-train 高低层指令的精神, 见 PI 主页 pi.website]。

### 1.5 [补漏] 数据工厂规模化(2026 重要趋势)

- **Scanford**:在 Stanford 图书馆部署 mobile manipulator 自动采集,VLM 书本识别 32.0% → 71.8%,节省 18.7 人小时 [Scanford, arXiv 2511.19647]。
- **Tutor Intelligence Data Factory One**(波士顿,~100 台 Sonny 人形)— 把"数据-模型-能力飞轮"做成产品 [Forbes, 2026-05-05]。
- **数据飞轮"必须自有全栈"**论点 [Tutor Intelligence, Forbes 2026-05-05]。

### 1.6 合规与版权

- 数据集 license 商用限制 [找不到 2026 综述,工程实践常识]。
- Ego4D 涉及人脸 [找不到出处,数据集发布说明]。
- 中国《数据出境安全评估办法》 [找不到出处,法规常识]。

---

## 二、动作表征与跨本体统一

### 2.1 动作空间选择

| 选项 | 出处 |
|---|---|
| EEF-relative 6D + gripper(默认) | [Rethinking VLA Scaling 关键结论, arXiv 2602.09722;π0/π0.6/π0.7 实践, PI06 model card];[X-VLA 跨 6 仿真 3 真机, arXiv 2510.10274] |
| Joint-space 增量 | [Helix 02 S0 层, figure.ai/news/helix-02] |
| 混合 EEF + joint residual | [GR00T N1.7, NVIDIA/Isaac-GR00T] |
| Latent action(VQ-VAE) | [Being-H0.5, arXiv 2601.12993;Ψ₀, arXiv 2603.12263] |

### 2.2 动作 Tokenizer

- ⭐ **FAST / FAST+**:DCT 频域 + BPE,~10× 压缩,π0-FAST 训到 10k h、匹敌 diffusion VLA、训练快 5× [FAST, arXiv 2501.09747, RSS'25]。
- LeRobot 集成 `lerobot/fast-action-tokenizer` [HuggingFace lerobot/fast-action-tokenizer 模型卡]。
- 朴素 binning 在高频任务上 token 数爆炸 [FAST 论文论证, arXiv 2501.09747]。

### 2.3 Action Head 选型

- Autoregressive token / FAST [FAST, arXiv 2501.09747;OpenVLA 经典做法, 上轮已弃用]。
- Flow Matching:π0/π0.6/π0.7,63 ms/chunk [PI06 model card]。
- Diffusion:RDT、π0.5 早期 [上轮已用,2026 主流转 flow,出处 PI06 model card]。
- Latent Action 预训练:LAPA / Ψ₀ / Being-H0.5 [Ψ₀, arXiv 2603.12263;Being-H0.5, arXiv 2601.12993]。
- 三层 S0/S1/S2:Helix 02 / GR00T N2 [Helix 02 blog;GR00T N2, techbytes.app]。

### 2.4 跨本体身份注入

- Soft Prompt embodiment id [X-VLA, arXiv 2510.10274]。
- Per-embodiment readout head / proprio adapter [找不到 2026 严格出处,源自 CrossFormer/HPT 思路, 上轮已注其 CoRL'24/NeurIPS'24]。
- **embodiment id 要在 proprio 入口 + action head 出口两处都注入** [找不到出处,工程经验,X-VLA 论文里有部分 ablation 暗示, arXiv 2510.10274]。

### 2.5 [补漏] 3D / 几何先验显式建模

- **PointVLA**:点云模块化注入预训 VLA,小样本多任务 [PointVLA, arXiv 2503.07511]。
- **Any3D-VLA**:统一 simulator/sensor/model-estimated 点云,跨域 3D 表征 [Any3D-VLA, arXiv 2602.00807]。
- **ST-VLA**:4D 时空理解 + ST-Human 数据集(300k episodes) [ST-VLA, arXiv 2603.13788]。
- **StemVLA**:未来 3D + 历史 4D [StemVLA, arXiv 2602.23721]。
- **Avi**:把动作生成重新 frame 成 3D 感知 + 几何变换,用预测点云 [Avi, arXiv 2510.21746]。

### 2.6 [补漏] 触觉表征

- **AnySkin**(磁性触觉皮肤,跨实例泛化) [AnySkin, arXiv 2409.08276v2]。
- **VISK**(Visuo-Skin,触觉 token + 视觉 token,接触任务 +27.5%) [VISK, arXiv 2410.17246]。
- **AnyTouch 2**(2026.02):跨光学触觉传感器,ToucHD 数据 2,426,174 接触样本 [AnyTouch2, gewu-lab.github.io/AnyTouch2 + arXiv 2602.09617]。

---

## 三、模型架构

### 3.1 三层架构

- 整体形态参考 Helix 02 + GR00T N2 [Helix 02 blog, figure.ai/news/helix-02; GR00T N2, techbytes.app]。
- 频率 1–5 Hz / 50–200 Hz / 500–1000 Hz [Helix 02:1 kHz/200 Hz, figure.ai/news/helix-02]。

### 3.2 关键架构选择

- **Backbone Qwen3-VL / Cosmos-Reason2 / Gemma3** [GR00T N1.7 用 Cosmos-Reason2-2B/Qwen3-VL, NVIDIA/Isaac-GR00T;π0.6 用 Gemma3-4B, PI06 model card]。
- **MoE / MoT(2B 边 + 32B 云)** [HY-Embodied-0.5, arXiv 2604.07430]。
- **World Model 显式建模** [GR00T N2, techbytes.app;GigaWorld-Policy, arXiv 2603.17240;MWM, arXiv 2604.19683]。
- **触觉融合直接进 S1** [Helix 02 blog "整合所有 onboard 传感器(视觉/触觉/proprio)"]。
- **3D 几何先验 / affordance head 适合冷启动** [GeneralVLA, arXiv 2602.04315]。

### 3.3 参数规模

- π0.6 ≈ 4.86B(Gemma3-4B + 860M action expert),H100 单卡 63 ms/chunk [PI06 model card]。
- GR00T N1.7 ≈ 3B,Jetson Thor 可跑 [Isaac-GR00T GitHub README;GR00T N2 用 Jetson Thor, techbytes.app]。
- HY-Embodied-0.5 32B 云端 [arXiv 2604.07430]。
- π0.7 / Helix 02 7B–20B 估算 [找不到出处,基于 PI/Figure 公开信息推测]。
- 一次训两个尺寸,边端走蒸馏 + 量化 [找不到 2026 严格出处,与 HY-Embodied 2B+32B 思路一致, arXiv 2604.07430]。

---

## 四、训练 pipeline 与课程

### 4.1 5 阶段课程

```text
Stage 1  VLM 通识预训练
Stage 2  Latent-Action 预训练
Stage 3  跨本体真机 + 仿真大规模动作预训练
Stage 4  目标本体后训练(SFT)
Stage 5  Online RL & 持续学习
```

- π0.5 co-training 思路 [PI 主页 pi.website / openpi GitHub README]。
- Green-VLA 5 阶段 curriculum [arXiv 2602.00919]。
- Ψ₀ 两段式(先人后机) [arXiv 2603.12263]。

### 4.2 训练超参与坑点

- **学习率分层**:VLM 1e-5、action expert 5e-4–1e-3,差 20–100× [找不到 2026 严格出处;OpenVLA 论文里有提及, 上轮已注;π0/GR00T 公开训练脚本中可见, openpi GitHub]。
- **Loss 权重平衡**(GradNorm/PCGrad) [找不到出处,工程惯例]。
- **Action Chunk 长度 0.5–1 s** [FAST 论文 + π0 实践, arXiv 2501.09747;PI06 model card]。
- **Receding Horizon 推 50 执行 10–20** [openpi GitHub π0 默认实现]。
- **FSDP / ZeRO-3 + bf16 + fp8** [找不到出处,通用大模型训练实践]。
- **数据 streaming(WebDataset / Datasets v3.0)** [LeRobot v0.4 release blog 强调 Datasets v3.0;huggingface/blog lerobot-release-v040.md]。

### 4.3 RL 后训练

- 进度 reward [TT-VLA, arXiv 2601.06748]。
- VLM-as-critic [OmniGuide, arXiv 2604.20100;MWM, arXiv 2604.19683]。
- 人类干预 reward(HG-DAgger) [SOP, arXiv 2601.03044]。
- On-policy distillation 优于纯 PPO/GRPO [VLA-OPD, arXiv 2603.26666v1]。
- chunk-level RL + 多维过程奖励防遗忘 [LifeLong-RFT, arXiv 2602.10503]。
- DIVL + QAM、16 台双臂 8 任务 95% [LWD, arXiv 2605.00416]。
- [补漏] **outcome-driven RL 是突破监督上限的路径** [SimpleVLA-RL, openreview TQhSodCM4r]。

---

## 五、仿真基础设施与 Sim2Real

### 5.1 仿真平台

| 平台 | 出处 |
|---|---|
| Isaac Lab / Isaac Sim | [找不到出处,通用知识] |
| GR00T-Sim | [Isaac-GR00T README] |
| RoboCasa | [上轮已注 CoRL'24,2026 仍主流;huggingface/lerobot 集成] |
| RoboTwin 2.0(强 DR) | [RoboTwin 2.0, arXiv 2506.18088] |
| MuJoCo MJX | [找不到出处,通用知识] |
| DexMimicGen | [找不到 2026 严格出处, NVIDIA 公开项目] |

### 5.2 Sim2Real 技术

- ⭐ **DexSim2Real**:VLM 当渲染 critic + 触觉视觉 cross-attention + LLM 课程,真机 78.2%、gap 8.3% [DexSim2Real, arXiv 2605.05241]。
- ⭐ **RoboTwin 2.0 强 DR**:5 轴 + LLM 任务合成,只需 10 真机演示 +367%,纯 sim zero-shot +228% [RoboTwin 2.0, arXiv 2506.18088]。
- **MolmoBot**:1.8M 仿真轨迹 + 94k 环境,无真机微调 79.2% [MolmoBot, arXiv 2603.16861]。
- **SimDist**:仿真预训世界模型,真机 15–30 分钟适应 [SimDist, arXiv 2603.15759]。
- **FM-DR**(VLM 自动调 DR 参数) [DexSim2Real 中的 FM-DR 模块, arXiv 2605.05241]。
- [补漏] **Approximate Information States**:用 state abstraction 形式化 sim2real,允许仿真省略关键细节 [Approximate IS, arXiv 2604.15289]。

### 5.3 Sim2Real 工程坑

- 物理参数随机化 > 视觉随机化(对接触任务) [找不到 2026 严格出处,RoboTwin 2.0 间接验证 5 轴含杂物/桌高, arXiv 2506.18088]。
- 触觉传感器仿真极难 + dropout 训练 [找不到 2026 严格出处,DexSim2Real 用真机触觉训练而非纯仿真触觉, arXiv 2605.05241]。
- 延迟 / 抖动 / 相机参数随机化必做 [找不到出处,工程经验]。

---

## 六、评测体系

### 6.1 多层评测金字塔

- LIBERO+ / RoboCasa / RoboTwin 2.0 [Being-H0.5 LIBERO 98.9 / RoboCasa 53.9, arXiv 2601.12993;RoboTwin 2.0, arXiv 2506.18088]。
- SimplerEnv [simpler-env/SimplerEnv GitHub]。
- RoboArena 双盲两两(2026 全年开放) [robo-arena.github.io]。
- RobotArena ∞(real-to-sim + VLM 打分 + 人类偏好) [arXiv 2510.23571]。

### 6.2 评测维度

- in-distribution / 任务 / 环境 / 物体 / 本体 / 指令 / 长程 / 鲁棒性 / OOD 检测率 [找不到一篇综述全部覆盖;**鲁棒性 28 种扰动来自 STRONG-VLA, arXiv 2604.10055**;**OOD 检测来自 ReconVLA, arXiv 2604.16677**;其他维度是工程惯例]。
- **本体泛化**对应 zero-shot 跨本体核心,**OXE-AugE 报告未见 robot×gripper +24%~+45%** [arXiv 2512.13100]。

### 6.3 SimplerEnv MMRV 指标

- Mean Maximum Rank Violation [SimplerEnv 项目页 simpler-env.github.io, CoRL'24]。

---

## 七、部署与推理优化

### 7.1 延迟预算

- 真机闭环 50 Hz → 单次 < 20 ms [找不到出处,工程惯例]。
- π0.6 在 H100 63 ms/chunk(5 步 flow + 3 路相机) [PI06 model card]。
- Jetson 上 75% latency 来自 action-generation memory-bound [VLA edge bottleneck, arXiv 2603.02271]。

### 7.2 优化技术

- ⭐ **QuantVLA**:PTQ,70% 内存节省、1.22× 加速,success 不掉 [QuantVLA, arXiv 2602.20309v1]。
- ⭐ **Shallow-π** 蒸馏:18 → 6 层、>2× 加速、task drop < 1% [Shallow-π / VLA edge, arXiv 2601.20262]。
- KV cache 共享 + chunk reuse [找不到出处, π0/Helix 是惯例,见 openpi]。
- Flow 步数压缩 [找不到出处,LeRobot issue 中讨论,严格论文我未见]。
- TensorRT / vLLM / SGLang [找不到出处,通用部署工具]。
- `NVlabs/vla-perf` benchmark [GitHub NVlabs/vla-perf]。

### 7.3 边/云协同

- HY-Embodied-0.5 2B 边 + 32B 云 [arXiv 2604.07430]。
- 网络抖动时 S2 降级 [找不到出处,工程经验]。

---

## 八、鲁棒性 / 安全 / 失败恢复

### 8.1 OOD 检测

- ⭐ **ReconVLA**:Conformal prediction 给 calibrated 不确定性 [ReconVLA, arXiv 2604.16677]。
- 阈值触发后回退 / 求助人类 [找不到出处,工程惯例]。

### 8.2 自纠错与重试

- ⭐ **CycleVLA**:子任务回溯 + Minimum Bayes Risk [CycleVLA, arXiv 2601.02295]。
- 子任务转换点插 checkpoint [CycleVLA 显式做法, arXiv 2601.02295]。

### 8.3 多模态扰动鲁棒性

- ⭐ **STRONG-VLA**:28 种视觉/文本扰动课程,跨架构 +16.49% [STRONG-VLA, arXiv 2604.10055]。

### 8.4 极端感知条件

- **E-VLA**:event camera + RGB,黑暗 0% → 90% [E-VLA, arXiv 2604.04834]。

### 8.5 安全约束

- 力 / 力矩硬限位、几何 / 关节安全空间、E-stop watchdog [找不到出处,机器人安全工程惯例]。
- 指令安全过滤(VLM 安全分类器拦截危险指令) [找不到 2026 严格出处,通用 LLM safety 实践]。

### 8.6 数据偏置 / 失败模式溯源

- 训练数据保留失败示范 + 原因标签 [RoboMIND 5k 失败示范, arXiv 2412.13877]。
- 部署后失败 episode 全程回流到训练集 [SOP, arXiv 2601.03044;LWD, arXiv 2605.00416;Scanford, arXiv 2511.19647]。

---

## 九、工程基础设施

### 9.1 数据管道

- LeRobot Datasets v3.0 [huggingface/blog lerobot-release-v040.md;huggingface/lerobot v0.5.0]。
- WebDataset / MosaicML Streaming [找不到出处,通用 ML infra]。
- DDP-aware shuffling [找不到出处,通用]。
- Lazy 解码 + GPU 解码(NVIDIA DALI / torchcodec) [找不到出处,通用]。
- 数据版本管理 DVC / Pachyderm / lakeFS,实验绑数据 hash [找不到出处,通用 MLOps]。

### 9.2 训练框架

- FSDP / DeepSpeed ZeRO-3 [找不到出处,通用]。
- 3D 并行(TP+PP+DP)在 30B+ [找不到出处,通用]。
- bf16 + fp8 [找不到出处,通用]。
- Async checkpoint [找不到出处,通用]。
- 梯度 spike 自动 rollback [找不到出处,大模型训练惯例]。
- 持续训(streaming pretraining) [SOP/LWD 思路, arXiv 2601.03044 / 2605.00416]。

### 9.3 算力规模(数字参考)

- 复现 OpenVLA-7B ~26k GPU-hour [找不到 2026 严格出处,OpenVLA 论文中报数, 上轮已弃用为 SOTA]。
- 复现 π0 级别 ~100–200k GPU-hour [找不到出处,基于 PI 公开发布推测]。
- 复现 π0.6/0.7 ~500k GPU-hour [找不到出处,基于 PI06 model card 4.86B 规模推测]。
- "超大"≥ 1M GPU-hour [找不到出处,基于 HY-32B / GR00T-N2 规模估算]。

### 9.4 实验管理

- Trackio / W&B / Tensorboard [Trackio 是 HuggingFace ML 实验追踪,见 huggingface trackio 项目]。
- 实验三方绑定(代码+数据+超参 hash) [找不到出处,通用 MLOps]。

---

## 十、机器人硬件 / 遥操数据采集

### 10.1 硬件选型

- 至少 5 类本体覆盖 [找不到出处,工程经验,但与 Being-H0.5 30 本体 / X-VLA 多本体一致, arXiv 2601.12993 / arXiv 2510.10274]。
- 同型号 3–5 台同时采集 [找不到出处,工程经验]。

### 10.2 遥操方式

- VR + IK 遥操 [找不到 2026 严格出处,通用做法, AgiBot World Colosseo 提及, arXiv 2503.06669]。
- **力控遥操**捕捉接触动力学 [AgiBot World Colosseo 强调"force-controlled teleoperation", arXiv 2503.06669;agibot.com/research]。
- 中等任务 ~5 分钟/条、高难任务 ~20 分钟/条;1 万小时 ≈ 60 人/年 [找不到出处,基于 AgiBot/Tutor 公开报道反推, Forbes 2026-05-05]。
- 产业级 50–200 人遥操员 [Tutor Data Factory ~100 台 Sonny, Forbes 2026-05-05]。

### 10.3 数据采集 SOP

- 3 路以上 RGB + 触觉 + proprio + 力/力矩 + 语言指令 [AgiBot World 多模态规格,RGB(D)/触觉/LiDAR/IMU, agibot.com/research;arXiv 2503.06669]。
- 失败示范也要采(5–10%) [RoboMIND, arXiv 2412.13877]。
- 多采集者多样性(每任务 ≥ 5 人) [找不到 2026 严格出处,工程经验]。

### 10.4 [补漏] 数据飞轮(关键!)

- **机器人在野部署 → 自动采集 → 模型自我提升"飞轮"** [Robot-Powered Data Flywheels, arXiv 2511.19647;Scanford 项目页 scanford-robot.github.io]。
- **F-ACIL**:把数据分解为(物体 / 动作 / 环境)因子空间,实现组合泛化,5–10× 更少演示 +45% [F-ACIL, arXiv 2603.25583]。
- **数据飞轮要起效必须自有全栈**(模型 + 机器人 + 客户) [Tutor Intelligence, Forbes 2026-05-05]。

---

## 十一、团队、流程、路线规划

### 11.1 团队结构

- 数据采集 50–200 人、数据工程 5–10 人、算法 10–20 人、仿真 5–8 人、部署 3–5 人、控制 5–10 人、评测 3–5 人、安全合规 1–3 人 [找不到出处,基于 PI/NVIDIA/AgiBot 公开招聘信息和团队披露的估算,见 pi.website / agibot.com / Forbes 2026-05-05]。
- 最低 30–50 人,世界级 200+ [找不到出处,行业经验估算]。

### 11.2 路线规划(12–24 月)

- 各阶段时长 [找不到出处,基于 PI/AgiBot 公开发布节奏推测,π0(2024)→π0.5(2025.04)→π0.6(2025.11)→π0.7(2026.04), 见 pi.website]。

### 11.3 项目失败常见原因

- 6 条 postmortem 经验 [找不到出处,综合项目经验+社区讨论]。

---

## 十二、合规、伦理、法律

- 公开数据 license 限制 [找不到 2026 综述出处,各数据集发布说明]。
- ISO 10218 / ISO/TS 15066 协作机器人 [找不到 2026 严格出处,标准文档]。
- EU Machinery Regulation 2023/1230 [找不到出处,法规文档]。
- EU AI Act 高风险类目 [找不到出处,法规文档]。
- 中国《数据出境安全评估办法》 [找不到出处,法规文档]。

---

## 补漏总览(本轮新增的关键维度)

为方便快速 cross-check,把这次补漏的部分集中列一遍:

| 补漏要点 | 出处 |
|---|---|
| Scaling 不只是数据量,还要 quality alignment + action unification + RL refinement | [Green-VLA, arXiv 2602.00919] |
| Outcome-driven RL 突破监督上限 | [SimpleVLA-RL, openreview TQhSodCM4r] |
| 大规模数据飞轮(Scanford / Tutor Data Factory) | [arXiv 2511.19647;Forbes 2026-05-05] |
| F-ACIL 因子化数据分解 → 组合泛化,+45% / 5–10× 更少演示 | [arXiv 2603.25583] |
| 3D / 几何先验显式建模(PointVLA / Any3D-VLA / ST-VLA / StemVLA / Avi) | [arXiv 2503.07511;2602.00807;2603.13788;2602.23721;2510.21746] |
| 触觉表征(AnySkin / VISK / AnyTouch 2 / ToucHD 2.4M) | [arXiv 2409.08276;2410.17246;2602.09617] |
| Approximate Information States 形式化 sim2real | [arXiv 2604.15289] |
| Tutor Intelligence "数据飞轮必须自有全栈"论点 | [Forbes 2026-05-05] |

---

## 注意:仍标"找不到出处"的项目

这些是我基于工程实践给的总结性建议,目前没有找到 2026 论文/blog 严格背书,但属于行业惯例:

- 数据 schema 细节(相机外参对齐、joint+EEF 双存)
- DoReMi 风格学到的混合权重在 VLA 上的对照实验
- 学习率分层数值(VLM 1e-5 vs action 1e-3)
- Loss 权重 GradNorm/PCGrad
- Action Chunk Receding Horizon 50/10–20
- KV cache 共享、Flow 步数压缩具体数字
- 量化 / 蒸馏在生产部署的硬性数字(QuantVLA 给了 1.22×/70%)
- 推理网络抖动降级策略
- 安全 watchdog 工程做法
- 大部分通用 ML infra(FSDP、DALI、DVC、Pachyderm)
- 算力 GPU-hour 数字(基于公开模型规模反推)
- 团队人数估算
- 法规条款细节

如果你需要,我可以专门针对其中几条**到文献里再深挖一轮**,把"找不到出处"逐项打掉。

---

# 世界模型考虑点.涉及DM0.V1

**User**

我要用超多的数据训练一个超大的能用于机器人领域,具身智能领域和VLA领域的世界模型,, 请问我该怎么做?要考虑那些方面和细节点.  列一下有些什么相关的, SOTA的2026年的论文, 文章或研究成果可以借鉴. 可上网搜索.

---

**Cursor**

让我先看一下你工作区里的相关文件,理解一下你的项目背景,再做针对性的搜索和分析。

了解你的项目背景了——你是在做 DM0(Dexmal & StepFun, arXiv 2602.14974)这条"具身原生 VLA"路线。现在你问的是**专门训世界模型(World Model)**,这是另一条与 VLA 紧密相关但相对独立的技术线。让我先搜一下 2026 年这块的最新进展。

非常好,看了你工作区的 DM0 项目背景(arXiv 2602.14974, Feb 2026, "Embodied-Native VLA"),你之前问的是 VLA,现在专门聚焦"超大具身世界模型"——这是一条**与 VLA 紧密相关但目标不同**的路线。

下面我做一份针对**2026 年具身世界模型(Embodied World Model, EWM)**的深度分析,所有论点都会标 2026 出处。

---

## 一、先把"世界模型"在 2026 重新定义清楚

2026 年具身领域对"世界模型"已经分化出 **5 类**截然不同的形态,选错形态比训不动更危险。先要选对形态:

| 形态 | 输入 → 输出 | 用途 | 代表 2026 工作 |
|---|---|---|---|
| **像素级视频生成 WM** | (o, a, l) → 未来视频帧 | 数据生成、sim2real、policy eval | NVIDIA Cosmos Predict 2.5、Genie 3、ABot-PhysWorld |
| **Latent / token-space WM** | (o_t) → z_t → ẑ_{t+1} | imagination-based RL、规划 | Dreamer V4、CoLA-World、DIAL |
| **3D / 4D 几何 WM** | (o, a) → 4D 点云/NeRF/Gaussian | 物理一致性、几何精确控制 | Kinema4D、Phys4D、Neural Fields as WM、DeepVerse |
| **Action-Centric / WAM**(World–Action Model) | (o, l) → (a, o') 联合 | 直接做策略 + 想象 | GR00T N2 WAM、GigaWorld-Policy、STARRY、Cosmos Policy |
| **Discrete / Mask WM** | (o, a) → 语义 mask / 离散 token | 抗噪、加速、评测 | Mask World Model、dWorldEval |

> 一个判断窗口:**如果你的下游主要服务于 VLA**,优先 **WAM(形态 4)** 或 **Latent WM(形态 2)**;**如果是数据飞轮**,选 **像素 WM(形态 1)**;**如果做精细物理操作**,选 **4D WM(形态 3)** [GR00T N2, NVIDIA blog;World-VLA-Loop, arXiv 2602.06508;Kinema4D, arXiv 2603.16669]。

---

## 二、训练超大具身世界模型需要考虑的全部维度

下面 14 个维度,**每条都标 2026 出处**(找不到出处的标注"工程经验")。

### 1. 任务与目标定义(决定整个项目的上限)

- **明确世界模型的"用途"**:数据飞轮 / policy eval / 想象 RL / 规划 / sim2real,**4 种用途的训练数据/损失/评测都不一样** [WorldArena 揭示 14 个模型存在 "perception-functionality gap":高画质 ≠ 强具身能力, arXiv 2602.08971]。
- **决定**条件输入维度:语言、动作、proprio、相机轨迹、目标 bbox、点云,缺一种就限制一类用途 [Cosmos Predict 2.5 同时支持 Text2World/Image2World/Video2World, research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5;Action Images 把 7-DoF 动作渲染成"动作图像"作为像素条件, arXiv 2604.06168]。
- **决定**输出形态:RGB 像素 / latent token / 3D 点云 / 4D Gaussian / 语义 mask / 离散 keyframe [Mask World Model 用语义 mask 替代 RGB,LIBERO/RLBench 全面优于 RGB WM, arXiv 2604.19683;dWorldEval 用稀疏 keyframe + progress token, arXiv 2604.22152]。
- **决定**预测视野:秒级(1–5s)/ 分钟级 / 1000+ 帧 [Cosmos Predict 2.5 = 30 秒, research.nvidia.com;Infinite-World = 1000+ 帧, arXiv 2602.02393;HorizonDrive = 分钟级, arXiv 2605.11596]。

### 2. 数据(规模、组合、对齐)

- **数据量起步线**:Cosmos Predict 2.5 用了 **2 亿条高质量视频片段** [research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5];ABot-PhysWorld 用了 **3M 操作片段 + 物理标注** [arXiv 2603.23376];Kinema4D 用 **Robo4D-200k**(20 万 episodes,4D 标注) [arXiv 2603.16669]。
- **数据组合(2026 共识)**:必须是 **互联网视频 + 真机机器人数据 + 仿真合成 + 第一人称人类视频** 的混合,任何单一来源都训不出泛化 [Cosmos 平台说明覆盖"机器人/AV/工业" 多 domain, nvidia.com/cosmos;DM0 同样混合 OXE/RoboMind/Agibot/Galaxea + LIBERO + RoboTwin 2.0 + Habitat, dm0.md §1.1 + DM0 论文]。
- **物理标注**(关键差异点):2026 顶级 WM 不再只标语言,而是把"接触/物理事件"显式标注 [ABot-PhysWorld 用"physics-aware annotations"对抗 object penetration / anti-gravity 错误, arXiv 2603.23376]。
- **4D 标注**:URDF + 关节 + 末端 + 点云对齐 [Kinema4D 用 URDF 推 4D 控制轨迹再投射到点图, arXiv 2603.16669]。
- **多模态时序对齐**(RGB / depth / proprio / action / 触觉):必须做严格的时间戳同步 + 重采样到统一基频 [找不到 2026 单篇综述出处,工程惯例;DM0 用 keyframe sampling, dm0.md §1.3]。
- **长程数据**:对训"分钟级"WM,必须有 ≥ 60s 的连续轨迹数据 [HorizonDrive 显式收集分钟级驾驶序列, arXiv 2605.11596;Infinite-World 1000+ 帧训练 schedule, arXiv 2602.02393]。
- **数据过滤与配比**(同 VLA 路线):去重、模糊过滤、本体不相关样本剔除 [Naive 跨域 mixing 会负迁移, Rethinking VLA Scaling, arXiv 2602.09722;Cosmos Curator 是 NVIDIA 官方过滤工具, nvidia.com/cosmos]。

### 3. 模型架构(2026 主流四选一)

| 架构 | 优点 | 出处 |
|---|---|---|
| **DiT(Diffusion Transformer)像素生成** | 视觉质量最高 | [Cosmos Predict 2.5 = 2B/14B DiT, research.nvidia.com;ABot-PhysWorld 14B DiT, arXiv 2603.23376] |
| **Block-causal Transformer + tokenizer** | 实时交互、低延迟 | [Dreamer V4, emergentmind.com/papers/2509.24527;Genie 3 实时 720p 20-24 fps, deepmind.google/genie] |
| **Autoregressive 离散 token** | 与 VLA 共享 backbone 友好 | [DeepVerse 4D autoregressive, openreview pdf;dWorldEval discrete diffusion, arXiv 2604.22152] |
| **Hybrid(latent + pixel + 3D)** | 兼顾几何精确与视觉真实 | [MosaicMem 显式 3D patch lift + 隐式条件, arXiv 2603.17117;Phys4D 三阶段 hybrid, arXiv 2603.03485] |

- **Tokenizer**:Causal video tokenizer 是 2026 的事实标准 [Dreamer V4 用 masked-autoencoding causal tokenizer, emergentmind.com;Cosmos 有专用 video tokenizer, github.com/nvidia-cosmos/cosmos-predict2.5]。
- **条件注入**:cross-attention / control net / latent concat,**动作条件首选 cross-attn**,不是 prompt embed [Action Images 把动作渲染成像素图直接当输入, arXiv 2604.06168;Cosmos Policy 把动作和状态当 latent video frames,统一进 DiT, huggingface.co/blog/nvidia/cosmos-policy-for-robot-control]。
- **记忆 / 长程**:HPMC 分层 pose-free 记忆压缩器 [Infinite-World, arXiv 2602.02393];3D patch + 隐式 hybrid memory [MosaicMem, arXiv 2603.17117]。

### 4. 训练目标 / Loss 设计

- **Diffusion / Flow Matching loss**(主流) [Cosmos Predict 2.5, research.nvidia.com;ABot-PhysWorld DiT, arXiv 2603.23376]。
- **Shortcut Forcing**(Dreamer V4 创新):泛化 diffusion forcing,直接预测干净 latent,**100× 数据效率** [Dreamer V4, emergentmind.com/papers/2509.24527]。
- **Inverse Dynamics Reward**(物理可执行性对齐):用 IDM 当 reward 对齐生成视频 [EVA, arXiv 2603.17808]。
- **DPO + 解耦判别器**(物理一致性后训练) [ABot-PhysWorld 物理 DPO, arXiv 2603.23376]。
- **多视角 fidelity reward**(防止多步 rollout 坍塌) [Persistent Robot World Models RL post-training, arXiv 2603.25685]。
- **Self-Corrective Rollout(SRR)**:训练时强迫从"被预测污染过的历史"重建未来,极大降低 drift [HorizonDrive SRR,FID -52% / FVD -37%, arXiv 2605.11596]。
- **3D / 几何辅助 loss**(显式 depth / pointmap 监督) [DeepVerse 4D 几何约束, openreview;Phys4D 几何投射, arXiv 2603.03485]。
- **Mask / 语义级 loss**(去 RGB 噪声) [MWM, arXiv 2604.19683]。

### 5. WM 与 VLA 的联合训练(决定 WM 能不能"为下游服务")

- ⭐ **World-VLA-Loop**:闭环 — VLA 失败 rollout 反馈精炼 WM,WM 再优化 VLA,**真机 +36.7%** [arXiv 2602.06508]。
- ⭐ **VLAW**:迭代共改进,**+39.2% 绝对成功率,合成数据 +11.6%** [arXiv 2602.12063]。
- **DIAL**:把"意图"和"动作"在 latent WM 里解耦 [arXiv 2603.29844]。
- **GR00T N2 WAM**:World Action Model 直接预测物理状态,98% zero-shot 操作未见物体 [techbytes.app GR00T N2]。
- **GigaWorld-Policy**:**比 π0.5 在 RoboTwin 2.0 上 +95%、推理快 9×** [arXiv 2603.17240]。
- **STARRY**:WM 和动作联合 denoise,RoboTwin 双臂 93.82%,真机 42.5% → 70.8%(对比 π0.5) [arXiv 2604.26848]。
- **Cosmos Policy**:NVIDIA 把动作和状态当 latent video frames,统一 DiT 同时做 visuomotor 控制 / WM / value 规划 [huggingface.co/blog/nvidia/cosmos-policy-for-robot-control]。
- **CoLA-World**:Latent Action 和 WM **协同进化**,解决两阶段方法的表征坍塌 [arXiv 2510.26433]。

### 6. 物理一致性(2026 真正的瓶颈)

- 物理违例(物体穿模、反重力、违反守恒)是当前所有 SOTA WM 的最大缺陷 [ABot-PhysWorld 显式声明此问题, arXiv 2603.23376]。
- 解法 1:**物理对齐 DPO 后训练** [ABot-PhysWorld, arXiv 2603.23376]。
- 解法 2:**4D Gaussian / NeRF 几何先验** [Phys4D, arXiv 2603.03485;Neural Fields as WM, arXiv 2602.18690]。
- 解法 3:**仿真 ground RL** [Phys4D 第三阶段, arXiv 2603.03485]。
- 解法 4:**URDF + IK 显式注入** [Kinema4D, arXiv 2603.16669]。
- 解法 5:**逆动力学一致性约束** [EVA inverse dynamics reward, arXiv 2603.17808]。

### 7. 长程一致性 / 记忆 / Drift 抑制

- ⭐ **Hierarchical Pose-free Memory(HPMC)**:递归压缩历史 latent,**1000 帧仍一致** [Infinite-World, arXiv 2602.02393]。
- ⭐ **Self-Corrective Rollout(SRR)**:训练时显式让模型从坏历史恢复 [HorizonDrive, arXiv 2605.11596]。
- ⭐ **RL on rollouts**:用多视角 fidelity reward 后训练,**98% 偏好对比** [Persistent Robot WM, arXiv 2603.25685]。
- **Hybrid spatial memory**:3D patch lift + 隐式条件 [MosaicMem, arXiv 2603.17117]。
- **离散稀疏 keyframe**:dWorldEval 用稀疏关键帧 + progress token 做评测 [arXiv 2604.22152]。

### 8. 训练流程与课程

- **3 阶段训练**(Phys4D 范式):
  1. 伪监督预训(纯视频)
  2. 物理 SFT(仿真数据)
  3. 仿真 ground RL [Phys4D, arXiv 2603.03485]。
- **2 阶段 latent + RL**(Dreamer V4):先训 WM,再"在 imagination 内"训 policy,**完全离线就能拿钻石** [Dreamer V4, emergentmind.com]。
- **联合协同**(CoLA-World):LAM + WM 同时训,有 warm-up 防坍塌 [arXiv 2510.26433]。
- **Cosmos Predict → Cosmos Policy 后训练**:先训通用世界模型,再把"动作和状态当 latent video frames"做 robot 后训练 [huggingface.co/blog/nvidia/cosmos-policy-for-robot-control]。

### 9. 参数规模与算力

- **Cosmos Predict 2.5**:2B / 14B 双尺寸 [research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5]。
- **ABot-PhysWorld**:14B DiT [arXiv 2603.23376]。
- **Dreamer V4**:**110M 在 4090 (8GB) 上实时跑** [github.com/machines-in-motion/dreamer-v4]。
- **Genie 3**:具体规模未披露,**720p @ 20–24 fps 实时** [deepmind.google/blog/genie-3-a-new-frontier-for-world-models]。
- **算力起步**:200M 视频片段训 14B DiT 至少 **2k–8k H100 × 周级** [找不到出处,基于 Cosmos 2.5 公开数据点反推]。

### 10. 评测体系(2026 必须用 WorldArena)

- ⭐ **WorldArena**(CVPR 2026 Challenge,清华 fib-lab):16 个指标 / 6 子维度 / **EWMScore** 单一指标;3 类评测:像素质量、具身功能、人评 [arXiv 2602.08971;world-arena.ai;cvpr2026challenge.world-arena.ai;github tsinghua-fib-lab/WorldArena]。
- 关键发现:**"perception-functionality gap"——画质好 ≠ 具身有用** [arXiv 2602.08971]。
- **dWorldEval**:用 WM 自身做 policy 大规模评测,跨千级环境/任务 [arXiv 2604.22152]。
- **物理 metric**:object penetration rate / anti-gravity rate / 守恒违反率 [ABot-PhysWorld 提议, arXiv 2603.23376]。
- **下游 utility metric**:把 WM 当 sim2real / RL 想象环境后,policy 真实成功率涨多少 [VLAW +39.2%, arXiv 2602.12063;World-VLA-Loop +36.7%, arXiv 2602.06508]。

### 11. 部署与推理

- **实时性硬指标**:Genie 3 = 720p / 20–24 fps [deepmind.google/genie];Dreamer V4 = 4090 上实时 [github.com/machines-in-motion/dreamer-v4]。
- **数据飞轮服务态**(Cosmos Curator + Dataset Search + Evaluator)是 2026 的标准范式 [nvidia.com/cosmos]。
- **像素 WM 用作 RL imagination**:必须能 ≥ 100Hz 出虚拟帧才有 RL 性价比 [Dreamer V4, emergentmind.com]。
- **WM 当 policy evaluator**:dWorldEval 显式服务此场景 [arXiv 2604.22152]。
- **Sim2Real 服务**(像素 WM 的最重要应用):Cosmos Transfer 2.5 配合 CARLA / Isaac Sim 把 sim 转 photoreal [nvidia.com/cosmos]。

### 12. 与你的 DM0 项目对接(关键)

DM0 论文里提供的"Embodied Spatial Scaffolding"已经在做**轻量级世界模型雏形**:
- "End-effector trajectory prediction"≈ 1D 轨迹 WM [DM0 论文 §2.3, dm0.md §1.3]
- "Goal bounding box prediction"≈ 2D 空间 WM [DM0 论文 §2.3]
- "Subtask prediction"≈ 语义层 WM [DM0 论文 §2.3]
- 全部通过 L_AR 自回归文本预测实现 [embd_pt_sota_princpl.md L140-145]

如果要把 DM0 升级成"含真实 WM 的 VLA",**最自然的路径是按 Cosmos Policy 风格** — 把 action / state 当作 latent video frames 跟视频帧统一进 DiT [huggingface.co/blog/nvidia/cosmos-policy-for-robot-control];或按 World-VLA-Loop 做闭环 RL [arXiv 2602.06508]。

### 13. 工程基础设施

- 数据 pipeline / streaming(同 VLA 路线)[找不到出处,工程经验]。
- **视频解码 GPU 加速**(NVIDIA DALI / torchcodec)对 200M 视频片段是必需 [找不到出处,工程惯例]。
- DiT 训练特殊问题:**显存爆炸**(高清长视频),要用 stage-wise + sequence parallel [Cosmos 训练要点, github.com/nvidia-cosmos/cosmos-predict2.5]。
- Async checkpoint / loss spike rollback(同大模型训练) [找不到出处,工程惯例]。

### 14. 安全 / 合规

- **生成式 WM 风险**:被用来伪造视频、deepfake [找不到出处,通用 AI safety 议题]。
- **Cosmos 用 NVIDIA Open Model License** [research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5]。
- **Genie 3 仅限 18+ 美国 Ultra 订阅用户**(说明 Google 评估了滥用风险) [deepmind.google blog;wikipedia.org/wiki/Genie_3]。
- 生成视频水印 / provenance(C2PA) [找不到出处,行业实践]。

---

## 三、2026 必读 SOTA 清单(纯 World Model,不重复 VLA)

> 全部为 **2025 Q4 – 2026 Q2** 的工作,标注首发月份/单位/链接关键词。

### A. 通用具身世界基础模型(Generalist EWM)

| 模型 | 时间 | 单位 | 关键贡献 | 出处 |
|---|---|---|---|---|
| ⭐ **NVIDIA Cosmos Predict 2.5** | 2026 | NVIDIA | 2B/14B DiT,200M video clips,统一 Text/Image/Video → World,30 秒预测 | research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5 |
| ⭐ **NVIDIA Cosmos Transfer 2.5** | 2026 | NVIDIA | sim2real,配 CARLA / Isaac Sim | nvidia.com/cosmos |
| ⭐ **NVIDIA Cosmos Reason** | 2026 | NVIDIA | physics-aware VLM,可与 WM 对接 | nvidia.com/cosmos |
| ⭐ **NVIDIA Cosmos Policy** | 2026 | NVIDIA | 把 action/state 当 latent video frames 进 DiT,统一 visuomotor + WM + value | huggingface.co/blog/nvidia/cosmos-policy-for-robot-control |
| ⭐ **Google Genie 3** | 2025.08 announce / 2026.01 公开 | DeepMind | 720p 20–24 fps 实时交互式 3D 世界 | deepmind.google/blog/genie-3-a-new-frontier-for-world-models |
| **Waymo World Model**(Genie 3 衍生) | 2026.02 | Waymo | 自动驾驶 edge case 仿真 | wikipedia.org/wiki/Genie_3 |
| ⭐ **Dreamer V4** | 2025.09 | DeepMind | block-causal Transformer,Shortcut Forcing,**Minecraft 钻石纯 offline,4090 上实时** | emergentmind.com/papers/2509.24527;github.com/machines-in-motion/dreamer-v4 |

### B. 机器人专用 / VLA 集成 WM

| 模型 | 出处 | 亮点 |
|---|---|---|
| ⭐ **GR00T N2 (WAM)** | techbytes.app GR00T N2 | World Action Model,Jetson Thor,98% zero-shot |
| ⭐ **GigaWorld-Policy** | arXiv 2603.17240 | RoboTwin 2.0 比 π0.5 +95%,推理快 9× |
| ⭐ **STARRY** | arXiv 2604.26848 | RoboTwin 双臂 93.82%,真机 42.5→70.8% vs π0.5 |
| ⭐ **World-VLA-Loop** | arXiv 2602.06508 | VLA 与 WM 闭环 RL,真机 +36.7% |
| ⭐ **VLAW** | arXiv 2602.12063 | 迭代共改进,+39.2% 绝对 |
| **DIAL** | arXiv 2603.29844 | 意图 / 动作解耦的 latent WM |
| **HiPolicy** | arXiv 2604.06067 | 多频率分层 |
| **CoLA-World** | arXiv 2510.26433 | LAM + WM 协同进化,防坍塌 |
| **OmniGuide** | arXiv 2604.20100 | 推理时 WM 引导 |
| **DriveWorld-VLA** | arXiv 2602.06521 | 自动驾驶 |

### C. Action-Conditioned 视频生成(把 WM 当 policy)

| 模型 | 出处 | 亮点 |
|---|---|---|
| **Action Images** | arXiv 2604.06168 | 把 7-DoF 动作渲成像素图,视频骨干即 zero-shot policy |
| **EVA** | arXiv 2603.17808 | inverse dynamics reward 对齐物理可执行性 |
| **PhysGen** | arXiv 2603.00110 | 预训练视频模型当物理仿真器 |
| **World Guidance(WoG)** | arXiv 2602.22010 | 用压缩条件空间建模未来 |

### D. 物理 / 4D 一致性

| 模型 | 出处 | 亮点 |
|---|---|---|
| ⭐ **ABot-PhysWorld** | arXiv 2603.23376 | 14B DiT + 物理 DPO + 解耦判别器,3M 物理标注片段 |
| ⭐ **Kinema4D** | arXiv 2603.16669 | URDF → 4D 控制,Robo4D-200k 数据 |
| ⭐ **Phys4D** | arXiv 2603.03485 | 3 阶段(伪监督 + 物理 SFT + 仿真 RL) |
| **Neural Fields as World Models** | arXiv 2602.18690 | NeRF / 神经场作为 WM |
| **DeepVerse** | openreview pdf 92cdb70... | 4D autoregressive,几何约束 |
| **Mask World Model** | arXiv 2604.19683 | 语义 mask 替 RGB,LIBERO/RLBench 全胜 |
| **Causal World Modeling** | arXiv 2601.21998 | 因果建模 |

### E. 长程一致性 / 记忆

| 模型 | 出处 | 亮点 |
|---|---|---|
| ⭐ **Infinite-World** | arXiv 2602.02393 | HPMC 分层 pose-free 记忆,**1000+ 帧** |
| ⭐ **HorizonDrive** | arXiv 2605.11596 | SRR 自纠正,FID -52% / FVD -37%,分钟级 |
| ⭐ **Persistent Robot WM** | arXiv 2603.25685 | RL 后训练 + 多视角 fidelity reward,98% 偏好 |
| **MosaicMem** | arXiv 2603.17117 | 3D patch + 隐式 hybrid 记忆 |

### F. 评测与下游 utility

| 资源 | 出处 |
|---|---|
| ⭐ **WorldArena**(CVPR 2026 Challenge) | arXiv 2602.08971;world-arena.ai;github tsinghua-fib-lab/WorldArena |
| ⭐ **dWorldEval** | arXiv 2604.22152 |
| **CVPR 2026 Challenge** Track 1 / Track 2 | cvpr2026challenge.world-arena.ai |

### G. 开源代码库 / 平台

| 资源 | 出处 |
|---|---|
| ⭐ **NVIDIA Cosmos**(代码 + 模型) | github.com/nvidia-cosmos/cosmos-predict2.5;nvidia.com/cosmos |
| ⭐ **WorldArena** | github.com/tsinghua-fib-lab/WorldArena |
| **Dreamer V4 PyTorch 实现** | github.com/machines-in-motion/dreamer-v4 |

---

## 四、推荐训练路线(基于以上 SOTA)

```text
Stage 0  选定形态 + 用途
        ├─ 数据飞轮服务?      → Cosmos Predict 2.5 风格(像素 DiT)
        ├─ Imagination RL?    → Dreamer V4 风格(latent)
        ├─ 物理精细操作?      → Phys4D / Kinema4D 风格(4D)
        └─ 与 VLA 一起训?     → Cosmos Policy / World-VLA-Loop / GigaWorld 风格

Stage 1  数据建设
        ├─ 互联网视频(去重 + 美学过滤)10⁸+ 片段
        ├─ 真机机器人(OXE / AgiBot World 2026 / 自采)10⁴ h+
        ├─ 仿真合成(RoboTwin 2.0 / Isaac Lab / GR00T-Sim)10⁵+ 轨迹
        ├─ 第一人称人类(Ego4D / EgoExo4D / UniHand-2.0)
        └─ 物理标注 + 4D 标注(Robo4D-200k 风格)
        关键工具: Cosmos Curator + Dataset Search + Evaluator

Stage 2  WM 主体训练
        ├─ Causal video tokenizer 训练
        ├─ DiT / Block-causal Transformer 大规模预训(Diffusion / Shortcut Forcing)
        ├─ 长程记忆模块(HPMC / MosaicMem)
        └─ 物理 SFT(Phys4D 第二阶段)

Stage 3  对齐 / 后训练(2026 关键差异点)
        ├─ Inverse Dynamics 对齐(EVA)
        ├─ 物理 DPO + 解耦判别器(ABot-PhysWorld)
        ├─ Self-Corrective Rollout(HorizonDrive SRR)
        ├─ 多视角 fidelity RL(Persistent Robot WM)
        └─ 仿真 ground RL(Phys4D 第三阶段)

Stage 4  与 VLA 联合训练 / 闭环
        ├─ Cosmos Policy 风格(action 当 latent video frame)
        ├─ World-VLA-Loop / VLAW 风格(失败回流共改进)
        └─ STARRY 风格(action 与未来 latent 联合 denoise)

Stage 5  评测 + 部署
        ├─ WorldArena(CVPR 2026 Challenge 同协议)
        ├─ dWorldEval 跨千任务自动评测
        ├─ 真机 utility(把 WM 当 sim2real,policy 真涨多少)
        └─ Cosmos Curator/Evaluator 数据飞轮服务态
```

---

## 五、与 DM0 路线对接的具体建议

如果你打算把这个 WM 跟 DM0 做整合,有 **3 种不同强度**的方案:

| 方案 | 改动量 | 路径 | 出处 |
|---|---|---|---|
| **轻量** | 在 DM0 的 `prefix_out` 后多加一个"未来视频帧 latent 预测"head,通过 L_AR 监督(但输出是 video latent token) | 与 DM0 当前的 Embodied Spatial Scaffolding 一致 [embd_pt_sota_princpl.md] |
| **中等** | 把 Cosmos Policy 的"action/state as latent video frames"思路套进 DM0 的 merged attention,共享 DiT | [huggingface.co/blog/nvidia/cosmos-policy-for-robot-control] |
| **重度** | 训独立 14B 级 WM,做 VLAW / World-VLA-Loop 闭环,DM0 当 VLA、WM 当训练数据/评测引擎 | [arXiv 2602.12063;arXiv 2602.06508] |

> 个人建议:**先做"中等"** —— Cosmos Policy 思路与 DM0 的"text-as-everything + merged attention"哲学最契合,且 NVIDIA 已开源参考实现,验证成本低。

---

## 六、容易踩的坑(2026 最新 postmortem 总结)

1. **画质好不等于有用** [WorldArena 实测 14 个模型存在 perception-functionality gap, arXiv 2602.08971]。
2. **不做物理对齐必然产生穿模 / 反重力** [ABot-PhysWorld 显式声明, arXiv 2603.23376]。
3. **多步 rollout 必坍塌**,除非显式做 SRR 或多视角 RL [HorizonDrive arXiv 2605.11596;Persistent Robot WM arXiv 2603.25685]。
4. **跨域 naive mixing 仍是 negative transfer** [Rethinking VLA Scaling, arXiv 2602.09722]。
5. **不做"下游 utility"评测,WM 只能当 demo,不能当生产力** [dWorldEval 是首批提出 policy 级评测的工作, arXiv 2604.22152]。
6. **训练-推理一致性容易丢**(尤其是动作条件投射方式) [找不到 2026 严格出处,工程经验]。

---

需要我接下来做哪些?

- 把上面这些 SOTA **按"DM0 兼容性"打分**,给出"哪些可以直接借鉴/借哪些模块"的对比表;
- 或者基于你现有算力 / 数据预算,给一份**专属的 6 阶段训练配方**(数据混合权重、超参、并行策略、评测节点);
- 或者**精读** Cosmos Policy / World-VLA-Loop / STARRY 这几篇里你最关心的一篇,做带页码的笔记。

告诉我你倾向哪个方向。

---

# 世界模型考虑点.忽略DM0和历史.V2

**User**

我要用超多的数据训练一个超大的能用于机器人领域,具身智能领域和VLA领域的世界模型. 列一下有些什么相关的, SOTA的2026年的论文, 文章或研究成果可以借鉴. 可上网搜索, 要把 DreamZero 及其相关文章也纳入搜索与分析范围. 然后回答一下如果我要做这种世界模型该怎么做? 要考虑那些方面和细节点?  回答的每个观点后面都加个括号, 后面注明该观点的出处, 出处可以是多个. 注意, 不要受DM0影响! 不要受任何前面问答的影响! 搜索与考虑的面和范围要更广些, 分析更深入更实用!

---

**Cursor**

好的,我重新独立做一份调研。先搜 DreamZero 及其周边,再扩展到世界模型的更多 2026 维度。

下面是一份**完全独立、不受先前对话和 DM0 影响**的 2026 具身世界模型训练深度指南。所有论点都标 2026 出处。

---

## 一、2026 思维范式的根本转变(决定能否成功的 8 条铁律)

1. **VLA 路线在"未见动作 / 未见技能"上已经触顶**;World Action Model(WAM)在真机泛化上比 SOTA VLA 高 **2× 以上**(seen task progress 62.2% vs 27.4%) [DreamZero, arXiv 2602.15922]。
2. **改善机器人能力 = 改善视频生成质量**——DreamZero 的核心结论是"policy 失败几乎全部源自视频预测错误,而不是动作预测错误" [DreamZero §5.1, arXiv 2602.15922]。
3. **训练数据要追求"多样、非重复"而不是"密集重复演示"**——这与传统 VLA 数据哲学正相反 [DreamZero §4.1 / Q1 实验, arXiv 2602.15922;A Comprehensive Survey on World Models for Embodied AI, arXiv 2510.16732]。
4. **像素重建 ≠ 决策有用**——WorldArena 在 14 个模型上实测 "perception-functionality gap":高画质模型在具身任务上反而可能输 [WorldArena, arXiv 2602.08971]。**重建型 VAE latent 在策略下游甚至输给 V-JEPA 2.1 / Web-DINO / SigLIP 2 这种语义 latent** [Reconstruction or Semantics?, arXiv 2605.06388]。
5. **跨本体迁移可以"只用视频、无需动作标签"**——DreamZero 用 12 分钟人类视频或 20 分钟 YAM 视频就能 +42% [DreamZero §5.1 Q4, arXiv 2602.15922]。
6. **新本体 30 分钟 play data 即可适配**——DreamZero 在 AgiBot 预训后,只用 30 分钟 YAM play data 就能保持 zero-shot 泛化 [DreamZero §5.1 Q5, arXiv 2602.15922]。
7. **WM 已成为"数据引擎"而非单纯仿真器**——GigaWorld-0、Nimbus、InternDataEngine、Genie Sim 3.0 都把 WM 当作生成大规模训练数据的工具 [GigaWorld-0, arXiv 2511.19861;Nimbus, arXiv 2601.21449;InternDataEngine GitHub InternRobotics/InternDataEngine;Genie Sim 3.0, arXiv 2601.02078]。
8. **WM 可以用 RL 后训练显著提涨下游 VLA 真机表现**——WoVR 把 LIBERO 从 39.95% → 69.2%(+29.3 pt),真机 61.7% → 91.7%(+30.0 pt) [WoVR, arXiv 2602.13977]。

---

## 二、世界模型的"形态选择"(选错形态比训不动更危险)

2026 已经分化出 **6 大形态**,每种用途不同:

| 形态 | 输入 → 输出 | 主要用途 | 代表 2026 工作 |
|---|---|---|---|
| **A. WAM(World Action Model,联合视频+动作)** | (o, l, q) → (video, action) 联合生成 | 直接当 zero-shot 策略 | DreamZero [arXiv 2602.15922]、GigaWorld-Policy [arXiv 2603.17240]、STARRY [arXiv 2604.26848]、Cosmos Policy [huggingface.co/blog/nvidia/cosmos-policy-for-robot-control] |
| **B. 像素级视频生成 WM(Action-conditioned video DiT)** | (o, a, l) → 未来视频帧 | 数据引擎、sim2real、policy eval | Cosmos Predict 2.5 [research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5]、Genie 3 [deepmind.google/blog/genie-3]、Wan2.2 [github Wan-Video/Wan2.2]、ABot-PhysWorld [arXiv 2603.23376] |
| **C. Latent / 语义 WM** | o_t → z_t → ẑ_{t+1} | Imagination RL、规划、特征学习 | V-JEPA 2 / 2.1 [ai.meta.com/vjepa;arXiv 2603.14482]、Dreamer V4 [emergentmind.com/papers/2509.24527]、VideoWorld 2 [arXiv 2602.10102]、DIAL [arXiv 2603.29844] |
| **D. 3D / 4D / 高斯泼溅 WM** | (o, a) → 4D 点云/Gaussian/NeRF | 物理一致性、几何精确控制 | GWM (Gaussian World Model) [arXiv 2508.17600]、Phys4D [arXiv 2603.03485]、Kinema4D [arXiv 2603.16669]、Physically Embodied Gaussian Splatting [arXiv 2406.10788] |
| **E. Discrete / Mask WM** | (o, a) → 语义 mask / 离散 token | 抗噪、加速、policy eval | Mask World Model [arXiv 2604.19683]、dWorldEval [arXiv 2604.22152] |
| **F. Reflective / Closed-loop WM** | (o, l, action plan) → think-act-reflect | 长程任务规划 | SPIRAL [arXiv 2603.08403]、World-VLA-Loop [arXiv 2602.06508] |

> **2026 共识级建议**:如果你目标是"用于机器人/VLA 领域的通用世界模型",**形态 A(WAM)是当下确定性最强的方向**——既能做 policy 又能做仿真,且在 DreamZero / GigaWorld-Policy / STARRY 上都已经超过纯 VLA SOTA [DreamZero, arXiv 2602.15922;GigaWorld-Policy, arXiv 2603.17240;STARRY 真机 42.5→70.8% vs π0.5, arXiv 2604.26848]。

---

## 三、训练超大具身世界模型必须考虑的全部维度

每条都加 2026 出处。共 **20 大维度**。

### 1. 任务 / 用途定义(决定整个项目)

- **明确世界模型的"用途分类"**:Decision-Coupled vs General-Purpose [Survey, arXiv 2510.16732 三轴分类法]。
- **明确"功能场景"**:policy learning / planning / 仿真 / 评估 / 数据生成 [World Model for Robot Learning Survey, arXiv 2605.00080]。
- **明确"输出形态 + 视野"**:Cosmos Predict 2.5 = 30 秒预测 [research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5];Genie 3 = 实时 720p [deepmind.google/blog/genie-3];Infinite-World = 1000+ 帧 [arXiv 2602.02393];HorizonDrive = 分钟级 [arXiv 2605.11596];Context Forcing 把上下文做到 20 秒 [arXiv 2602.06028]。
- **决定**条件输入维度:语言、动作、proprio、相机轨迹、历史多视角观测 [DreamZero 用 (visual context, language, proprio), arXiv 2602.15922 §3.1;Wan2.1-I2V-14B 用 image + text, deepwiki.com/Wan-Video/Wan2.1]。

### 2. 数据规模与组合(2026 起步线极高)

- **互联网视频 + 真机 + 仿真 + 第一人称人类视频** 四源混合是 2026 默认 [V-JEPA 2 用 1M+ hours 互联网视频, ai.meta.com/research/publications/v-jepa-2;Cosmos Predict 2.5 用 200M video clips, research.nvidia.com;GigaWorld-0 同时融视频+3D, arXiv 2511.19861]。
- **真机数据数量级**:DreamZero 仅用 **500 hours teleoperation(AgiBot G1)+ DROID** 就达到 SOTA [DreamZero §4.1, arXiv 2602.15922];远小于 π0.5 / GR00T N1.6 的"几千小时" [DreamZero baselines, arXiv 2602.15922]。
- **关键数据采集哲学**:**长 episode、多子任务、多技能、不重复**——DreamZero 每条 episode 平均 4.4 分钟、42 个子任务,跨 22 个真实环境(家、餐厅、超市、咖啡店、办公室) [DreamZero §4.1 Figure 6, arXiv 2602.15922]。
- **跨本体只用视频(无动作)足够**:用其它机器人或人类视频做 video-only co-train,1:1 mix 10K steps 即可 +42% [DreamZero §5.1 Q4, arXiv 2602.15922]。
- **物理标注**:ABot-PhysWorld 用 3M 条带"物理事件标注"的操作片段 [arXiv 2603.23376];Kinema4D 用 Robo4D-200k(20 万 episodes、4D 标注、URDF 推 4D 控制轨迹) [arXiv 2603.16669]。
- **Action filtering 必做**:DreamZero 显式过滤 idle actions [DreamZero §4.1, arXiv 2602.15922]。

### 3. 视频 Tokenizer / Latent 选择(常被低估)

- **重建型 VAE 训练上更稳但下游策略上输给语义 latent**——V-JEPA 2.1 / Web-DINO / SigLIP 2 在 planning / policy / representation quality 全面优于 VAE [Reconstruction or Semantics?, arXiv 2605.06388]。
- **Wan-VAE(3D 时空 VAE)** 是当前像素 WM 的事实工业标准 [Wan-Video/Wan2.1, deepwiki.com/Wan-Video/Wan2.1/9.2-model-architecture-details;Wan2.2 GitHub, github Wan-Video/Wan2.2]。
- **Causal video tokenizer + masked autoencoding** 是 latent WM 主流(Dreamer V4) [Dreamer V4, emergentmind.com/papers/2509.24527]。
- **Block-causal attention** 让 video / action token 在时间块内对齐而跨时间自回归 [LeWAM, GitHub ErykHalicki/LeWAM]。

### 4. 模型架构(2026 主流四选一)

| 架构 | 出处 |
|---|---|
| **DiT(Diffusion Transformer)+ Image-to-Video** —— 视觉质量最高 | [Cosmos Predict 2.5 = 2B/14B DiT, research.nvidia.com;DreamZero 14B Wan2.1-I2V backbone, arXiv 2602.15922;Wan2.1 = ~40-50 块 DiT, hidden 2048-4096+, 16-32 heads, deepwiki Wan2.1] |
| **Autoregressive DiT + flow matching**(2026 新主流) | [DreamZero AR + flow matching, arXiv 2602.15922 §3.1] |
| **Block-causal Transformer + causal tokenizer** —— 实时交互 | [Dreamer V4, emergentmind.com;Genie 3 实时 20-24 fps, deepmind.google/genie] |
| **MoE / MoT video DiT** —— 容量大、推理省 | [Wan2.2 MoE, github Wan-Video/Wan2.2;LingBot-VA Mixture-of-Transformers, 同 Causal World Modeling, arXiv 2601.21998] |

- **AR vs Bidirectional 决策**:DreamZero 实测 AR 给"更平滑机器人动作 + 更高视频-动作对齐",且 KV cache 让长上下文可行 [DreamZero §3.1, arXiv 2602.15922]。
- **AR 视频 + 非 AR 动作**:DreamZero 显式让动作不做自回归(避免 closed-loop 错误传播),只让视频做 AR [DreamZero §3.1, arXiv 2602.15922]。
- **多视角拼接 → 单帧**:不改 backbone,把所有视角 concat 进单帧,**最小侵入** [DreamZero §3.1, arXiv 2602.15922]。
- **新增组件最少化**:DreamZero 只额外加 state encoder / action encoder / action decoder,backbone 全部冻结的部分仅 text encoder + image encoder + VAE [DreamZero §4.1 Training, arXiv 2602.15922]。
- **MoE 与多尺寸**:Wan2.2 提供 5B 高效版(720p@24fps,24GB GPU 跑得动)+ 14B 大版本 [localaimaster.com/blog/wan-video-generation-guide]。

### 5. 动作条件注入方式(决定下游可控性)

- **Cross-attention adapter + zero-init 输出** 是把现成视频模型改成 WM 最稳的方式 [ActionConditioningAdapter on CogVideoX-5B, GitHub A-SHOJAEI/action-conditioned-video-worldmodel;LoRA rank 16 配套使用]。
- **联合 denoise (video latent + action latent) 共享 timestep** 训练初期收敛更快 [DreamZero §3.1 Training Objective, arXiv 2602.15922]。
- **解耦 noise schedule(Flash):video bias 高噪 + action 均匀** 让"少步推理"训-推一致 [DreamZero-Flash:Beta(7,1), 16→1 步,arXiv 2602.15922 §3.2.5]。
- **Multi-Agent Condition Module**(多体协作):Agent Identity Embedding + Adaptive Action Weighting [MultiWorld, arXiv 2604.18564]。
- **Action Cross-Attention block 内置每层** 而非只 prepend [Wan-style cross-attention 每层都有, deepwiki Wan2.1]。

### 6. 训练目标 / Loss(2026 关键差异点)

- **Flow Matching loss(主流)**:线性插值 + 速度预测 [DreamZero §3.1, arXiv 2602.15922 公式 (2)(3)];Cosmos Predict 2.5 同样 flow matching [research.nvidia.com]。
- **联合 velocity 预测**:同一个 head 同时预测 video 和 action 的 velocity [DreamZero L(θ), arXiv 2602.15922 公式 (3)]。
- **Teacher Forcing chunk-wise**:用干净的"前一 chunk"条件 noisy 当前 chunk [DreamZero §3.1, arXiv 2602.15922]。
- **Diffusion Forcing / Self-Forcing 的局限**:无法建模真实 rollout 误差累积 [LIVE 综述, huggingface.co/papers/2602.03747]。
- **Cycle-Consistency loss**(2026 新方向):前向 rollout 后反向重建初始状态,loss 在终态上,**消除 teacher 蒸馏需求** [LIVE, huggingface.co/papers/2602.03747]。
- **Causal Forcing**(2026):用 AR teacher 做 ODE 初始化,弥合 bidirectional → causal 鸿沟 [Causal Forcing, arXiv 2602.02214]。
- **Context Forcing**:用 long-context teacher 训 long-context student,**有效上下文 20 秒,2-10× 长于已有方法** [Context Forcing, arXiv 2602.06028]。
- **Inverse Dynamics Reward**:用 IDM 当 reward 对齐生成视频和"可执行动作" [EVA, arXiv 2603.17808]。
- **Physics DPO + 解耦判别器**:专门训"反 object penetration / 反重力" [ABot-PhysWorld, arXiv 2603.23376]。
- **Multi-view fidelity reward**(防多步坍塌):Persistent Robot WM 用 RL 后训练 + 多视角真实度 reward,98% 偏好 [arXiv 2603.25685]。
- **Mask / 语义 loss 替代 RGB loss**:LIBERO/RLBench 全胜 [Mask World Model, arXiv 2604.19683]。

### 7. 长程一致性 / 记忆机制(WM 真正的瓶颈)

- ⭐ **Hierarchical Pose-free Memory Compressor (HPMC)**:递归压缩历史 latent,1000+ 帧仍一致 [Infinite-World, arXiv 2602.02393]。
- ⭐ **Self-Corrective Rollout (SRR)**:训练时强迫模型从"被预测污染"的历史里恢复未来,**FID -52% / FVD -37%** [HorizonDrive, arXiv 2605.11596]。
- ⭐ **Hybrid spatial memory**:3D patch lift + 隐式条件混合,minute-level 导航与 AR rollout [MosaicMem, arXiv 2603.17117]。
- ⭐ **Sliding Window Re-encoding**:稳定 AR 推理 [RoboAlign-R1, arXiv 2605.03821]。
- ⭐ **KV cache 替换 GT 帧**:闭环时用真实观测替换预测帧进 KV cache,**消除 AR 视频生成固有的 compounding error** [DreamZero §3.1 Inference, arXiv 2602.15922]。

### 8. 物理一致性(2026 难点 #1)

- 主流 WM 都会出现 object penetration、anti-gravity、违反守恒 [ABot-PhysWorld 显式定义, arXiv 2603.23376]。
- 解法 1:**物理对齐 DPO 后训练 + 解耦判别器** [ABot-PhysWorld, arXiv 2603.23376]。
- 解法 2:**4D Gaussian / NeRF 几何先验** [GWM, arXiv 2508.17600;Phys4D, arXiv 2603.03485;Physically Embodied Gaussian Splatting 30Hz, arXiv 2406.10788]。
- 解法 3:**仿真 ground RL** [Phys4D 第三阶段, arXiv 2603.03485]。
- 解法 4:**URDF + IK 显式注入** [Kinema4D 把 URDF 推 4D 控制轨迹再投到 pointmap, arXiv 2603.16669]。
- 解法 5:**逆动力学一致性约束** [EVA, arXiv 2603.17808]。

### 9. 实时性 / 推理优化(决定能否上车)

DreamZero 提供 **2026 最完整的工程范本**——从 5.7s/chunk 优化到 150ms/chunk,38× 加速 [DreamZero Table 1, arXiv 2602.15922]:

- **CFG Parallelism**:把条件/无条件两次 forward 拆到 2 GPU,−47% per-step latency [arXiv 2602.15922]。
- **DiT Caching**:利用 flow velocity 的方向一致性,cosine similarity 阈值复用 cache,16 → 4 步 [arXiv 2602.15922]。
- **torch.compile + CUDA Graphs**:消除 CPU overhead [arXiv 2602.15922]。
- **NVFP4 量化**(Blackwell):权重 + 激活,QKV/Softmax 保 FP8,非线性保 FP16 [arXiv 2602.15922]。
- **cuDNN attention backend** + scheduler 移到 GPU 消 CPU-GPU 同步 [arXiv 2602.15922]。
- **DreamZero-Flash**:解耦 noise schedule,16→1 步,350ms→150ms [arXiv 2602.15922]。
- **Action Chunk Smoothing**:2× 上采样 + Savitzky-Golay 滤波 + 下采样 [arXiv 2602.15922]。
- **Asynchronous Closed-Loop**:推理与执行并行,只需 200ms 即可保 30Hz 闭环 [DreamZero §3.2.2, arXiv 2602.15922]。

> 14B 模型 + GB200 + Flash 后仍能跑 7Hz [DreamZero §3.2.6, arXiv 2602.15922]。

### 10. 训练阶段课程

- **3 阶段(物理范式)**:伪监督预训 → 物理 SFT(仿真) → 仿真 ground RL [Phys4D, arXiv 2603.03485]。
- **2 阶段(latent 范式)**:训 WM → "在 imagination 内"训 policy [Dreamer V4, emergentmind.com/papers/2509.24527]。
- **3 阶段(WAM 范式,DreamZero)**:Wan2.1-I2V backbone → Joint video-action 预训 100K steps × global batch 128 → DreamZero-Flash 后训 [DreamZero §4.1, arXiv 2602.15922]。
- **5 阶段(Cosmos 范式)**:Curator 数据筛选 → 训 Cosmos Predict → Cosmos Reason 加 physics → Cosmos Transfer 做 sim2real → Cosmos Policy 做后训练 [nvidia.com/cosmos]。

### 11. 与 VLA 的耦合方式

- **方式 A:WAM 直接当策略**(DreamZero) — 联合预测视频+动作,跳过纯 VLA [arXiv 2602.15922]。
- **方式 B:WM 后训练 VLA** — WoVR 把 LIBERO 39.95→69.2%,真机 61.7→91.7% [arXiv 2602.13977]。
- **方式 C:闭环共改进** — World-VLA-Loop:VLA 失败 rollout 反馈精炼 WM,**真机 +36.7%** [arXiv 2602.06508];VLAW:**+39.2% 绝对** [arXiv 2602.12063]。
- **方式 D:Reflective / Hierarchical** — SPIRAL 用 PlanAgent + CriticAgent 做 think-act-reflect [arXiv 2603.08403]。
- **方式 E:统一 latent video frame** — Cosmos Policy 把 action/state 当 latent video frames 一起进 DiT [huggingface.co/blog/nvidia/cosmos-policy-for-robot-control]。

### 12. World Model 当 RL "想象环境"

- **Dreamer V4** 在 Minecraft 完全 offline 拿钻石,**100× labeled data 节省** [emergentmind.com/papers/2509.24527]。
- **WoVR** 用 WM 做 sim,后训 VLA,LIBERO +29.3pt [arXiv 2602.13977]。
- 注意:像素 WM 必须 ≥ 100Hz 出虚拟帧才有 RL 性价比 [Interactive World Simulator 在 RTX 4090 单卡跑 15fps × 10+ 分钟, arXiv 2603.08546]。

### 13. RL / Reward 后训练 WM 自身

- **RoboAlign-R1**:多模态 teacher judge(6 维)→ 蒸馏出 student reward → RL 后训练,**+10.1% 综合 / +7.5% 操作准确度 / +4.6% 指令遵循** [arXiv 2605.03821]。
- **RLVR-World**:可验证奖励 RL,显著提升视频 WM [NeurIPS 2025, proceedings.neurips.cc]。
- **RLIR (RL with Inverse Rewards)**:用 IDM 反推动作可验证性,GRPO 后训练,5-10% 涨 action-following [openreview.pdf ad1bca0...]。
- **Persistent Robot WM**:多步 rollout 上做 RL,**98% 偏好对比** [arXiv 2603.25685]。

### 14. 多视角 / 多模态扩展

- **MultiWorld**:多代理 + 多视角 + Global State Encoder [arXiv 2604.18564]。
- **Wan2.2-S2V-14B**:加入音频驱动 [github Wan-Video/Wan2.2;huggingface Wan-AI/Wan2.2-S2V-14B]。
- **触觉**:DreamZero 在 future work 把触觉/力反馈点名为"WAMs may align actions with other predictive modalities"明确未来扩展 [arXiv 2602.15922 §2.2]。

### 15. 数据飞轮 / WM-as-Data-Engine(2026 最大商业价值)

- ⭐ **GigaWorld-0**:WM 当 data engine,**纯生成数据训出来的策略不需任何真实交互就能在真机表现** [arXiv 2511.19861]。
- **Nimbus**:4 层异步架构(轨迹规划 / 渲染 / 存储分离),**端到端吞吐 2-3× 提升** [arXiv 2601.21449]。
- **InternDataEngine**:统一物理仿真 + 语义任务生成 + 高吞吐调度,2026.03 发布 [GitHub InternRobotics/InternDataEngine]。
- **Genie Sim 3.0**(智元):LLM 驱动 3D 环境生成 + 5 task suites + **10,000+ 小时合成数据 + 200+ 任务** [arXiv 2601.02078;agibot.com/article/231/detail/55.html]。
- **Cosmos Curator + Dataset Search + Evaluator**:NVIDIA 官方数据飞轮三件套 [nvidia.com/cosmos]。
- **Interactive World Simulator**:**单卡 RTX 4090 跑 10+ 分钟 / 15 fps**,生成数据训的策略与真机数据等效 [arXiv 2603.08546]。

### 16. 评测体系(2026 必上的 4 件套)

- ⭐ **WorldArena**(CVPR 2026 Challenge,清华 fib-lab) — 16 metrics / 6 子维度 + EWMScore + 3 类(像素 / 具身功能 / 人评) [arXiv 2602.08971;world-arena.ai;cvpr2026challenge.world-arena.ai;github tsinghua-fib-lab/WorldArena]。
- ⭐ **RoboArena** — 真机 DROID 双盲两两 + 7 机构众包 [proceedings.mlr.press/v305/atreya25a]。
- ⭐ **Genie Sim 3.0**(智元) — 5 task suites,与真机配对验证 [arXiv 2601.02078]。
- ⭐ **PolaRiS** — Real-to-Sim,把真实视频扫描转可交互仿真,"与真机相关性高于已有 sim" [arXiv 2512.16881;github arhanjain/PolaRiS]。
- **dWorldEval** — discrete diffusion + sparse keyframe + progress token,跨千任务自动 policy eval [arXiv 2604.22152]。
- **关键评测原则**:"unseen environment + unseen objects 才是 default,不是 nice-to-have" [DreamZero §4.1 Evaluation Protocol, arXiv 2602.15922]。
- **任务粒度定义**:DreamZero 明确"motion + object 类型组合"决定 seen/unseen [DreamZero, arXiv 2602.15922]。

### 17. 算力 / 规模(2026 真实数据点)

- **DreamZero**:14B,batch 128 × 100K steps × 2 套(AgiBot + DROID) [arXiv 2602.15922 §4.1]。
- **Cosmos Predict 2.5**:2B / 14B 双尺寸,200M 视频片段 [research.nvidia.com]。
- **Wan2.1-I2V-14B**:训练在"billions of images and videos" [huggingface.co/docs/diffusers/en/api/pipelines/wan]。
- **Wan2.2-14B**:训练数据比 Wan2.1 多 65.6% 图 / 83.2% 视频 [github Wan-Video/Wan2.2]。
- **Open-Sora 2.0**:11B,**总训练成本 \$200K**,与 HunyuanVideo 同级 [arXiv 2503.09642]。
- **Dreamer V4**:110M 在 4090 (8GB) 实时 [github machines-in-motion/dreamer-v4]。
- **V-JEPA 2**:1M+ hours 互联网视频 [ai.meta.com/research/publications/v-jepa-2]。
- **关键洞察**:"larger pretrained video diffusion models produce higher-quality video predictions, which directly translates to superior downstream action execution" [DreamZero, arXiv 2602.15922]。

### 18. 训练稳定性 / 工程

- **共享 timestep 在训练初期更快收敛** [DreamZero §3.1 Training Objective vs Zhu et al. 2025/Kim et al. 2026, arXiv 2602.15922]。
- **不要用 LoRA**(至少在 Wan2.1-I2V backbone 上):DreamZero 实测 LoRA suboptimal,**只能全量更新 DiT 块** [DreamZero §4.1, arXiv 2602.15922]。
- **冻结 text/image encoder + VAE**,只更新 DiT + state/action encoder/decoder [DreamZero, arXiv 2602.15922]。
- **Trajectory-level 更新 + attention masking** 让当前 noisy chunk 只 attend 干净历史 chunk [DreamZero §3.1 公式 (3) + Figure 14, arXiv 2602.15922]。

### 19. 安全 / 合规

- **Cosmos NVIDIA Open Model License** [research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5]。
- **Genie 3 限 18+ 美国 Ultra 订阅**(显示 Google 评估了滥用风险) [deepmind.google/genie]。
- **生成视频水印 / C2PA provenance** —— 找不到 2026 严格出处,行业实践通识。

### 20. 综述参考

- ⭐ **A Comprehensive Survey on World Models for Embodied AI** [arXiv 2510.16732] — 三轴分类法。
- ⭐ **World Model for Robot Learning: A Comprehensive Survey** [arXiv 2605.00080]。
- **From Perception to Action: Spatial AI Agents and World Models** [arXiv 2602.01644]。
- **Embodied AI from LLMs to World Models** [清华 mn.cs.tsinghua.edu.cn/xinwang]。

---

## 四、2026 必读 SOTA 清单(纯具身世界模型,按主题归类)

### A. WAM(World Action Model)— 2026 主流

| 模型 | 出处 | 关键亮点 |
|---|---|---|
| ⭐ **DreamZero** | arXiv 2602.15922 + dreamzero0.github.io + github dreamzero0/dreamzero | 14B Wan2.1-I2V backbone,7Hz 闭环,38× 加速,RoboArena #1,MolmoSpaces #1,seen 62.2% / unseen 39.5% |
| ⭐ **GigaWorld-Policy** | arXiv 2603.17240 | Action-centered WAM,RoboTwin 2.0 比 π0.5 +95%,推理快 9× |
| ⭐ **STARRY** | arXiv 2604.26848 | RoboTwin 双臂 93.82%,真机 42.5→70.8% vs π0.5 |
| **GR00T N1.6 / N2 / WAM** | NVIDIA blog | DreamZero 主要 baseline |
| **LeWAM** | github ErykHalicki/LeWAM | 开源 WAM 参考实现 |
| **Cosmos Policy** | huggingface.co/blog/nvidia/cosmos-policy-for-robot-control | NVIDIA 官方 WAM,action 当 latent video frame |

### B. WM × VLA 闭环 / 共改进

| 模型 | 出处 | 亮点 |
|---|---|---|
| ⭐ **WoVR** | arXiv 2602.13977 | LIBERO 39.95→69.2%,真机 61.7→91.7% |
| **World-VLA-Loop** | arXiv 2602.06508 | 真机 +36.7% |
| **VLAW** | arXiv 2602.12063 | 绝对 +39.2% |
| **DriveWorld-VLA** | arXiv 2602.06521 | 自动驾驶 |
| **DIAL** | arXiv 2603.29844 | intent / action 解耦 |
| **SPIRAL** | arXiv 2603.08403 | think-act-reflect 闭环 |
| **CoLA-World** | arXiv 2510.26433 | LAM + WM 协同进化 |

### C. 通用视频 WM 基座(可作为 backbone)

| 模型 | 出处 | 亮点 |
|---|---|---|
| ⭐ **Wan2.1-I2V-14B-480P** | github Wan-Video/Wan2.1 + deepwiki Wan2.1 | DreamZero 实际 backbone,DiT + Wan-VAE |
| ⭐ **Wan2.2** | github Wan-Video/Wan2.2 | MoE 架构,5B 高效 + 14B 高质,Wan2.2-S2V-14B 加音频 |
| ⭐ **Cosmos Predict 2.5** | research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5 | 2B/14B,200M videos,30s 预测 |
| **Cosmos Transfer 2.5** | nvidia.com/cosmos | sim2real photoreal |
| **Cosmos Reason** | nvidia.com/cosmos | physics-aware VLM |
| **Genie 3** | deepmind.google/blog/genie-3 | 720p 实时 20-24 fps |
| **Open-Sora 2.0** | arXiv 2503.09642 | 11B,$200K |
| **HunyuanVideo / MovieGen** | (业界 baseline,公开资料未直接搜到 2026 更新) | 找不到 2026 出处 |

### D. Latent / Semantic WM

| 模型 | 出处 | 亮点 |
|---|---|---|
| ⭐ **V-JEPA 2** | ai.meta.com + arXiv 2506.09985 | 1M+ hours,Franka 上 zero-shot pick-and-place |
| ⭐ **V-JEPA 2.1** | arXiv 2603.14482 | 2026.03,dense feature,real-robot grasp +20% |
| ⭐ **Dreamer V4** | emergentmind.com/papers/2509.24527 | block-causal Transformer + Shortcut Forcing,**Minecraft 钻石全 offline** |
| **VideoWorld 2** | arXiv 2602.10102 | dLDM 解耦动作动力学与视觉外观,+70% 长程 |
| **Reconstruction or Semantics?** | arXiv 2605.06388 | 实证语义 latent > VAE latent |

### E. 物理一致性 / 4D / 高斯

| 模型 | 出处 | 亮点 |
|---|---|---|
| ⭐ **GWM (Gaussian World Model)** | arXiv 2508.17600 | 3D Gaussian 预测下未来,Latent DiT + 3D VAE |
| ⭐ **Physically Embodied Gaussian Splatting** | arXiv 2406.10788 | dual Gaussian-Particle,30Hz 实时,带物理纠正 |
| ⭐ **ABot-PhysWorld** | arXiv 2603.23376 | 14B DiT + 物理 DPO + 解耦判别器,3M 物理标注 |
| **Kinema4D** | arXiv 2603.16669 | URDF + Robo4D-200k(20 万 episodes 4D 标注) |
| **Phys4D** | arXiv 2603.03485 | 3 阶段(伪监督 + 物理 SFT + 仿真 RL) |
| **Mask World Model** | arXiv 2604.19683 | 语义 mask 替 RGB |
| **Causal World Modeling** | arXiv 2601.21998 | 因果建模 + Mixture-of-Transformers |

### F. 长程一致性 / 训练 forcing 范式

| 模型 | 出处 | 亮点 |
|---|---|---|
| ⭐ **LIVE** | huggingface.co/papers/2602.03747 | Cycle-consistency 消除 teacher 蒸馏需求 |
| ⭐ **Causal Forcing** | arXiv 2602.02214 | AR teacher ODE 初始化,跨 bidirectional → causal |
| ⭐ **Context Forcing** | arXiv 2602.06028 | 长上下文 student,**有效 20s,2-10×** |
| **Vid2World** | arXiv 2505.14357 | 把视频 diffusion 改造成交互式 WM |
| **Infinite-World** | arXiv 2602.02393 | HPMC,1000+ 帧 |
| **HorizonDrive** | arXiv 2605.11596 | SRR,FID -52% / FVD -37%,分钟级 |
| **MosaicMem** | arXiv 2603.17117 | 3D patch + 隐式 hybrid memory |
| **Persistent Robot WM** | arXiv 2603.25685 | RL 后训练 + 多视角 fidelity reward,98% 偏好 |

### G. RL / Reward 后训练 WM

| 模型 | 出处 | 亮点 |
|---|---|---|
| ⭐ **RoboAlign-R1** | arXiv 2605.03821 | 多模态 teacher judge,**+10.1% / +7.5% / +4.6%** |
| ⭐ **RLVR-World** | NeurIPS 2025 | 可验证奖励 RL |
| **EVA** | arXiv 2603.17808 | inverse dynamics reward |
| **RLIR** | openreview ad1bca0... | GRPO + IDM,+5-10% action-following |

### H. 数据引擎 / 飞轮

| 资源 | 出处 |
|---|---|
| ⭐ **GigaWorld-0** | arXiv 2511.19861 + giga-world-0.github.io |
| ⭐ **Genie Sim 3.0**(智元) | arXiv 2601.02078 + agibot.com/article/231/detail/55.html |
| ⭐ **InternDataEngine** | github InternRobotics/InternDataEngine |
| **Nimbus** | arXiv 2601.21449 |
| **Interactive World Simulator** | arXiv 2603.08546 |
| **NVIDIA Cosmos Curator/Search/Evaluator** | nvidia.com/cosmos |

### I. 评测 / Benchmark

| 资源 | 出处 |
|---|---|
| ⭐ **WorldArena**(CVPR 2026 Challenge) | arXiv 2602.08971 + cvpr2026challenge.world-arena.ai + github tsinghua-fib-lab/WorldArena |
| ⭐ **RoboArena** | proceedings.mlr.press/v305/atreya25a |
| ⭐ **Genie Sim 3.0** | arXiv 2601.02078 |
| ⭐ **PolaRiS** | arXiv 2512.16881 + github arhanjain/PolaRiS |
| **dWorldEval** | arXiv 2604.22152 |

### J. 多视角 / 多智能体 / 创新架构

| 模型 | 出处 | 亮点 |
|---|---|---|
| **MultiWorld** | arXiv 2604.18564 | multi-agent + multi-view |
| **GigaWorld-Policy** | arXiv 2603.17240 | action-centered causal sequence |

### K. 综述

| 资源 | 出处 |
|---|---|
| **A Comprehensive Survey on World Models for Embodied AI** | arXiv 2510.16732 |
| **World Model for Robot Learning Survey** | arXiv 2605.00080 |
| **From Perception to Action: Spatial AI Agents and World Models** | arXiv 2602.01644 |

---

## 五、2026 推荐训练路线(基于以上 SOTA)

```text
─────────────────────────── Stage 0 ───────────────────────────
形态选择 + 用途定义
   ├─ 主要做策略?       → WAM (DreamZero / GigaWorld-Policy / STARRY)
   ├─ 数据飞轮?         → 像素 DiT (Cosmos Predict 2.5 / GigaWorld-0)
   ├─ Imagination RL?   → Latent (Dreamer V4 / V-JEPA 2.1)
   ├─ 精细物理?         → 4D / Gaussian (GWM / Phys4D / Kinema4D)
   └─ 多智能体协作?     → MultiWorld / 自定义

─────────────────────────── Stage 1 ───────────────────────────
基座选择(2026 共识:不要从零开始)
   优先级:
   • Wan2.2 / Wan2.1-I2V-14B(开源,DreamZero 验证可行)
   • Cosmos Predict 2.5(NVIDIA 全栈)
   • Dreamer V4(轻量 latent)
   • V-JEPA 2.1(语义 latent)
关键决策:
   • 全量更新 DiT,不要 LoRA(DreamZero 验证)
   • 冻结 text encoder + image encoder + VAE

─────────────────────────── Stage 2 ───────────────────────────
数据建设
   • 互联网视频 10⁸+ 片段(用 Cosmos Curator 或自建)
   • 真机机器人:DROID + AgiBot + 自采;500h 起步,优先"长 episode + 多子任务 + 不重复"
   • 第一人称人类视频:Ego4D / EgoExo4D
   • 仿真数据:Genie Sim 3.0(10k h)+ RoboCasa + Isaac Lab
   • 物理标注 + 4D 标注:Robo4D-200k 风格

─────────────────────────── Stage 3 ───────────────────────────
WAM 主体训练
   • Joint video-action flow matching
   • 共享 timestep(初期更快收敛)
   • Teacher Forcing chunk-wise
   • Block-causal attention,AR 视频 + 非 AR 动作
   • 多视角 concat 进单帧,backbone 不动
   • Trajectory-level update + attention masking
   • 100K steps × global batch 128(DreamZero 数字)

─────────────────────────── Stage 4 ───────────────────────────
对齐 / 后训练(2026 关键)
   • DreamZero-Flash:解耦 noise schedule,16→1 步推理
   • Inverse Dynamics 对齐(EVA)
   • 物理 DPO(ABot-PhysWorld)
   • Self-Corrective Rollout(HorizonDrive)
   • 多视角 fidelity RL(Persistent Robot WM)
   • 多模态 reward 对齐(RoboAlign-R1)

─────────────────────────── Stage 5 ───────────────────────────
长程 / 记忆 增强
   • 加 HPMC(Infinite-World)或 MosaicMem(3D + 隐式 hybrid)
   • Context Forcing 推到 20 秒 effective context
   • Cycle-Consistency loss(LIVE)

─────────────────────────── Stage 6 ───────────────────────────
推理优化 + 部署(DreamZero 范本)
   ① CFG Parallelism(双 GPU 拆条件/无条件)→ -47% latency
   ② DiT Caching(velocity cosine 阈值)→ 16→4 步
   ③ torch.compile + CUDA Graphs
   ④ NVFP4 量化(Blackwell)
   ⑤ DreamZero-Flash → 1 步,150ms / chunk
   ⑥ Asynchronous Closed-Loop(200ms 内即可保 30Hz 闭环)
   ⑦ Action Chunk 后处理:2× upsample + Savitzky-Golay 滤波

─────────────────────────── Stage 7 ───────────────────────────
评测 + 飞轮
   • WorldArena(CVPR 2026 Challenge 协议)
   • RoboArena 真机 leaderboard
   • Genie Sim 3.0 + PolaRiS 配对验证
   • dWorldEval 跨千任务自动 policy eval
   • 用 WM 当 data engine 反哺训练(GigaWorld-0 / Nimbus)
```

---

## 六、容易踩的坑(基于 2026 公开 postmortem)

1. **画质好 ≠ 决策有用**——14 个 SOTA WM 都有 perception-functionality gap [WorldArena, arXiv 2602.08971]。
2. **不做物理对齐必然产生穿模 / 反重力** [ABot-PhysWorld, arXiv 2603.23376]。
3. **AR 多步 rollout 必坍塌**,除非显式做 SRR / cycle / RL [HorizonDrive, arXiv 2605.11596;Persistent Robot WM, arXiv 2603.25685;LIVE, huggingface.co/papers/2602.03747]。
4. **盲目追求重复演示数据反而害事**——WAM 偏好"多样、非重复"长 episode [DreamZero §4.1, arXiv 2602.15922]。
5. **VAE latent 训出来 PSNR 高但下游策略低**——必须切语义 latent(V-JEPA 2.1 / SigLIP 2 / Web-DINO) [Reconstruction or Semantics?, arXiv 2605.06388]。
6. **LoRA 在视频 backbone 上 suboptimal**——Wan2.1-I2V 上必须全量更新 DiT [DreamZero §4.1, arXiv 2602.15922]。
7. **不解耦 video / action noise schedule,少步推理必崩** [DreamZero-Flash 训-推 mismatch 分析, arXiv 2602.15922 §3.2.5]。
8. **AR 视频 + AR 动作两者都自回归会爆错误传播**——只让视频 AR、动作不 AR [DreamZero §3.1, arXiv 2602.15922]。
9. **不替换 KV cache 中的预测帧**会让 closed-loop 发散——必须用 GT 观测替换 [DreamZero §3.1 Inference, arXiv 2602.15922]。
10. **不做 CFG Parallelism 直接两次 forward**会浪费一半算力 [DreamZero, arXiv 2602.15922]。

---

## 七、给"2026 实战"的 5 条最重要建议

1. **基座站在巨人肩上**:用 **Wan2.2 14B** 或 **Cosmos Predict 2.5 14B** 起步,**不要从零训** [github Wan-Video/Wan2.2;research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5;DreamZero 实证 14B > 5B, arXiv 2602.15922]。
2. **数据"长 + 多样 + 不重复"** > 数量;500 小时真机 + 海量互联网视频 + 无标签人类视频已经够 [DreamZero 用 500h AgiBot 即超 SOTA VLA, arXiv 2602.15922;V-JEPA 2 用 1M+ 互联网视频 + <62h 机器人, ai.meta.com]。
3. **闭环必做 KV cache GT 替换 + Asynchronous Execution**,这是 14B WM 能 7Hz 跑的关键 [DreamZero §3.1/§3.2.2, arXiv 2602.15922]。
4. **必做 Flash-style decoupled noise schedule** 才能把 diffusion 步数从 16 压到 1,**380ms→150ms 是部署红线** [DreamZero §3.2.5, arXiv 2602.15922]。
5. **用 WorldArena + RoboArena + Genie Sim 3.0 + PolaRiS 四件套评测**,不要只看像素 metric——任何一个跑分高但下游低的都是"功能性陷阱" [arXiv 2602.08971;proceedings.mlr.press/v305/atreya25a;arXiv 2601.02078;arXiv 2512.16881]。

---

如果你需要,我可以接着做以下任意一项:

- 把 **DreamZero 论文**做一份**带页码的精读笔记**(架构图 / 公式 / 表 1 加速;数据收集 SOP;实验细节);
- 给你一份**针对你算力预算和数据规模**的**专属配方**(Wan2.2 vs Cosmos vs V-JEPA 2.1 该选哪个,各阶段 batch / steps / 评测节点);
- 或对比 **DreamZero / GigaWorld-Policy / STARRY / Cosmos Policy / Helix 02** 五大主流 WAM 路线,出一张**模块级差异表**。

