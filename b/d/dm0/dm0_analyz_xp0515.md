# Analysis and summary of DM0 paper
_Exported on 5/16/2026 at 09:02:58 GMT+8 from Cursor (3.4.20)_

---

**User**

请阅读和深入分析 @b/d/dm0/dm0_An_Embodied-Native_Vision-Language-Action_Model_towards_Physical_AI.pdf , 并根据 @embd_pt_sota_princpl.md 中提到的各个方面各个细节对"DM0 An Embodied-Native Vision-Language-Action Model towards Physical AI"这篇论文进行详细的解释和总结, 解释与总结写到 @b/d/dm0/dm0.md 中.

---

**Cursor**

我已经深入阅读了这篇论文,包括架构、训练流程、数据构成和实验结果。现在我将根据 `embd_pt_sota_princpl.md` 中提到的各个维度,详细的解读和总结写到 `dm0.md` 中。

我已经完成了对 `DM0 An Embodied-Native Vision-Language-Action Model towards Physical AI` 这篇论文的深入阅读,并严格按照 `embd_pt_sota_princpl.md` 中列出的 6 大维度(数据及其处理 / 模型结构 / 系统 / Pre-Training / Mid-Training / 微调 / RL / 评测与消融)逐项对照,把详细解读写入了 `dm0.md`。

主要工作要点:

- **逐维度对照**:对原则文件里每一项(数据来源、多样性、格式、Tokenizer、跨本体身份注入、Action Head、Loss、训练阶段划分、超参、并行精度、推理加速、Receding Horizon、安全约束、Reward 设计、消融实验等)都按照"DM0 是怎么做的 / 没做"做了明确对应。
- **原文核对**:所有关键数字都来自论文正文(2B 参数、Qwen3-1.7B 主干、PE 视觉编码、728×728 图像、Action chunk=50、255-bin 量化 token、1.2T tokens / 200M / 50M 三阶段数据量、AdamW 超参、64×H20 / 8×H20 / 16×H20 训练规模、RoboChallenge Specialist 62.0% 与 Generalist 37.3/49.08 等)。
- **写出"做得好"和"披露不足"两部分**:在第 9 节明确指出 DM0 未涉及的方面(数据飞轮、触觉/音频/深度、embodiment id embedding、系统层面、RL、消融实验、OOD 评测等),便于后续与其它 VLA 论文做横向对比。
- **三大核心设计独立成段**:把 Hybrid Gradient Strategy(KI 借鉴)、Embodied Spatial Scaffolding 层次化 CoT、Action 双视图对齐(VLM 离散 token + Action Expert 连续 Flow Matching)讲清楚。

如果你之后想做和其它 VLA 论文(π0.5 / GR00T N1 / GigaBrain-0 / Spirit-v1.5 等)的横向对照,可以直接复用 `dm0.md` 的小节结构。

---

**User**

请阅读和深入分析 @b/d/dm0/dm0_An_Embodied-Native_Vision-Language-Action_Model_towards_Physical_AI.pdf , 看看有哪些是研究机器人领域的具身智能和VLA或WAM所要考虑的方面和细节在  @embd_pt_sota_princpl.md 中是被漏掉的, 请补在该文档的后面.

---

**Cursor**

我重新过了一遍 DM0 全文,把它讨论或暗示但 `embd_pt_sota_princpl.md` 中没有覆盖到的具身 / VLA / WAM 维度整理出来,并按照原文档的风格(##、+、-)追加到文档末尾。

我已经完成了对 DM0 论文的二次精读,并把原 `embd_pt_sota_princpl.md` 中没有覆盖到的具身智能 / VLA / WAM 维度,以一致的风格(`##` + `+` + `-`)追加在文档末尾。新增的所有维度都来自 DM0 论文中明确出现或隐式暗示的设计选择。

新增内容按以下 11 个主题组织(都是原文档没有充分覆盖的):

1. **训练范式与设计哲学** — Embodied-Native vs Pretrain-then-Adapt 等顶层范式选择,物理先验 vs 语义先验的二分。
2. **梯度路由与数据条件化训练** — Knowledge Insulation / 数据条件化梯度流 / 双向能力保持。
3. **动作表征的多视图对齐** — 离散+连续双重监督、动作词表设计、Flow Matching 训练超参等。
4. **辅助任务与层次化监督设计** — 子任务 → bbox → 轨迹 → 动作的层次 curriculum,信息瓶颈视角,进度监督。
5. **推理模式与运行时配置** — 同模型多种推理模式(直接动作 vs CoT 后再动作)、训推一致性、CoT 延迟管理。
6. **模型结构补充** — 视觉编码器选择、VLM 与 Action Expert 的耦合机制、多视角图像的工程化处理、图像 token 压缩。
7. **数据工程补充** — episodic JSONL、对话模板增广、关键帧采样、跨阶段数据保留、隐私脱敏、数据 license 等。
8. **训练流程补充** — 单阶段内多 phase LR、Catastrophic Forgetting 显式缓解、序列长度预算、训练效率指标。
9. **评测的更多维度** — Specialist vs Generalist 双设定、复合分指标、基准时效性、参数量对齐对比、跨领域涌现能力、真机/仿真权衡、评测协议开源化。
10. **工程化与开源化** — 开源策略、复现门槛、团队组织、硬件合规与地缘约束等。
11. **World Model (WAM) 专项** — WAM 目的、与 VLA 集成方式、输入输出空间、训练数据、预测目标、评估指标、部署模式。
12. **模态扩展路线** — 多模态引入时机、tokenizer/encoder 选择、对齐任务设计、模态缺失鲁棒性。
13. **Scaling 与未来方向** — 范式级 scaling law、多尺寸训练计划、跨本体 scaling、与 WM/RL 协同 scaling 等。

如果你后续要把这份扩展版的清单用到其它 VLA / WAM 论文的对照评估,可以直接按这个 13 + 原 8 个 ##(共 21 个)模块逐项打勾。

---

**User**

请深入分析一下 @b/d/dm0/dm0.md 及其对应的代码库`d:/SRC/Robot/dexbotic/`, 注意dm0.md中提到的"# 要参考的资料". 基于现有代码库,详细回答一下 dm0.md 提到的如下内容在实际代码中是如何实现的, 把详细的分析和实现写到 @b/d/dm0/dm0_1.md 中. 需要分析与回答实际代码(在 `d:/SRC/Robot/dexbotic/` 目录中)是如何实现的内容如下:
```
**Embodied Spatial Scaffolding 的层次化辅助任务**(都通过 L_AR 监督):
1. **Subtask prediction**:预测细粒度子任务描述
2. **Goal bounding box prediction**:预测目标物体/区域的 bbox
3. **End-effector trajectory prediction**:预测主视角下未来若干帧的 EEF 2D 轨迹
4. **Discrete action prediction**:预测离散 action token

> 这里**没有**单独的 affordance head,也没有专门的 reward / value head;一切辅助监督都通过自回归文本预测来实现 —— 这是一种"text-as-everything"的简洁设计。
```

---

**Cursor**

我先做一个快速的预读以理解上下文,然后再制定计划。需要看两件事:
1. `dm0.md` 中的"要参考的资料"在哪里、引用了什么
2. `d:/SRC/Robot/dexbotic/` 代码库的整体结构

我先对 DM0 代码结构做精准的预读,以便制定有依据的分析计划。

我做最后一次精准探查,确认 `dm0_prog_arch.py` 是否就是 Spatial Scaffolding 的实现入口,以及 conversation 模板里有没有 bbox/trajectory 这些字段。

发现重要事实:`dm0_prog_arch.py` 实际是 **"Progress prediction"**(进度估计) 的实现,不是论文讲的"Progressive scaffolding"。让我对全代码库做最后的关键词验证,确认论文里 4 层级辅助任务的开源覆盖度。

预读完成,我已经掌握了核心事实。这里有一个**重要发现**需要在计划中明确体现:

**关键发现**:`dm0.md` 第 200-207 行所讲的论文 4 层级辅助任务(Subtask / Goal bbox / EEF 2D 轨迹 / 离散 action token),**在开源的 `d:/SRC/Robot/dexbotic/` 代码库里几乎没有任何对应实现**:

- 关键字 `subtask` / `goal_bbox` / `waypoint` / `2d_traj` / `spatial_scaffold` / `cot` / `255-bin` 在整个 `dexbotic/` + `playground/` 都**没有任何匹配**。
- `dm0_arch.py` 是基础版:仅 Qwen3-VLM + Action Expert(连续 Flow Matching),`action_expert.model.embed_tokens = None`(第 80 行)直接证明 action expert 不预测离散 action token。
- `dm0_prog_arch.py` 的 "Prog" 实际是 **"Progress prediction"**(进度估计的数值回归),通过 `progress_in_proj/out_proj` 实现,这对应论文 3.3 节里 Specialist 用的 **progress supervision**,**不是**论文 2.3 节的 4 层级 CoT。
- `docs/DM0.md` 也未提及任何 spatial scaffolding / CoT / 子任务预测。

也就是说:**论文里"text-as-everything"的辅助文本监督,在开源 dexbotic 中被简化为"只回归连续动作 + 可选 progress 数值通道"** —— 这正是 b/d/dm0/dm0_1.md 应该清楚说明的核心结论。

这个事实清晰且重要,无须澄清。我直接给出计划。

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

让我快速查看 `dm0_exp.py` 来完整理解训练入口和 forward 调用链。

所有验证完成。关键发现:

- `subtask` 关键字仅出现在 **数据转换脚本** 的 `extra` 元数据字段中(用于 lerobot/so101 数据转 dexdata),**完全不进入** DM0 训练流水线 (`data_keys` 只包含 `input_ids, labels, action, image, state, image_masks`)。
- `action_token` 在 pi0/pi05/oft/gr00tn1 模型中存在,但 **DM0 模型完全没有**。
- `goal_bbox`/`waypoint`/`scaffold`/`chain.of.thought` 等**全库零匹配**。

现在我开始撰写 `b/d/dm0/dm0_1.md`。

所有 todo 已完成。`b/d/dm0/dm0_1.md` 已生成且通过 lint 检查。

## 完成情况摘要

我按计划分 7 个步骤完成了对 `dm0.md` 第 200-207 行 (Embodied Spatial Scaffolding 4 层级辅助任务) 在 `d:/SRC/Robot/dexbotic/` 中的实现分析,写入 `b/d/dm0/dm0_1.md` (~570 行)。

### 关键发现 (写入 dm0_1.md 第 0 章)

1. **论文承诺的 4 层级辅助任务在开源 dexbotic 中均未实现** — 关键字 `goal_bbox` / `waypoint` / `scaffold` / `chain.of.thought` / `255-bin` 全库零匹配; `subtask` 仅出现在 2 个数据转换脚本的 `extra` 元数据字段里, 训练流水线根本不读; `action_token` 在 pi0/pi05/oft/gr00tn1 模块里有, **DM0 模块完全没有**。

2. **`dm0_prog_arch.py` 的 "Prog" 是 "Progress prediction" 而不是 "Progressive Scaffolding"** — 它是 0-1 数值进度估计的辅助通道 (`progress_in_proj` / `progress_out_proj`), 对应论文 3.3 节里只占一行的 progress supervision, 而且开源版本只支持推理 (文件头注释明确 "for inference only"), **没有 forward 方法**, 不可直接训练。

3. **"text-as-everything" 在开源代码中被退化为 "action-MSE-as-everything"** — `DM0Model.__init__` 第 80 行 `self.action_expert.model.embed_tokens = None` 永久断绝离散 token 路径; `DM0ForCausalLM.forward` 末尾只有 `loss = action_loss` 一行, 即使 `labels` 在签名里也只是被透传忽略。

### 文档亮点

- **完整引用**: 全部代码引用使用精确的 `startLine:endLine:filepath` 格式 (覆盖 16 处关键代码位置)
- **2 张 mermaid 图**: (a) DM0 架构数据流, (b) 数据/标签到模型的流向 + 3 处断点标注
- **零匹配证据表** (附录 C): 把全库扫描命令与命中数列成表格作为论断的经验证据
- **4 项辅助任务对照总表** (3.5 节): 一目了然
- **7 章完整结构**: 核心结论 / 文件清单 / 架构概览 / 4 项逐项对照 / 等价物 / 数据流追踪 / "text-as-everything" 偏差分析 / 复现 roadmap

---

**User**

请深入分析一下 @b/d/dm0/dm0.md 及其对应的代码库`d:/SRC/Robot/dexbotic/`, 注意dm0.md中提到的"# 要参考的资料". 基于现有代码库,详细回答一下 dm0.md 提到的 `**L_AR (自回归交叉熵)**:监督 VLM 预测具身推理文本 + 离散 action token` 在实际代码 (在 `d:/SRC/Robot/dexbotic/` 目录中) 中是如何实现的, 把详细的分析和对应代码如何实现的详细回答写到 @b/d/dm0/dm0_1.md 后面. 

---

**Cursor**

我先做精准的预读, 确认两件事:
1. `dm0.md` 中 L_AR 的具体出现位置 (用于精确引用)
2. dexbotic 代码库里其他模型 (非 DM0) 的 L_AR 实现, 用于横向对比

我已经掌握了关键事实, 可以做出有据可查的计划。核心发现:

- `DexboticForCausalLM.forward()` (基类, `dexbotic_arch.py:429-491`) 里**有**标准 LM loss 实现 (`self.loss_function(logits, labels, vocab_size)` 第 488 行)。
- `DM0ForCausalLM.forward()` 完全 **override 掉** 了基类 forward, 没调 `super()`, 也没调 `loss_function`, 只算 MSE Flow Matching loss。
- `cross_entropy` / LM loss 在 dexbotic 的其他 6 个模型里都有: `oft_discrete_arch.py`, `hybrid_pi05_arch.py`, `muvla_arch.py`, `navila/loss.py`, `navila/navila_arch.py`, `cogact/hybrid_cogact_arch.py`, 这些可以作为 "如果 DM0 想加 L_AR 应该长什么样" 的具体范本。
- `DM0Tokenization` (前一轮已读) 已经生成了正确的 `labels` 和 `loss_mask`, 但 `DM0ForCausalLM.forward()` 不消费。

dm0.md 中 L_AR 出现的关键位置是 **194 行** (公式) 和 **282 行** (Pretrain 阶段说明) 以及 **316 行** (Mid-train Loss), 我可以精确引用。

请求清晰, 不需要澄清。直接出计划。

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

已按计划完成全部 6 个 todo。`dm0_1.md` 末尾追加的第 8 章内容如下（无 lint 警告）：

## 完成摘要

新增章节 [`dm0_1.md` 第 8 章](dm0_1.md): **L_AR (自回归交叉熵) 在 dexbotic 代码库中的实现分析**, 共 9 个小节 (8.0-8.8):

| 小节 | 内容 | 核心证据来源 |
|---|---|---|
| **8.0** | 4 点核心结论一段话说完 | — |
| **8.1** | `dm0.md` 中 `L_AR` 的 5 处出现位置定位 | dm0.md L192-657 |
| **8.2** | DM0 forward 跳过 LM loss 的 5 项证据 | `dm0_arch.py:77-80, 128-143, 406-424, 495-511` |
| **8.3** | 基类标准 LM loss 实现 (被 DM0 绕过) | `dexbotic_arch.py:482-496` |
| **8.4** | 数据层准备好 labels 但无人消费 | `process.py:407-414, 451-454, 471-475` |
| **8.5** | **4 个兄弟模型 L_AR 范本横向对比** + 对比小表 | `oft_discrete_arch.py:10-13, 187-191`; `hybrid_pi05_arch.py:455-512`; `cogact/hybrid_cogact_arch.py:126-188`; `navila/loss.py:11-17` |
| **8.6** | "L_AR 链路 5 环节断点图" mermaid + 状态判定表 | 整合 8.2-8.4 证据 |
| **8.7** | (推测) 3 条最小补丁路径 (A 最便宜 / B 最完整 / C 最直接) + 工作量估计表 | — |
| **8.8** | "四端预留, 一处断路" 状态卡 + 与第 1-7 章的孪生定位 | — |

## 关键发现 (与第 1-7 章互补)

第 1-7 章回答 "Embodied Spatial Scaffolding 4 项辅助任务在哪 → 全缺失"。第 8 章回答更上位的问题: **承接这 4 项辅助任务的 `L_AR` 主干为何接不上去**:

- **DM0 自己的 forward 主动断路** ([`dm0_arch.py:495-502`](d:/SRC/Robot/dexbotic/dexbotic/model/dm0/dm0_arch.py)): 唯一 loss = `F.mse_loss`
- **基础设施"四端全部预留"**: 数据格式 (labels + IGNORE_INDEX 协议) / 模型签名 (labels 形参) / 基类实现 (`self.loss_function`) / 兄弟模型范本 (Pi0.5 hybrid 的 `L_AR + L_FM` 共存写法)
- **最便宜补法只需 ≈15 行** (路径 A: 复用基类 `loss_function`); **最完整补法 ≈200 行** (路径 B: 含离散 action token 词表扩展 + 4 项辅助任务 conversation 模板)

全部代码引用采用 `startLine:endLine:filepath` 格式, 总行数从原 684 行扩到 ≈ 1100 行, ReadLints 检查无警告。

---

**User**

上网查一下2026年有些什么带开源代码的, 并且使用了类似 DM0 的 `离散 action token` 和 `L_AR loss` 方法的论文, 请把论文和代码地址列出来. 

---

**Cursor**

下面是我从 arxiv / ICLR 2026 / ICRA 2026 / OpenReview / GitHub 综合检索到的"2026 年"开源 VLA 论文列表, 全部命中 DM0 论文里 `离散 action token + L_AR (自回归交叉熵)` 这条技术线. 按"和 DM0 相似度"从高到低排序:

---

## 一、严格 2026 年 (arxiv 26xx 或 2026 会议接收)

### 1. AR-VLA: True Autoregressive Action Expert for VLA Models (Mar 2026)

- **论文**: [arXiv:2603.10126](https://arxiv.org/abs/2603.10126), [HTML 版](https://arxiv.org/html/2603.10126v1)
- **代码**: 官方 repo 暂未发现; 同作者团队相关实现见 [alibaba-damo-academy/RynnVLA-002](https://github.com/alibaba-damo-academy/RynnVLA-002)
- **跟 DM0 的对应**: 跟 DM0 论文里"离散 action token + 自回归 cross-entropy"几乎完全是同一招, 区别在 AR-VLA 强调**跨时间步**保持自回归 (DM0 是在单步内做自回归), 用 Hybrid KV-Cache + Dynamic Temporal Re-anchoring 同步 V-L-A 三模态.

### 2. Fast-dVLA: Accelerating Discrete Diffusion VLA to Real-Time (Mar 2026)

- **论文**: [arXiv:2603.25661](https://arxiv.org/abs/2603.25661)
- **项目页**: <https://chris1220313648.github.io/Fast-dVLA/>
- **底层基础代码 (Discrete Diffusion VLA)**: [Liang-ZX/DiscreteDiffusionVLA](https://github.com/Liang-ZX/DiscreteDiffusionVLA)
- **跟 DM0 的对应**: 同样是"action 离散化为 token + cross-entropy 监督", 但用 discrete diffusion 替代严格自回归, LIBERO 96.6% / 4.1× 加速.

### 3. ActionCodec: What Makes for Good Action Tokenizers (Feb 2026)

- **论文**: [arXiv:2602.15397](https://arxiv.org/abs/2602.15397)
- **代码**: [ZibinDong/actioncodec](https://github.com/ZibinDong/actioncodec)
- **跟 DM0 的对应**: **最值得参考** —— 用 Perceiver + VQ/RVQ 做 action tokenizer, repo 同时给出 4 种 policy 范式: AR (next-token, 即 DM0 论文里的 L_AR) / PD (并行解码) / KI (Knowledge Isolation, 即 DM0 论文 3.4 节灵感来源) / BAR (块自回归). SmolVLM2-2.2B 微调后 LIBERO 97.4% (无机器人预训练).

### 4. FASTer / FASTerVLA (ICLR 2026 Poster)

- **论文**: [OpenReview k6nTUFoqeT](https://openreview.net/forum?id=k6nTUFoqeT), arxiv [2512.04952](https://arxiv.org/abs/2512.04952)
- **相关代码**: [uynitsuj/FASTer](https://github.com/uynitsuj/FASTer) (DCT 早停实现; 官方 repo 暂未挂出)
- **跟 DM0 的对应**: 学习型 action tokenizer (FASTerVQ) + 块自回归解码 (FASTerVLA), 等价于 DM0 论文的"离散 action token + L_AR"加一层 chunk-wise 并行.

### 5. UniVLA (BAAI 版, ICLR 2026)

- **论文**: [arXiv:2506.19850](https://arxiv.org/abs/2506.19850), [ICLR 2026 海报](https://iclr.cc/virtual/2026/poster/10009646)
- **代码**: [baaivision/UniVLA](https://github.com/baaivision/UniVLA)
- **跟 DM0 的对应**: vision/language/action 全部表示为离散 token, 在统一 transformer 里做 next-token prediction —— **本质就是把 DM0 论文 4 项辅助任务 + L_AR 做到极致**, LIBERO 95.5%, 超过 π0-FAST 的 85.5%.
- **注意**: 还有一个**同名**的 OpenDriveLab UniVLA (`OpenDriveLab/UniVLA`), 那是另一篇论文, 路径不同.

### 6. Discrete Diffusion VLA (ICLR 2026)

- **论文**: [arXiv:2508.20072](https://arxiv.org/abs/2508.20072), [OpenReview YWeNCMxdhM](https://openreview.net/pdf?id=YWeNCMxdhM)
- **代码**: [Liang-ZX/DiscreteDiffusionVLA](https://github.com/Liang-ZX/DiscreteDiffusionVLA)
- **跟 DM0 的对应**: 与 DM0 共享"action 离散化为 token, cross-entropy 监督"的思路; 区别是把严格 left-to-right 自回归换成 discrete diffusion + 自适应解码顺序. LIBERO 96.3%.

### 7. Unified Diffusion VLA (JD3P, ICLR 2026)

- **论文**: [arXiv:2511.01718](https://arxiv.org/abs/2511.01718)
- **代码**: [OpenHelix-Team/Unified-Diffusion-VLA](https://github.com/OpenHelix-Team/Unified-Diffusion-VLA) (基于 BAAI UniVLA fork)
- **跟 DM0 的对应**: Joint Discrete Denoising Diffusion Process, 把多模态 (含 action) 全部 token 化后联合去噪, 比 AR 推理快 4×.

### 8. LLaVA-VLA (ICRA 2026, OpenHelix Team)

- **论文 / 项目**: [OpenHelix-Team/LLaVA-VLA](https://github.com/OpenHelix-Team/LLaVA-VLA)
- **跟 DM0 的对应**: 极简实现 —— 直接用 LLaVA-v1.5-7B 接 256-bin 离散动作 + cross-entropy, 7 小时微调即可 (CALVIN 3.68 平均 task 长度). 是 DM0 那条 L_AR 路径**最便宜的开源样板**.

---

## 二、虽然不是 2026 但与 DM0 离散动作 + L_AR 直接同源, 且是 DM0 论文里的对照基线

### 9. π₀-FAST + FAST Tokenizer (Jan 2025) — DM0 论文最重要的基线之一

- **论文**: [arXiv:2501.09747](https://arxiv.org/abs/2501.09747)
- **代码**: [Physical-Intelligence/openpi](https://github.com/physical-intelligence/openpi)
- **Tokenizer**: [physical-intelligence/fast (HuggingFace)](https://huggingface.co/physical-intelligence/fast)
- **跟 DM0 的对应**: DCT + BPE 把动作压成离散 token + 标准 LM cross-entropy, 比 diffusion 训快 5×. **dexbotic 仓库的 `pi05` 模块直接基于此**, 也是 DM0 论文 4.1 节明确对标的对象.

### 10. OpenVLA (Jun 2024, 长青基线)

- **论文**: [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- **代码**: [openvla/openvla](https://github.com/openvla/openvla); 精简版 [stanford-iliad/openvla-mini](https://github.com/stanford-iliad/openvla-mini)
- **跟 DM0 的对应**: 7B Llama2 主干 + 256-bin 离散动作 + 自回归 cross-entropy, **就是 DM0 论文 `L_AR + 离散 action token` 的最早开源完整范式**.

### 11. WorldVLA → RynnVLA-002 (阿里达摩院, Jun & Nov 2025, Apache-2.0)

- **WorldVLA 论文**: [arXiv:2506.21539](https://arxiv.org/abs/2506.21539); 代码: [alibaba-damo-academy/WorldVLA](https://github.com/alibaba-damo-academy/WorldVLA)
- **RynnVLA-002 论文**: [arXiv:2511.17502](https://arxiv.org/abs/2511.17502); 代码: [alibaba-damo-academy/RynnVLA-002](https://github.com/alibaba-damo-academy/RynnVLA-002); 模型: [HF Alibaba-DAMO-Academy/RynnVLA-002](https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-002)
- **跟 DM0 的对应**: image / text / action **共享同一 vocabulary** + 自回归生成, 本质就是 DM0 论文里"用一个 LLM 同时跑 L_AR 文本 + L_AR action token"思想的极致版. RynnVLA-002 同时给出离散 action / 连续 action 两条头, LIBERO 97.4%.

### 12. VQ-VLA (ICCV 2025)

- **论文 / 项目页**: <https://xiaoxiao0406.github.io/vqvla.github.io/>
- **代码**: [xiaoxiao0406/VQ-VLA](https://github.com/xiaoxiao0406/VQ-VLA)
- **跟 DM0 的对应**: VQ 离散动作 tokenizer + 自回归 cross-entropy, 比 OpenVLA 快 3×, 长程任务成功率 +30%.

### 13. HybridVLA (Mar 2025, 多次更新到 v3)

- **论文**: [arXiv:2503.10631](https://arxiv.org/abs/2503.10631)
- **跟 DM0 的对应**: 把 next-token prediction (走 cross-entropy / L_AR) 与 diffusion (走 MSE / L_FM) 同时放进一个 LLM, 跟 DM0 公式 `L_total = λ·L_AR + L_FM` 思路完全一致. (代码地址多次变更, 建议在 [Awesome-VLA](https://github.com/yueen-ma/Awesome-VLA) 里追踪最新链接.)

---

## 三、和 DM0 思路最贴近的"3 个首推"

如果只看"**离散 action token + L_AR cross-entropy + 真正能跑的开源代码**", 我建议你优先精读这 3 个:

| 排名 | 项目 | 推荐理由 |
|---|---|---|
| 🥇 **1** | **ActionCodec** ([code](https://github.com/ZibinDong/actioncodec)) | 同时给出 AR / PD / KI / BAR 4 种 policy 范式, 等于把 DM0 论文里的 "L_AR + Knowledge Insulation" 拆成 4 个 ablation, 直接照搬就能给 dexbotic 的 dm0 模块补上 L_AR |
| 🥈 **2** | **π₀-FAST + openpi** ([code](https://github.com/physical-intelligence/openpi)) | dexbotic 的 `pi05` 模块就是它的 PyTorch 版, 跟 DM0 共享同一基础设施, 改动量最小 |
| 🥉 **3** | **UniVLA (BAAI)** ([code](https://github.com/baaivision/UniVLA)) | 把 DM0 论文 3.3 节 "4 项辅助任务全走 L_AR" 真正端到端实现的开源样本, ICLR 2026 接收, 性能也最强 |

需要我把其中任何一份 (比如 ActionCodec 的 AR 头, 或 openpi 的 FAST tokenizer 调用链) 详细对照到 dexbotic 的 DM0 代码里, 给出"如何最小改动补上 L_AR"的具体行号补丁草案吗?

