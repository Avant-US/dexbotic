# "Text-as-Everything" 在 dexbotic 代码库中的跨架构实现全景与 DM0 落地方案

> **定位**: 本文与 [dm0_1.md](./dm0_1.md) (审计视角: 论文承诺 vs 代码缺失) 和 [dm0_lAr.md](./dm0_lAr.md) (L_AR 单点深挖) 形成三角互补. 本文的核心视角是 **正向工程**: 现有代码已经做了什么、怎么做的、DM0 如何复用.
> 代码库: `d:/SRC/Robot/dexbotic/` (Dexmal 官方开源)
> 论文: DM0: An Embodied-Native Vision-Language-Action Model towards Physical AI

---

## 0. TL;DR 概要

DM0 论文的 "text-as-everything" 核心思想: **所有辅助监督 (子任务描述 / 目标 bbox / EEF 轨迹 / 离散动作) 都通过 VLM 的自回归文本预测 (L_AR) 来实现**, 而不是设计专门的 head. 当前 DM0 开源代码只有 `L_FM` (Flow Matching MSE), 但:

1. **完整的 L_AR 基础设施已经散布在 6+ 个兄弟架构中**, 从数据管线到模型 forward 到训练日志, 每一环都有现成范本可以直接复用.
2. **DM0 自身的 `prefix_out` 已经被 `_merged_attention_forward` 计算并返回** (`dm0_arch.py:486`), 只是从未被传入 `lm_head` — 加 L_AR 的代码改动从 "重新设计" 降级为 "接一根线".
3. **三层落地方案**:
   - **方案一** (15 行改动): 在 forward 末尾加 `self.lm_head(prefix_out)` → `cross_entropy` → `loss = action_loss + text_loss`. 一天完成.
   - **方案二** (~150 行改动): 在方案一基础上加离散 action token 词表 + 双管线数据. 一周完成.
   - **方案三** (~400+ 行 + 新文件): 完整复刻论文 §2.3 的 4 层级 Scaffolding + Knowledge Insulation. 4-8 周完成.

本文不重复已有文档的审计结论, 聚焦于 **跨架构机制追踪** (第 2-3 章) 、**DM0 merged attention 梯度流新分析** (第 4 章) 和 **三层具体实现方案** (第 5 章).

---

## 1. "Text-as-Everything" 设计哲学的技术解构

### 1.1 论文核心洞察

论文 §2.3 的 Embodied Spatial Scaffolding 不是"用文本描述动作", 而是一种更深层的设计意图:

> **让 VLM backbone 在生成辅助文本的过程中 internalize 空间理解**.

具体地:
- **Subtask prediction** 迫使 VLM 理解任务分解和时序逻辑
- **Goal bbox prediction** 迫使 VLM 建立 "语言描述 → 图像区域" 的 grounding 能力
- **EEF trajectory prediction** 迫使 VLM 理解 "图像空间中的运动规划"
- **Discrete action token prediction** 迫使 VLM 在词表空间编码低级控制信号

这 4 项辅助任务本身可能对最终 action 质量贡献有限, 但它们对 VLM backbone 的 **表征学习** 贡献显著 — 论文的核心假设是: 一个能用文本 "想清楚" 这些空间信息的 VLM, 给 Action Expert 提供的 prefix hidden states 也会更 informative.

### 1.2 为什么不是"简单加个 CE loss"

直觉上, "text-as-everything" 的实现似乎只需要在 forward 末尾加一行 `cross_entropy` 就够了. 但实际工程中有 5 个层面的问题需要解决:

1. **注意力掩码的方向性**: DM0 的 cumsum 机制 (`dm0_utils.py:37-38`) 保证 **prefix 不能 attend 到 suffix**. 这意味着 `prefix_out` 是纯粹的 VLM 编码, 不含任何 action 信息泄漏 — 这是 L_AR 能独立监督 VLM 的前提. 但反过来也意味着 L_AR 的文本预测完全看不到 action context.

2. **Merged attention 的梯度路径**: VLM 和 Action Expert 在每一层共享注意力头 (Q/K/V concat → 共同 softmax). L_FM 的梯度会通过共享注意力权重回传到 VLM 的 Q/K/V 投影 — 即使不加 L_AR, VLM 也在被 L_FM 间接训练. 加上 L_AR 后, 两条梯度在 VLM 的注意力参数处汇合, 需要考虑梯度冲突.

3. **DM0Tokenization 弹空 assistant turn**: `process.py:407-414` 会主动删除空的尾部 assistant turn. 当前 DM0 的 SFT 数据中 assistant turn 都是空的 (因为 DM0 不做文本预测), 所以 labels 全是 `IGNORE_INDEX`. 加 L_AR 前必须先修复数据层, 否则 CE loss 梯度恒零.

4. **混合 batch 的数值稳定性**: 如果 batch 中既有纯 VL 数据 (只有 text, 没有 action) 又有具身数据 (有 action), 需要 `has_text` / `has_action` per-sample 掩码来避免对空维度求均值时的 NaN/Inf. HybridPi05 和 HybridCogACT 已有这套机制.

5. **L_AR 与 L_FM 的梯度量级差异**: CE loss 在词表维度 (vocab_size ≈ 151k) 上的梯度量级 vs MSE loss 在动作维度 (action_dim = 32) 上的梯度量级可能相差数个数量级. 论文的 `λ=1` 可能需要在实际训练中微调.

### 1.3 本文组织方式

- **第 2 章**: 横扫整个代码库的 "文本监督基础设施" — 7 种 Tokenization、5 种对话模板、动作量化管线、特殊 token 注册、Collator、Trainer
- **第 3 章**: 提炼 6 种 L_AR 实现 **模式** (而非逐架构列举), 每种模式给出可复用的代码路径
- **第 4 章**: DM0 merged attention 的梯度流拓扑分析 (新分析, 不在已有文档中)
- **第 5 章**: 三层落地方案, 具体到文件、行号、diff 代码 (最详细的部分)
- **第 6-7 章**: 风险缓解 + 实验路线图

---

## 2. 跨架构的文本监督基础设施全图

"Text-as-everything" 的完整实现链条跨越 6 个层次. 本章逐层扫描整个代码库, 标记 DM0 在每一层的状态 (✅ 已就绪 / ⚠️ 需微调 / ❌ 需新增).

### 2.1 令牌化管线: 7 种 Tokenization 类的 loss_mask 生成

代码库包含 7 种 Tokenization 实现 (`dexbotic/tokenization/process.py`), 它们在 `loss_mask` 生成逻辑上有关键差异:

| Tokenization 类 | 行号 | loss_mask 策略 | 返回字段 | 使用者 |
|---|---|---|---|---|
| `LLMTokenization` | 79-91 | 通过 `tokenize_dexbotic()` 标准处理 | `input_ids, labels` | DiscreteVLA, CogACT, OFT 等 |
| `NaVILATokenization` | 94-130 | 直接 mask human 部分 | `input_ids, labels` | NaVILA |
| `UniNaVidTokenization` | 133-298 | vicuna 模板 + 视频特殊 token | `input_ids, labels, prompt` | UniNaVid |
| `Pi0Tokenization` | 301-314 | SentencePiece 编码 | `input_ids, labels` | Pi0 |
| `Pi05Tokenization` | 317-365 | 多轮对话 + 角色前缀 | `input_ids, labels` | Pi0.5, HybridPi05 |
| **`DM0Tokenization`** | **368-483** | **per-token loss_mask (assistant=True)** | **`input_ids, labels, token_mask, ar_mask, loss_mask`** | **DM0** |
| `GR00TN1Tokenization` | 485-554 | 全 IGNORE_INDEX (不做文本监督) | `input_ids, labels` | GR00TN1 |

**DM0Tokenization 的特殊之处**:

1. **返回 5 个字段** (其他类只返回 2-3 个), 其中 `ar_mask` 和 `loss_mask` 是为 merged attention 架构预留的接口:
   - `token_mask`: 标记哪些位置是真实 token (非 padding)
   - `ar_mask`: 标记哪些位置参与自回归 (当前全为 0, 即不参与 AR — 但字段已预留)
   - `loss_mask`: 标记哪些位置计算 loss (只有 assistant 内容为 True)

2. **labels 生成** (`process.py:474-475`):
   ```python
   input_ids = np.asarray(tokens)
   labels = np.where(np.asarray(loss_mask), input_ids, IGNORE_INDEX)
   ```
   这意味着 `labels` 字段已经按 transformers 协议出厂, 可以直接传入 `F.cross_entropy(logits, labels, ignore_index=-100)`.

3. **空 assistant turn 弹出** (`process.py:407-414`):
   ```python
   if (conversations and conversations[-1].get("from") == "gpt"
       and not conversations[-1].get("value")):
       conversations.pop()
   ```
   这导致当前 SFT 数据的 labels 全为 `IGNORE_INDEX`. **方案一需要注释或条件化此段**.

**DM0 状态**: ⚠️ 字段齐全, labels 协议正确, 但数据内容为空.

### 2.2 对话模板层: 5 种 Conversation 模板

`dexbotic/tokenization/conversation.py` 第 197-270 行注册了 5 种对话模板:

| 模板名 | 角色格式 | sep | sep2 | 使用者 |
|---|---|---|---|---|
| `dexbotic` | USER / ASSISTANT | ` ` | `<\|endoftext\|>` | 默认 |
| **`step`** | **USER / ASSISTANT** | **` `** | **`<\|im_end\|>`** | **DM0** |
| `llama_3` | Llama-3 格式 | `<\|eot_id\|>` | `<\|end_of_text\|>` | Llama 系列 |
| `qwen2-chat` | ChatML `<\|im_start\|>user` | `<\|im_end\|>` | `<\|im_end\|>` | Qwen 系列 |
| `vicuna` | USER / ASSISTANT | ` ` | `</s>` | Vicuna, UniNaVid |

所有模板都是通用的 USER/ASSISTANT 格式, **没有结构化的辅助输出标签** (如 `<subtask>`, `<bbox>`, `<traj>`). 但模板系统本身有充分的扩展能力:
- `conv_templates` 是一个全局 dict, 新模板可通过 `register_conv_template()` 注册
- 每个模板的 `messages` 字段支持多轮对话, assistant turn 的 `value` 可以是任意文本

**DM0 使用 "step" 模板** (`dm0_exp.py:294`), sep2 为 Qwen3 的 `<|im_end|>` token.

**DM0 状态**: ✅ 模板系统可扩展, 无需结构性改动.

### 2.3 动作量化管线: ActionNormAnd2String vs ActionNorm

这是 "text-as-everything" 数据侧的 **核心分叉点**:

```
                      ┌── ActionNorm ──────────────── 连续 [-1,1] ──→ Flow Matching
连续动作 action[T,D] ─┤
                      └── ActionNormAnd2String ──── 离散文本 " 128 64 255 ..." ──→ L_AR
```

**ActionNormAnd2String** (`action.py:281-398`) 是一个三步管线:

```python
# Step 1: 归一化到 [-1, 1]  (line 379-385)
action = np.clip(action, min, max)
action = (action - min) / (max - min + 1e-8) * 2 - 1

# Step 2: 量化到 [0, vocab_size-1]  (line 387-391)
action = np.round((action + 1) / 2 * (vocab_size - 1))  # vocab_size=255
action = np.clip(action, 0, vocab_size - 1)

# Step 3: 格式化为字符串  (line 393-398)
action_str = [''.join([string_format.format(value=int(_)) for _ in action[i]])
              for i in range(len(action))]
# 输出: [" 128 64 255 12 198 45 127", " 129 65 254 13 197 46 126", ...]
```

关键设计:
- `vocab_size=255` (默认) 对应 255 个 bin, 每个 bin 宽度 ≈ 0.0079
- `string_format=' {value}'` (默认) 用空格分隔, 每个时间步的 D 维动作拼成一个字符串
- 输出写入 `episode_data_dict['answer']`, 后续被 `ToConversation` transform 放入对话的 assistant turn

**ActionNorm** (`action.py:229-278`) 只做 Step 1 (归一化), 不做 Step 2-3. **DM0 使用 ActionNorm**, 因此动作只作为连续浮点数进入 Flow Matching, 不生成文本.

关键事实: **两个类共享 `_norm_action` 的归一化逻辑** (line 379-385 vs line 240-253 对比, 数学一致). 这保证了连续动作和离散 token 的归一化空间是对齐的 — 如果未来要做方案二 (同时输出连续 + 离散), 两个管线的归一化不会产生不一致.

**DM0 状态**: ❌ 使用 ActionNorm 而非 ActionNormAnd2String. 方案二需要新建双输出 transform.

### 2.4 特殊 Token 注册: add_special_tokens

`base_exp.py:355-367` 的 `TokenizerConfig.add_special_tokens` 是词表扩展的统一入口:

```python
def add_special_tokens(self, special_token_format, vocab_size, tokenizer, model):
    if not self.use_special_tokens:  # 默认 False → 不注册
        return tokenizer
    special_tokens = [special_token_format.format(i) for i in range(vocab_size)]
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer
```

**调用链**: `BaseExp._initialize_train()` 调用此方法, 传入 `ActionConfig.string_format` (如 `' {value}'` 或 `'<act_{value}>'`) 和 `ActionConfig.vocab_size` (如 255).

默认 `use_special_tokens=False` 意味着大多数模型 **不注册特殊 action token**, 而是让 tokenizer 把 `" 128 64 255"` 这样的字符串按普通文本分词 (每个数字可能被拆成多个 subword token).

如果设为 `True` 且 `string_format='<act_{value}>'`, 则会注册 `<act_0>` 到 `<act_254>` 共 255 个单 token, 每个动作维度精确对应一个 token — 这是论文 "离散 action token" 路径的严格实现.

**DM0 状态**: ❌ `DM0TokenizerConfig` 继承默认 `use_special_tokens=False`. 方案二需要改为 True.

### 2.5 Collator 层: has_text / has_action 映射键

`dexbotic/data/collator.py:49-57` 的 `DataCollatorForSupervisedDataset` 已经包含了 `has_text` 和 `has_action` 的映射:

```python
mapping_keys = {
    'image': 'images',
    'actions': 'actions',
    'action': 'actions',
    'state': 'states',
    'reward': 'reward',
    'image_masks': 'image_masks',
    'has_action': 'has_action',   # ← 已就绪
    'has_text': 'has_text',       # ← 已就绪
}
```

只要 dataset 的每个 instance 包含 `has_action` 和 `has_text` 字段, Collator 就会自动 `torch.stack` 它们并传入 model forward.

**DM0 状态**: ✅ Collator 已支持. 但 `DM0DataConfig.data_keys` (`dm0_exp.py:272-279`) 不包含 `has_text` / `has_action`, 需要方案二中添加.

### 2.6 Trainer 层: *_loss 自动发现与日志

`dexbotic/exp/trainer.py:170-186` 的 `DexboticTrainer` 有一个优雅的自动发现机制:

```python
def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
    loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
    loss_keys = [_ for _ in outputs if _.endswith("_loss")]  # 自动发现所有 *_loss 字段
    for loss_key in loss_keys:
        if outputs[loss_key] is None or torch.isclose(...):
            if loss_key not in self.loss_cache:
                self.loss_cache[loss_key] = 0.0
            continue
        self.loss_cache[loss_key] = outputs[loss_key].detach().item()  # 缓存用于日志
    return (loss, outputs) if return_outputs else loss

def log(self, logs, start_time=None):
    logs.update(self.loss_cache)  # 自动合并到 wandb/tensorboard 日志
    super().log(logs, start_time)
```

这意味着: **只要 model forward 返回的 `CausalLMOutputDexbotic` 中填了 `text_loss` 和 `action_loss`, Trainer 就会自动把它们分开记录到训练日志中**, 无需任何 Trainer 侧改动.

`CausalLMOutputDexbotic` (`dexbotic_arch.py:26-34`) 已经定义了这两个字段:

```python
@dataclass
class CausalLMOutputDexbotic(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[...] = None
    hidden_states: Optional[...] = None
    attentions: Optional[...] = None
    text_loss: Optional[torch.FloatTensor] = None    # ← 已定义
    action_loss: Optional[torch.FloatTensor] = None   # ← 已定义
```

当前 DM0 的 forward 只填了 `loss=action_loss`, 没有分别填 `text_loss` 和 `action_loss`.

**DM0 状态**: ✅ Trainer 和输出结构完全就绪, 只需 forward 中分别填充.

### 2.7 基础设施状态总表

| 层次 | 组件 | DM0 状态 | 改动需求 |
|---|---|---|---|
| 令牌化 | DM0Tokenization + labels + loss_mask | ⚠️ 字段齐全但数据为空 | 方案一: 条件不弹空 assistant |
| 对话模板 | "step" 模板 (USER/ASSISTANT) | ✅ 可扩展 | 方案三: 加结构化辅助模板 |
| 动作量化 | ActionNormAnd2String (255-bin) | ❌ DM0 用 ActionNorm | 方案二: 新建双输出 transform |
| 词表扩展 | add_special_tokens | ❌ DM0 默认 False | 方案二: 改为 True |
| Collator | has_text / has_action 映射 | ✅ 已支持 | 方案二: data_keys 中添加 |
| Trainer | *_loss 自动发现 + 日志 | ✅ 完全就绪 | 无需改动 |

---

## 3. 跨架构 L_AR 实现模式图谱

本章不按架构逐个列举 (那是 [dm0_lAr.md](./dm0_lAr.md) 的做法), 而是提炼 **6 种可复用的 L_AR 实现模式**, 每种模式给出关键代码路径和适用场景.

### 3.1 模式 A: 基类继承 — 零行改动获得标准 L_AR

**代表**: DexboticForCausalLM (基类), DiscreteVLA

**核心代码** (`dexbotic_arch.py:482-488`):
```python
hidden_states = outputs.hidden_states[-1]
logits = self.lm_head(hidden_states)
loss = None
if labels is not None:
    loss = self.loss_function(logits, labels, self.model.backbone.vocab_size)
```

`self.loss_function` 是 HuggingFace `PreTrainedModel` 的标准接口, 内部调用 `torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)`.

**DiscreteVLA 如何复用**: 完全不重写 forward. 训练时继承基类的 CE loss, 推理时用 `self.generate()` 自回归生成文本 → `re.findall(r'\d+', ...)` 提取数字 → `(actions / (vocab_size - 1)) * 2 - 1` 反量化回连续动作 (`discrete_vla_arch.py:52-58`).

**优势**: 零代码改动, "文本即一切" 的最纯粹体现.
**限制**: 不支持双 loss (无法同时做 L_AR + L_FM), 不支持 per-sample `has_text` 掩码.
**DM0 适用性**: 不直接适用 — DM0 重写了 forward, 且需要同时保留 L_FM.

### 3.2 模式 B: 并行分类 — lm_head 做 K-bin 分类

**代表**: OFT-Discrete (`oft_discrete_arch.py:20-204`)

**核心代码** (`oft_discrete_arch.py:168, 187-191`):
```python
# 把 action hidden states 投影到 vocab 维度
predicted_actions = self.lm_head(action_hidden_states)  # [B, chunk*dim, vocab_size]

# 对每个动作维度做 K-bin 分类 (K=256)
predicted_actions_flat = predicted_actions.reshape(-1, predicted_actions.size(-1))
discrete_action_labels_flat = discrete_action_labels.reshape(-1)
loss = nn.functional.cross_entropy(predicted_actions_flat, discrete_action_labels_flat, reduction="mean")
```

**核心区别**: OFT-Discrete 不是自回归的. 它在 LLM 输出的 action 位置插入 placeholder embedding, 经过 Transformer 后用 lm_head 并行预测所有动作维度的离散 bin. 推理用 argmax 而非自回归生成 (`oft_discrete_arch.py:222-223`):
```python
discrete_action_indices = torch.argmax(predicted_logits[:, :, -self.config.num_bins + 1:], dim=-1)
```

**优势**: 推理速度 O(1), 不依赖自回归解码.
**限制**: 无法建模动作维度之间的条件依赖; 必须共用 lm_head (词表包含 text token + action bin).
**DM0 适用性**: 如果 DM0 想加离散 action 预测但不想做自回归, 这是最直接的参考.

### 3.3 模式 C: 双 Loss + Merged Attention — ★最接近论文★

**代表**: HybridPi05 (`hybrid_pi05_arch.py:339-533`)

这是 dexbotic 代码库中 **唯一同时实现 `L_AR` + `L_FM` 并通过 merged attention 融合 VLM 和 Action Expert** 的架构, 与 DM0 论文的 `L_total = L_AR + L_FM` 公式一一对应.

**完整 loss 计算路径** (3 个阶段):

**阶段 1 — text_loss (L_AR)** (`hybrid_pi05_arch.py:455-479`):
```python
# 从 VLM 侧的输出计算文本 logits
text_logits = self.lm_head(prefix_out)                    # line 455

# Per-token CE loss + IGNORE_INDEX 掩码
target_tokens = labels[:, 1:]                              # 标准 shift
text_len = input_ids.shape[1]
pred_tokens = text_logits[:, -text_len:-1]                 # 对齐 prefix 长度
token_loss = F.cross_entropy(pred_tokens.transpose(1, 2),
                             target_tokens, reduction="none")  # line 462-463
token_mask = torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)  # line 465

# Per-sample 聚合 + has_text 掩码
sample_loss = (token_loss * token_mask).sum(dim=-1) / torch.clamp(
    token_mask.sum(dim=-1), min=1.0)                       # line 466-468
has_text_mask = has_text.reshape(-1).to(sample_loss.device).float()
text_loss = (sample_loss * has_text_mask).sum() / (has_text_mask.sum() + 1e-6)
```

**阶段 2 — action_loss (L_FM)** (`hybrid_pi05_arch.py:481-504`):
```python
action_logits = self.model.action_out_proj(
    suffix_out[:, -self.model.config.chunk_size:])          # line 484-485
per_sample_action_loss = F.mse_loss(
    action_logits, u_t, reduction="none").mean(dim=[1, 2])  # line 487-489
has_action_mask = has_action.reshape(-1).to(...).float()
action_loss = (per_sample_action_loss * has_action_mask).sum() / (
    has_action_mask.sum() + 1e-6)                           # line 502-503
```

**阶段 3 — loss 组合** (`hybrid_pi05_arch.py:506-512`):
```python
loss = None
if text_loss is not None and action_loss is not None:
    loss = text_loss + action_loss    # λ=1, 隐式
elif text_loss is not None:
    loss = text_loss
elif action_loss is not None:
    loss = action_loss
```

**DM0 适用性**: ★★★★★ — DM0 和 HybridPi05 共享 merged attention 架构 (QKV concat → 共享 softmax → 分离 FFN). `prefix_out` 的获取方式完全一致. DM0 的方案一/二可以直接照搬此模式的 loss 计算逻辑.

### 3.4 模式 D: 双 Loss + 分离架构

**代表**: HybridCogACT (`hybrid_cogact_arch.py:52-203`)

**核心差异**: VLM 和 Action Head 是 **分离的** — 不共享 Transformer 层. Action Head 只接收 VLM 最后一个非 padding 位置的 hidden state (`cognition_features`):

```python
# 提取 cognition features  (line 146-157)
cumulative_sum = attention_mask.cumsum(dim=1)
last_unmask_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]
                      ).float().argmax(dim=1)
cognition_features = last_hidden_state.gather(1, expanded_indices.unsqueeze(1))

# text_loss 用基类的 loss_function  (line 129-141)
text_labels = labels.clone()
has_text = has_text.bool().view(-1)
if ~has_text.any():
    text_labels[~has_text] = IGNORE_INDEX
text_loss = self.loss_function(logits, text_labels, vocab_size) * has_text.any().float()

# action_loss 用 diffusion head  (line 174-180)
action_loss = self.model.action_head_module.loss(
    actions_repeated, cognition_features_repeated, reduction="none")
action_loss = (action_loss.mean(dim=[1,2]) * has_action_repeated).sum() / (...)
```

**DM0 适用性**: ★★★ — loss 组合逻辑可复用, 但 cognition features 提取方式不适用于 DM0 (DM0 用 merged attention, 不用分离架构).

### 3.5 模式 E: 软标签变体 — 相邻 bin 高斯 soft target

**代表**: NaVILA (`navila/loss.py:11-70`)

**核心思想**: 当 action token 被量化到离散 bin 时, 相邻 bin 之间有连续的距离关系 (bin 127 和 bin 128 在物理空间上几乎相同). 标准 hard CE 会把 "预测 127 但 label 是 128" 视为完全错误, 造成不必要的梯度震荡. soft CE 用高斯分布为相邻 bin 分配概率:

```python
# 对 soft token 位置构造高斯分布  (loss.py:64-67)
dist = torch.exp(-((target - soft_tokens) ** 2) / (2 * std**2))
targets_indices[k][soft_tokens] = dist / dist.sum()  # 归一化

# 对非 soft token 位置用标准 hard CE  (loss.py:53-56)
loss = cross_entropy(outputs[indices], targets[indices], reduction="sum")
```

`soft_tokens` 是一个 token ID 列表, 指定哪些 token 走软标签 (如 `self.config.time_token_ids`).

**DM0 适用性**: ★★ — 如果方案二中用了离散 action token, 可以把 255 个 action token ID 作为 `soft_tokens` 传入此函数, 减少 bin 边界处的梯度震荡. 但方案一不需要.

### 3.6 模式 F: 纯 L_FM — DM0 的当前状态

**代表**: DM0 (`dm0_arch.py:406-511`)

```python
# 唯一的 loss  (line 500-502)
action_loss = F.mse_loss(v_t, u_t, reduction="mean")
loss = action_loss
```

`prefix_out` 在 line 486 已经被计算并解包, 但之后完全不使用. 这是本文三层方案的出发点.

### 3.7 六种模式统一对照表

| 维度 | 模式 A (基类) | 模式 B (并行分类) | 模式 C (双Loss+MA) | 模式 D (双Loss+分离) | 模式 E (软标签) | 模式 F (DM0) |
|---|---|---|---|---|---|---|
| L_AR 调用 | `self.loss_function` | `F.cross_entropy` (K-bin) | `F.cross_entropy` (per-token) | `self.loss_function` | `soft_cross_entropy` | **无** |
| L_FM 调用 | 无 | 无 | `F.mse_loss` | diffusion `.loss()` | 无 | `F.mse_loss` |
| per-sample 掩码 | 无 | 无 | `has_text` + `has_action` | `has_text` + `has_action` | 无 | 无 |
| IGNORE_INDEX | 自动 (HF) | 手动 (label 构造) | 手动 (token_mask) | 自动 (label clone) | 自动 (传参) | N/A |
| reduction | mean (HF 默认) | mean | none → 手动 | mean (HF) + none | sum / count | mean |
| λ 权重 | N/A | N/A | 隐式 1 (相加) | 隐式 1 (相加) | N/A | N/A |
| Merged Attention | 否 | 否 | **是** | 否 | 否 | **是** |
| DM0 复用难度 | 低 (但不支持双 loss) | 中 | **极低** (架构相同) | 中 (需适配) | 低 (工具函数) | — |

---

## 4. DM0 Merged Attention 的梯度流拓扑分析

本章是新分析, 不在已有文档 (dm0_1.md, dm0_lAr.md) 中. 理解梯度流拓扑是判断 "加 L_AR 后会不会把 VLM 搞坏" 的关键.

### 4.1 _compute_merged_layer: 共享注意力 + 分离 FFN

`dm0_arch.py:145-268` 的 `_compute_merged_layer` 是 DM0 merged attention 的核心计算单元. 每层 Transformer 的计算流程:

```
VLM 侧 (module_list[0] = self.model.llm):
  prenorm_embeds_vlm = layer_vlm.input_layernorm(input_embeds_vlm)        # line 166
  Q_vlm = layer_vlm.self_attn.q_norm(layer_vlm.self_attn.q_proj(...))    # line 173-177
  K_vlm = layer_vlm.self_attn.k_norm(layer_vlm.self_attn.k_proj(...))    # line 178-182
  V_vlm = layer_vlm.self_attn.v_proj(...)                                # line 183-187

Action Expert 侧 (module_list[1] = self.model.action_expert.model):
  prenorm_embeds_ae = layer_ae.input_layernorm(input_embeds_ae)
  Q_ae = layer_ae.self_attn.q_norm(layer_ae.self_attn.q_proj(...))
  K_ae = layer_ae.self_attn.k_norm(layer_ae.self_attn.k_proj(...))
  V_ae = layer_ae.self_attn.v_proj(...)

拼接 QKV (按 head 维度拼接):                                              # line 197-199
  Q = torch.cat([Q_vlm, Q_ae], dim=2)   # [B, num_heads_vlm+num_heads_ae, S, head_dim]
  K = torch.cat([K_vlm, K_ae], dim=2)
  V = torch.cat([V_vlm, V_ae], dim=2)

共享注意力计算:                                                            # line 233-240
  attn_output = eager_attention_forward(Q, K, V, attention_mask)
  attn_output = attn_output.view(B, total_seq_len, hidden_dim)

分离后处理:                                                               # line 246-268
  对每个 module:
    attn_embeds = attn_output[:, start:start+seq_len, :]                  # 按 seq_len 切片
    attn_embeds = layer.self_attn.o_proj(attn_embeds)                     # 各自的 o_proj
    residual = input_embeds + attn_embeds                                 # 各自的残差连接
    postnorm = layer.post_attention_layernorm(residual)                   # 各自的 layernorm
    mlp_out = layer.mlp(postnorm)                                         # 各自的 FFN
    output = residual + mlp_out                                           # 各自的残差
```

**关键架构特征**:
- **共享**: softmax 运算, RoPE 位置编码
- **分离**: input_layernorm, Q/K/V 投影, o_proj, post_attention_layernorm, MLP/FFN, 残差连接
- 注意力头按 `dim=2` (head 维度) 拼接, 意味着 VLM 的 Q 会和 Action Expert 的 K 计算注意力分数 — 这是 "merged" 的精髓

### 4.2 cumsum 注意力掩码: 方向性保证

`dm0_utils.py:37-38` 的核心公式:
```python
cumsum = torch.cumsum(attn_mask, dim=1)
attn_mask_2d = cumsum[:, None, :] <= cumsum[:, :, None]
```

用一个具体例子说明. 假设:
- prefix 有 4 个 token, `attn_mask = [1, 0, 0, 0]` (第一个 token 开新段, 后续共享)
- suffix 有 3 个 token, `attn_mask = [1, 0, 0]` (第一个 token 开新段, 后续共享)

拼接后: `full_attn_mask = [1, 0, 0, 0, 1, 0, 0]`

cumsum: `[1, 1, 1, 1, 2, 2, 2]`

`cumsum[:, None, :] <= cumsum[:, :, None]` 生成 7×7 的注意力矩阵:

```
         col(key):   p0  p1  p2  p3  s0  s1  s2
                     (1)  (1)  (1)  (1)  (2)  (2)  (2)
row(query):
  p0 (1)              T    T    T    T    F    F    F
  p1 (1)              T    T    T    T    F    F    F
  p2 (1)              T    T    T    T    F    F    F
  p3 (1)              T    T    T    T    F    F    F
  s0 (2)              T    T    T    T    T    T    T
  s1 (2)              T    T    T    T    T    T    T
  s2 (2)              T    T    T    T    T    T    T
```

**解读**:
- **prefix ↔ prefix**: 全部互相可见 (cumsum 都是 1, 1≤1 为 True) — 双向注意力
- **suffix → prefix**: 全部可见 (cumsum 2 vs 1, 1≤2 为 True) — suffix 可以 attend 到所有 prefix
- **prefix → suffix**: 全部不可见 (cumsum 1 vs 2, 2≤1 为 False) — **prefix 看不到 suffix**
- **suffix ↔ suffix**: 全部互相可见 (cumsum 都是 2, 2≤2 为 True) — 双向注意力

这个方向性有两个重要推论:

1. **`prefix_out` 是纯粹的 VLM 编码**: prefix token 从未接触过任何 suffix (action) 信息, 因此 prefix_out 可以安全地用于 L_AR 文本预测, 不会出现 "action 信息泄漏到文本 logits" 的问题.

2. **suffix 对 prefix 的依赖是单向的**: suffix (action) 可以 attend 到 prefix (VLM 编码), 但反过来不行. 这意味着 VLM 给 Action Expert 的信息是 "只读" 的 — 完全符合论文 "VLM 提供 rich representation, Action Expert 基于此回归动作" 的设计.

> 注: 上面的例子中 prefix 内部是 **全连接** (非因果) 注意力. 实际 DM0 的 prefix 中还有图像 token 的 `ar_mask` 设置 (部分 token `ar_mask=0` 表示双向, 部分 `ar_mask=1` 表示因果), 但核心的 prefix→suffix 不可见性质不变.

### 4.3 梯度流推导: L_FM 已经在训练 VLM

当前 DM0 只有 `L_FM = MSE(v_t, u_t)`. 反向传播的梯度路径:

```
L_FM
  ↓
action_out_proj (∂L/∂W_out)
  ↓
suffix_out[:, -chunk_size:]
  ↓
_merged_attention_forward 反向:
  每一层反向传播:
    ↓ 反向通过 Action Expert 的 MLP + 残差
    ↓ 反向通过 Action Expert 的 o_proj
    ↓ 反向通过共享 softmax
    ↓ 在共享 softmax 的反向中:
       ├── 梯度流向 Q_ae, K_ae, V_ae → Action Expert 的 Q/K/V proj + layernorm
       └── 梯度流向 Q_vlm, K_vlm, V_vlm → ★VLM 的 Q/K/V proj + layernorm★
```

**关键发现**: 即使没有 L_AR, **VLM 的 Q/K/V 投影矩阵和 input_layernorm 已经在被 L_FM 的梯度更新**. 这是因为 merged attention 中 VLM 的 K/V 会被 suffix query 用到 (suffix→prefix 可见), 从而产生对 VLM K/V 投影的梯度.

不过, VLM 的 MLP/FFN 和 post_attention_layernorm **不受 L_FM 影响** — 因为它们在分离后处理阶段, 只接收 prefix 侧的 attn_output 切片, 而 prefix 侧的 loss 为零.

加上 L_AR 后:

```
L_AR
  ↓
lm_head (∂L/∂W_lm)
  ↓
prefix_out
  ↓
_merged_attention_forward 反向:
  每一层反向传播:
    ↓ 反向通过 VLM 的 MLP + 残差    ← ★新增梯度路径★
    ↓ 反向通过 VLM 的 o_proj
    ↓ 反向通过共享 softmax
    ↓ 梯度流向 Q_vlm, K_vlm, V_vlm → VLM 的 Q/K/V proj + layernorm
```

**两条梯度路径的汇合点**: VLM 的 Q/K/V 投影矩阵和 input_layernorm. 在这些参数上, L_FM 的梯度 (间接的, 通过 suffix query 的 attention) 和 L_AR 的梯度 (直接的, 通过 prefix output) 会相加.

**VLM MLP/FFN 的独占权**: L_AR 加入后, VLM 的 MLP/FFN 将**只被 L_AR 更新** (L_FM 的梯度无法到达此处). 这些参数约占 VLM 总参数的 2/3, 因此 L_AR 对 VLM 的训练控制力远大于 L_FM.

### 4.4 Knowledge Insulation: 三种可行方案

论文提到 "Action Expert 的梯度不回传到 VLM" (Knowledge Insulation). 代码中未显式实现 (没有 `.detach()` 或 `stop_gradient`). 如果要实现, 有三种方案:

**方案 KI-A: 在共享注意力反向中截断 (精确但侵入性大)**:

在 `_compute_merged_layer` 中, 对 VLM 侧参与 suffix attention 的 K/V 做 detach:

```python
# 在 line 197-199 之前:
if self.training and self.config.knowledge_insulation:
    # 只 detach VLM 的 K/V 被 suffix 使用的路径
    key_list[0] = key_list[0].detach()    # VLM 的 K
    value_list[0] = value_list[0].detach()  # VLM 的 V
```

效果: L_FM 的梯度被阻断在 VLM 的 K/V proj 处. VLM 的 Q proj 仍然可以被 L_FM 更新 (因为 VLM 的 Q 不被 suffix 使用 — prefix→suffix 不可见). 但这会阻断 VLM 的 attention 参数学习 "如何为 Action Expert 提供有用的 context".

**方案 KI-B: 分离学习率 (工程友好, 等效近似)**:

```python
# 在训练配置中:
optimizer_groups = [
    {"params": vlm_params, "lr": 1e-5},        # VLM 低学习率
    {"params": action_expert_params, "lr": 1e-4},  # Action Expert 高学习率
]
```

效果: 不阻断梯度, 但通过学习率差异让 L_FM 对 VLM 的影响变小. 这是工程上最简单的方案, 也是很多 VLA 训练的常见做法 (如 OpenVLA, Octo 等).

**方案 KI-C: has_text + has_action 掩码实现数据级隔离**:

HybridPi05 的 `has_text` / `has_action` 掩码 (`hybrid_pi05_arch.py:470-503`) 本质上实现了一种 "数据级别的梯度路由": 当一个样本的 `has_text=False` 时, 该样本的 text_loss 为零, VLM 不会被这个样本的 L_AR 更新; 当 `has_action=False` 时, 该样本的 action_loss 为零, Action Expert 不会被更新.

这不是真正的 "梯度阻断" (L_FM 仍然可以通过 shared attention 影响 VLM), 但在实践中, 如果大部分样本要么是纯 VL (只有 text) 要么是纯具身 (只有 action), 两种 loss 对 VLM 的影响就被自然分离了.

### 4.5 prefix_out 的零成本接口

回到 `dm0_arch.py:486`:

```python
(prefix_out, suffix_out), _ = self._merged_attention_forward(
    module_list=module_list,
    attention_mask=attn_mask,
    position_ids=positions,
    past_key_values=None,
    input_embeds_list=[prefix_hidden_states, suffix_hidden_states],
    use_cache=False,
)
```

`prefix_out` 已经被解包到局部变量中, 包含 VLM 经过全部 Transformer 层 + final layer norm 后的输出. **它的形状是 `[B, prefix_seq_len, hidden_size]`**, 与 `lm_head` 的输入要求完全匹配.

在 line 486 和 line 495 (开始计算 action_loss) 之间, `prefix_out` 是一个 **悬空的、已计算但未使用的张量**. 加 L_AR 只需要在这个间隙中插入 `text_logits = self.lm_head(prefix_out)`, 然后计算 CE loss. 这就是为什么方案一只需要 15 行改动的原因.

---

## 5. DM0 实现 Text-as-Everything 的三层方案

### 5.1 方案一: 最小改动 — 仅加 VLM 端 L_AR

**目标**: 让 DM0 的训练 forward 同时计算 `text_loss` (L_AR) 和 `action_loss` (L_FM), 返回 `loss = text_loss + action_loss`.

**改动文件清单**:

| 文件 | 改动描述 | 行数 |
|---|---|---|
| `dexbotic/model/dm0/dm0_arch.py` | forward 末尾加 text_loss 计算 + 返回值补充 | +12 行 |
| `dexbotic/tokenization/process.py` | DM0Tokenization 条件不弹空 assistant turn | ~2 行 |
| `dexbotic/constants.py` | (无需改动, IGNORE_INDEX=-100 已定义) | 0 |

#### 5.1.1 dm0_arch.py forward 改动 (diff 格式)

在 `dm0_arch.py` 文件头新增 import:

```python
# 在现有 import 区域添加:
from dexbotic.constants import IGNORE_INDEX
```

在 forward 方法中, 替换 line 495-511:

```python
        # ===== 原代码 (line 495-511) =====
        # Compute flow matching loss
        if actions.dtype == torch.float32:
            suffix_out = suffix_out.to(torch.float32)
        suffix_out_final = suffix_out[:, -self.model.config.chunk_size :]
        v_t = self.model.action_out_proj(suffix_out_final)
        action_loss = F.mse_loss(v_t, u_t, reduction="mean")

-       loss = action_loss

+       # Compute text loss (L_AR) on prefix output
+       text_loss = None
+       if labels is not None:
+           text_logits = self.lm_head(prefix_out)
+           shift_logits = text_logits[:, :-1, :].contiguous()
+           shift_labels = labels[:, 1:].contiguous()
+           text_loss = F.cross_entropy(
+               shift_logits.view(-1, self.config.llm_config.vocab_size),
+               shift_labels.view(-1),
+               ignore_index=IGNORE_INDEX,
+           )
+
+       # Combine losses: L_total = L_AR + L_FM  (λ=1)
+       loss = action_loss
+       if text_loss is not None:
+           loss = loss + text_loss

        outputs = CausalLMOutputDexbotic(
            loss=loss,
            logits=v_t,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
+           text_loss=text_loss,
+           action_loss=action_loss,
        )
        return outputs
```

**代码说明**:
- `self.lm_head` 已在 `_real_init` (line 140-142) 中创建, 权重初始化来自 Qwen3 的 LM head (或通过 `tie_weights` 共享 embedding)
- `shift_logits[:, :-1]` 和 `shift_labels[:, 1:]` 是标准的自回归 shift, 让 position i 预测 position i+1 的 token
- `ignore_index=IGNORE_INDEX` 自动跳过非 assistant 位置 (human prompt / system prompt / padding)
- `text_loss=text_loss, action_loss=action_loss` 让 Trainer 自动分别记录

#### 5.1.2 process.py 改动

**选项 A (推荐)**: 条件化空 assistant 弹出

```python
        # process.py line 407-414, 改为:
        conversations = list(conversations)
        # 只在 assistant 完全没有内容且没有 action 标签时才弹出
        # (保留空 assistant 以维持 label 张量对齐)
        # if (conversations and conversations[-1].get("from") == "gpt"
        #     and not conversations[-1].get("value")):
        #     conversations.pop()
```

**选项 B (更安全)**: 在数据侧为每个 assistant turn 填入任务描述 echo

在数据 transform pipeline 中, 在 `ToConversation` 之前或之后, 把空 assistant turn 的 `value` 设为任务 prompt 的 echo:

```python
# 伪代码 — 在 action pipeline 中添加:
class FillAssistantEcho:
    def __call__(self, episode_data_dict):
        if 'answer' not in episode_data_dict or not episode_data_dict['answer']:
            episode_data_dict['answer'] = [episode_data_dict['prompt'][0]] * len(...)
        return episode_data_dict
```

这样 assistant turn 的内容就是任务指令本身 (如 "pick up the blue cup"), labels 中 assistant 位置不再全是 IGNORE_INDEX, L_AR 就能产生有效梯度 — 迫使 VLM "复述" 任务指令, 这是最简单的文本监督.

#### 5.1.3 预期效果

- **训练日志**: wandb/tensorboard 中会同时出现 `text_loss` 和 `action_loss` 两条曲线
- **VLM 表征**: L_AR 会更新 VLM 的全部参数 (包括 MLP/FFN), 使 VLM 的 prefix_out 包含更丰富的语义信息
- **Action 质量**: 间接改善 — 更好的 prefix_out → suffix 通过 attention 获得更 informative 的 context → 更好的 action 预测
- **推理无变化**: 推理路径 (`inference_action`) 不使用 `lm_head`, 因此推理速度和行为不变

#### 5.1.4 风险评估

| 风险 | 严重度 | 缓解措施 |
|---|---|---|
| L_AR 梯度与 L_FM 梯度量级失衡 | 中 | 引入 `lambda_text` 权重系数: `loss = action_loss + lambda_text * text_loss` |
| prefix_out 形状与 labels 不对齐 | 低 | prefix_out 包含图像 token, labels 只有文本 token — shift 和 IGNORE_INDEX 已自动处理 |
| VLM 知识遗忘 (灾难性遗忘) | 低 (方案一风险最小) | "复述任务指令" 是最温和的 L_AR, 不会引入大量新 token 分布 |

#### 5.1.5 验证测试方案

1. **单元测试**: 构造一个带非空 assistant 的 mini-batch (2 个样本), forward 后验证:
   - `text_loss` 非零且有限
   - `action_loss` 与不加 L_AR 时的值相同 (因为 suffix 路径未改动)
   - `loss ≈ action_loss + text_loss`

2. **集成测试**: 在 LIBERO-Spatial 上训练 1000 步:
   - 对比 baseline (无 L_AR) 和 方案一 (有 L_AR) 的 loss 曲线
   - 确认 `text_loss` 单调下降 (VLM 学会了复述指令)
   - 确认 `action_loss` 没有显著恶化

3. **回归测试**: 传入 `labels=None` 时, 确认行为与原 forward 完全一致 (text_loss=None, loss=action_loss)

---

### 5.2 方案二: 中等改动 — L_AR + 离散 Action Token

**目标**: 在方案一基础上, 让 VLM 不仅做文本监督, 还要预测离散化的 action token. 这对应论文 §2.3 的 "Discrete action prediction" 辅助任务.

**额外改动文件清单** (在方案一基础上):

| 文件 | 改动描述 | 行数 |
|---|---|---|
| `dexbotic/exp/dm0_exp.py` | DM0TokenizerConfig.use_special_tokens = True + DM0DataConfig.data_keys 扩展 | +5 行 |
| `dexbotic/exp/dm0_exp.py` | DM0ActionConfig: 新建双输出 Pipeline | +30 行 |
| `dexbotic/data/dataset/transform/action.py` | 新增 `ActionNormDual` transform | +50 行 |
| `dexbotic/model/dm0/dm0_arch.py` | forward 加 has_text/has_action per-sample 掩码 | +25 行 |
| `dexbotic/data/dataset/transform/language.py` | 新增 `ToConversationWithAction` transform | +20 行 |

#### 5.2.1 词表扩展

```python
# dm0_exp.py, DM0TokenizerConfig:
@dataclass
class DM0TokenizerConfig(TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)
    use_special_tokens: bool = field(default=True)   # 改为 True

# DM0ActionConfig:
@dataclass
class DM0ActionConfig(ActionConfig):
    vocab_size: int = field(default=255)             # 255 个 action bin
    string_format: str = field(default='<act_{value}>')  # 注册为特殊 token
    ...
```

启用后, `base_exp.py:364-366` 会自动注册 `<act_0>` 到 `<act_254>` 并调用 `model.resize_token_embeddings(len(tokenizer))`.

#### 5.2.2 数据管线双输出 Transform

```python
# action.py 新增:
class ActionNormDual:
    """同时输出归一化连续动作 (给 FM) 和离散 token 字符串 (给 L_AR)."""

    def __init__(self, statistic_mapping, vocab_size=255, string_format='<act_{value}>'):
        self.vocab_size = vocab_size
        self.statistic_mapping = statistic_mapping
        self.string_format = string_format

    def __call__(self, episode_data_dict, **kwargs):
        if 'action' not in episode_data_dict:
            return episode_data_dict

        action = episode_data_dict['action']
        prompt = episode_data_dict['prompt'][0]
        dataset = episode_data_dict['meta_data']['dataset']

        statistic_mapping = self._get_stats(dataset, prompt)

        # 归一化到 [-1, 1] (与 ActionNorm 相同的逻辑)
        normalized = self._norm_action(action, statistic_mapping['min'], statistic_mapping['max'])
        episode_data_dict['action'] = normalized  # 连续值 → Flow Matching

        # 量化 + 字符串化 (与 ActionNormAnd2String 相同的逻辑)
        binned = np.round((normalized + 1) / 2 * (self.vocab_size - 1))
        binned = np.clip(binned, 0, self.vocab_size - 1)
        action_str = [''.join([self.string_format.format(value=int(v)) for v in binned[i]])
                      for i in range(len(binned))]
        episode_data_dict['answer'] = action_str  # 离散字符串 → L_AR

        return episode_data_dict
```

#### 5.2.3 forward 加 per-sample 掩码 (仿 HybridPi05)

在 `dm0_arch.py` forward 签名中添加 `has_text` 和 `has_action` 参数:

```python
    def forward(
        self,
        ...
        image_masks: Optional[torch.BoolTensor] = None,
+       has_text: Optional[torch.BoolTensor] = None,
+       has_action: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> CausalLMOutputDexbotic:
```

text_loss 计算改为:

```python
        text_loss = None
        if labels is not None:
            text_logits = self.lm_head(prefix_out)
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Per-token loss
            token_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.llm_config.vocab_size),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(shift_labels.shape)

            # Per-sample aggregation with has_text mask
            token_mask = (shift_labels != IGNORE_INDEX).float()
            sample_loss = (token_loss * token_mask).sum(dim=-1) / torch.clamp(
                token_mask.sum(dim=-1), min=1.0
            )

            if has_text is None:
                has_text_mask = torch.ones(batch_size, device=sample_loss.device)
            else:
                has_text_mask = has_text.reshape(-1).float()

            text_loss = (sample_loss * has_text_mask).sum() / (has_text_mask.sum() + 1e-6)
```

action_loss 计算也需要加 `has_action` 掩码:

```python
        # Per-sample action loss with has_action mask
        per_sample_action_loss = F.mse_loss(v_t, u_t, reduction="none").mean(dim=[1, 2])

        if has_action is None:
            has_action_mask = torch.ones(batch_size, device=per_sample_action_loss.device)
        else:
            has_action_mask = has_action.reshape(-1).float()

        action_loss = (per_sample_action_loss * has_action_mask).sum() / (
            has_action_mask.sum() + 1e-6
        )
```

#### 5.2.4 DM0DataConfig 扩展

```python
# dm0_exp.py:
@dataclass
class DM0DataConfig(DataConfig):
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids",
            "labels",
            "action",
            "image",
            "state",
            "image_masks",
+           "has_text",       # per-sample text 标志
+           "has_action",     # per-sample action 标志
        ]
    )
```

#### 5.2.5 预期效果

- **VLM 学习离散 action 语义**: L_AR 不再只是 "复述指令", 而是预测 `<act_128> <act_64> <act_255> ...` 这样的 action token 序列. VLM 被迫在词表空间中编码低级控制信号.
- **混合训练支持**: 通过 `has_text` / `has_action` 掩码, 可以混合纯 VL 数据 (只有 text) 和具身数据 (有 text + action). 这对应论文 §4.1 的 "VL 数据走 L_AR, 具身数据走 L_AR + L_FM" 混训策略.
- **推理无变化**: 推理仍然走 Flow Matching denoising, 不走自回归生成.

---

### 5.3 方案三: 完整复刻 — 4 层级 Scaffolding + Knowledge Insulation

**目标**: 完整实现论文 §2.3 的 Embodied Spatial Scaffolding + §3.4 的 Knowledge Insulation.

**额外改动清单** (在方案二基础上):

| 文件 | 改动描述 | 行数 |
|---|---|---|
| `dexbotic/data/dataset/transform/scaffolding.py` | **新建文件**: AddSubtaskText, AddGoalBBox, AddEEFTrace2D | +150 行 |
| `dexbotic/tokenization/conversation.py` | 注册 dm0_scaffolding 多段 assistant 模板 | +30 行 |
| `dexbotic/tokenization/process.py` | DM0Tokenization 支持多段 assistant turn | +30 行 |
| `dexbotic/model/dm0/dm0_arch.py` | _compute_merged_layer 加 Knowledge Insulation (KI-A 或 KI-B) | +10 行 |
| `dexbotic/exp/dm0_exp.py` | DM0ActionConfig Pipeline 加入新 Transform | +20 行 |
| 数据脚本 | bbox/trace 标注生成、模板增广 | +200 行 |

#### 5.3.1 新增数据 Transform 设计

```python
# scaffolding.py (新文件)

class AddSubtaskText:
    """从 episode 元数据提取子任务描述, 写入 assistant turn.

    论文示例: "pick up the blue dinosaur toy from the right bin
              and place it in the left bin"

    实现: 从 jsonl 的 extra.subtask 字段读取.
    """
    def __call__(self, episode_data_dict):
        subtask = episode_data_dict.get('meta_data', {}).get('subtask', '')
        if subtask:
            episode_data_dict['subtask_text'] = subtask
        return episode_data_dict


class AddGoalBBox:
    """将目标物体 3D 位置投影到主视角 2D, 归一化到 [0, 1000].

    论文格式: "[xmin, ymin, xmax, ymax]"

    需要: 相机内参 K, 相机外参 T, 目标物体 3D bbox.
    """
    def __init__(self, camera_intrinsics, image_size=(640, 480)):
        self.K = camera_intrinsics
        self.image_size = image_size

    def __call__(self, episode_data_dict):
        if 'goal_bbox_3d' not in episode_data_dict:
            return episode_data_dict
        # 3D → 2D 投影 + 归一化到 [0, 1000]
        bbox_3d = episode_data_dict['goal_bbox_3d']
        bbox_2d = self._project(bbox_3d)
        bbox_str = f"[{bbox_2d[0]}, {bbox_2d[1]}, {bbox_2d[2]}, {bbox_2d[3]}]"
        episode_data_dict['goal_bbox_text'] = bbox_str
        return episode_data_dict


class AddEEFTrace2D:
    """将 EEF 3D 轨迹投影到主视角 2D 屏幕坐标.

    论文格式: "(x,y) (x,y) (x,y) ..."

    需要: 相机内参 K, EEF 未来若干帧的 3D 位置.
    """
    def __init__(self, camera_intrinsics, num_waypoints=10, image_size=(640, 480)):
        self.K = camera_intrinsics
        self.num_waypoints = num_waypoints
        self.image_size = image_size

    def __call__(self, episode_data_dict):
        if 'eef_positions' not in episode_data_dict:
            return episode_data_dict
        eef_3d = episode_data_dict['eef_positions'][:self.num_waypoints]
        waypoints_2d = [self._project_point(p) for p in eef_3d]
        trace_str = ' '.join([f"({x},{y})" for x, y in waypoints_2d])
        episode_data_dict['eef_trace_text'] = trace_str
        return episode_data_dict
```

#### 5.3.2 多模板对话系统

```python
# conversation.py 新增:
conv_dm0_scaffolding = Conversation(
    system="You are a robot assistant that thinks step-by-step before acting.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|im_end|>",
)
register_conv_template(conv_dm0_scaffolding, override=False)

# assistant turn 的内容格式 (由数据 transform 填充):
# "<subtask>pick up the blue cup from the table</subtask>
#  <bbox>[342, 198, 456, 312]</bbox>
#  <traj>(0.32,0.54) (0.38,0.51) (0.45,0.48)</traj>
#  <act><act_128><act_64><act_255><act_12><act_198><act_45><act_127></act>"
```

#### 5.3.3 Knowledge Insulation 实现 (KI-A 方案)

```python
# dm0_arch.py, _compute_merged_layer 中 line 197 之前:

        # Knowledge Insulation: detach VLM's K/V from L_FM gradient path
        if self.training and getattr(self.config, 'knowledge_insulation', False):
            key_list[0] = key_list[0].detach()
            value_list[0] = value_list[0].detach()

        query_states = torch.cat(query_list, dim=2)
        key_states = torch.cat(key_list, dim=2)
        value_states = torch.cat(value_list, dim=2)
```

这两行 detach 会阻断 L_FM 通过 suffix attention → VLM K/V proj 的梯度回传. VLM 将只被 L_AR 更新 (通过 prefix_out → lm_head), 完全符合论文的 "Knowledge Insulation" 设计.

#### 5.3.4 数据策展需求

| 数据需求 | 来源 | 估算量 |
|---|---|---|
| 子任务文本 | 已有 (jsonl extra.subtask) | 直接复用 |
| 目标 bbox | 需要相机标定 + 目标检测 | 中等 (可用现有检测模型自动标注) |
| EEF 2D 轨迹 | 需要相机标定 + EEF 3D 位置 | 中等 (需要 FK/IK + 相机投影) |
| 离散 action token | 自动 (ActionNormDual 生成) | 零 |
| 语言增广 (500 模板) | 需要人工/LLM 生成 | 高 |

#### 5.3.5 工作量估算

| 阶段 | 内容 | 时间 |
|---|---|---|
| 1 | 数据 Transform + Tokenizer 词表 + 模板库 | 1-2 周 |
| 2 | 模型 loss + head + Knowledge Insulation | 1 周 |
| 3 | 数据策展 (bbox/trace 标注, 语言增广) | 2-4 周 |
| 4 | 训练超参 + 多机调试 + 收敛验证 | 1-2 周 |
| 5 | LIBERO / Table30 评测 | 1 周 |
| **合计** | | **4-8 周** |

---

### 5.4 三方案对比决策矩阵

| 维度 | 方案一 (最小) | 方案二 (中等) | 方案三 (完整) |
|---|---|---|---|
| **改动行数** | ~15 行 | ~150 行 | ~400+ 行 + 新文件 |
| **改动文件** | 2 个 | 5 个 | 8+ 个 |
| **新增数据** | 否 (echo 现有 prompt) | 否 (自动量化) | 是 (bbox, trace, 子任务) |
| **词表扩展** | 否 | 是 (255 action token) | 是 (255 + bbox + 特殊标签) |
| **论文覆盖** | `L_total = L_AR + L_FM` | + §2.3(iv) 离散 action | + §2.3 全部 + §3.4 KI |
| **VLM 受益** | 弱 (指令复述正则化) | 中 (动作语义编码) | 强 (空间理解内化) |
| **Action 质量提升** | 间接 (更好的 prefix_out) | 间接+直接 | 显著 |
| **推理变化** | 无 | 无 | 可选 (加文本推理输出) |
| **风险等级** | 低 | 中 | 高 |
| **工作量** | 1 天 | 1 周 | 4-8 周 |
| **推荐场景** | 快速验证 / PoC | Mid-train 复刻 | 完整论文复现 |

**推荐路径**: 方案一 → 消融验证 → 方案二 → 消融验证 → 根据结果决定是否需要方案三.

---

## 6. 实现风险与缓解策略

### 6.1 L_AR 与 L_FM 梯度量级差异

**风险**: CE loss 在 vocab_size ≈ 151k 维度上的梯度 vs MSE loss 在 action_dim = 32 维度上的梯度, 量级可能差 1-3 个数量级. 如果 L_AR 梯度远大于 L_FM, 会导致 Action Expert 训练不充分.

**缓解**:
- 引入 `lambda_text` 权重: `loss = action_loss + lambda_text * text_loss`, 初始 lambda_text=1.0 (论文值), 如果 text_loss >> action_loss 则下调到 0.1-0.5
- 监控 `text_loss / action_loss` 比值, 保持在 0.1-10 范围内
- 使用 gradient clipping (`max_grad_norm=1.0`, Trainer 默认已有)

### 6.2 VLM 知识遗忘 (灾难性遗忘)

**风险**: L_AR 如果引入大量新 token 分布 (如 255 个 `<act_X>` token), 可能覆盖 VLM 预训练获得的通用语言能力.

**缓解**:
- **方案一风险最低**: "复述指令" 不引入新 token, 只强化已有语言能力
- **方案二**: 使用 `has_text` 掩码, 对纯 VL 数据只走 L_AR (保持语言能力), 对具身数据走 L_AR + L_FM
- **方案三**: 混入 VL-only 数据 (论文 §4.2 的 mid-train 数据比例管理)
- 定期在 VQA benchmark 上评测 VLM 保留率

### 6.3 注意力掩码正确性

**风险**: 加 L_AR 后, 如果 prefix 长度因新增 assistant 文本而变长, position_ids 和 KV cache 需要对齐.

**缓解**:
- 方案一不改变 prefix 长度 (assistant 文本已经被 tokenize 进 input_ids, 只是 labels 从 IGNORE_INDEX 变成真实值)
- 方案二/三如果在 assistant turn 增加 action token, 需要确保 DM0Tokenization 的 `token_mask`, `ar_mask`, `loss_mask` 长度与 `input_ids` 对齐
- 用 assertion 验证: `assert len(token_mask) == len(input_ids) == len(labels)`

### 6.4 混合 batch 的数值稳定性

**风险**: `has_text_mask.sum()` 或 `has_action_mask.sum()` 为零时, 除法分母为零.

**缓解**: HybridPi05 已用 `+ 1e-6` 保护 (`hybrid_pi05_arch.py:478, 503`). DM0 方案二需要同样加入此保护.

---

## 7. 实验验证路线图

### 7.1 消融实验设计

| 实验组 | 配置 | 对应 |
|---|---|---|
| A (baseline) | DM0 原版, L_FM only | 当前代码 |
| B (方案一) | L_AR (echo prompt) + L_FM | 方案一改动 |
| C (方案二-basic) | L_AR (离散 action token) + L_FM | 方案二 (不含 has_text/has_action) |
| D (方案二-full) | L_AR + L_FM + has_text/has_action | 方案二完整 |
| E (方案三-partial) | L_AR + L_FM + KI | 方案三 (仅加 Knowledge Insulation) |

### 7.2 评测指标

| 指标 | 衡量什么 |
|---|---|
| LIBERO-Spatial 成功率 | 空间理解能力 (与 Scaffolding 直接相关) |
| LIBERO-Goal 成功率 | 目标理解能力 |
| `text_loss` 收敛曲线 | VLM 是否学到了文本监督信号 |
| `action_loss` 收敛曲线 | Action Expert 性能是否退化 |
| VQA benchmark F1 | VLM 通用语言能力保留率 |
| `text_loss / action_loss` 比值 | 两种 loss 的相对重要性 |

### 7.3 实施优先级

```
方案一 (1 天) → LIBERO-Spatial 评测 (1 天)
  ├── 如果 action_loss 无退化 → 方案二 (1 周) → 评测 (2 天)
  │     ├── 如果成功率提升显著 → 方案三 (4-8 周)
  │     └── 如果提升不大 → 停在方案二
  └── 如果 action_loss 退化 → 调 lambda_text → 重试方案一
```

---

## 附录 A: 跨架构代码引用索引

| 文件 | 行号 | 内容 |
|---|---|---|
| `dexbotic/model/dexbotic_arch.py` | 27-34 | `CausalLMOutputDexbotic` 定义 (含 text_loss, action_loss) |
| `dexbotic/model/dexbotic_arch.py` | 487-488 | 基类 L_AR: `self.loss_function(logits, labels, vocab_size)` |
| `dexbotic/model/dm0/dm0_arch.py` | 77-80 | Action Expert `embed_tokens = None` |
| `dexbotic/model/dm0/dm0_arch.py` | 140-142 | `lm_head` 创建 (仅 tie_weights) |
| `dexbotic/model/dm0/dm0_arch.py` | 145-268 | `_compute_merged_layer` (共享注意力 + 分离 FFN) |
| `dexbotic/model/dm0/dm0_arch.py` | 197-199 | Q/K/V 按 head 维度拼接 |
| `dexbotic/model/dm0/dm0_arch.py` | 233-240 | 共享 `eager_attention_forward` |
| `dexbotic/model/dm0/dm0_arch.py` | 254-266 | 按 seq_len 切片 → 分离 o_proj + FFN |
| `dexbotic/model/dm0/dm0_arch.py` | 270-298 | `_merged_attention_forward` (逐层循环 + final norm) |
| `dexbotic/model/dm0/dm0_arch.py` | 406-511 | 训练 forward (prefix_out 在 486 行获取但未使用) |
| `dexbotic/model/dm0/dm0_arch.py` | 486 | `(prefix_out, suffix_out), _ = self._merged_attention_forward(...)` |
| `dexbotic/model/dm0/dm0_arch.py` | 500 | `action_loss = F.mse_loss(v_t, u_t)` |
| `dexbotic/model/dm0/dm0_arch.py` | 502 | `loss = action_loss` (唯一 loss, 无 L_AR) |
| `dexbotic/model/dm0/dm0_utils.py` | 37-38 | cumsum 注意力掩码: `cumsum[:, None, :] <= cumsum[:, :, None]` |
| `dexbotic/model/dm0/dm0_prog_arch.py` | 93-95 | progress_in_proj / progress_out_proj (辅助预测参考) |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 455 | `text_logits = self.lm_head(prefix_out)` |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 462-463 | `F.cross_entropy(pred_tokens.transpose(1,2), target_tokens, reduction="none")` |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 465 | `token_mask = torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)` |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 470-478 | per-sample `has_text_mask` 加权 |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 484-503 | action_loss: `F.mse_loss` + `has_action_mask` |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 506-512 | `loss = text_loss + action_loss` (λ=1) |
| `dexbotic/model/cogact/hybrid_cogact_arch.py` | 129-141 | text_loss: `self.loss_function` + `has_text.any().float()` |
| `dexbotic/model/cogact/hybrid_cogact_arch.py` | 174-180 | action_loss: diffusion `.loss()` |
| `dexbotic/model/cogact/hybrid_cogact_arch.py` | 182-188 | loss 组合 (同 HybridPi05) |
| `dexbotic/model/oft/oft_discrete_arch.py` | 11-14 | `num_bins = 256` 配置 |
| `dexbotic/model/oft/oft_discrete_arch.py` | 168 | `predicted_actions = self.lm_head(action_hidden_states)` |
| `dexbotic/model/oft/oft_discrete_arch.py` | 187-191 | `F.cross_entropy` 并行分类 |
| `dexbotic/model/discrete_vla/discrete_vla_arch.py` | 34-41 | `self.generate()` 自回归推理 |
| `dexbotic/model/discrete_vla/discrete_vla_arch.py` | 52-58 | regex 提取 + 反量化 |
| `dexbotic/model/navila/loss.py` | 11-70 | `soft_cross_entropy` 高斯软标签 |
| `dexbotic/model/navila/loss.py` | 64-67 | Gaussian: `exp(-((target - soft_tokens)^2) / (2*std^2))` |
| `dexbotic/tokenization/process.py` | 368-483 | DM0Tokenization 全文 |
| `dexbotic/tokenization/process.py` | 407-414 | 空 assistant turn 弹出 |
| `dexbotic/tokenization/process.py` | 451-454 | loss_mask: assistant=True, human=False |
| `dexbotic/tokenization/process.py` | 474-475 | `labels = np.where(loss_mask, input_ids, IGNORE_INDEX)` |
| `dexbotic/data/dataset/transform/action.py` | 281-398 | `ActionNormAnd2String` 三步管线 |
| `dexbotic/data/dataset/transform/action.py` | 387-391 | `_action2bin`: `round((action+1)/2 * (vocab_size-1))` |
| `dexbotic/data/dataset/transform/action.py` | 393-398 | `_bin2string`: format + join |
| `dexbotic/data/collator.py` | 49-57 | `mapping_keys` 含 has_action, has_text |
| `dexbotic/exp/base_exp.py` | 344 | `use_special_tokens: bool = False` |
| `dexbotic/exp/base_exp.py` | 355-367 | `add_special_tokens` 词表扩展 |
| `dexbotic/exp/base_exp.py` | 398-417 | 基类 `ActionConfig.build_action_process_func` |
| `dexbotic/exp/dm0_exp.py` | 230-257 | `DM0ActionConfig` (用 ActionNorm) |
| `dexbotic/exp/dm0_exp.py` | 268-312 | `DM0DataConfig` (data_keys 无 has_text/has_action) |
| `dexbotic/exp/trainer.py` | 170-182 | `compute_loss`: 自动发现 `*_loss` 字段 |
| `dexbotic/exp/trainer.py` | 184-186 | `log`: 自动合并 loss_cache 到日志 |

## 附录 B: cumsum 注意力掩码数学推导

### 完整示例

设 prefix 4 个 token, suffix 3 个 token (chunk_size=3):
- `prefix_attn_mask = [1, 0, 0, 0]` (一个因果段)
- `suffix_attn_mask = [1, 0, 0]` (一个因果段)

**Step 1**: 拼接
```
full_attn_mask = [1, 0, 0, 0, 1, 0, 0]
```

**Step 2**: cumsum
```
cumsum = [1, 1, 1, 1, 2, 2, 2]
```

**Step 3**: `cumsum[:, None, :] <= cumsum[:, :, None]` (列 ≤ 行)

```
       key: p0  p1  p2  p3  s0  s1  s2
            (1) (1) (1) (1) (2) (2) (2)
query:
  p0 (1)    T   T   T   T   F   F   F
  p1 (1)    T   T   T   T   F   F   F
  p2 (1)    T   T   T   T   F   F   F
  p3 (1)    T   T   T   T   F   F   F
  s0 (2)    T   T   T   T   T   T   T
  s1 (2)    T   T   T   T   T   T   T
  s2 (2)    T   T   T   T   T   T   T
```

**解读**:
- prefix 内部: **全连接** (每个 prefix token 可以 attend 到所有 prefix token)
- suffix → prefix: **全可见** (suffix 可以 attend 到所有 prefix — "读取 VLM 上下文")
- **prefix → suffix: 全不可见** (prefix 看不到 suffix — "VLM 编码不含 action 信息")
- suffix 内部: **全连接** (suffix token 之间互相可见)

### 梯度方向性推论

因为 prefix → suffix 不可见:
- forward: `prefix_out` 不依赖于 `suffix_hidden_states` 的值
- backward: L_FM 对 `prefix_out` 没有直接梯度 (因为 `prefix_out` 不出现在 L_FM 的计算图中)

但 L_FM **间接** 影响 VLM:
- `suffix_out` 依赖于 VLM 的 K/V (通过 suffix → prefix attention)
- L_FM → suffix_out → attention weights → VLM 的 K_proj / V_proj 权重

加入 L_AR 后:
- `prefix_out` 出现在 L_AR 的计算图中
- L_AR → prefix_out → VLM 的全部参数 (Q/K/V proj + layernorm + MLP)
- VLM 的 Q/K/V proj 同时被 L_AR 和 L_FM 更新 (梯度叠加)
- VLM 的 MLP/FFN **只被 L_AR 更新** (L_FM 的梯度无法到达)

## 附录 C: 关键类/函数签名速查

```python
# DM0ForCausalLM.forward
def forward(
    self,
    input_ids: torch.LongTensor = None,       # [B, text_seq_len]
    attention_mask: Optional[torch.Tensor] = None,  # [B, text_seq_len]
    labels: Optional[torch.LongTensor] = None,      # [B, text_seq_len] (IGNORE_INDEX=-100)
    actions: Optional[torch.FloatTensor] = None,     # [B, chunk_size, action_dim]
    states: Optional[torch.FloatTensor] = None,      # [B, state_dim]
    images: Optional[torch.FloatTensor] = None,      # [B, n_img, C, H, W]
    image_masks: Optional[torch.BoolTensor] = None,  # [B, n_img]
    # 方案二新增:
    # has_text: Optional[torch.BoolTensor] = None,   # [B]
    # has_action: Optional[torch.BoolTensor] = None,  # [B]
) -> CausalLMOutputDexbotic:
    # 返回: loss, logits, text_loss, action_loss, ...

# CausalLMOutputDexbotic
@dataclass
class CausalLMOutputDexbotic(ModelOutput):
    loss: Optional[torch.FloatTensor] = None       # 总 loss = text_loss + action_loss
    logits: torch.FloatTensor = None               # 预测输出
    text_loss: Optional[torch.FloatTensor] = None  # L_AR (cross-entropy)
    action_loss: Optional[torch.FloatTensor] = None  # L_FM (MSE)

# ActionNormAnd2String.__call__
def __call__(self, episode_data_dict: dict) -> dict:
    # 输入: episode_data_dict['action'] = np.array [T, D]
    # 输出: episode_data_dict['action'] = normalized [-1, 1]
    #        episode_data_dict['answer'] = [" 128 64 255 ...", ...]

# DM0Tokenization.__call__
def __call__(self, conversations: List[Dict], **kwargs):
    # 输入: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
    # 输出: {"input_ids": np.array, "labels": np.array, "token_mask": np.array,
    #         "ar_mask": np.array, "loss_mask": np.array}

# DexboticTrainer.compute_loss
def compute_loss(self, model, inputs, return_outputs=False):
    # 自动发现 outputs 中所有 *_loss 字段, 缓存到 self.loss_cache
    # 在 self.log() 中自动合并到 wandb/tensorboard 日志
```
