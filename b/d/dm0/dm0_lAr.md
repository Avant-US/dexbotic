# DM0 "L_AR 自回归交叉熵" 代码实现深度分析

> 对应论文: DM0: An Embodied-Native Vision-Language-Action Model towards Physical AI  
> 代码库: `d:/SRC/Robot/dexbotic/`  
> 关联文档: [dm0.md](./dm0.md), [dm0_actTkn.md](./dm0_actTkn.md)

---

## 0. 核心结论 (TL;DR)

1. **L_AR 本质**：标准的 next-token prediction 交叉熵 loss，只在 assistant 回复位置的 token 上计算，其余位置（system prompt、user 输入、padding）通过 `IGNORE_INDEX = -100` 屏蔽。

2. **DM0 代码中没有 L_AR**：DM0 模型 (`dm0_arch.py`) 的 forward 方法只计算 Flow Matching MSE loss (`F.mse_loss(v_t, u_t)`)。论文描述的 `L_total = L_AR + L_FM` 双 Loss 在 DM0 模块中并未实现。

3. **L_AR 的完整实现分散在 5 个架构中**：
   - `DexboticForCausalLM`（基类）— 标准 HuggingFace CE
   - `DiscreteVLA` — 继承基类，纯自回归动作预测
   - `OFT-Discrete` — 并行分类交叉熵
   - **`HybridPi05`** — **最接近论文的双 Loss 设计**：`text_loss`(L_AR) + `action_loss`(L_FM)
   - `HybridCogAct` — 类似双 Loss 设计
   - `NaVILA` — soft cross-entropy 变体

4. **Knowledge Insulation 未显式实现**：代码中没有 `.detach()`、`stop_gradient` 等显式的梯度阻断操作。

5. **Embodied Spatial Scaffolding 未实现**：代码中没有子任务预测、目标框预测、EEF 轨迹预测等论文描述的 4 层级辅助任务。

---

## 1. 论文中 L_AR 的定义与作用

### 1.1 公式定义

论文定义的总损失函数：

$$\mathcal{L}_{total}(\theta) = \lambda \mathcal{L}_{AR}(\theta) + \mathcal{L}_{FM}(\theta), \quad \lambda = 1$$

其中 L_AR（自回归交叉熵）：

$$\mathcal{L}_{AR}(\theta) = -\mathbb{E}_D [\log \pi_\theta(\hat{l} \mid o_t, l)]$$

- $o_t$：观测（多视角图像 + proprioception）
- $l$：语言指令
- $\hat{l}$：模型预测的输出 token 序列（包括文本推理 + 离散 action token）

### 1.2 L_AR 的双重监督作用

L_AR 同时监督两类输出：

| 监督目标 | 内容 | 示例 |
|---|---|---|
| **具身推理文本** | 子任务描述、目标框坐标、EEF 轨迹点 | `"pick up the red cup"`, `"[0.3, 0.5, 0.7, 0.9]"` |
| **离散 action token** | 量化为 255-bin 的动作值 | `" 250 0 250 110 250 250 0"` |

论文将这种"一个 Loss 监督两种输出"的设计称为 **"text-as-everything"** — 一切辅助监督都通过自回归文本预测来实现。

### 1.3 Embodied Spatial Scaffolding 的 4 层辅助任务

论文描述 L_AR 还负责监督 Embodied Spatial Scaffolding 的 4 个层级：

1. **Subtask prediction** — 预测细粒度子任务描述
2. **Goal bounding box prediction** — 预测目标物体/区域的 bbox
3. **End-effector trajectory prediction** — 预测主视角下未来若干帧的 EEF 2D 轨迹
4. **Discrete action prediction** — 预测离散 action token

这些辅助任务全部以文本形式在 VLM 的自回归生成中监督，共享同一个 L_AR Loss。

---

## 2. L_AR 的数据侧实现 — 从连续动作到可监督标签

### 2.1 端到端流程总览

```
连续动作值 [T, D]          例: [0.15, -0.32, 0.87, 0.01, -0.56, 0.44, 0.0]
      │
      ▼ ① AddAction (predict_length=1)
状态偏移得到动作标签
      │
      ▼ ② DeltaAction (enable=True)
计算相对动作 delta_action = action - state
      │
      ▼ ③ AddTrajectory (trajectory_length=50)
滚动窗口堆叠为 action chunk [50, D]
      │
      ▼ ④ ActionNormAnd2String (vocab_size=255)
      │    ├── _norm_action(): 归一化到 [-1, 1]
      │    ├── _action2bin():  量化到 [0, 254]
      │    └── _bin2string():  格式化为字符串 " 250 0 250 110 250 250 0"
      │
      ▼ ⑤ ToConversation
构建对话格式 {"from":"gpt", "value":" 250 0 250 110 250 250 0"}
      │
      ▼ ⑥ DM0Tokenization / tokenize_dexbotic
token 化 + loss_mask 构建 → labels 张量
      │
      ▼ ⑦ DataCollatorForSupervisedDataset
batch padding (padding_value=IGNORE_INDEX=-100)
      │
      ▼ 模型 forward → cross_entropy(logits, labels, ignore_index=-100)
```

**关键：DM0 使用 `ActionNorm` 而非 `ActionNormAnd2String`，因此跳过了步骤 ④ 的量化和字符串化，连续值直接送给 Action Expert 做 Flow Matching。**

### 2.2 ActionNormAnd2String：量化核心

**文件**: `dexbotic/data/dataset/transform/action.py` lines 281-398

三步操作：

**Step 1 — 归一化 (lines 379-385)**
```python
def _norm_action(self, action, min, max):
    action = np.clip(action, min, max)
    action = (action - min) / (max - min + 1e-8) * 2 - 1  # → [-1, 1]
    return action
```

**Step 2 — 离散化 (lines 387-391)**
```python
def _action2bin(self, action, vocab_size):
    action = np.round((action + 1) / 2 * (vocab_size - 1))  # → [0, vocab_size-1]
    action = np.clip(action, 0, vocab_size - 1)
    return action
```

**Step 3 — 字符串化 (lines 393-398)**
```python
def _bin2string(self, action, string_format):
    # string_format 默认为 ' {value}'
    action_str = [''.join([string_format.format(value=int(_))
                          for _ in action[i]]) for i in range(len(action))]
    return action_str
```

输出示例：`" 250 0 250 110 250 250 0"` — 每个维度一个空格分隔的整数。

### 2.3 DM0Tokenization：loss_mask 构建

**文件**: `dexbotic/tokenization/process.py` lines 368-483

`DM0Tokenization` 类负责将对话格式转换为可训练的 input_ids 和 labels：

```
对话格式:
  System: "A chat between a curious user..."
  USER: "<image>\nWhat action should the robot take to {prompt}?"
  ASSISTANT: " 250 0 250 110 250 250 0"
```

**loss_mask 规则** (lines 436-454):

| 对话部分 | loss_mask | 说明 |
|---|---|---|
| System prompt | `False` | 不监督 |
| Role 标记 ("USER:", "ASSISTANT:") | `False` | 不监督 |
| Human 内容（指令、图像占位符） | `False` | 不监督 |
| **Assistant 内容（动作 token 字符串）** | **`True`** | **被监督** |
| Padding | `False` | 不监督 |

**Labels 构建** (line 475):
```python
input_ids = np.asarray(tokens)
labels = np.where(np.asarray(loss_mask), input_ids, IGNORE_INDEX)
# 即：labels[i] = input_ids[i] if loss_mask[i] else -100
```

这意味着 **L_AR 只在 assistant 的回复 token 上计算 loss**，所有其他位置被 `-100` 屏蔽。

### 2.4 DataCollatorForSupervisedDataset：batch 级 padding

**文件**: `dexbotic/data/collator.py` lines 10-68

```python
labels = torch.nn.utils.rnn.pad_sequence(
    labels, batch_first=True, padding_value=IGNORE_INDEX  # -100
)
```

Batch 内不同长度的 labels 用 `-100` 对齐，确保 padding 位不参与 loss 计算。

### 2.5 ActionNorm vs ActionNormAnd2String — 关键区别

| 特性 | ActionNorm (DM0 用) | ActionNormAnd2String (DiscreteVLA 用) |
|---|---|---|
| 文件 | action.py:229-278 | action.py:281-398 |
| 归一化 | ✓ → [-1, 1] | ✓ → [-1, 1] |
| 量化 | ✗ | ✓ → [0, vocab_size-1] |
| 字符串化 | ✗ | ✓ → `" 250 0 ..."` |
| 输出类型 | float numpy array | string |
| 用途 | Action Expert (Flow Matching) | VLM (自回归文本预测) |
| 使用架构 | DM0, Pi0.5 | DiscreteVLA, 及支持 L_AR 的架构 |

**这是 DM0 不使用 L_AR 的数据侧根源**：DM0 的 `DM0ActionConfig` (dm0_exp.py:244-264) 使用 `ActionNorm` 而非 `ActionNormAnd2String`，输出的是连续浮点值而非离散 token 字符串。

---

## 3. L_AR 的模型侧实现 — 6 种架构变体

### 架构总览

| 架构 | 文件 | L_AR 类型 | L_FM 类型 | Loss 组合 | 论文对应度 |
|---|---|---|---|---|---|
| DexboticForCausalLM（基类） | `dexbotic_arch.py:487` | HF standard CE | 无 | 单一 | 基础组件 |
| DiscreteVLA | `discrete_vla_arch.py` | 继承基类 CE | 无 | 单一 | 纯 L_AR |
| OFT-Discrete | `oft_discrete_arch.py:187` | 并行分类 CE | 无 | 单一 | 变体 |
| **HybridPi05** | **`hybrid_pi05_arch.py:462`** | **F.cross_entropy** | **MSE** | **text+action** | **最接近** |
| HybridCogAct | `hybrid_cogact_arch.py:138` | HF CE | Diffusion | text+action | 接近 |
| NaVILA | `navila_arch.py:481` | soft CE | 无 | 单一 | 变体 |
| **DM0** | **`dm0_arch.py:500`** | **无** | **MSE** | **单一** | **缺失 L_AR** |

---

### 3.1 DexboticForCausalLM 基类 — 最基础的 L_AR

**文件**: `dexbotic/model/dexbotic_arch.py` lines 429-496

基类实现了最标准的自回归交叉熵 loss：

```python
# line 469-480: LLM forward pass
outputs = self.model.backbone(
    input_ids=input_ids,
    labels=labels,      # 包含 action token IDs 或 -100
    ...
)

# line 482-483: 计算 logits
hidden_states = outputs.hidden_states[-1]
logits = self.lm_head(hidden_states)    # [B, seq_len, vocab_size]

# line 487-488: 计算 loss
if labels is not None:
    loss = self.loss_function(logits, labels, self.model.backbone.vocab_size)
```

`self.loss_function` 是 HuggingFace Transformers 提供的标准交叉熵，内部使用 `ignore_index=-100`，只在 `labels != -100` 的位置上计算 loss。

**这就是 L_AR 的核心实现**：next-token prediction 交叉熵，与 GPT 训练完全一致。

---

### 3.2 DiscreteVLA — 纯自回归离散动作预测

**文件**: `dexbotic/model/discrete_vla/discrete_vla_arch.py` lines 12-58

```python
class DiscreteVLAForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = DexboticConfig
```

DiscreteVLA 继承基类 `DexboticForCausalLM`，**训练时直接使用基类的 forward 和 L_AR**。

**推理流程**：
1. `self.generate()` — 自回归生成 token（line 35-41）
2. `tokenizer.decode()` — 解码为字符串（line 44）
3. `_discrete_action_to_continuous()` — 反量化（lines 52-57）：
   ```python
   actions = re.findall(r'\d+', action_str)[:7]     # regex 提取数字
   actions = np.array([int(a) for a in actions])     # → [0, 254]
   actions = (actions / (vocab_size - 1)) * 2 - 1    # → [-1, 1]
   ```
4. `_denorm()` — 反归一化回原始物理量

**特点**：
- 训练：纯 L_AR，无 L_FM
- 推理：自回归生成（逐 token），速度慢
- 量化精度：取决于 vocab_size（默认 255 bins）

---

### 3.3 OFT-Discrete — 并行离散分类

**文件**: `dexbotic/model/oft/oft_discrete_arch.py` lines 20-204

OFT-Discrete 不是自回归文本预测，而是并行分类预测：

**训练** (lines 164-191):
```python
# 从 input_ids 中剥离 action token，插入 placeholder 嵌入
placeholder_action_token_ids = self.model.action_head.action_query.expand(...)
action_embeds = self.model.llm.get_input_embeddings()(placeholder_action_token_ids.long())

# 通过 lm_head 投影到词表空间
predicted_actions = self.lm_head(action_hidden_states)
# Shape: (batch_size, chunk_size * action_dim, vocab_size)

# 交叉熵 loss — 每个位置独立分类
predicted_actions_flat = predicted_actions.reshape(-1, predicted_actions.size(-1))
discrete_action_labels_flat = discrete_action_labels.reshape(-1)
loss = nn.functional.cross_entropy(
    predicted_actions_flat,          # [B*C*D, vocab_size]
    discrete_action_labels_flat,     # [B*C*D]
    reduction="mean",
)
```

**与标准 L_AR 的区别**：
- 不是自回归（不按序列顺序逐 token 预测），而是 **并行分类**
- action token 作为独立的分类任务，每个位置从 `num_bins=256` 类中选一个
- 使用 `lm_head` 的词表投影 + cross_entropy，但没有文本生成能力
- Config 中 `num_bins=256`（line 13），比 DiscreteVLA 的 255 多 1 个 bin

---

### 3.4 HybridPi05 — 最接近论文的双 Loss 设计

**文件**: `dexbotic/model/pi05/hybrid_pi05_arch.py` lines 430-529

这是 **最接近 DM0 论文描述的 `L_total = L_AR + L_FM` 双 Loss 设计** 的实现。

#### text_loss (L_AR) — lines 457-479

```python
text_loss = None
if labels is not None and input_ids is not None:
    target_tokens = labels[:, 1:]                    # Shifted labels（next-token prediction）
    text_len = input_ids.shape[1]
    pred_tokens = text_logits[:, -text_len:-1]       # 取对应 prefix 的 logits

    # Per-token 交叉熵
    token_loss = F.cross_entropy(
        pred_tokens.transpose(1, 2),    # [B, vocab_size, seq_len]
        target_tokens,                   # [B, seq_len]
        reduction="none"                 # 不立即 reduce — 用于后续掩码
    )

    # Per-token 掩码：仅保留 labels != IGNORE_INDEX 的位置
    token_mask = torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)

    # Per-sample 平均 loss
    sample_loss = (token_loss * token_mask).sum(dim=-1) / torch.clamp(
        token_mask.sum(dim=-1), min=1.0
    )

    # Per-sample 掩码：has_text 控制哪些样本有文本监督
    if has_text is None:
        has_text_mask = torch.ones(...)
    else:
        has_text_mask = has_text.reshape(-1).to(sample_loss.device).float()

    text_loss = (sample_loss * has_text_mask).sum() / (has_text_mask.sum() + 1e-6)
```

**关键设计**：
- `reduction="none"` + 手动 per-token/per-sample 掩码 — 比基类更精细的控制
- `has_text` 掩码 — 支持混合 batch（部分样本有文本、部分只有动作）
- Shifted labels（`labels[:, 1:]`）— 标准 next-token prediction

#### action_loss (L_FM) — lines 481-504

```python
action_loss = None
action_logits = None
if suffix_out is not None and u_t is not None:
    action_logits = self.model.action_out_proj(
        suffix_out[:, -self.model.config.chunk_size :]    # 取最后 chunk_size 个位置
    )
    # Per-sample MSE
    per_sample_action_loss = F.mse_loss(
        action_logits, u_t, reduction="none"
    ).mean(dim=[1, 2])

    # Per-sample 掩码：has_action 控制哪些样本有动作监督
    if has_action is None:
        has_action_mask = torch.ones(...)
    else:
        has_action_mask = has_action.reshape(-1).to(...).float()

    action_loss = (per_sample_action_loss * has_action_mask).sum() / (
        has_action_mask.sum() + 1e-6
    )
```

#### Loss 组合 — lines 506-512

```python
loss = None
if text_loss is not None and action_loss is not None:
    loss = text_loss + action_loss       # λ=1（隐式）
elif text_loss is not None:
    loss = text_loss
elif action_loss is not None:
    loss = action_loss
```

**与论文公式的精确对应**：

| 论文 | HybridPi05 代码 |
|---|---|
| $\mathcal{L}_{AR}$ | `text_loss` (F.cross_entropy, reduction="none" + 掩码) |
| $\mathcal{L}_{FM}$ | `action_loss` (F.mse_loss, reduction="none" + 掩码) |
| $\mathcal{L}_{total} = \lambda \mathcal{L}_{AR} + \mathcal{L}_{FM}$ | `loss = text_loss + action_loss` (λ=1 隐式) |

#### Merged Attention 架构

HybridPi05 使用 **Merged Attention**，VLM (llm) 和 Action Expert 共享 Transformer 层的注意力计算：

```python
# line 481-484
module_list = [self.model.llm, self.model.action_expert]
(prefix_out, suffix_out), past_key_values, ... = self._inner_forward_mot(
    module_list, [prefix_tokens, suffix_tokens], ...
)
```

- `prefix`：VLM 处理的文本/图像 token → `text_logits = self.lm_head(prefix_out)` → `text_loss`
- `suffix`：Action Expert 处理的动作 token → `action_logits = action_out_proj(suffix_out)` → `action_loss`

#### 输出结构

```python
return CausalLMOutputDexbotic(
    loss=loss,
    text_loss=text_loss,       # 单独返回，供 Trainer 日志记录
    action_loss=action_loss,   # 单独返回，供 Trainer 日志记录
    logits=action_logits if action_logits is not None else text_logits,
    ...
)
```

---

### 3.5 HybridCogAct — 另一种双 Loss 实现

**文件**: `dexbotic/model/cogact/hybrid_cogact_arch.py` lines 126-188

结构类似 HybridPi05，但有几个关键差异：

**text_loss** (lines 126-141):
```python
text_loss = None
if labels is not None:
    text_labels = labels.clone()
    has_text = has_text.bool().view(-1)
    if ~has_text.any():
        text_labels[~has_text] = IGNORE_INDEX    # 无文本的样本全部屏蔽
    text_loss = (
        self.loss_function(logits, text_labels, self.model.backbone.vocab_size)
        * has_text.any().float()                  # 整个 batch 没有文本时 loss=0
    )
```

**action_loss** (lines 174-180):
```python
with torch.amp.autocast("cuda", dtype=torch.float32):
    action_loss = self.model.action_head_module.loss(
        actions_repeated, cognition_features_repeated, reduction="none"
    )
action_loss = (action_loss.mean(dim=[1, 2]) * has_action_repeated).sum() / (
    has_action_repeated.sum() + 1e-6
)
```

**与 HybridPi05 的差异**：
- text_loss 使用 HuggingFace 标准 `loss_function` 而非手动 `F.cross_entropy`
- action_loss 是 diffusion-based loss（通过 `action_head_module.loss()`），不是直接的 MSE
- `cognition_features`（line 155-157）从 VLM 最后一个非 padding 位置提取，作为 Action Head 的条件
- VLM 和 Action Expert 是 **分离的**（不共享 Transformer 层），不像 HybridPi05 的 Merged Attention

---

### 3.6 NaVILA — soft cross-entropy 变体

**文件**: `dexbotic/model/navila/navila_arch.py` lines 473-490  
**文件**: `dexbotic/model/navila/loss.py` lines 11-70

NaVILA 使用 **soft cross-entropy** 处理时间相关的 token：

```python
# navila_arch.py:476-486
if (self.training and hasattr(self.config, "time_token_ids") and self.config.time_token_ids):
    loss = soft_cross_entropy(
        logits, new_labels,
        soft_tokens=self.config.time_token_ids,
        std=getattr(self.config, "soft_ce_std", 1.0),
    )
else:
    loss = self.loss_function(logits, new_labels, self.model.backbone.vocab_size)
```

**soft_cross_entropy 实现** (loss.py:11-70):
- **非 soft token**：标准 hard cross-entropy
- **soft token（时间 token）**：构建高斯分布 soft target
  ```python
  # 对每个 soft token，构建以 target 为中心的高斯分布
  dist = torch.exp(-((target - soft_tokens) ** 2) / (2 * std**2))
  targets_indices[k][soft_tokens] = dist / dist.sum()
  loss += cross_entropy(outputs[indices], targets_indices, reduction="sum")
  ```

**动机**：时间 token 具有连续语义（相邻 bin 相似），hard CE 会惩罚"差一个 bin"和"差一百个 bin"一样多，soft CE 让相邻 bin 获得部分概率质量，降低量化边界处的梯度噪声。

---

### 3.7 DM0 — 没有 L_AR

**文件**: `dexbotic/model/dm0/dm0_arch.py` lines 495-502

```python
# Flow matching loss
if actions.dtype == torch.float32:
    suffix_out = suffix_out.to(torch.float32)
suffix_out_final = suffix_out[:, -self.model.config.chunk_size :]
v_t = self.model.action_out_proj(suffix_out_final)
action_loss = F.mse_loss(v_t, u_t, reduction="mean")

loss = action_loss    # ← 只有 L_FM，没有 L_AR
```

**没有 L_AR 的 3 条证据**：

1. **数据侧**：`DM0ActionConfig` (dm0_exp.py:244-264) 使用 `ActionNorm` 而非 `ActionNormAnd2String` — 连续值不被量化为离散 token
2. **模型侧**：`dm0_arch.py` 的 forward 只返回 `loss = action_loss`（MSE），无 `text_loss` 字段
3. **嵌入侧**：`dm0_arch.py:80` 设置 `self.action_expert.model.embed_tokens = None` — Action Expert 不接受离散 token 输入

**DM0 的输出结构**:
```python
outputs = CausalLMOutputDexbotic(
    loss=loss,           # 只有 action_loss (MSE)
    logits=v_t,          # 速度场预测
    # 注意：没有 text_loss 和 action_loss 分别返回
)
```

---

## 4. Knowledge Insulation (知识绝缘) 的实现状态

### 4.1 论文描述

论文 §5.1 描述了 **Hybrid Gradient Strategy**（借鉴 Knowledge Insulation, Driess et al., 2025）：

> 对具身数据：Action Expert 的梯度 **不回传** 到 VLM（insulate / 隔离），防止侵蚀 VLM 的语义知识。  
> 对非具身数据：VLM 正常更新。

这意味着 L_FM 的梯度应该被阻断在 Action Expert 内部，不影响 VLM backbone。

### 4.2 代码现状：未发现显式的梯度阻断

在所有架构中，**没有发现** `.detach()`、`torch.no_grad()`、`stop_gradient` 或任何显式的梯度阻断操作：

- **HybridPi05 的 Merged Attention** (`_inner_forward_mot`)：VLM 和 Action Expert 共享 Transformer 层的注意力计算，梯度可以双向流动
- **HybridCogAct**：`cognition_features` 从 VLM 输出提取，直接传给 Action Head，没有 `.detach()`
- **DM0 的 Merged Attention** (`_merged_attention_forward`)：同样没有梯度阻断

### 4.3 可能的隐式梯度路由

虽然没有显式的梯度阻断，但以下机制可能提供 **间接的/部分的** 梯度隔离效果：

1. **`has_text` / `has_action` 掩码**：在混合 batch 中，只有文本的样本不计算 `action_loss`，只有动作的样本不计算 `text_loss` — 这限制了每个样本上的梯度来源
2. **Merged Attention 的前缀/后缀分离**：prefix（VLM）和 suffix（Action Expert）在注意力中是分开的序列，suffix 不能直接注意到 prefix 的 hidden states（通过因果掩码）
3. **参数分组**：VLM 和 Action Expert 使用不同的参数集合，即使共享注意力，参数更新的主要来源取决于哪个 loss 对哪些参数有梯度

### 4.4 论文 vs 代码的差距

| 维度 | 论文描述 | 代码实现 |
|---|---|---|
| 梯度策略 | "Action Expert 梯度不回传到 VLM" | 无显式阻断 |
| 实现方式 | 显式的 Knowledge Insulation | 可能通过架构结构隐式实现 |
| 可能解释 | — | 完整的 KI 实现可能在私有分支中 |

---

## 5. Embodied Spatial Scaffolding 的实现状态

### 5.1 论文描述

论文描述 Embodied Spatial Scaffolding 提供层次化的辅助监督：

```
子任务描述 → 目标物体 bbox → 2D EEF 轨迹 → 离散 action token
```

所有层级全部通过 L_AR 以文本形式监督。

### 5.2 代码现状：未实现

**证据 1** — 对话模板简单（`process.py:368-483`）：

DM0 使用简单的 USER/ASSISTANT 模板，没有多层级结构：
```
System: "A chat between a curious user and an artificial intelligence assistant..."
USER: "<image>\nWhat action should the robot take to {prompt}?"
ASSISTANT: (动作内容)
```

**证据 2** — 数据配置不包含 Scaffolding 数据键（`dm0_exp.py:268-312`）：

```python
class DM0DataConfig(DataConfig):
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids", "labels", "action",
            "image", "state", "image_masks",
        ]
    )
```

没有 `subtask`、`goal_bbox`、`trajectory` 等键。

**证据 3** — Prompt 模板单一（`language.py:4`）：

```python
defalut_prompt_template = "<image>\nWhat action should the robot take to {prompt}?"
```

没有多层级预测格式。

**证据 4** — DM0Prog 只有 progress（`dm0_prog_arch.py`）：

DM0 的 "Prog" 变体只增加了 **progress prediction**（进度估计），是单一浮点输出，不是论文描述的 4 层 Scaffolding。

### 5.3 可能的解释

Embodied Spatial Scaffolding 可能：
- 存在于 Dexmal 内部的私有分支中
- 是论文的设计愿景，但开源代码中尚未完整实现
- 通过数据侧的 prompt 模板多样化间接实现（500 条模板），但不是代码层面的显式实现

---

## 6. CausalLMOutputDexbotic 输出结构与训练器

### 6.1 输出数据类

**文件**: `dexbotic/model/dexbotic_arch.py` lines 27-34

```python
@dataclass
class CausalLMOutputDexbotic(ModelOutput):
    loss: Optional[torch.FloatTensor] = None           # 组合后的总 loss
    logits: torch.FloatTensor = None                    # 预测 logits
    past_key_values: Optional[...] = None               # KV cache
    hidden_states: Optional[...] = None                 # 中间 hidden states
    attentions: Optional[...] = None                    # 注意力权重
    text_loss: Optional[torch.FloatTensor] = None       # L_AR（仅 Hybrid 架构）
    action_loss: Optional[torch.FloatTensor] = None     # L_FM（仅 Hybrid 架构）
```

**各架构返回的字段**：

| 架构 | loss | text_loss | action_loss |
|---|---|---|---|
| DexboticForCausalLM | CE | — | — |
| DiscreteVLA | CE | — | — |
| OFT-Discrete | CE (分类) | — | — |
| HybridPi05 | text+action | ✓ | ✓ |
| HybridCogAct | text+action | ✓ | ✓ |
| NaVILA | CE/soft CE | — | — |
| DM0 | MSE | — | — |

### 6.2 DexboticTrainer 的 loss 处理

**文件**: `dexbotic/exp/trainer.py` lines 170-186

```python
def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
    loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

    # 自动发现并缓存所有 *_loss 字段
    loss_keys = [_ for _ in outputs if _.endswith("_loss")]
    for loss_key in loss_keys:
        if outputs[loss_key] is None or torch.isclose(
            outputs[loss_key], torch.zeros_like(outputs[loss_key])
        ):
            if loss_key not in self.loss_cache:
                self.loss_cache[loss_key] = 0.0
            continue
        self.loss_cache[loss_key] = outputs[loss_key].detach().item()
    return (loss, outputs) if return_outputs else loss

def log(self, logs, start_time=None):
    logs.update(self.loss_cache)    # 将 text_loss, action_loss 等追加到日志
    super().log(logs, start_time)
```

Trainer 会自动收集 `text_loss` 和 `action_loss` 并记录到训练日志中，方便监控双 Loss 的训练动态。

---

## 7. 论文 vs 代码差异总结表

| 维度 | DM0 论文描述 | 代码实现（dexbotic 开源） |
|---|---|---|
| **L_AR 在 DM0 中** | L_total = λ·L_AR + L_FM, λ=1 | DM0 只有 L_FM (MSE) |
| **L_AR 监督内容** | 具身推理文本 + 离散 action token | 仅在其他架构中实现（DiscreteVLA 等） |
| **双 Loss 设计** | DM0 架构 | 实现在 HybridPi05 和 HybridCogAct 中 |
| **λ 权重** | λ=1（显式） | HybridPi05 中 `text_loss + action_loss`（隐式 λ=1） |
| **Knowledge Insulation** | Action Expert 梯度不回传 VLM | 无显式梯度阻断代码 |
| **Spatial Scaffolding** | 4 层辅助任务通过 L_AR 监督 | 未实现（仅有 progress prediction 变体） |
| **Soft cross-entropy** | 未提及 | NaVILA 实现了 soft CE 变体 |
| **Per-sample 掩码** | 未详述 | HybridPi05 实现了 `has_text`/`has_action` 掩码 |
| **Loss 日志** | 未详述 | Trainer 自动收集 `*_loss` 字段 |

---

## 8. 开发者指南 — 若要在 DM0 中启用 L_AR

如果要在 DM0 中实现论文描述的完整 `L_total = L_AR + L_FM`，需要以下修改：

### 8.1 数据侧修改

**DM0ActionConfig** (`dm0_exp.py`):
- 将 `ActionNorm` 替换为 `ActionNormAnd2String`
- 配置 `vocab_size=255`, `string_format=' {value}'`
- 修改数据管线以同时输出：
  - 连续动作值（给 Action Expert 的 L_FM）
  - 离散 action token 字符串（给 VLM 的 L_AR）

### 8.2 Tokenizer 修改

**add_special_tokens** (`base_exp.py:355-367`):
- 注册 255 个特殊 action token 到 tokenizer
- 相应地 `model.resize_token_embeddings(len(tokenizer))`

### 8.3 模型侧修改

**dm0_arch.py forward 方法**:
1. 在 merged attention 的 prefix 分支末端添加 `text_logits = self.lm_head(prefix_out)`
2. 计算 text_loss：
   ```python
   text_loss = F.cross_entropy(text_logits, labels, ignore_index=-100)
   ```
3. 组合 loss：
   ```python
   loss = text_loss + action_loss
   ```
4. 返回 `text_loss` 和 `action_loss` 供日志记录

### 8.4 Knowledge Insulation（可选）

如要实现论文描述的梯度隔离：
```python
# 方案 A：在 prefix_out 传给 suffix 之前 detach
suffix_condition = prefix_out.detach()  # 阻断 L_FM 梯度回传到 VLM

# 方案 B：对 has_action 样本，detach VLM 参数的梯度
# 更复杂，需要自定义 backward hook
```

### 8.5 参考实现

可以直接参考 `HybridPi05ForCausalLM` (`hybrid_pi05_arch.py:430-529`) 的实现模式：
- `text_loss` 的计算方式（per-token/per-sample 掩码）
- `action_loss` 的计算方式
- `has_text`/`has_action` 的混合 batch 处理
- Loss 组合与返回

---

## 附录 A：完整代码引用索引

| 文件 | 行号 | 内容 |
|---|---|---|
| `dexbotic/model/dexbotic_arch.py` | 27-34 | `CausalLMOutputDexbotic` 数据类（含 text_loss, action_loss 字段） |
| `dexbotic/model/dexbotic_arch.py` | 424-426 | `DexboticForCausalLM._real_init`：lm_head 初始化 |
| `dexbotic/model/dexbotic_arch.py` | 429-496 | `DexboticForCausalLM.forward`：基类 L_AR 实现 |
| `dexbotic/model/dexbotic_arch.py` | 487-488 | `loss = self.loss_function(logits, labels, vocab_size)` |
| `dexbotic/model/dm0/dm0_arch.py` | 35-42 | `DM0Config`：action_dim=32, chunk_size=50 |
| `dexbotic/model/dm0/dm0_arch.py` | 80 | `self.action_expert.model.embed_tokens = None` |
| `dexbotic/model/dm0/dm0_arch.py` | 495-502 | DM0 forward loss：只有 `F.mse_loss(v_t, u_t)` |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 455 | `text_logits = self.lm_head(prefix_out)` |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 457-479 | `text_loss`(L_AR)：F.cross_entropy + per-token/per-sample 掩码 |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 462-464 | `F.cross_entropy(pred_tokens.transpose(1,2), target_tokens, reduction="none")` |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 465 | `token_mask = torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)` |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 470-475 | `has_text` per-sample 掩码 |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 481-504 | `action_loss`(L_FM)：F.mse_loss + per-sample 掩码 |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 506-512 | `loss = text_loss + action_loss`（λ=1 隐式） |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 525-529 | 返回 CausalLMOutputDexbotic（含 text_loss, action_loss） |
| `dexbotic/model/cogact/hybrid_cogact_arch.py` | 126-141 | HybridCogAct text_loss 实现 |
| `dexbotic/model/cogact/hybrid_cogact_arch.py` | 155-157 | cognition_features 从 VLM 最后非 padding 位置提取 |
| `dexbotic/model/cogact/hybrid_cogact_arch.py` | 174-180 | HybridCogAct action_loss 实现（diffusion-based） |
| `dexbotic/model/cogact/hybrid_cogact_arch.py` | 182-188 | Loss 组合：text_loss + action_loss |
| `dexbotic/model/discrete_vla/discrete_vla_arch.py` | 12-13 | `DiscreteVLAForCausalLM` 继承 `DexboticForCausalLM` |
| `dexbotic/model/discrete_vla/discrete_vla_arch.py` | 34-41 | 推理：`self.generate()` 自回归生成 |
| `dexbotic/model/discrete_vla/discrete_vla_arch.py` | 52-57 | `_discrete_action_to_continuous`：反量化 |
| `dexbotic/model/oft/oft_discrete_arch.py` | 13 | `OFTDiscreteConfig`：`num_bins=256` |
| `dexbotic/model/oft/oft_discrete_arch.py` | 125-129 | 插入 placeholder action 嵌入 |
| `dexbotic/model/oft/oft_discrete_arch.py` | 168 | `predicted_actions = self.lm_head(action_hidden_states)` |
| `dexbotic/model/oft/oft_discrete_arch.py` | 187-191 | `F.cross_entropy(predicted_flat, labels_flat)` |
| `dexbotic/model/navila/loss.py` | 11-70 | `soft_cross_entropy` 完整实现 |
| `dexbotic/model/navila/loss.py` | 64-67 | 高斯 soft target 构建 |
| `dexbotic/model/navila/navila_arch.py` | 476-486 | NaVILA 条件选择 soft/hard CE |
| `dexbotic/tokenization/process.py` | 368-483 | `DM0Tokenization` 类完整实现 |
| `dexbotic/tokenization/process.py` | 401-405 | System prompt 的 loss_mask=False |
| `dexbotic/tokenization/process.py` | 436 | Human role tokens 的 loss_mask=False |
| `dexbotic/tokenization/process.py` | 451-454 | Assistant 回复 loss_mask=True vs Human 内容 loss_mask=False |
| `dexbotic/tokenization/process.py` | 474-475 | `labels = np.where(loss_mask, input_ids, IGNORE_INDEX)` |
| `dexbotic/data/dataset/transform/action.py` | 281-398 | `ActionNormAnd2String` 类 |
| `dexbotic/data/dataset/transform/action.py` | 379-385 | `_norm_action()`：归一化到 [-1, 1] |
| `dexbotic/data/dataset/transform/action.py` | 387-391 | `_action2bin()`：量化到 [0, vocab_size-1] |
| `dexbotic/data/dataset/transform/action.py` | 393-398 | `_bin2string()`：格式化为字符串 |
| `dexbotic/data/dataset/transform/action.py` | 229-278 | `ActionNorm`：仅归一化（DM0 用） |
| `dexbotic/data/collator.py` | 10-68 | `DataCollatorForSupervisedDataset` |
| `dexbotic/data/collator.py` | 29-32 | `pad_sequence(..., padding_value=IGNORE_INDEX)` |
| `dexbotic/exp/trainer.py` | 170-182 | `compute_loss`：自动收集 `*_loss` 字段 |
| `dexbotic/exp/trainer.py` | 184-186 | `log`：将 loss_cache 追加到日志 |
| `dexbotic/exp/base_exp.py` | 355-367 | `TokenizerConfig.add_special_tokens()` |
| `dexbotic/exp/base_exp.py` | 370-417 | `ActionConfig`：vocab_size=255 默认 |
| `dexbotic/exp/dm0_exp.py` | 244-264 | `DM0ActionConfig`：使用 `ActionNorm`（非 `ActionNormAnd2String`） |
| `dexbotic/exp/dm0_exp.py` | 268-312 | `DM0DataConfig`：data_keys 不含 Scaffolding 字段 |
| `dexbotic/constants.py` | — | `IGNORE_INDEX = -100`, `IMAGE_TOKEN_INDEX = -200` |

## 附录 B：L_AR 变体间的关键差异速查

| 差异维度 | 基类 CE | HybridPi05 CE | OFT-Discrete CE | Soft CE (NaVILA) |
|---|---|---|---|---|
| 函数 | `self.loss_function()` | `F.cross_entropy(..., reduction="none")` | `F.cross_entropy(..., reduction="mean")` | `soft_cross_entropy()` |
| Token shift | HF 内部处理 | 手动 `labels[:, 1:]` | 无 shift（并行分类） | 手动 `[..., :-1, :]` |
| 掩码粒度 | ignore_index=-100 | per-token mask + per-sample has_text | 无（全部 action token 参与） | ignore_index=-100 + soft target |
| Reduction | HF 默认 (mean) | 手动 per-sample mean → batch weighted mean | mean | sum / count |
| 混合 batch | 不支持 | ✓ has_text 掩码 | 不适用 | 不支持 |
| 监督目标 | 文本 + action token | 文本 + action token | 仅 action token（并行分类） | 文本 + time token（soft） |
