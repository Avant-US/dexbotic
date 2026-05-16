# DM0 "255-bin Action Token 量化" 在 dexbotic 代码库中的实现分析

> 分析对象: DM0 论文 (arXiv:2602.14974) 中的 **"动作：量化为 255-bin 的特殊 action tokens（供 VLM 自回归预测）"**   
> 代码库: `d:/SRC/Robot/dexbotic/` (Dexmal 官方开源 dexbotic 工具箱)   
> 关联文档: [dm0.md](./dm0.md) — DM0 论文详细解读

---

## 0. 核心结论 (TL;DR)

**结论一**：255-bin action token 量化的完整代码基础设施 **存在** 于 dexbotic 代码库中，但 **不在 DM0 模块里**。量化管线 `ActionNormAnd2String`（`action.py:281-398`）和特殊 token 注册 `TokenizerConfig.add_special_tokens`（`base_exp.py:355-367`）是公共基础设施，被 DiscreteVLA、OFT-Discrete 等架构使用。DM0 模块自身使用的是 `ActionNorm`（仅连续归一化，不经过量化）。

**结论二**：代码库中存在 **两种并行的离散化方案**：
1. **"文本化"方案**（`vocab_size=255`）：`ActionNormAnd2String` 将 bin 编号格式化为字符串 `" 0" ... " 254"`，注册为特殊 token，由 VLM 自回归逐 token 生成。用于 DiscreteVLA。
2. **"并行解码"方案**（`num_bins=256`）：`DiscreteActionHead` 将 hidden states 投影为 logits `(batch, chunk×dim, num_bins)`，用交叉熵训练，argmax/multinomial 采样一次输出所有 action。用于 OFT-Discrete。

**结论三**：**DM0 模型完全不使用离散 action token**。`dm0_arch.py` 中 `action_expert.model.embed_tokens = None`（第 80 行）从架构上断绝了 token 路径；forward 只有 `F.mse_loss`（Flow Matching）；数据管线只用 `ActionNorm` 不用 `ActionNormAnd2String`。论文描述的 "VLM 预测离散 action token（L_AR）" **在开源 DM0 中未实现**。最接近论文 L_total = L_AR + L_FM 描述的实现是 **Hybrid Pi0.5 架构**（`hybrid_pi05_arch.py`），它同时计算 text_loss（交叉熵）+ action_loss（Flow Matching MSE）。

---

## 1. 什么是 Action Token，为什么是 255 个 bin

### 1.1 定义

**Action Token** 是将机器人连续动作向量（如末端执行器的 xyz 位移、旋转、夹爪开合等）**量化为离散整数索引**，然后作为 **特殊 token 注册到 LLM 词表** 中的一种表征方式。这使得动作预测可以复用 VLM 的自回归文本生成机制 —— 模型像生成下一个"字"一样生成下一个"动作 bin"。

例如，一个 7 维 EEF 动作 `[0.12, -0.34, 0.56, 0.78, -0.91, 0.45, 0.0]`（归一化到 [-1, 1] 后），经过 255-bin 量化后变为 `[142, 84, 198, 226, 11, 184, 127]`，然后格式化为字符串 `" 142 84 198 226 11 184 127"` 作为 VLM 的"回答"来训练。

### 1.2 为什么这样做

DM0 论文的核心设计是 **双表征对齐（Dual Representation Alignment）**：同一段动作序列同时表示成两种形式：

1. **给 VLM 的离散 action token**（255-bin 量化）→ 用自回归交叉熵损失 L_AR 监督
2. **给 Action Expert 的连续 action 值** → 用 Flow Matching MSE 损失 L_FM 监督

L_AR 为 VLM 提供"温和的"梯度信号，鼓励它学习"动作相关的语义表达"；L_FM 为 Action Expert 提供精确的连续回归监督。两者联合训练：L_total = L_AR + L_FM。

### 1.3 为什么是 255

`vocab_size=255` 意味着 bin 编号从 0 到 254，共 255 个离散值。映射公式 `round((x+1)/2 × 254)` 将 [-1, 1] 均匀映射到 [0, 254]。

- **分辨率**：2 / 254 ≈ 0.00787 —— 在归一化 [-1, 1] 空间内，每个 bin 的宽度约 0.008，对于机器人末端执行器控制已足够精细
- **OFT-Discrete 用 256 bins**（编号 0-255），分辨率 2 / 255 ≈ 0.00784，差异可忽略
- 这些是实现选择，不是基础约束；论文中说的"255-bin"对应代码中 `ActionConfig.vocab_size = 255`

**代码出处**：
- `dexbotic/exp/base_exp.py:393` — `vocab_size: int = field(default=255)`
- `dexbotic/model/oft/oft_discrete_arch.py:13` — `num_bins: Optional[int] = 256`

---

## 2. 数据侧完整量化管线

### 2.1 管线总览

对于使用离散 action token 的模型（DiscreteVLA、OFT-Discrete），数据处理管线如下：

```
raw state → AddAction → DeltaAction → AddTrajectory → ActionNormAnd2String → tokenize
                                                         ↑ 三步合一: 归一化 + 量化 + 字符串化
```

管线的构建代码位于 `dexbotic/exp/base_exp.py:398-417`（`ActionConfig.build_action_process_func()`）：

```python
action_config = Pipeline([
    ToDict(),
    ToNumpy(),
    AddAction(predict_length=1),          # 状态偏移得到动作
    DeltaAction(enable=self.delta),       # 计算增量动作
    AddTrajectory(trajectory_length=...,  # 轨迹分块
                  padding_mode=...,
                  padding_action=...),
    ActionNormAnd2String(                 # 归一化 + 量化 + 字符串化
        statistic_mapping=statistic_mapping,
        vocab_size=self.vocab_size,       # 默认 255
        string_format=self.string_format), # 默认 ' {value}'
    LoadMultiModal(),
    AddPromptTemplate(prompt_template=self.prompt_template),
    ReplaceAnswer(default_answer=self.replace_with_default_answer),
    ToList(),
])
```

> **关键区分**：DM0 模型 **不使用此管线**。DM0 使用自己的 `DM0ActionConfig.build_action_process_func()`（`dm0_exp.py:247-264`），其中用的是 `ActionNorm`（仅连续归一化），不经过量化和字符串化。详见 §2.7。

### 2.2 AddAction: 状态偏移得到动作

**文件**: `dexbotic/data/dataset/transform/action.py:61-90`

```python
class AddAction:
    def __init__(self, predict_length: int = 1):
        self.predict_length = predict_length

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        state = episode_data_dict["state"]
        action = state[self.predict_length:]      # 向后偏移 1 步
        episode_data_dict["action"] = action
        episode_data_dict["abs_action"] = action
        # 裁剪其他 key 保持等长
        for key in episode_data_dict.keys():
            if key == "meta_data": continue
            episode_data_dict[key] = episode_data_dict[key][:len(action)]
        return episode_data_dict
```

**逻辑**：将 proprioceptive state 序列向后偏移 `predict_length`（默认 1）步，得到"下一步的目标状态"作为 action label。这是 DM0 论文中提到的 "用 proprioceptive state 通过 temporal shifting 构造下一步动作标签" 的实现。

### 2.3 DeltaAction: 计算增量动作

**文件**: `dexbotic/data/dataset/transform/action.py:93-153`

```python
class DeltaAction:
    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        state = episode_data_dict['state']
        action = episode_data_dict['action']
        delta_action = action - state           # 增量 = 目标 - 当前
        # 周期维度（如角度）的 wrap 处理
        if periodic_mask is not None:
            for dim in periodic_mask:
                delta_action[..., dim] = np.where(
                    delta_action[..., dim] > periodic_range / 2,
                    delta_action[..., dim] - periodic_range,
                    delta_action[..., dim])
        # 非增量维度（如夹爪）保持绝对值
        delta_action[..., non_delta_mask] = action[..., non_delta_mask]
        episode_data_dict['action'] = delta_action
        return episode_data_dict
```

**逻辑**：
- `delta_action = action - state`：将绝对动作转为增量表示
- `non_delta_mask`（如夹爪开合）：这些维度不做差值，保持绝对值
- `periodic_mask`（如关节角度）：做 wrap 处理，避免 ±π 跳变

### 2.4 AddTrajectory: 轨迹分块

**文件**: `dexbotic/data/dataset/transform/action.py:156-226`

```python
class AddTrajectory:
    def __init__(self, trajectory_length: int = 50, flatten=True, padding_mode='zero'):
        ...
    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        action = episode_data_dict['action']
        trajectory = np.stack([action[i:i+1] for i in range(trajectory_length)], ...)
        # 不足 trajectory_length 时做 padding
        ...
```

**逻辑**：为每个时间步构造一个 `trajectory_length` 长的 action chunk（滑动窗口）。

- DM0 使用 `trajectory_length=50, flatten=False, padding_mode="last"`
- DiscreteVLA 通常用 `trajectory_length=1`（单步预测）
- OFT-Discrete 用 `trajectory_length=8` 或 `16`
- `flatten=True` 时输出 shape 为 `[N, T×D]`；`flatten=False` 时为 `[N, T, D]`

### 2.5 ActionNormAnd2String: 归一化 + 量化 + 字符串化（三步合一）

**文件**: `dexbotic/data/dataset/transform/action.py:281-398`

这是离散 action token 量化的**核心类**。包含三个步骤：

#### Step 1: `_norm_action` — 归一化到 [-1, 1]

```python
def _norm_action(self, action, min, max) -> np.array:
    min = min.reshape(1, -1)
    max = max.reshape(1, -1)
    action = np.clip(action, min, max)               # 先裁剪到 [min, max]
    action = (action - min) / (max - min + 1e-8) * 2 - 1  # 线性映射到 [-1, 1]
    return action
```

- `statistic_mapping` 提供每个数据集、每个 prompt 的 min/max 统计量
- 支持层级查找：prompt 级 → dataset 级 → default 级

#### Step 2: `_action2bin` — 量化为离散 bin 索引

```python
def _action2bin(self, action, vocab_size) -> np.array:
    action = np.round((action + 1) / 2 * (vocab_size - 1))  # [-1,1] → [0, vocab_size-1]
    action = np.clip(action, 0, vocab_size - 1)
    return action
```

**量化公式**：

$$bin = \text{round}\left(\frac{a_{norm} + 1}{2} \times (N - 1)\right), \quad N = 255$$

将 [-1, 1] 均匀映射到 {0, 1, 2, ..., 254}。

#### Step 3: `_bin2string` — 格式化为字符串

```python
def _bin2string(self, action, string_format) -> list[str]:
    action_str = [''.join([string_format.format(value=int(_))
                          for _ in action[i]]) for i in range(len(action))]
    return action_str
```

- 默认 `string_format = ' {value}'`（注意前面有空格）
- 7 维动作量化后变成形如 `" 142 84 198 226 11 184 127"` 的字符串
- 每个 bin 编号前有一个空格分隔

#### 整体调用流程（`__call__`）

```python
def __call__(self, episode_data_dict, **kwargs):
    action = episode_data_dict['action']
    prompt = episode_data_dict['prompt'][0]
    dataset = episode_data_dict['meta_data']['dataset']
    # 1. 查找该 dataset/prompt 的归一化统计量
    statistic_mapping = self._lookup_stats(dataset, prompt)
    # 2. 归一化
    normalized_action = self._norm_action(action, statistic_mapping['min'], statistic_mapping['max'])
    episode_data_dict['action'] = normalized_action   # 连续归一化值保留
    # 3. 量化
    bin_action = self._action2bin(normalized_action, self.vocab_size)
    # 4. 字符串化
    action_str = self._bin2string(bin_action, self.string_format)
    # 5. 写入 answer 字段（供 tokenizer 编码为 label）
    if self.add_answer and "answer" not in episode_data_dict:
        episode_data_dict['answer'] = action_str
    return episode_data_dict
```

**关键输出**：
- `episode_data_dict['action']`：连续归一化后的值（用于 Flow Matching 等连续路径）
- `episode_data_dict['answer']`：量化后的字符串（用于 VLM 自回归预测）

### 2.6 ActionNorm: 仅归一化，不量化

**文件**: `dexbotic/data/dataset/transform/action.py:229-278`

```python
class ActionNorm:
    def _normalize(self, data, stats):
        if self.use_quantiles:
            return ((data - stats["min"]) / (stats["max"] - stats["min"] + 1e-6) * 2.0 - 1.0)
        else:
            return ((data - stats["mean"]) / (stats["std"] + 1e-6))
```

**与 `ActionNormAnd2String` 的区别**：
- `ActionNorm` 只做归一化到 [-1, 1]（或 z-score），**不做量化，不做字符串化**
- 输出仍然是连续的浮点数，直接送入 Action Expert 做 Flow Matching
- **这是 DM0 使用的归一化方式**

### 2.7 DM0 vs 其他模型的数据管线对比

DM0 的管线定义在 `dexbotic/exp/dm0_exp.py:247-264`：

```python
class DM0ActionConfig(ActionConfig):
    trajectory_length: int = field(default=50)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline([
            ToDict(),
            ToNumpy(),
            AddAction(predict_length=1),
            PadState(ndim=32, axis=-1),           # DM0 特有: 动作填充到 32 维
            PadAction(ndim=32, axis=-1),
            AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
            DeltaAction(enable=True),
            ActionNorm(statistic_mapping=..., use_quantiles=True),  # ← 仅归一化
            LoadMultiModal(return_masks=True),
            ToList(),
        ])
        return action_config
```

对比表：

| 管线 | 模型 | 用 ActionNormAnd2String? | 用 ActionNorm? | 输出 answer 字符串? | 输出 action 张量? |
|---|---|---|---|---|---|
| `base_exp.py` 默认 | DiscreteVLA/base | 是 (vocab_size=255) | 否 | 是 | 是（归一化后连续值） |
| `dm0_exp.py` | **DM0** | **否** | **是** (quantile) | **否** | 是（归一化后连续值） |
| `oft_discrete_exp.py` | OFT-Discrete | 是（继承 base） | 否 | 是 | 是 |

> **这是理解 DM0 开源代码最关键的一点**：DM0 的数据管线 **根本不生成离散 action token 字符串**。数据以连续浮点张量的形式直接送入 Action Expert 做 Flow Matching 回归。

---

## 3. 模型侧: 特殊 Token 注册与 Embedding 扩展

### 3.1 TokenizerConfig.add_special_tokens

**文件**: `dexbotic/exp/base_exp.py:355-367`

```python
class TokenizerConfig(Config):
    use_special_tokens: bool = field(default=False)   # ← 默认关闭！
    use_fast_tokenizer: bool = field(default=True)

    def add_special_tokens(self,
                           special_token_format: str,    # ' {value}' → ' 0', ' 1', ...
                           vocab_size: int,              # 255
                           tokenizer,
                           model):
        if not self.use_special_tokens:
            return tokenizer                             # 默认直接返回

        special_tokens = [special_token_format.format(i) for i in range(vocab_size)]
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer
```

**逻辑**：
1. 生成 255 个特殊 token：`[" 0", " 1", " 2", ..., " 254"]`
2. `special_tokens=True` 标记为特殊 token，使 BPE/SentencePiece 不会将其拆分
3. `resize_token_embeddings` 扩展 LLM 的 embedding 矩阵以容纳新 token

**关键守卫**：`use_special_tokens` 默认为 `False`。当 `ActionNormAnd2String` 的 `string_format = ' {value}'` 生成的字符串（如 `" 127"`）被送入不启用特殊 token 的 tokenizer 时，它会被 BPE 按普通文本分词（例如拆成 `" "` + `"1"` + `"2"` + `"7"` 等子 token）。启用特殊 token 后，`" 127"` 会被识别为单个完整 token。

### 3.2 调用时机

**文件**: `dexbotic/exp/base_exp.py:803-807`

```python
self.tokenizer = self.tokenizer_config.add_special_tokens(
    self.data_config.action_config.string_format,
    self.data_config.action_config.vocab_size,
    self.tokenizer,
    self.model)
```

在 `_initialize_train()` 中调用，即模型初始化和 tokenizer 加载之后、训练开始之前。

### 3.3 LLM 词表的 resize

注册特殊 token 后，LLM 的 embedding 矩阵从 `original_vocab_size` 扩展为 `original_vocab_size + 255`。新增的 255 个 embedding 向量会被随机初始化，然后在训练中学习。

---

## 4. 三种架构变体中的离散 Action Token 使用

### 4.1 DiscreteVLA: 纯自回归文本生成

**文件**: `dexbotic/model/discrete_vla/discrete_vla_arch.py`

**训练**：标准 VLM 自回归训练。action bin 字符串嵌入在对话的"回答"部分，被 tokenizer 编码为 token ID 序列，用标准 next-token 交叉熵损失（继承自 `DexboticForCausalLM`）监督。

**推理**（第 24-50 行）：
```python
# 自回归生成文本
outputs = self.generate(input_ids, images=image_tensor,
                        max_new_tokens=1024, temperature=0.7, ...)
# 解码文本
outputs = tokenizer.decode(outputs, skip_special_tokens=False)
# 反量化
actions = self._discrete_action_to_continuous(outputs, vocab_size)
actions = self._denorm(actions, action_norms)
```

**反量化**（第 52-58 行）：
```python
def _discrete_action_to_continuous(self, action_str: str, vocab_size: int):
    actions = re.findall(r'\d+', action_str)[:7]        # 正则提取前 7 个数字
    actions = np.array([int(action) for action in actions], dtype=np.float32).reshape(1, -1)
    actions = (actions / (vocab_size - 1)) * 2 - 1      # [0, 254] → [-1, 1]
    return actions
```

**反量化公式**：

$$a_{norm} = \frac{bin}{N - 1} \times 2 - 1, \quad N = 255$$

**特点**：
- 简单：复用 VLM 的文本生成能力
- 慢：逐 token 自回归生成，7 维 action 需要生成至少 7 个 token
- 脆弱：依赖正则解析，生成的文本可能包含无效内容（代码中有 retry 机制，最多尝试 40 次）

### 4.2 OFT-Discrete: 并行离散解码

**文件**: `dexbotic/model/oft/oft_discrete_arch.py`（完整 283 行）  
**文件**: `dexbotic/model/oft/action_model/model.py:275-347`（`DiscreteActionHead`）

**架构**：基于 OFT（Open Foundation Transformer），使用 `num_bins=256`。

#### DiscreteActionHead 的量化/反量化

```python
class DiscreteActionHead(nn.Module):
    def __init__(self, input_dim=4096, vocab_size=32000, action_dim=7,
                 action_chunk=16, num_bins=256, ...):
        self.action_query = torch.ones(1, action_chunk * action_dim)  # 占位 token

    def discretize_actions(self, actions):
        """[-1,1] → [0, num_bins-1]"""
        actions = torch.clamp(actions, -1, 1)
        discrete_actions = ((actions + 1) / 2 * (self.num_bins - 1)).round().long()
        return discrete_actions

    def discrete_tokens_to_continuous(self, token_ids):
        """[0, num_bins-1] → [-1,1]"""
        actions = (actions / (self.num_bins - 1)) * 2 - 1
        return actions
```

#### 训练 Forward（`oft_discrete_arch.py:26-204`）

1. **剥离 action token**（第 67-106 行）：从 `input_ids` 中识别并分离 action token 区域
   - 结构：`[prefix tokens] + [action tokens (chunk_size × action_dim)] + [suffix token]`
   - 保留 prefix + suffix 送入 LLM，action labels 单独提取
2. **插入占位 embedding**（第 125-130 行）：在 action 位置插入可学习的 `action_query` embedding
3. **LLM forward**（第 145-154 行）：获取 hidden states
4. **提取 action hidden states**（第 157-159 行）：从 LLM 输出中截取 action 位置的 hidden states
5. **投影到 logits**（第 168 行）：`self.lm_head(action_hidden_states)` → shape `(batch, chunk×dim, vocab_size)`
6. **交叉熵损失**（第 187-191 行）：
   ```python
   loss = nn.functional.cross_entropy(
       predicted_actions_flat,              # (batch × chunk × dim, vocab_size)
       discrete_action_labels_flat,         # (batch × chunk × dim,)
       reduction="mean")
   ```

#### 推理（`oft_discrete_arch.py:206-282`）

**Argmax 推理**（`inference_action`，第 206-235 行）：
```python
discrete_action_indices = torch.argmax(
    predicted_logits[:, :, -self.config.num_bins + 1:], dim=-1)  # 取最后 num_bins-1 列
predicted_actions = self.model.action_head.discrete_tokens_to_continuous(discrete_action_indices)
```

**带温度采样推理**（`generate_action`，第 237-282 行）：
```python
scaled_logits = predicted_logits / temperature
probs = torch.softmax(scaled_logits, dim=-1)
probs_flat = probs.reshape(-1, probs.shape[-1])
sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
```

**特点**：
- 快：单次 forward pass 并行输出所有 `chunk_size × action_dim` 个 action bin
- 灵活：支持 argmax（确定性）和 multinomial sampling（随机性）两种推理模式
- 精确：直接操作 logits，无字符串解析的脆弱性

### 4.3 DM0: 纯 Flow Matching，无离散 Action Token

**文件**: `dexbotic/model/dm0/dm0_arch.py`

DM0 模型 **完全不使用离散 action token**。以下是代码中的关键证据：

#### 证据 1: Action Expert 的 embed_tokens 被设为 None（第 80 行）

```python
class DM0Model(DexboticVLMModel):
    def __init__(self, config: DM0Config):
        super().__init__(config)
        self.action_expert = Qwen3ForCausalLM(action_model_config)
        self.action_expert.model.embed_tokens = None   # ← 永久禁用 token embedding
```

这意味着 Action Expert 无法接受 token ID 输入，只能通过 `action_in_proj`（连续值投影层）接受连续动作输入。

#### 证据 2: Forward 只有 Flow Matching Loss（第 495-502 行）

```python
def forward(self, ..., actions, ...):
    # Flow matching interpolation
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions    # 噪声插值
    u_t = noise - actions                                            # 目标速度场
    ...
    # 只有 Flow Matching loss
    v_t = self.model.action_out_proj(suffix_out_final)
    action_loss = F.mse_loss(v_t, u_t, reduction="mean")            # ← MSE，非交叉熵
    loss = action_loss                                               # ← 唯一的 loss
```

没有 `lm_head`、没有 `cross_entropy`、没有 `labels` 的消费 —— 只有纯粹的 Flow Matching 回归。

#### 证据 3: 数据管线不做量化（`dm0_exp.py:247-264`）

如 §2.7 所述，DM0 的数据管线用 `ActionNorm`（连续归一化）替代了 `ActionNormAnd2String`（量化+字符串化）。

#### DM0 的推理：Euler 采样（第 513-641 行）

DM0 的推理不是"生成 token"，而是经典的扩散模型去噪过程：

```python
def inference_action(self, ..., diffusion_steps=10, ...):
    noise = torch.normal(0, 1, size=(batch, chunk_size, action_dim))
    time = 1.0
    dt = -1.0 / diffusion_steps
    # 先算 prefix 的 KV cache
    _, kv_cache = self._merged_attention_forward(...)
    # Euler 采样循环
    while time >= -dt / 2:
        noise, time = self._denoise_step(noise, time, dt, ...)
    return noise  # 去噪后的连续动作
```

每一步去噪调用 `_denoise_step`，它复用 prefix 的 KV cache，只重新计算 suffix（action）部分：

```python
def _denoise_step(self, x_t, time, dt, ...):
    suffix_hidden, ..., ... = self.get_suffix_hidden_states(x_t, time)
    (_, suffix_out), _ = self._merged_attention_forward(
        ..., past_key_values=kv_cache, input_embeds_list=[None, suffix_hidden], ...)
    v_t = self.model.action_out_proj(suffix_out[:, -chunk_size:])
    return x_t + v_t * dt, time + dt   # Euler 更新
```

### 4.4 Hybrid Pi0.5: 双 Loss — 最接近论文 L_total = L_AR + L_FM

**文件**: `dexbotic/model/pi05/hybrid_pi05_arch.py:455-512`

这是代码库中最接近 DM0 论文描述的 **"L_AR + L_FM 联合训练"** 的实现：

```python
# --- L_AR: 自回归交叉熵（Text Loss，包含 action token 字符串） ---
text_logits = self.lm_head(prefix_out)
text_loss = None
if labels is not None:
    pred_tokens = text_logits[:, -text_len:-1]
    token_loss = F.cross_entropy(
        pred_tokens.transpose(1, 2), target_tokens, reduction="none")
    token_mask = torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)
    text_loss = (token_loss * token_mask * has_text_mask).sum() / ...

# --- L_FM: Flow Matching MSE（Action Loss） ---
action_loss = None
if suffix_out is not None and u_t is not None:
    action_logits = self.model.action_out_proj(suffix_out[:, -chunk_size:])
    action_loss = F.mse_loss(action_logits, u_t, reduction="none").mean(dim=[1, 2])
    action_loss = (action_loss * has_action_mask).sum() / ...

# --- L_total = L_AR + L_FM ---
loss = None
if text_loss is not None and action_loss is not None:
    loss = text_loss + action_loss
elif text_loss is not None:
    loss = text_loss
elif action_loss is not None:
    loss = action_loss
```

**关键设计**：
- `has_text` / `has_action` 掩码允许混合 batch（有的样本只有文本、有的只有动作、有的两者都有）
- `text_loss` 通过 `lm_head`（VLM 的语言头）计算，监督的 target 包括对话文本中嵌入的 action token 字符串
- `action_loss` 通过 `action_out_proj`（Action Expert 的输出投影）计算，监督的 target 是连续 velocity field
- `CausalLMOutputDexbotic` 输出中分别返回 `text_loss` 和 `action_loss`，便于日志和调试

---

## 5. 量化 / 反量化数学公式

### 5.1 正向量化（Continuous → Discrete）

**第一步：归一化**（将原始动作值映射到 [-1, 1]）

$$a_{norm} = \frac{a_{raw} - a_{min}}{a_{max} - a_{min}} \times 2 - 1$$

**代码位置**：`action.py:379-385`（`ActionNormAnd2String._norm_action`）

**第二步：量化**（将 [-1, 1] 映射到 {0, 1, ..., N-1}）

$$bin = \text{round}\left(\frac{a_{norm} + 1}{2} \times (N - 1)\right), \quad \text{clip to } [0, N-1]$$

其中 $N = 255$（文本化方案）或 $N = 256$（并行解码方案）。

**代码位置**：
- 数据侧：`action.py:387-391`（`ActionNormAnd2String._action2bin`）
- 模型侧：`action_model/model.py:303-313`（`DiscreteActionHead.discretize_actions`）

### 5.2 反向量化（Discrete → Continuous）

**第一步：逆量化**（将 {0, 1, ..., N-1} 映射回 [-1, 1]）

$$a_{norm} = \frac{bin}{N - 1} \times 2 - 1$$

**代码位置**：
- `action_model/model.py:325-347`（`DiscreteActionHead.discrete_tokens_to_continuous`）
- `discrete_vla_arch.py:52-58`（`DiscreteVLAForCausalLM._discrete_action_to_continuous`）

**第二步：逆归一化**（将 [-1, 1] 映射回原始动作空间）

$$a_{raw} = a_{min} + \frac{a_{norm} + 1}{2} \times (a_{max} - a_{min})$$

**代码位置**：`dexbotic/model/dexbotic_arch.py` 中的 `_denorm` 方法

### 5.3 量化误差分析

每个维度的最大量化误差（在归一化 [-1, 1] 空间内）：

$$\epsilon_{max} = \frac{1}{N - 1}$$

| 方案 | N | 最大误差（归一化空间） | 分辨率 |
|---|---|---|---|
| 文本化 (vocab_size=255) | 255 | 1/254 ≈ 0.00394 | 2/254 ≈ 0.00787 |
| 并行解码 (num_bins=256) | 256 | 1/255 ≈ 0.00392 | 2/255 ≈ 0.00784 |

在原始动作空间中，最大误差为 $\epsilon_{raw} = \frac{a_{max} - a_{min}}{N - 1}$，取决于归一化范围。

### 5.4 代码位置对照表

| 操作 | 数据侧 | 模型侧 (DiscreteActionHead) | 模型侧 (DiscreteVLA) |
|---|---|---|---|
| 归一化 | `action.py:379-385` `_norm_action` | N/A（数据侧完成） | N/A（数据侧完成） |
| 量化 | `action.py:387-391` `_action2bin` | `model.py:303-313` `discretize_actions` | N/A（数据侧完成） |
| 字符串化 | `action.py:393-398` `_bin2string` | N/A | N/A |
| 反量化 | N/A（推理侧） | `model.py:325-347` `discrete_tokens_to_continuous` | `discrete_vla_arch.py:52-58` |
| 逆归一化 | N/A（推理侧） | `dexbotic_arch.py` `_denorm` | `dexbotic_arch.py` `_denorm` |

---

## 6. 双表征对齐: 离散 Token (VLM) vs 连续值 (Action Expert)

### 6.1 论文的设计理念

DM0 论文描述的核心设计是**同一段动作序列同时用两种表征**：

```
同一个 action chunk (50 步 × 7 维)
    ├── 离散路径 → 255-bin 量化 → action tokens → VLM 自回归预测 → L_AR (交叉熵)
    └── 连续路径 → 归一化到 [-1,1] → Action Expert → Flow Matching → L_FM (MSE)
```

**L_AR 的作用**：为 VLM backbone 提供"温和的"梯度信号。由于 Hybrid Gradient Strategy（知识隔离）中 Action Expert 的梯度 **不回传** 到 VLM，如果没有 L_AR，VLM 就不会从动作数据中学到任何东西。L_AR 通过让 VLM 预测离散 action token，鼓励它学习"动作相关的语义表达"。

**L_FM 的作用**：为 Action Expert 提供精确的连续回归监督。Flow Matching 直接在连续空间中学习 velocity field，避免了量化误差。

### 6.2 代码中的实现状态

| 架构 | 离散路径 (L_AR) | 连续路径 (L_FM) | 双路径对齐 |
|---|---|---|---|
| DiscreteVLA | 有（自回归交叉熵） | 无 | 无 |
| OFT-Discrete | 有（并行交叉熵） | 无 | 无 |
| DM0 | **无** | 有（Flow Matching MSE） | **无** |
| Hybrid Pi0.5 | 有（text_loss 包含 action token） | 有（action_loss MSE） | **有** |

> Hybrid Pi0.5 是唯一同时具备两条路径的架构，也是最接近论文描述的实现。

### 6.3 Hybrid Gradient Strategy 在代码中的体现

在 DM0 架构中，虽然没有 L_AR，但 **Merged Attention** 机制实现了 VLM 和 Action Expert 的交互（`dm0_arch.py:480-493`）：

```python
module_list = [self.model.llm, self.model.action_expert.model]
(prefix_out, suffix_out), _ = self._merged_attention_forward(
    module_list=module_list,
    attention_mask=attn_mask,
    position_ids=positions,
    input_embeds_list=[prefix_hidden_states, suffix_hidden_states],
    ...)
```

- `prefix_hidden_states`（VLM 的图像+语言 embedding）和 `suffix_hidden_states`（Action Expert 的噪声动作+时间 embedding）共享注意力计算
- 但 loss 只在 suffix_out 上计算（`F.mse_loss(v_t, u_t)`），所以梯度主要流向 Action Expert
- VLM 通过注意力机制接收 Action Expert 的信息，但不直接被动作 loss 监督

---

## 7. 论文描述 vs 代码实现差异总结

| 维度 | 论文描述 | 开源代码实现 |
|---|---|---|
| DM0 使用 255-bin action tokens | 是（L_AR 监督 VLM 预测离散 token） | **否** — DM0 模块无量化路径 |
| VLM 预测 action tokens | 是（自回归生成） | 仅在 DiscreteVLA 中有（非 DM0） |
| 并行离散解码 | 论文未描述此变体 | 在 OFT-Discrete 中实现（num_bins=256） |
| 双 Loss L_AR + L_FM | 是（DM0 的核心设计） | 仅在 Hybrid Pi0.5 中实现（非 DM0 模块） |
| Action Expert embed_tokens | 应该可用（VLM+AE 双路径） | `= None`，在 DM0 中被永久禁用 |
| 特殊 token 注册 | 论文隐含需要 | 基础设施存在，但 `use_special_tokens=False` 默认关闭 |
| action_dim | 论文样例显示 7 维（EEF） | DM0 代码中默认 32 维（padding 到 32） |
| chunk_size | 论文说 Horizon = 50 | DM0 代码中 `chunk_size = 50`，一致 |

**可能的解释**：
1. DM0 开源代码可能是论文训练代码的**简化版本**，省略了 L_AR 路径
2. 或者 L_AR 路径已在内部实现但**尚未开源**
3. 量化基础设施的存在说明团队 **确实构建了** 离散 action token 系统，但在 DM0 模型定义中选择了只保留 Flow Matching 路径

---

## 8. 开发者指南: 若要在 DM0 中启用离散 Action Token

基于代码库中已有的基础设施，要为 DM0 添加论文描述的 L_AR 路径，需要修改以下部分：

### 8.1 数据侧

将 `dm0_exp.py` 中 `DM0ActionConfig.build_action_process_func()` 的 `ActionNorm` 替换为 `ActionNormAnd2String`，或同时保留两者（连续值用于 Flow Matching，字符串用于 L_AR）。

### 8.2 Tokenizer 侧

在 DM0 的 tokenizer 配置中设置 `use_special_tokens=True`，确保 `" 0"` 到 `" 254"` 被注册为特殊 token。

### 8.3 模型侧

1. 移除 `dm0_arch.py:80` 的 `self.action_expert.model.embed_tokens = None`
2. 在 `DM0ForCausalLM.forward()` 中添加交叉熵损失：
   ```python
   # 在 prefix_out 上计算 text logits
   text_logits = self.lm_head(prefix_out)
   # 对 labels 中的 action token 区域计算交叉熵
   text_loss = F.cross_entropy(text_logits[..., :-1, :].view(-1, vocab_size),
                                labels[..., 1:].view(-1),
                                ignore_index=IGNORE_INDEX)
   # 联合 loss
   loss = text_loss + action_loss
   ```
3. 参考 `hybrid_pi05_arch.py:455-512` 的实现模式，处理 `has_text` / `has_action` 掩码。

### 8.4 参考实现

最直接的参考是 `dexbotic/model/pi05/hybrid_pi05_arch.py`，它完整实现了 text_loss + action_loss 的双 Loss 架构。将其适配到 DM0 的 Merged Attention 框架中，主要工作量在于正确处理 prefix_out 中文本部分的 logits 提取和 label 对齐。

---

## 附录 A: 完整代码引用表

| 代码位置 | 内容 |
|---|---|
| `dexbotic/data/dataset/transform/action.py:61-90` | `AddAction` — 状态偏移得到动作 |
| `dexbotic/data/dataset/transform/action.py:93-153` | `DeltaAction` — 计算增量动作 |
| `dexbotic/data/dataset/transform/action.py:156-226` | `AddTrajectory` — 轨迹分块 |
| `dexbotic/data/dataset/transform/action.py:229-278` | `ActionNorm` — 仅归一化（DM0 用） |
| `dexbotic/data/dataset/transform/action.py:281-398` | `ActionNormAnd2String` — 归一化+量化+字符串化 |
| `dexbotic/data/dataset/transform/action.py:379-385` | `_norm_action` — 归一化到 [-1,1] |
| `dexbotic/data/dataset/transform/action.py:387-391` | `_action2bin` — 量化为 bin 索引 |
| `dexbotic/data/dataset/transform/action.py:393-398` | `_bin2string` — bin 索引转字符串 |
| `dexbotic/exp/base_exp.py:355-367` | `TokenizerConfig.add_special_tokens` — 特殊 token 注册 |
| `dexbotic/exp/base_exp.py:370-417` | `ActionConfig` — 动作配置（默认 vocab_size=255） |
| `dexbotic/exp/base_exp.py:393` | `vocab_size: int = field(default=255)` |
| `dexbotic/exp/base_exp.py:803-807` | 调用 `add_special_tokens` 的位置 |
| `dexbotic/exp/dm0_exp.py:244-264` | `DM0ActionConfig` — DM0 管线用 ActionNorm |
| `dexbotic/model/dm0/dm0_arch.py:35-42` | `DM0Config` — action_dim=32, chunk_size=50 |
| `dexbotic/model/dm0/dm0_arch.py:63-92` | `DM0Model` — Action Expert 初始化 |
| `dexbotic/model/dm0/dm0_arch.py:80` | `embed_tokens = None` — 禁用 token 路径 |
| `dexbotic/model/dm0/dm0_arch.py:406-511` | `forward` — 训练前向，只有 Flow Matching loss |
| `dexbotic/model/dm0/dm0_arch.py:495-502` | Flow Matching loss 计算 |
| `dexbotic/model/dm0/dm0_arch.py:513-583` | `inference_action` — Euler 采样推理 |
| `dexbotic/model/dm0/dm0_arch.py:585-641` | `_denoise_step` — 单步去噪 |
| `dexbotic/model/discrete_vla/discrete_vla_arch.py:24-50` | DiscreteVLA 自回归推理 |
| `dexbotic/model/discrete_vla/discrete_vla_arch.py:52-58` | `_discrete_action_to_continuous` 反量化 |
| `dexbotic/model/oft/action_model/model.py:275-347` | `DiscreteActionHead` — 模型侧量化/反量化 |
| `dexbotic/model/oft/action_model/model.py:303-313` | `discretize_actions` — 量化 |
| `dexbotic/model/oft/action_model/model.py:325-347` | `discrete_tokens_to_continuous` — 反量化 |
| `dexbotic/model/oft/oft_discrete_arch.py:13` | `num_bins: Optional[int] = 256` |
| `dexbotic/model/oft/oft_discrete_arch.py:26-204` | OFT-Discrete forward（并行解码+交叉熵） |
| `dexbotic/model/oft/oft_discrete_arch.py:168` | `lm_head` 投影到 action logits |
| `dexbotic/model/oft/oft_discrete_arch.py:187-191` | 交叉熵 loss 计算 |
| `dexbotic/model/oft/oft_discrete_arch.py:206-235` | Argmax 推理 |
| `dexbotic/model/oft/oft_discrete_arch.py:237-282` | Multinomial 采样推理 |
| `dexbotic/model/pi05/hybrid_pi05_arch.py:455-479` | text_loss（L_AR）计算 |
| `dexbotic/model/pi05/hybrid_pi05_arch.py:481-504` | action_loss（L_FM）计算 |
| `dexbotic/model/pi05/hybrid_pi05_arch.py:506-512` | L_total = text_loss + action_loss |
| `dexbotic/exp/discrete_vla_exp.py:85` | DiscreteVLA 推理中 vocab_size=255 |

## 附录 B: 关键类/函数签名速查

```python
# === 数据侧 ===

# 归一化 + 量化 + 字符串化（三步合一）
class ActionNormAnd2String:
    def __init__(self, statistic_mapping, vocab_size=255, string_format=' {value}', add_answer=True)
    def __call__(self, episode_data_dict) -> dict    # 输入/输出 episode dict
    def _norm_action(self, action, min, max) -> np.array           # [-1, 1]
    def _action2bin(self, action, vocab_size) -> np.array           # [0, N-1] 整数
    def _bin2string(self, action, string_format) -> list[str]       # 字符串列表

# 仅归一化（DM0 用）
class ActionNorm:
    def __init__(self, statistic_mapping, strict=True, use_quantiles=False)
    def _normalize(self, data, stats) -> np.array                  # [-1, 1] 或 z-score

# === 模型侧 ===

# 特殊 token 注册
class TokenizerConfig:
    use_special_tokens: bool = False
    def add_special_tokens(self, special_token_format, vocab_size, tokenizer, model) -> tokenizer

# 模型侧量化/反量化（OFT-Discrete 用）
class DiscreteActionHead(nn.Module):
    def __init__(self, input_dim, vocab_size, action_dim=7, action_chunk=16, num_bins=256)
    def discretize_actions(self, actions) -> torch.LongTensor       # [-1,1] → [0, N-1]
    def continuous_to_discrete_tokens(self, actions) -> torch.LongTensor  # → flatten
    def discrete_tokens_to_continuous(self, token_ids) -> torch.Tensor    # [0, N-1] → [-1,1]

# DiscreteVLA 反量化
class DiscreteVLAForCausalLM:
    def _discrete_action_to_continuous(self, action_str, vocab_size) -> np.array

# DM0 模型
class DM0ForCausalLM(DexboticForCausalLM):
    def forward(self, ..., actions, ...) -> CausalLMOutputDexbotic  # 只有 Flow Matching loss
    def inference_action(self, ..., diffusion_steps=10) -> torch.Tensor  # Euler 采样
```
