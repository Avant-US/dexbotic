# dm0_txtAsEvry2_op46.md vs dm0_txtAsEvry2_op47.md — 全维度深度对比

> 创建日期: 2026-05-16
> 关联文档:
> - [dm0_txtAsEvry2_op46.md](./dm0_txtAsEvry2_op46.md) — Opus 4.6 版全栈分析与实现方案
> - [dm0_txtAsEvry2_op47.md](./dm0_txtAsEvry2_op47.md) — Opus 4.7 版全栈实现方案
> - [dm0.md](./dm0.md) — DM0 论文解读
> - [dm0_1.md](./dm0_1.md) — DM0 代码初始分析
> - [dm0_analyz_xp0515.md](./dm0_analyz_xp0515.md) — 各模型 token 生成能力盘点

---

## 0. TL;DR

两篇文档是同一命题（让 DM0 恢复 token 输出能力）的独立分析，核心结论完全一致（自写 generate、接上 lm_head、覆盖 tokenizer max_length），但侧重点和深度各有不同。以下基于**实际代码验证**给出全维度对比。

**核心结论**：以 op46 为实施主本（代码完整、风险覆盖全），但用 op47 的损失合并公式替换 op46 的简化版（加入 `has_text` / `has_action` per-sample mask），并参考 op47 的序列图理解 hybrid 模式的数据流。

---

## 1. 总体相同点

两篇文档在以下方面**完全一致**（经代码验证全部正确）：

1. **问题诊断 — 三道断路**：
   - ① `generate()` 不可用（merged attention 使 HF `GenerationMixin` 无法直接调）
   - ② `lm_head` 是空壳（checkpoint 不含权重，注释明确说 "compatibility with parent class tie_weights"）
   - ③ `tokenizer.model_max_length=100` 截断过短（`DM0Tokenization._max_len` 直接读此值）

2. **三档方案结构**：A（零训练嫁接）→ B（SFT 微调+推理改造）→ C（全量混合训练），推荐方案 B

3. **首要参考对象**：`HybridPi05ForCausalLM.generate()`（`hybrid_pi05_arch.py:672-810`），因为它与 DM0 架构最同构（VLM + Action Expert + merged attention + flow matching）

4. **关键技术决策**：
   - 必须自写 `generate()` 而非用 HF 标准接口
   - `lm_head` 用 weight tying 绑定到 `embed_tokens`
   - `tokenizer.model_max_length` 在代码层覆盖到 2048
   - 用 `has_text` / `has_action` mask 隔离双 loss
   - 冻结策略分 3 阶段

5. **配置扩展**：都新增 `max_new_tokens` / `temperature` / `do_sample` / `return_text` 等 `InferenceConfig` 字段

---

## 2. 总体不同点

| 维度 | op46 (Opus 4.6) | op47 (Opus 4.7) |
|------|-----------------|-----------------|
| **定位** | "全栈分析与实现方案" — 训练+推理 | "全栈实现方案" — 偏推理侧 |
| **数学推导** | 完整公式推导（$\mathcal{L}_{\text{AR}}$, $\mathcal{L}_{\text{FM}}$, $T_{\text{output}}^{\max}$, token 容量公式） | 简要公式，够用但不展开 |
| **generate() 伪代码** | 适配 DM0 的 `_merged_attention_forward` 接口，显式处理 `position_ids` vs `position_embeddings` 差异 | 更贴近 HybridPi05 原始结构，列出差异点但未完全展开适配代码 |
| **forward 双 loss 代码** | §4.3 给出精确的插入位置和完整代码块（~25 行） | §4.2 给出同等逻辑但以架构图为主 |
| **`dm0_prog_arch.py` 对称改动** | 明确列为独立改动项（~105 行），含 progress 通道注意事项 | 提及但未展开 |
| **风险项** | 9 项（多出"config.json architectures 与实际 DM0Prog 不一致"） | 8 项 |
| **max_length 矩阵** | 6 层配置项逐一列出，含公式链 | 6 层配置项，公式略简 |
| **Flask 接口** | 单路由 `/process_frame` + `mode` 参数路由 | 多路由 `/process_frame` + `/process_text` + `/process_hybrid` |
| **代码量估算** | 方案 B 总 ~260 行（含 prog_arch 对称改动） | 方案 B 总 ~160 行（不含 prog_arch） |
| **Gantt 时间线** | 更紧凑（B-1 起步 5 天模型改造） | 稍宽松（B-1 起步 5+2 天） |

---

## 3. 深度对比：generate() 实现

### 3.1 接口适配

op46 给出了**完整的适配后伪代码**（§4.4，约 90 行），显式处理了 DM0 与 HybridPi05 的接口差异。通过实际代码验证，所有差异判断**全部正确**：

| 接口差异 | op46 判断 | 实际代码验证 | 准确性 |
|---------|----------|-------------|-------|
| `_merged_attention_forward` 接收 `position_ids` | 正确 | 签名：`position_ids: torch.LongTensor` (`dm0_arch.py:278`) | ✅ |
| `_inner_forward_mot` 接收 `position_embeddings` | 正确 | 签名：`position_embeddings: Optional[torch.Tensor]`，传入 `rotary_emb()` 返回的 `(cos, sin)` tuple (`hybrid_pi05_arch.py:129`) | ✅ |
| DM0 不需要 embedding 缩放 | 正确 | DM0 的 `get_prefix_hidden_states` 直接调 `embed_language_tokens(input_ids)` 无缩放；HybridPi05 显式做 `* hidden_size**0.5` (`hybrid_pi05_arch.py:288`) | ✅ |
| DM0 无 `adarms_cond` | 正确 | `_merged_attention_forward` 签名只有 6 个参数，无 `adarms_cond` (`dm0_arch.py:270-283`) | ✅ |
| 返回值数量 | 正确 | DM0 返回 `(decoder_embeds_list, past_key_values)` = 2 项；HybridPi05 返回 4 项 | ✅ |

op47 **没有给出适配后的完整伪代码**，而是引用 HybridPi05 原始代码 + 差异表。差异列表正确但需要开发者自己做适配。

### 3.2 Position 计算

两篇文档在 decode 阶段 position 计算上语义等价但写法不同：

```python
# op46:
decode_position = (context_len + step + 1 - 1).unsqueeze(1)  # 直接计算

# HybridPi05 原始 (op47 引用):
decode_position = context_mask.sum(dim=1, keepdim=True) - 1   # 用 mask sum
```

两种方式等价（`context_mask` 每步 cat 一个 True，sum 就是 prefix_len + step）。HybridPi05 方式在 hybrid 模式下更通用（text 生成完后 action 还要用同一个 mask）。

### 3.3 Decode Mask 构建

**op46** 绕过了 DM0 的 `make_attn_mask_2d` 接口，直接构建全 True mask：
```python
decode_mask_2d = torch.ones(batch_size, 1, full_len, device=device, dtype=torch.bool)
```

**HybridPi05 原始**（op47 引用）用 `context_mask`：
```python
decode_mask = make_attn_mask_4d(context_mask[:, None, :])
```

**op46 的做法有一个 padding 隐患**：全 True mask 会让新 token attend 到 batch 中的 padding 位置，而 HybridPi05 的 `context_mask` 方式在 padding 位置自动为 False。

**建议**：实现时应采用 HybridPi05 的 `context_mask` 方式。

### 3.4 verdict

| 维度 | op46 | op47 |
|------|------|------|
| 可直接复制实现 | ✅ 可以大段抄 | ❌ 需要自己适配 |
| 接口适配准确性 | ✅ 全部正确 | ✅ 差异识别正确但未展开 |
| decode mask 构建 | ⚠️ 全 True mask 有 padding 隐患 | 未展开，留给实现者 |
| hybrid 模式支持 | ✅ 有 `return_action` 分支 | ✅ 序列图更清晰 |

---

## 4. 深度对比：训练侧 forward 双 loss

### 4.1 代码精确性

op46 §4.3 给出精确插入方案，定位在 `dm0_arch.py:495-511` 后：

```python
text_logits = self.lm_head(prefix_out)
pred_tokens = text_logits[:, -text_len:-1, :]
target_tokens = labels[:, 1:]
token_loss = F.cross_entropy(pred_tokens.transpose(1, 2), target_tokens, reduction="none")
```

对照 HybridPi05 forward (`hybrid_pi05_arch.py:458-479`) 验证，关键对齐点**全部一致**：

| 对齐点 | HybridPi05 实际代码 | op46 方案 |
|--------|-------------------|----------|
| `text_logits = self.lm_head(prefix_out)` | line 458 ✅ | ✅ |
| `pred_tokens = text_logits[:, -text_len:-1]` | line 464 ✅ | ✅ |
| `target_tokens = labels[:, 1:]` | line 462 ✅ | ✅ |
| `torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)` | line 469 ✅ | ✅ |
| `has_text_mask` 处理 | lines 473-476 ✅ | ✅ |

op46 额外加了 `torch.amp.autocast("cuda", dtype=torch.float32)` 上下文管理器，强制 CE loss 在 fp32 下计算 — 这是 HybridPi05 没有但数值稳定性更好的实践。

op47 用架构图 + 改动表格展示逻辑，没有给出完整代码块。

### 4.2 损失合并

**op46**（简化版）：
```python
loss = action_loss
if text_loss is not None:
    loss = loss + text_loss
```

**op47**（完整版，对齐 HybridPi05）：
```python
loss = λ_AR · text_loss · 1_text + λ_FM · action_loss · 1_action
```

实际 HybridPi05 (`hybrid_pi05_arch.py:506-512`) 用 `has_text_mask` / `has_action_mask` 分别加权后再相加。**op47 的方式更严谨**，支持 batch 内混合训练（text-only + action-only 样本）。op46 的简化版在所有样本都同时有 text 和 action 时等价，但**不支持混合训练**。

### 4.3 隐含假设的验证

两篇文档都用 `text_logits[:, -text_len:-1]` 对齐 `labels[:, 1:]`，隐含假设 `labels` 长度等于 `input_ids.shape[1]`。

代码验证：`DM0Tokenization.__call__()` 返回的 `labels` 确实只覆盖文本 token，视觉 token 在 `get_prefix_hidden_states` 中单独处理。**假设正确**，但两篇文档都没有显式说明。

---

## 5. 深度对比：DM0Prog 继承问题

### 5.1 问题本质

`DM0ProgForCausalLM`（`dm0_prog_arch.py:133`）**没有覆写 `forward()` 方法**。继承链：

```
DM0ProgForCausalLM
  └─ DexboticForCausalLM (提供 forward: 标准 LM → backbone → lm_head → CE loss)
  └─ ActionOutputForCausalLM (mixin, 无 forward)
```

因此 `DM0ProgForCausalLM.forward()` 走的是 `DexboticForCausalLM.forward()`（`dexbotic_arch.py:429-496`）— **标准 LM 前向**，完全跳过 flow matching。

而 `DM0ForCausalLM` 覆写了 `forward()`（`dm0_arch.py:406-511`）— **flow matching 前向**（`_merged_attention_forward` → `action_out_proj` → MSE loss）。

### 5.2 Checkpoint 矛盾

`table30_generalist_aloha` checkpoint 中包含 `progress_in_proj` 和 `progress_out_proj` 权重（`model.safetensors.index.json:966-969`），但：

| 证据 | 事实 |
|------|------|
| `config.json` 的 `architectures` | `["DM0ForCausalLM"]`（不是 `DM0ProgForCausalLM`） |
| `DM0ForCausalLM` 的 `DM0Model` | 没有 progress 层 |
| `DM0ProgForCausalLM` | 有 progress 层但没有 flow matching forward |
| `DM0Exp` 和 `DM0InferenceConfig._load_model()` | 硬编码 `DM0ForCausalLM.from_pretrained()` (`dm0_exp.py:330`) |

**唯一合理解释**：训练时用了 `DM0ProgForCausalLM`（或未开源变体），发布 checkpoint 时 config 被错误标注。开源代码中 forward 遗漏可能因为：
- 内部版本有 forward，开源时遗漏
- progress 层通过独立训练脚本训练
- DM0Prog 只是推理变体，progress 层从其他训练阶段迁移

### 5.3 对 token 输出恢复的影响

| 影响维度 | 说明 |
|---------|------|
| **推理侧（用 DM0ForCausalLM）** | progress 权重被静默丢弃，与 token 输出无关 — generate() 只走 LLM 侧 |
| **推理侧（用 DM0ProgForCausalLM）** | generate() 也需要加到 `DM0ProgForCausalLM`，且要同时处理 progress 通道 |
| **训练侧** | 给 `DM0ProgForCausalLM` 加 forward() 需要处理 flow matching + progress + text loss 三通道，比"对称改动"更复杂 |
| **代码量** | `DM0ProgForCausalLM` 需要**从零写 forward()**，约 80-100 行 |

### 5.4 两篇文档的处理

**op46**：风险表第 9 项明确列出，建议 "加载时用 `DM0ProgForCausalLM` 或修正 `architectures`"。代码量估算为 dm0_prog_arch.py ~105 行。但称之为"对称改动"低估了 forward() 缺失的影响。

**op47**：改动清单提到 "DM0ProgForCausalLM 需要做对称改动"，但**完全没有识别出它缺少 forward()** 这个问题，也没有在风险表中覆盖。

**注**：[dm0_1.md](./dm0_1.md) 最早识别了这个问题："`DM0ProgForCausalLM` — 没有 forward 方法, 不可直接训练"。

---

## 6. 深度对比：Mask 构建

### 6.1 函数签名对比

| 特性 | DM0 `make_attn_mask_2d` | HybridPi05 `make_attn_mask` |
|------|------------------------|---------------------------|
| 文件 | `dm0_utils.py:12-40` | `hybrid_pi05_arch.py:24-30` |
| 参数 1 | `padding_mask: BoolTensor[B,N]` | `input_mask: BoolTensor[B,N]` |
| 参数 2 | `attn_mask: IntTensor[B,N]` | `ar_mask: BoolTensor[B,N]` |
| 返回 | `BoolTensor[B,N,N]` | `BoolTensor[B,N,N]` |

名字不同，但**核心算法完全等价**：

```python
# 两者共用的核心逻辑：
cumsum = torch.cumsum(attn_mask, dim=1)
mask_2d = cumsum[:, None, :] <= cumsum[:, :, None]
valid_2d = padding_mask[:, None, :] * padding_mask[:, :, None]
return mask_2d & valid_2d
```

### 6.2 `make_attn_mask_4d` 对比

```python
# DM0 (dm0_utils.py:78-92):
def make_attn_mask_4d(attn_mask_2d, dtype=torch.bfloat16):
    attn_mask_4d = torch.where(attn_mask_2d, 0.0, -2.3819763e38)[:, None, :, :]
    return attn_mask_4d.to(dtype)

# HybridPi05 (hybrid_pi05_arch.py:33-35):
def make_attn_mask_4d(attn_mask):
    attn_mask = torch.where(attn_mask, 0.0, -2.3819763e38)[:, None]
    return attn_mask
```

**完全一致**。DM0 版本多了 `dtype` 参数，但行为等同。

### 6.3 DM0 独有：`make_suffix_attn_mask_2d`

DM0 有 `make_suffix_attn_mask_2d`（`dm0_utils.py:43-75`），用于 `_denoise_step` 中构建 suffix 对 prefix+suffix 的 attention mask：

```python
# DM0: 拼接完整矩阵再切行
full_mask = make_attn_mask_2d([prefix + suffix], [prefix + suffix])  # [B, P+S, P+S]
return full_mask[:, -S:, :]                                           # [B, S, P+S]
```

HybridPi05 没有这个函数，在 `_denoise_step_with_cache` 中手工构建：
```python
# HybridPi05: 分块构建再拼列
suffix_mask = make_attn_mask(suffix_mask, suffix_ar_mask)                    # [B, S, S]
context_mask = context_mask.unsqueeze(1).repeat(1, S, 1)                     # [B, S, P]
full_mask = make_attn_mask_4d(torch.cat([context_mask, suffix_mask], dim=-1)) # [B, S, P+S]
```

| 方式 | 计算量 | 正确性 | 易出错性 |
|------|--------|-------|---------|
| DM0（切行） | $O((P+S)^2)$ — 构建了无用的 prefix-to-prefix 部分 | ✅ | 低（复用已验证函数） |
| HybridPi05（拼列） | $O(S \times P + S^2)$ — 只构建需要的行 | ✅ | 略高（手工拼接） |

### 6.4 关键架构差异：Prefix Attention 类型

**DM0** — prefix 内用 **causal** attention：
```python
# dm0_arch.py:340-348 (get_prefix_hidden_states)
attn_mask_list += [1] * num_img_tokens   # 全 1
attn_mask_list += [1] * num_lang_tokens  # 全 1
# cumsum = [1, 2, 3, ..., N] → causal mask
```

**HybridPi05** — prefix 内用 **bidirectional** attention：
```python
# hybrid_pi05_arch.py:285-295 (embed_prefix)
ar_mask = torch.tensor([False] * (num_img_tokens + num_lang_tokens))
# cumsum = [0, 0, 0, ..., 0] → bidirectional
```

这反映了底层 LLM 的差异：DM0 基于 Qwen3（标准 causal LM），HybridPi05 基于 Gemma（prefix 编码时用 bidirectional）。

**对 generate() 的影响**：decode 阶段，由于每步只有 1 个 query token + KV cache（cache 里没有未来 token），causal vs bidirectional 的差异**不影响 decode 的正确性**。但需要注意不能在 DM0 的 prefix 上假设 bidirectional 语义。

### 6.5 对 op46/op47 的影响

op46 的 generate 伪代码用全 True mask（`torch.ones(...)` ），没有处理 batch padding。HybridPi05 的 `context_mask[:, None, :]` 方式自然处理 padding。

**实现建议**：DM0 的 generate() decode mask 应**照搬 HybridPi05 的 `context_mask` 方式**，DM0 的 `make_attn_mask_4d` 接口兼容可直接使用。

---

## 7. 风险分析逐项对照

| # | 风险 | op46 | op47 | 代码验证 |
|---|------|------|------|---------|
| 1 | `lm_head ← embed_tokens` weight tie 效果差 | ✅ 列出 | ✅ 列出 | 合理。`table30` config `tie_word_embeddings: false` |
| 2 | merged attention KV cache 与自写 generate 不兼容 | ✅ 概率"低" | ❌ 未单独列 | HybridPi05 已验证可行 |
| 3 | `model_max_length=100` 截断 SFT assistant turn | ✅ 概率"高" | ✅ 概率"高" | **已验证**：`process.py:383` `_max_len = tokenizer.model_max_length` |
| 4 | pop empty assistant turn 误删 SFT 标签 | ✅ 概率"高" | ✅ 概率"高" | **已验证**：`process.py:407-414` 无条件 pop |
| 5 | HF `GenerationMixin.generate()` 被误调用 | ✅ 概率"中" | ✅ 概率"高" | 自写 `generate()` 遮蔽 mixin 同名方法 |
| 6 | SFT 破坏 flow matching 能力 | ✅ | ✅ | 冻结策略缓解 |
| 7 | action_expert KV cache 在 text decode 时占显存 | ✅ 概率"低" | ✅ 概率"低" | 一致 |
| 8 | 大 max_new_tokens 推爆 RoPE | ✅ 概率"极低" | ✅ 概率"极低" | 一致 |
| 9 | **config architectures 与 DM0Prog 不一致** | ✅ **op46 独有** | ❌ 未覆盖 | **已验证**：checkpoint 含 progress 权重但 config 标 `DM0ForCausalLM` |
| 10 | **向后兼容老 checkpoint** | ❌ 未单独列 | ✅ **op47 独有** | 默认 `mode='action'` 可缓解 |
| 11 | **stopping_criteria 接口偏差** | ❌ 未覆盖 | ✅ **op47 独有** | 自写 generate 需手动实现 stop 词 |

**风险覆盖完整性**：op46 (9项) > op47 (8项 + 2独有)。op46 的独有风险 #9 是最关键的发现。

---

## 8. 其他维度对比

### 8.1 Flask 接口设计

| 方案 | op46 | op47 |
|------|------|------|
| 路由设计 | 单路由 `/process_frame` + `mode` 参数 | 多路由 `/process_frame` + `/process_text` + `/process_hybrid` |
| 优点 | 向后兼容好、API 表面积小 | RESTful 清晰、职责单一 |
| 缺点 | 单路由承载多语义 | 新路由需客户端感知 |

当前代码（`dm0_exp.py:369-372`）只有一个 `/process_frame` 路由，**op46 的方案改动更小**。

### 8.2 代码量估算

| 模块 | op46 估算 | op47 估算 | 差异原因 |
|------|----------|----------|---------|
| dm0_arch.py | ~100 行 | ~100 行 | 一致 |
| dm0_prog_arch.py | ~105 行 | ~30 行 | op46 按独立改动计算；op47 只算增量 |
| dm0_exp.py | ~45 行 | ~50 行 | 基本一致 |
| process.py | ~7 行 | ~10 行 | 基本一致 |
| **总计** | **~260 行** | **~160 行** | 差 100 行，主要来自 prog_arch |

op46 的估算更诚实 — 计入了 `dm0_prog_arch.py` 的对称改动，而 op47 低估了这部分工作。实际上 `DM0ProgForCausalLM` 缺少 forward() 意味着"对称改动"的说法不准确，需要从零写 forward。

### 8.3 数学推导深度

**op46** 的数学部分更系统：
- 给出了 token 生成过程的完整公式链（prefix 编码 → 自回归解码 → 采样）
- 推导了 $T_{\text{output}}^{\max} = \min(\texttt{max\_new\_tokens}, P_{\max} - L_{\text{prefix}})$ 并代入实际数值
- max_length 配置矩阵给出了 6 层约束的层级关系和具体数值链

**op47** 的数学部分够用但更简略 — 给出了同样的公式但没有代入数值展开。

---

## 9. 各自优缺点汇总

### 9.1 op46 (Opus 4.6)

**优点**：
- ★★★★★ **全栈覆盖最完整** — forward 双 loss 代码、generate 伪代码、配置链、Flask 接口都可直接使用
- ★★★★★ **对 DM0 接口的适配最深** — generate 中显式处理了 `position_ids` vs `position_embeddings` 差异
- ★★★★★ **风险覆盖最全** — 9 项，含关键的 DM0Prog 架构不一致
- ★★★★ **数学严谨** — 公式推导链完整
- ★★★★ **包含 dm0_prog_arch.py 改动** — 不遗漏实际 checkpoint 的问题

**缺点**：
- ★★★ **篇幅过长** — 1200+ 行，阅读成本高
- ★★ **decode mask 有 padding 隐患** — 全 True mask 没有处理 batch padding
- ★★ **损失合并过于简化** — 不支持 batch 内混合训练

### 9.2 op47 (Opus 4.7)

**优点**：
- ★★★★★ **结构精炼** — 改动清单用表格呈现，扫一眼就知道改哪里
- ★★★★★ **序列图最清晰** — hybrid 模式的时序图画得更详细
- ★★★★★ **损失公式更通用** — 支持 has_text/has_action per-sample mask
- ★★★★ **HybridPi05 参考代码引用直接** — 给出具体行号范围
- ★★★★ **向后兼容性列为独立风险项**

**缺点**：
- ★★ **未覆盖 dm0_prog_arch.py** — 实际 checkpoint 使用的类，遗漏它方案不完整
- ★★ **generate 适配不够深入** — 列出差异但没有给出适配后的完整伪代码
- ★★ **代码量估算偏低** — 160 行没有计入 prog_arch 的改动
- ★ **完全遗漏 DM0Prog forward() 缺失** — 这是一个关键风险

---

## 10. 最终建议

### 10.1 实施蓝本选择

**以 op46 为主、op47 为辅**：

| 取用来源 | op46 | op47 |
|---------|------|------|
| forward 双 loss 代码 | ✅ 主 | |
| generate() 伪代码 | ✅ 主 | |
| 损失合并公式 | | ✅ 主（加入 has_text/has_action per-sample mask） |
| hybrid 模式数据流 | | ✅ 主（序列图更清晰） |
| decode mask 构建 | | ✅ 用 HybridPi05 的 `context_mask` 方式替换 op46 的全 True mask |
| 风险清单 | ✅ 主（9 项） | 补充 #10（向后兼容）和 #11（stopping_criteria） |
| Flask 接口 | ✅ 单路由 + mode 参数 | |

### 10.2 两篇文档都遗漏的问题

1. **DM0ProgForCausalLM 缺少 forward() 不只是"对称改动"** — 需要从零实现，处理 flow matching + progress + text loss 三通道
2. **decode mask 的 batch padding 处理** — 两篇都没有完整解决，应用 `context_mask` 方式
3. **`labels` 与 `prefix_out` 对齐的隐含假设** — 两篇都正确使用但没有显式验证
4. **prefix 内 causal vs bidirectional 的差异** — DM0 (causal) vs HybridPi05 (bidirectional)，两篇都没有提及但不影响 decode 正确性

### 10.3 评分总表

| 评价维度 | op46 | op47 |
|---------|------|------|
| 作为实施蓝本 | ★★★★★ | ★★★☆☆ |
| 风险覆盖 | ★★★★★ | ★★★★☆ |
| 接口适配准确性 | ★★★★★ | ★★★★☆ |
| 阅读效率 | ★★★☆☆ | ★★★★★ |
| 训练侧设计 | ★★★★☆ | ★★★★★ |
| 推理侧设计 | ★★★★★ | ★★★★☆ |
| 向后兼容考虑 | ★★★★☆ | ★★★★☆ |
