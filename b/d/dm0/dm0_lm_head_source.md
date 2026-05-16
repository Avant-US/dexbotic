# 寻找 DM0 "真正的" lm_head 权重 — 分析与可行方案

## Context

DM0 checkpoint (`DM0-table30_generalist_aloha`) 中缺失顶层 `lm_head.weight`，只有 `model.action_expert.lm_head.weight` (151936×1024, 属于 action expert) 和 `model.llm.embed_tokens.weight` (152701×2048, 属于 LLM)。前一轮分析已排除 `action_expert.lm_head` 作为文本生成头的可能性。问题是：**真正能用于文本生成的 lm_head 权重在哪里？**

---

## 核心发现：embed_tokens 就是你要找的 lm_head

### 推理链

| 步骤 | 事实 | 来源 |
|------|------|------|
| 1 | Qwen3-1.7B 原始模型使用 `tie_word_embeddings: true` | HuggingFace Qwen/Qwen3-1.7B config.json |
| 2 | 当 `tie_word_embeddings=true` 时, `lm_head.weight ≡ embed_tokens.weight`, HF 只保存一份 (embed_tokens), 加载时自动重建 lm_head | HuggingFace Transformers 源码 & 社区 Issues (#44060, #39812) |
| 3 | DM0 从 Qwen3-1.7B 初始化 → 初始化时 embed_tokens = lm_head (同一份权重) | `dexbotic/model/dexbotic_arch.py:62` `AutoModel.from_config(llm_config)` |
| 4 | DM0 将 vocab 从 151936 resize 到 152701 (加了765个特殊token) | `dexbotic/model/dexbotic_arch.py:87` `resize_token_embeddings(152701)` |
| 5 | DM0 创建了独立的 `self.lm_head` 但标注 "for compatibility with parent class tie_weights", 且设置 `tie_word_embeddings: false` | `dexbotic/model/dm0/dm0_arch.py:139-142` |
| 6 | DM0 训练只用了 L_FM, 没有 L_AR → 顶层 lm_head 从未参与训练 → checkpoint 中不存在 `lm_head.weight` | `dm0_arch.py:502` `loss = action_loss` |
| 7 | `freeze_llm=False` (默认值) → embed_tokens 在训练中接收了来自 L_FM 的梯度 | `dexbotic/exp/base_exp.py:289` |
| 8 | checkpoint 中 embed_tokens.weight 形状 = [152701, 2048] → 完全覆盖全词表 | `model.safetensors.index.json` |

**结论: `model.llm.embed_tokens.weight` 就是从 Qwen3-1.7B 继承下来的、经过 DM0 训练后的 lm_head 权重。它们在 Qwen3 中本来就是同一个张量。**

---

## 方案排序 (从最可行到最不可行)

### 方案 A: 直接 weight tying (最可行, 零训练成本)

**做法**: 把 lm_head.weight 绑定到 embed_tokens.weight

```python
# 在加载模型后:
model.lm_head.weight = model.model.llm.embed_tokens.weight
```

**优点**:
- 零额外训练, 即插即用
- embed_tokens 正是 Qwen3-1.7B 中与 lm_head 共享的那个张量
- 形状正确 [152701, 2048], 覆盖全部 token 包括 765 个特殊 token
- 符合 Qwen3 小模型 (0.6B/1.7B/4B) 的原始设计

**风险**:
- embed_tokens 在 DM0 训练中通过 L_FM 梯度有所漂移 (但 LLM 的核心文本表示能力应基本保留)
- 如果漂移过大, 生成文本质量会下降 → 可通过实测验证

### 方案 B: 从原始 Qwen3-1.7B 提取 (次选)

**做法**: 下载 `Qwen/Qwen3-1.7B`, 提取其 embed_tokens (=lm_head), 再拼接 DM0 的 765 个特殊 token 的 embedding

```python
from transformers import AutoModelForCausalLM

qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
original_lm_head = qwen.lm_head.weight.data  # [151936, 2048]

# 扩展到 152701 行: 前 151936 行用原始 Qwen3 权重, 后 765 行用 DM0 embed_tokens
dm0_embed = model.model.llm.embed_tokens.weight.data
extended_lm_head = torch.cat([
    original_lm_head,           # [151936, 2048] — 原始文本最优
    dm0_embed[151936:, :]       # [765, 2048] — DM0 训练过的特殊 token
], dim=0)
model.lm_head.weight.data = extended_lm_head
```

**优点**:
- 原始 151936 个文本 token 保证是未被 L_FM 污染的纯文本权重
- 765 个特殊 token (robot_id, action tokens 等) 用 DM0 的训练版本

**风险**:
- **分布不匹配**: DM0 的 hidden states 已经通过 L_FM 微调发生了偏移, 用原始 Qwen3 的 lm_head 可能产生不匹配
- 需要额外下载 Qwen3-1.7B (~3.4GB)

### 方案 C: 检查其他 DM0 HuggingFace 变体

**做法**: 检查 `DM0-base`, `DM0-libero`, 其他 specialist/generalist 模型的 `model.safetensors.index.json`, 看是否有变体保存了 `lm_head.weight`

**可能性**: 低。所有 DM0 变体共享相同的架构代码, L_AR 在任何变体中都未实现, 所以 lm_head 在所有变体中大概率都是缺失的。

### 方案 D: 小规模 L_AR 微调 (最稳健)

**做法**: 采用方案 A 或 B 初始化 lm_head, 然后在少量文本数据上做 L_AR 微调

**优点**: 最稳健, 可以修正任何漂移
**缺点**: 需要训练资源和合适的文本数据

---

## 推荐执行路径

1. **首先试方案 A** — weight tying, 一行代码改动, 立即可测试
2. 测试文本生成质量 (给模型一个简单的 prompt, 看输出是否通顺)
3. 如果质量不佳 → 试方案 B (用原始 Qwen3 lm_head + DM0 特殊 token)
4. 如果仍不佳 → 方案 D (L_AR 微调)

---

## 附: 为什么 DM0 团队没有保存 lm_head.weight?

不是他们故意剥离, 而是**它从来就不需要被保存**:
- DM0 的顶层 lm_head 是 `nn.Linear(2048, 152701)`, 在 `_real_init()` 中创建, 标注 "for compatibility"
- 训练中从未被 L_AR loss 更新 (L_AR 未实现)
- HuggingFace Trainer 的 `save_model()` 会保存所有有梯度的参数, 但这个 lm_head 虽然 `requires_grad=True`, 梯度始终为 0
- 有可能 DeepSpeed ZeRO-3 优化器在保存时自动跳过了零梯度参数, 或者 DM0 团队在后处理中剥离了它

真正的 "lm_head" 信息一直藏在 `embed_tokens.weight` 里 — 因为 Qwen3-1.7B 的 weight tying 机制。
