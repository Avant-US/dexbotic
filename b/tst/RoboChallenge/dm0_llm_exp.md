# DM0 语言生成恢复与 RoboChallenge 评测实验记录

## 1. 实验目标

本实验围绕 DM0 在开源 dexbotic 代码中的语言 token 生成能力恢复展开。目标不是重新训练模型，而是在现有 checkpoint 上尽量恢复/验证 DM0 生成有语义 token 的能力，并在 RoboChallenge `turn_on_faucet` 任务上记录：

- 输入侧：prompt、机器人/控制模式标签、图像/视频路径、状态向量抽样、token 长度等。
- 输出侧：自然语言 CoT、bbox、2D 轨迹、离散动作 token、连续动作预测、Action MSE 等。
- 运行侧：生成耗时、动作推理耗时、错误数、日志路径与 summary。

评测任务与路径：

- 任务：RoboChallenge `task_table30_turn_on_faucet` / `turn_on_faucet`
- 数据路径：`/mnt/g/DATA/RoboChallenge/task_table30_turn_on_faucet/turn_on_faucet`
- checkpoint 路径：`/mnt/g/CKPT/dexbotic/`
- Python 虚拟环境：`/mnt/r/VENV/venv_dm0_actual/`
- 评测脚本与日志目录：`b/tst/RoboChallenge/`

## 2. 背景结论

前置分析确认了几个关键点：

1. DM0 的连续动作生成主要由 Action Expert/Flow Matching 分支完成；这一路不是传统文本 token 生成。
2. `action_expert.lm_head` 不能替代顶层 VLM/LLM 的 `lm_head` 来做自然语言 token 生成。
3. `DM0-base` 的 `model.safetensors` 中存在顶层 `lm_head.weight`，且实测与 `embed_tokens.weight` 数值一致，可以作为 native `lm_head` 使用。
4. 若 specialist checkpoint 缺少顶层 `lm_head.weight`，则 `from_pretrained()` 会随机初始化该权重；必须在模型加载后显式调用 `tie_lm_head()`，让 `lm_head.weight` 重新引用 `model.llm.embed_tokens.weight`。
5. 开源 dexbotic 的 DM0 训练路径中没有完整启用论文里的自回归语言损失 `L_AR`，因此 specialist checkpoint 的文本 token 质量不应被期待为完整论文版 DM0 的 CoT 能力。

## 3. 代码改动记录

### 3.1 `dexbotic/model/dm0/dm0_arch.py`

主要改动：

- 为 `DM0ForCausalLM` 增加 `tie_lm_head()` 方法。
- 为 `DM0ForCausalLM` 增加自回归 `generate()` 方法，使其能基于图像和文本前缀逐 token 生成文本。
- 在 `generate()` 中修复 attention mask dtype 问题，将 `prefix_pad_mask` 显式转为 `bool`。

关键原因：

- specialist checkpoint 中可能没有顶层 `lm_head.weight`，加载时会出现新初始化；加载完成后再 tie 才能避免随机 head 参与生成。
- DM0 的 merged attention 与通用 HuggingFace `generate()` 不完全兼容，所以需要定制前缀编码、KV cache、位置编码、mask 与采样流程。

### 3.2 `b/tst/RoboChallenge/eval_dm0_turn_on_faucet.py`

该脚本用于早期 specialist checkpoint 评测，主要验证：

- `DM0-table30_turn_on_faucet` 能否加载并完成离线 open-loop action inference。
- `tie_lm_head()` 后能否调用 `generate()` 产出 token。
- `torch.bfloat16` 加载下，action 相关层需要转为 `float32`，否则 `inference_action()` 会出现 dtype mismatch。

关键修复：

- `attn_mask` 从 long 改为 bool。
- 加载后调用 `model.tie_lm_head()`。
- 将 `action_out_proj`、`action_in_proj`、`action_time_mlp_in`、`action_time_mlp_out` 转为 `float32`。

### 3.3 `b/tst/RoboChallenge/eval_dm0_base_turn_on_faucet.py`

该脚本用于 `DM0-base` 的第一轮评测，确认 `DM0-base` native `lm_head` 可以被加载使用。

观察：

- 不显式 tie 时，`DM0-base` 可以使用 safetensors 中自带的顶层 `lm_head.weight`。
- 但生成文本非常短，平均只有约 4 个 token，基本只出现离散 action token 或很快结束。

### 3.4 `b/tst/RoboChallenge/eval_dm0_base_v2.py`

这是最终主要评测脚本，新增了完整 prompt 控制、长文本生成、跨本体身份注入、CoT-then-Action 与 Scaffolding。

主要改动：

- 增加 `build_prompt_with_assistant()`，显式构造带 `ASSISTANT: ` 后缀的 prompt。
- 增加 `robot_tag` 与 `control_tag` 参数，用于跨本体身份注入，例如 `<robot_aloha><joint_control>`。
- 增加 `scaffold_prefix` 参数，例如 `subtask: `，引导模型输出更结构化的 CoT/轨迹内容。
- 增加 `generate_with_min_tokens()`，支持 `min_new_tokens`、`top_p`、`top_k`、`temperature`，避免过早采样 EOS。
- 增加 `append_cot_to_input_ids()`，实现 CoT-then-Action：先生成文本，再把生成 token 拼回输入，作为 action inference 条件。
- 日志中同时记录 CoT 串联后的 Action MSE 和不串联 CoT 的 baseline MSE。

## 4. 运行中遇到的问题与修复

| 问题 | 原因 | 修复 |
| --- | --- | --- |
| specialist checkpoint 加载后 `lm_head.weight` 新初始化 | checkpoint 缺少顶层 `lm_head.weight` | 加载后调用 `model.tie_lm_head()` |
| `make_attn_mask_4d` 报 Long/Bool dtype 错误 | attention pad mask 类型不稳定 | 在 `generate()` 中显式 `.bool()` |
| action inference 出现 Float/BFloat16 mismatch | bf16 模型权重与 float32 中间量混用 | action 相关投影层转 `float32` |
| DM0-base 生成太短 | prompt 缺少 `ASSISTANT: `，且 EOS 过早 | 显式追加 `ASSISTANT: `，并加入 `min_new_tokens` |
| 日志看不到 ALOHA 身份 | prompt 没有注入机器人/控制模式标签 | 加入 `<robot_aloha>` 与 `<joint_control>` |
| 无法直接比较 CoT 是否改善 action | 只有单一路径 MSE | 同时记录 CoT-then-Action MSE 与 baseline MSE |

## 5. 评测结果汇总

### 5.1 Specialist checkpoint：`DM0-table30_turn_on_faucet`

路径：

- checkpoint：`/mnt/g/CKPT/dexbotic/DM0-table30_turn_on_faucet`
- 日志：`b/tst/RoboChallenge/logs/eval_20260516_180602.jsonl`
- summary：`b/tst/RoboChallenge/logs/summary_20260516_180602.json`

配置与结果：

| 指标 | 数值 |
| --- | ---: |
| episodes | 20 |
| frames | 540 |
| frame interval | 30 |
| max new tokens | 128 |
| avg generated tokens | 128.0 |
| avg generation time | 3069.79 ms |
| avg action time | 358.29 ms |
| avg action MSE | 0.00218194 |

观察：

- 该 specialist checkpoint 的连续动作 MSE 很低，说明 action expert 对该任务有明显适配。
- 但文本生成依赖 tied `lm_head`，不代表 checkpoint 原生具备可靠 CoT 生成能力。
- 生成 token 长度达到上限，但语义质量不稳定，不能把它直接等同于论文版 DM0 的真实文本能力。

### 5.2 `DM0-base` 第一轮：native `lm_head`，无长文本修复

路径：

- checkpoint：`/mnt/g/CKPT/dexbotic/DM0-base`
- 日志：`b/tst/RoboChallenge/logs/eval_DM0-base_20260517_011147.jsonl`
- summary：`b/tst/RoboChallenge/logs/summary_DM0-base_20260517_011147.json`

配置与结果：

| 指标 | 数值 |
| --- | ---: |
| episodes | 20 |
| frames | 540 |
| errors | 0 |
| max new tokens | 128 |
| avg gen tokens | 4.074 |
| avg generation time | 359.37 ms |
| avg action time | 515.91 ms |
| avg action MSE | 2.27869 |
| tag types | `discrete_action_tokens` |

观察：

- `DM0-base` native `lm_head` 可正常加载并生成。
- 生成内容过短，基本不能形成 CoT、bbox 或 2D 轨迹。
- 主要原因是 prompt 没有稳定进入 assistant 回答态，模型很快输出结束符。

### 5.3 `DM0-base` native `lm_head` 复核

路径：

- 日志：`b/tst/RoboChallenge/logs/eval_DM0-base-native-lmhead_20260517_014321.jsonl`
- summary：`b/tst/RoboChallenge/logs/summary_DM0-base-native-lmhead_20260517_014321.json`

配置与结果：

| 指标 | 数值 |
| --- | ---: |
| episodes | 20 |
| frames | 540 |
| errors | 0 |
| avg gen tokens | 4.485 |
| avg generation time | 366.58 ms |
| avg action time | 510.27 ms |
| avg action MSE | 2.27554 |
| tag types | `discrete_action_tokens` |

观察：

- 与上一轮一致，说明 native `lm_head` 本身可用，但 prompt 与生成策略不足以诱导长 CoT。
- 这轮结果支持后续对 `ASSISTANT: ` 和 `min_new_tokens` 的修复判断。

### 5.4 `DM0-base v2`：`ASSISTANT: ` + `min_new_tokens`

路径：

- 日志：`b/tst/RoboChallenge/logs/eval_DM0-base_v2_20260517_023612.jsonl`
- summary：`b/tst/RoboChallenge/logs/summary_DM0-base_v2_20260517_023612.json`

配置与结果：

| 指标 | 数值 |
| --- | ---: |
| episodes | 10 |
| frames | 160 |
| errors | 0 |
| max new tokens | 256 |
| min new tokens | 32 |
| temperature / top_p / top_k | 0.7 / 0.95 / 50 |
| avg gen tokens | 71.65 |
| avg generation time | 1817.99 ms |
| avg action time | 361.98 ms |
| avg action MSE | 2.44487 |
| tag types | `discrete_action_tokens`, `xy_coords`, `bbox_xyxy` |

观察：

- 生成长度从约 4 token 增加到约 72 token。
- 日志中开始出现 `xy_coords`、`bbox_xyxy` 这类结构化语义 token。
- 只靠长文本诱导不一定改善 action MSE，说明「生成 CoT」和「让 action expert 使用 CoT」是两个不同问题。

### 5.5 `DM0-base v2`：跨本体身份注入 + CoT-then-Action + Scaffolding

路径：

- 日志：`b/tst/RoboChallenge/logs/eval_DM0-base_v2_20260517_032011.jsonl`
- summary：`b/tst/RoboChallenge/logs/summary_DM0-base_v2_20260517_032011.json`

配置：

| 配置项 | 数值 |
| --- | --- |
| checkpoint | `/mnt/g/CKPT/dexbotic/DM0-base` |
| robot tag | `<robot_aloha>` |
| control tag | `<joint_control>` |
| scaffold prefix | `subtask: ` |
| cot then action | `true` |
| max new tokens | 256 |
| min new tokens | 32 |
| temperature / top_p / top_k | 0.7 / 0.95 / 50 |
| episodes / frames | 10 / 160 |

结果：

| 指标 | CoT-then-Action | Baseline |
| --- | ---: | ---: |
| avg action MSE | 1.96946 | 2.48747 |
| median action MSE | 1.93992 | 2.47981 |
| p90 action MSE | 2.71065 | - |
| avg gen tokens | 82.15 | - |
| avg generation time | 2125.28 ms | - |
| avg action time | 369.56 ms | - |
| errors | 0 | - |

相对 baseline：

$$
\frac{2.4874663230 - 1.9694638558}{2.4874663230} \approx 20.82\%
$$

因此，在这组离线 open-loop Action MSE 指标上，CoT-then-Action + Scaffolding 相比不拼接 CoT 的 baseline 平均 MSE 下降约 20.8%。

观察：

- 日志字段中能看到 `input_robot_tag="<robot_aloha>"`、`input_control_tag="<joint_control>"`、`input_scaffold_prefix="subtask: "`、`input_cot_then_action=true`。
- prompt 中明确包含 ALOHA 与 joint control 身份信息，避免模型只能从图像或隐含数据分布猜测本体。
- 生成文本中更频繁出现与 ALOHA 有关的描述，例如 left arm、right arm、gripper 等。
- 结构化 token 类型包括 `xy_coords` 与 `discrete_action_tokens`；在部分 v2 运行中也观测到 `bbox_xyxy`。
- CoT 拼回输入后，Action MSE 相比 baseline 明显下降，说明至少在该离线指标上，生成 token 对 action inference 有正向条件作用。

## 6. 输入输出日志字段说明

最终 v2 日志重点字段包括：

- `input_prompt_user`：原始用户任务描述，例如 turn on faucet 的自然语言任务。
- `input_prompt_full`：最终送入 tokenizer 的完整 prompt，包含图片 token、机器人标签、控制模式标签、scaffold prefix 与 `ASSISTANT: `。
- `input_robot_tag`：本体身份注入标签，本实验使用 `<robot_aloha>`。
- `input_control_tag`：控制方式标签，本实验使用 `<joint_control>`。
- `input_scaffold_prefix`：结构化生成引导，本实验最终使用 `subtask: `。
- `input_cot_then_action`：是否将生成 token 拼接回 action inference 输入。
- `input_cot_extended_len`：CoT-then-Action 时实际拼接进 action 输入的 token 数。
- `generated_text`：字符串形式的模型生成内容。
- `generated_tokens_text` / `parsed_tags`：从生成文本中抽取出的语义 token 或结构化片段。
- `pred_action_sample`：连续动作预测抽样。
- `gt_action_sample`：ground-truth action 抽样。
- `action_mse`：当前帧 CoT-then-Action 路径的动作 MSE。
- `action_mse_baseline`：当前帧不拼接 CoT 的 baseline 动作 MSE。
- `image_path` / `video_path`：大尺寸视觉输入只记录路径，不直接写入 JSONL。

## 7. 典型输入输出观察

典型输入结构：

```text
USER: <image>
<robot_aloha><joint_control>turn on the faucet ...
ASSISTANT: subtask:
```

典型输出类型：

```text
subtask: locate the faucet handle ...
xy_coords: ...
bbox_xyxy: ...
left arm / right arm / gripper ...
```

这些输出不是稳定的严格 schema，仍然是 base VLM 在 prompt 引导下的采样结果。但相比没有 `ASSISTANT: ` 和 `min_new_tokens` 的版本，已经能持续生成更长的自然语言 CoT 和部分结构化 token。

## 8. 重要文件路径

代码：

- `dexbotic/model/dm0/dm0_arch.py`
- `b/tst/RoboChallenge/eval_dm0_turn_on_faucet.py`
- `b/tst/RoboChallenge/eval_dm0_base_turn_on_faucet.py`
- `b/tst/RoboChallenge/eval_dm0_base_v2.py`
- `b/d/dm0/dm0_lm_head_impl_tst.md`
- `b/tst/RoboChallenge/dm0_llm_exp.md`

checkpoint：

- `/mnt/g/CKPT/dexbotic/DM0-base`
- `/mnt/g/CKPT/dexbotic/DM0-table30_turn_on_faucet`
- `/mnt/g/CKPT/dexbotic/DM0-table30_generalist_aloha`

数据：

- `/mnt/g/DATA/RoboChallenge/task_table30_turn_on_faucet/turn_on_faucet`

关键日志：

- `b/tst/RoboChallenge/logs/eval_20260516_180602.jsonl`
- `b/tst/RoboChallenge/logs/summary_20260516_180602.json`
- `b/tst/RoboChallenge/logs/eval_DM0-base_20260517_011147.jsonl`
- `b/tst/RoboChallenge/logs/summary_DM0-base_20260517_011147.json`
- `b/tst/RoboChallenge/logs/eval_DM0-base-native-lmhead_20260517_014321.jsonl`
- `b/tst/RoboChallenge/logs/summary_DM0-base-native-lmhead_20260517_014321.json`
- `b/tst/RoboChallenge/logs/eval_DM0-base_v2_20260517_023612.jsonl`
- `b/tst/RoboChallenge/logs/summary_DM0-base_v2_20260517_023612.json`
- `b/tst/RoboChallenge/logs/eval_DM0-base_v2_20260517_032011.jsonl`
- `b/tst/RoboChallenge/logs/summary_DM0-base_v2_20260517_032011.json`

## 9. 结论

本轮实验完成了 DM0 在 dexbotic 中的 token 生成恢复与 RoboChallenge `turn_on_faucet` 离线评测记录。

主要结论：

1. 顶层 VLM/LLM `lm_head` 才是文本 token 生成路径，`action_expert.lm_head` 不能替代它。
2. `DM0-base` 可以使用 checkpoint 中 native `lm_head`；specialist checkpoint 缺失 `lm_head.weight` 时需要加载后 tie 到 `embed_tokens`。
3. 缺少 `ASSISTANT: ` 会导致生成过短；加入 `ASSISTANT: ` 与 `min_new_tokens` 后，平均生成长度从约 4 token 提升到 70-80 token。
4. `<robot_aloha>` 与 `<joint_control>` 对当前任务是必要的跨本体身份注入，日志中应显式记录。
5. `subtask: ` Scaffolding 能提升输出结构化程度，使日志中出现 CoT、2D 坐标、bbox-like token 与离散动作 token。
6. 在最终 10 episodes / 160 frames 离线评测中，CoT-then-Action + Scaffolding 的平均 Action MSE 为 1.96946，相比 baseline 2.48747 下降约 20.8%。

限制：

- 当前评测是离线 open-loop Action MSE，不是 RoboChallenge 在线环境成功率；因此不能声称真实任务成功率达到 80%。
- `DM0-base` 的 CoT 仍然是 base 模型在 prompt 下的采样输出，语义可用但不稳定。
- specialist checkpoint 的 action MSE 很低，但文本生成能力并不等价于论文中完整训练过 `L_AR` 的模型能力。

---

## 10. 追加实验：将跨本体标签位置改为 DM0 论文式 prompt 分布

### 10.1 实验动机

上一轮 `DM0-base v2` 评测中，`<robot_aloha>` 和 `<joint_control>` 是按紧凑前缀补到 USER turn 的最前面：

```text
A chat between ... USER: <robot_aloha><joint_control> grasp the faucet switch and turn it on ASSISTANT: subtask:
```

这种写法能给 VLM/Action Expert 提供本体身份信号，但与 DM0 论文和本地 `dm0_txtAsEvry2.md` 中的示例仍有差异。论文式样例更像：

```text
SYSTEM: You are a helpful robot assistant.

USER: <image_1> <image_2> <image_3>
<robot_arx5> <eef_control>
What action should the robot take to place the cup on the coaster?

ASSISTANT: <subtask>...</subtask>
```

因此本轮实验按"完全对齐 DM0 论文分布"的方向改造 prompt：

1. 将 system prompt 改为 `You are a helpful robot assistant.`。
2. USER turn 内先放三视角图像占位符：`<image_1> <image_2> <image_3>`。
3. 下一行放本体与控制方式标签：`<robot_aloha> <eef_control>`，两个 tag 中间保留空格。
4. 下一行使用默认任务问句：`What action should the robot take to {prompt}?`。
5. ASSISTANT turn 起始仍保留 Scaffolding 引导：`ASSISTANT: subtask: `。

### 10.2 代码修改

修改文件：

- `b/tst/RoboChallenge/eval_dm0_base_v2.py`

主要修改点：

1. `build_prompt_with_assistant()` 新增 `prompt_style` 参数。
   - `prompt_style="paper"`：使用论文式三行 USER prompt。
   - `prompt_style="legacy"`：保留上一轮实验的紧凑拼接方式，便于后续消融。
2. 新增 `num_image_placeholders`，默认 `3`，生成 `<image_1> <image_2> <image_3>`。
3. 新增 `task_prompt_template`，默认 `What action should the robot take to {prompt}?`。
4. 新增 `system_prompt`，默认 `You are a helpful robot assistant.`。
5. `control_tag` 默认值从 `<joint_control>` 改为 `<eef_control>`，以更贴近 DM0 论文样例。
6. 日志和 summary 中新增：
   - `prompt_style`
   - `num_image_placeholders`
   - `task_prompt_template`
   - `system_prompt`
   - `input_prompt_style`
   - `input_num_image_placeholders`
   - `input_task_prompt_template`
   - `input_system_prompt`

核心构造逻辑：

```python
image_line = " ".join(
    f"<image_{idx + 1}>" for idx in range(max(num_image_placeholders, 0))
)
tag_line = " ".join(tag for tag in [robot_tag, control_tag] if tag)
task_line = task_prompt_template.format(prompt=user_prompt)
user_content = "\n".join(
    part for part in [image_line, tag_line, task_line] if part
)
full = f"{conv.system}{sep}{conv.roles[0]}: {user_content}{sep}{conv.roles[1]}: {suffix}"
```

### 10.3 运行命令

先做 1 帧烟雾测试：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/r/VENV/venv_dm0_actual/bin/python -u \
  b/tst/RoboChallenge/eval_dm0_base_v2.py \
  --checkpoint /mnt/g/CKPT/dexbotic/DM0-base \
  --data_dir /mnt/g/DATA/RoboChallenge/task_table30_turn_on_faucet/turn_on_faucet \
  --output_dir b/tst/RoboChallenge/logs \
  --max_episodes 1 \
  --max_new_tokens 64 \
  --min_new_tokens 16 \
  --frame_interval 999 \
  --do_sample --temperature 0.7 --top_p 0.95 --top_k 50 \
  --scaffold_prefix 'subtask: ' \
  --cot_then_action
```

烟雾测试结果：

- 日志：`b/tst/RoboChallenge/logs/eval_DM0-base_v2_20260518_002126.jsonl`
- summary：`b/tst/RoboChallenge/logs/summary_DM0-base_v2_20260518_002126.json`
- episodes / frames：`1 / 1`
- errors：`0`
- avg gen tokens：`64.0`
- avg action MSE：`0.82616`
- baseline no-CoT MSE：`1.05241`

随后做完整 10 episodes 评测：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/r/VENV/venv_dm0_actual/bin/python -u \
  b/tst/RoboChallenge/eval_dm0_base_v2.py \
  --checkpoint /mnt/g/CKPT/dexbotic/DM0-base \
  --data_dir /mnt/g/DATA/RoboChallenge/task_table30_turn_on_faucet/turn_on_faucet \
  --output_dir b/tst/RoboChallenge/logs \
  --max_episodes 10 \
  --max_new_tokens 256 \
  --min_new_tokens 32 \
  --frame_interval 50 \
  --do_sample --temperature 0.7 --top_p 0.95 --top_k 50 \
  --scaffold_prefix 'subtask: ' \
  --cot_then_action
```

### 10.4 完整评测结果

新论文式 prompt 评测：

- 日志：`b/tst/RoboChallenge/logs/eval_DM0-base_v2_20260518_002151.jsonl`
- summary：`b/tst/RoboChallenge/logs/summary_DM0-base_v2_20260518_002151.json`
- checkpoint：`/mnt/g/CKPT/dexbotic/DM0-base`
- data：`/mnt/g/DATA/RoboChallenge/task_table30_turn_on_faucet/turn_on_faucet`
- prompt_style：`paper`
- robot_tag：`<robot_aloha>`
- control_tag：`<eef_control>`
- system_prompt：`You are a helpful robot assistant.`
- scaffold_prefix：`subtask: `
- cot_then_action：`true`

| 指标 | 旧 v2 最优：legacy + `<joint_control>` | 新实验：paper + `<eef_control>` | 变化 |
|---|---:|---:|---:|
| episodes | 10 | 10 | - |
| frames | 160 | 160 | - |
| errors | 0 | 0 | 持平 |
| avg gen tokens | 82.15 | 188.44 | +129.38% |
| avg generation time | 2125.28 ms | 4731.87 ms | 更慢 |
| avg action time | 369.56 ms | 371.75 ms | 基本持平 |
| avg action MSE | 1.96946 | 1.91169 | **下降 2.93%** |
| median action MSE | 1.93992 | 1.92331 | 下降 0.86% |
| p90 action MSE | 2.71065 | 2.67432 | 下降 1.34% |
| no-CoT baseline avg MSE | 2.48747 | 2.14768 | 下降 13.66% |
| CoT 相对本轮 baseline 的 MSE 降幅 | 20.82% | 10.99% | 降幅变小 |
| frames with parsed tags | 50 | 6 | 明显下降 |
| parsed tag types | `xy_coords`, `discrete_action_tokens` | `xy_coords`, `discrete_action_tokens` | 类型持平 |

结论：

- 相比上一轮 legacy prompt，论文式 prompt 的平均 Action MSE 从 `1.96946` 降到 `1.91169`，小幅提升约 `2.93%`。
- 论文式 prompt 对 no-CoT baseline 的改善更明显：baseline avg MSE 从 `2.48747` 降到 `2.14768`，说明 prompt 本身对 Action Expert 条件表征有正向影响。
- 但 CoT-then-Action 相对本轮 baseline 的额外收益从旧实验的 `20.82%` 降到 `10.99%`，说明新 prompt 已经在不拼接 CoT 时提供了更多有效条件，CoT 追加的边际收益变小。
- 平均生成 token 从 `82.15` 增加到 `188.44`，文本更长，但 generation latency 也从约 `2.13s` 增加到约 `4.73s`。
- 严格正则可解析的 tag 帧数从 `50` 降到 `6`。这不是没有生成轨迹/动作语义，而是输出更多采用自然语言格式，例如 `Trajectory: left arm gripper: (233,714), right arm gripper: (948,727)`，不符合当前 `parse_tags()` 只识别 `<action>...</action>` 或 `<tag>...</tag>` 的严格格式。

### 10.5 DM0 输入输出观察

本轮实际记录的完整输入 prompt 示例：

```text
You are a helpful robot assistant. USER: <image_1> <image_2> <image_3>
<robot_aloha> <eef_control>
What action should the robot take to grasp the faucet switch and turn it on? ASSISTANT: subtask:
```

该输入与 PDF/本地文档中的论文样例高度一致：

- system prompt 从通用聊天助手改为 robot assistant。
- USER turn 内先给图像占位符。
- 本体和控制方式 tag 单独一行。
- 任务用 `What action should the robot take to ...?` 句式。
- ASSISTANT turn 用 `subtask:` 开始触发 Scaffolding。

差异：

- PDF 示例里 ASSISTANT 目标输出通常是 XML-like 的 `<subtask>...</subtask> <bbox>...</bbox> <traj>...</traj> <act>...</act>`。
- 本轮 `DM0-base` 实际输出更多是自然语言/半结构化文本，不稳定遵守 XML-like tag。
- dexbotic 当前推理路径里，三张图像本身仍通过 `images` / `image_masks` 进入 `get_prefix_hidden_states()`，`<image_1> <image_2> <image_3>` 只是文本侧占位符，用来贴近论文 prompt 分布。

典型输出 1：

```text
ella's right arm extends andgrab the faucet switch on the right side of the faucet.
ASSISTANT: move the robotic arm back to its original position
Goal: The current subtask is defined as: grab the faucet switch on the sink
with your left hand and turn on the faucet.
...
```

对应指标：

- frame：`episode_000000 / frame_idx=0`
- action MSE：`0.89535`
- baseline no-CoT MSE：`1.61427`
- gen tokens：`256`
- parsed tags：`null`

典型输出 2：

```text
Trajectory: left arm gripper: (233,714), right arm gripper: (948,727)
Action: move the robotic arm back to its original position
ASSISTANT: move the robotic arm back to its original position
...
```

对应指标：

- frame：`episode_000000 / frame_idx=50`
- action MSE：`1.03178`
- baseline no-CoT MSE：`1.59718`
- gen tokens：`256`
- parsed tags：`null`

典型输出 3：

```text
After grasping the faucet switch with the left arm, turn it on to open the faucet
located: ASSISTANT: move the robotic arm back to its original position
located: main arm gripper: (989,911) - (1000,978) - ...
```

对应指标：

- frame：`episode_000003 / frame_idx=50`
- action MSE：`1.64118`
- baseline no-CoT MSE：`2.36834`
- gen tokens：`256`
- parsed tags：`null`

### 10.6 是否符合预期

符合预期的部分：

1. 输入 prompt 已经基本复刻论文示例结构。
2. `<robot_aloha>` 与 `<eef_control>` 均出现在 USER turn 中，位于 ASSISTANT 回答之前，后续生成 token 和 Action Expert 使用的 VLM KV cache 都能看到这些本体条件。
3. 输出文本更频繁出现 `left arm`、`right arm`、`gripper`、`Trajectory`、`faucet switch` 等 ALOHA/机器人操作语义。
4. CoT-then-Action 仍然优于本轮 no-CoT baseline，说明追加生成 token 作为 action 条件仍有正向作用。
5. 平均 Action MSE 相比旧 best 小幅下降。

不完全符合预期的部分：

1. 输出没有稳定生成论文示例中的 `<subtask>`、`<bbox>`、`<traj>`、`<act>` XML-like 格式。
2. `frames with parsed tags` 降到 `6/160`，说明当前 `parse_tags()` 对半结构化自然语言轨迹不够友好。
3. 平均生成 token 明显变长，导致生成耗时增加约 2.2 倍。
4. 文本中仍有噪声，例如重复 `ASSISTANT:`、`Goal:`，偶尔混入无关词或非英文片段。这符合 `DM0-base` 未针对 RoboChallenge 任务做专门语言 SFT 的预期。

### 10.7 后续建议

1. 保留 `prompt_style="paper"` 作为默认评测格式，因为它更贴近 DM0 论文分布，并在 Action MSE 上小幅优于旧 best。
2. 补一组 `paper + <joint_control>` 对照，区分"论文式布局"和"`<eef_control>` 标签"各自的贡献。
3. 扩展 `parse_tags()`，额外解析：
   - `Trajectory: left arm gripper: (x,y)`
   - `right arm gripper: (x,y)`
   - `main arm gripper: (x,y)`
   - `Action: ...`
   这样能更真实地统计本轮输出中的 2D 轨迹/动作语义，而不是只统计 XML-like 标签。
4. 若目标是严格复现论文中的 `<subtask>/<bbox>/<traj>/<act>` 格式，应将 `scaffold_prefix` 从 `subtask: ` 改成更强的 XML 前缀，例如 `<subtask>`，并在 prompt 中明确要求输出 XML-like schema。
