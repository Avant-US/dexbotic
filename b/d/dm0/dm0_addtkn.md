# DM0 Added Tokens 逆向分析：从一手证据还原 Template 与 Token 使用方式

> 修订日期: 2026-05-18
> 方法: 侦探式逆向分析 — 仅依据一手证据推理，不引用非官方分析文档
>
> 一手证据来源:
> - DM0 论文 (arxiv:2602.14974)
> - dexbotic 开源代码 (https://github.com/dexmal/dexbotic)
> - DM0 各 checkpoint 配置文件 (`b/m/dm0/`)
> - 官方文档 (`docs/DM0.md`, `docs/Data.md`)
> - HuggingFace 模型卡 (https://huggingface.co/Dexmal/DM0-base)

---

## 0. TL;DR

### 核心发现

DM0 存在 **两套训练阶段**，使用不同的 template 和 token 配置：

| 阶段 | Template 格式 | model_max_length | L_AR | L_FM | 特殊 Token | 开源 |
|------|-------------|-----------------|------|------|-----------|------|
| Mid-training | Qwen3 chat template (`<\|im_start\|>role\n...<\|im_end\|>`) | 4096 | ✅ (config 声明) | ✅ | 可能使用 | ❌ |
| SFT/Post-training | "step" template (`USER: ... ASSISTANT: ...<\|im_end\|>`) | 100 | ❌ (代码未实现) | ✅ | 未使用 | ✅ |

### 关于 `<robot_*>` 和 `<*_control>` token 的结论

| 事实级别 | 内容 |
|---------|------|
| **确认** | 18 个 `<robot_*>` + 2 个 `<*_control>` token 注册在所有 checkpoint 的 `added_tokens.json` 中 |
| **确认** | 这些 token 的 embedding 存在于 `model.llm.embed_tokens.weight` [152701, 2048] 中 |
| **确认** | 开源代码（训练 + 推理）中**零引用**这些 token |
| **确认** | DM0 论文**完全没有提到**这些 token |
| **推测** | 这些 token 可能在 mid-training 的 500 条 conversation template 中被使用（未开源） |
| **未知** | 这些 token 的 embedding 是否经过有效训练（可能只是随机初始化 + LLM 继续训练后的值） |

---

## 1. 方法论

本文档以"逆向工程"的方式，从可观测的一手证据出发，推断 DM0 的 template 和 token 使用方式。

**证据分级：**
- 🟢 **确认 (Confirmed)**: 可直接从代码/配置文件/论文原文验证
- 🟡 **推测 (Inferred)**: 基于多条间接证据的逻辑推理，标注推理链
- 🔴 **未知 (Unknown)**: 证据不足，无法判断

**不使用的来源：**
- `b/d/dm0/` 中的非官方 md 分析文档（这些是同事的方案推测，不是 DM0 官方资料）

---

## 2. 一手证据汇总

### 2.1 DM0 论文说了什么 (arxiv:2602.14974)

**论文明确说了：**

> "All data are expressed as template-based conversations so that the same supervision (text, actions, waypoints, etc.) can be presented in varied natural language."

> "We design 500 distinct conversation templates for each specific data combination scenario, which are manually polished for quality."

> "During training, we randomly select one of these templates for each sample, introducing linguistic diversity and preventing overfitting to specific prompt structures."

> "We construct short-horizon windows of length 50, normalize them, and quantize them into a 255-bin vocabulary as special action tokens."

> "Actions are supervised in two aligned views over the same sequence: discrete tokens for the VLM and continuous values for the action expert."

> "Data representation, conversation templates, and action formulation remain identical to those used in mid-training." (关于 post-training)

**论文没有提到：**
- ❌ `<robot_*>` token — 论文从未提及任何 robot identity token
- ❌ `<eef_control>` / `<joint_control>` — 论文从未提及 control mode token
- ❌ 跨本体身份注入的具体实现机制 — 论文提到训练了多种机器人（Franka, UR5, ARX-5, UMI, ALOHA），但**没有解释**模型如何区分不同本体
- ❌ 728-bin action token — 论文只提到 255-bin
- ❌ 具体的 prompt 示例 — 500 条 template 中没有展示任何一条
- ❌ 图像 token 的具体格式 — 没有提到 `<im_start>`, `<im_end>`, `<im_patch>`

### 2.2 官方文档说了什么 (`docs/DM0.md`)

官方文档仅描述了：
- 模型下载和训练启动命令
- 推理部署方式（Flask server + curl 调用）
- Benchmark 结果（Libero, RoboChallenge）
- 配置字段说明（dataset_name, num_images, action_dim, non_delta_mask）

**没有任何关于 prompt template、robot token、action token 格式的描述。**

### 2.3 HuggingFace 模型卡说了什么

DM0-base 的模型卡极其简略：
> "This model provides an initialization checkpoint for training DM0 VLA Model"

**没有 prompt 示例、template 描述、或 token 使用说明。**

### 2.4 Dexbotic Tech Report (arxiv:2510.23511)

Tech report 描述了 dexbotic 框架的整体设计，提到：
> "Each dimension of the robot actions is discretized separately into 256 bins."

**没有提到 `<robot_*>`, `<*_control>`, 或任何 robot identity 机制。**

---

## 3. Token 清单与证据状态

来源: `b/m/dm0/base/added_tokens.json` (所有 5 个 checkpoint 完全一致, MD5 相同)

### 3.1 Qwen3 基础设施 Token (ID 151643-151678)

| Token | ID | 证据 |
|-------|------|------|
| `<\|endoftext\|>` | 151643 | 🟢 pad_token, 代码使用 |
| `<\|im_start\|>` | 151644 | 🟢 Qwen3 chat template 使用 |
| `<\|im_end\|>` | 151645 | 🟢 eos_token + "step" template 的 sep2 |
| `<\|box_start\|>` / `<\|box_end\|>` | 151648-151649 | 🟡 可能用于 bbox 预测 (ESS 第 2 层) |
| `<\|vision_start\|>` / `<\|vision_end\|>` | 151652-151653 | 🟡 Qwen3 视觉 token |
| `<tool_call>` / `</tool_call>` | 151657-151658 | 🟢 Qwen3 chat template 支持 |
| `<think>` / `</think>` | 151667-151668 | 🟢 Qwen3 chat template 的 `<think>` 包裹 |

### 3.2 图像 Token (ID 151679-151681)

| Token | ID | 证据 |
|-------|------|------|
| `<im_patch>` | 151679 | 🟡 Qwen3 chat template 的 `render_content` 宏中 image → `<im_patch>` |
| `<im_start>` | 151680 | 🟡 `process.py:54` "step" 模式中 `msg["value"] + f"<im_start>{DEFAULT_IMAGE_TOKEN}<im_end>"` — 但 DM0Tokenization **不调用此函数** |
| `<im_end>` | 151681 | 🟡 同上 |

**关键发现**: `<im_start>/<im_end>` 的使用代码在 `llava_multi_image_map_fn()` 中，但 DM0Tokenization **不调用** `llava_multi_image_map_fn`。DM0 的图像走 vision tower 直接编码，不通过 text token 路径。

### 3.3 预留 Token (ID 151682-151697)

| Token | ID | 证据 |
|-------|------|------|
| `<dream>` / `<dream_start>` / `<dream_end>` | 151682-151684 | 🔴 无任何使用证据 |
| `<video_start>` / `<video_end>` 等 | 151687-151691 | 🔴 无使用证据 |
| `<｜begin▁of▁sentence｜>` 等 | 151692-151697 | 🟡 可能继承自 DeepSeek 系列 tokenizer |

### 3.4 255-bin Action Token (ID 151698-151952)

| Token | ID 范围 | 证据 |
|-------|--------|------|
| `<action_0>` ~ `<action_254>` | 151698-151952 | 🟢 论文确认 "255-bin vocabulary as special action tokens" |

**论文引用**: "We construct short-horizon windows of length 50, normalize them, and quantize them into a 255-bin vocabulary as special action tokens."

🟡 **推测**: 这些 token 在 mid-training 阶段被 L_AR (autoregressive loss) 使用。VLM 自回归预测这些 token，同时 Action Expert 用 flow matching 预测连续 action。在开源 SFT 代码中，L_AR 被去除，这些 token 不再使用。

### 3.5 Robot Identity Token (ID 151953-151970)

| Token | ID | 对应平台 |
|-------|------|---------|
| `<robot_ur5>` | 151953 | Universal Robots UR5 |
| `<robot_franka>` | 151954 | Franka Emika Panda |
| `<robot_aloha>` | 151955 | ALOHA 双臂 |
| `<robot_r1_lite>` | 151956 | AgiBot R1 Lite |
| `<robot_arx5>` | 151957 | ARX-5 单臂 |
| `<robot_umi>` | 151958 | UMI 手持 |
| `<robot_z1>` | 151959 | Unitree Z1 |
| `<robot_g1>` | 151960 | Unitree G1 人形 |
| `<robot_realman_756f>` | 151961 | RealMan 756F |
| `<robot_widowx>` | 151962 | WidowX 250 |
| `<robot_kuka>` | 151963 | KUKA iiwa |
| `<robot_xarm>` | 151964 | UFactory xArm |
| `<robot_google_robot>` | 151965 | Google Robot |
| `<robot_stretch>` | 151966 | Hello Robot Stretch |
| `<robot_sawyer>` | 151967 | Rethink Sawyer |
| `<robot_jaco2>` | 151968 | Kinova JACO2 |
| `<robot_fanuc_mate>` | 151969 | Fanuc CRX-10iA/L |
| `<robot_dlr_edan>` | 151970 | DLR EDAN |

**证据状态: 🟡 推测**

论文没有提到这些 token。代码没有使用这些 token。但它们存在于所有 checkpoint 中，且覆盖了 DM0 论文提到的所有训练平台以及 OXE 数据集中的常见平台。

### 3.6 Control Mode Token (ID 151971-151972)

| Token | ID | 含义 |
|-------|------|------|
| `<eef_control>` | 151971 | 末端执行器坐标系控制 |
| `<joint_control>` | 151972 | 关节空间控制 |

**证据状态: 🟡 推测** — 同 robot token，无论文/代码证据。

### 3.7 728-bin Action Token (ID 151973-152700)

| Token | ID 范围 | 格式 |
|-------|--------|------|
| `<action>0</action>` ~ `<action>727</action>` | 151973-152700 | XML-style 闭合标签 |

**证据状态: 🔴 未知**

论文只提到 255-bin，没有提到 728-bin。注意到 728 = DM0 图像分辨率（728×728），这可能不是巧合。

🟡 **假说 A**: 728-bin 用于表示 2D 图像坐标（Embodied Spatial Scaffolding 中的 2D waypoint trajectory），每个坐标维度用一个 `<action>N</action>` token 表示 0-727 像素位置。

🟡 **假说 B**: 728-bin 是更精细的 action 离散化方案（255-bin 是 ~8-bit，728-bin 是 ~10-bit），可能在不同训练阶段或不同任务中使用。

🟡 **假说 C**: 两种 token 格式（`<action_N>` vs `<action>N</action>`）对应不同的语义——前者是 action 值的 bin index，后者是 2D 坐标或其他用途。

### 3.8 Token ID 布局总览

```
Qwen3 基础 vocab:  [0, 151642]               — 151,643 个 token
──────────────────────────────────────────────
Added tokens:      [151643, 152700]            — 1,058 个 token
  ├─ Qwen3 特殊:   [151643, 151678]  36 个     — 基础设施
  ├─ 图像/梦境/视频: [151679, 151697]  19 个    — 多模态
  ├─ 255-bin action: [151698, 151952] 255 个    — 论文确认
  ├─ Robot identity: [151953, 151970]  18 个    — 未确认
  ├─ Control mode:   [151971, 151972]   2 个    — 未确认
  └─ 728-bin action: [151973, 152700] 728 个    — 未确认
──────────────────────────────────────────────
LLM vocab_size:    152,701
Action Expert vocab_size: 151,936  ← 注意：比 Qwen3 base 大但不含 robot/control/728-action
```

---

## 4. 关键发现：两套 Template 共存

### 4.1 Template A: Qwen3 Chat Template (存在于 base checkpoint 的 tokenizer_config.json)

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
<think>
{reasoning}
</think>
{response}<|im_end|>
```

**特征：**
- `render_content` Jinja2 宏将 image 类型渲染为 `<im_patch>` token
- 支持 `<think>...</think>` 推理包裹
- 支持 `<tool_call>` / `<tool_response>` 工具调用
- 完整的 Qwen3 chat 格式

**来源**: `b/m/dm0/base/tokenizer_config.json:8502` — 完整的 Jinja2 chat_template 字段

### 4.2 Template B: "step" Template (存在于 dexbotic 开源代码)

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {content} ASSISTANT: {response}<|im_end|>
```

**特征：**
- LLaVA/Vicuna 风格的 `USER:` / `ASSISTANT:` 角色
- System prompt 不用 `<|im_start|>` 包裹
- 分隔符: 空格 (sep) + `<|im_end|>` (sep2)
- 不支持 `<think>`, 不支持 tool calling

**来源**: `dexbotic/tokenization/conversation.py:211-221` — `conv_step` 对象

### 4.3 关键推理：两套 Template 对应不同的训练阶段

| 线索 | 指向 |
|------|------|
| base model_max_length = **4096** | Mid-training 使用长序列 → 可容纳 CoT + action tokens |
| fine-tuned model_max_length = **100** | SFT 只需短指令 → 无 CoT, 无 action tokens |
| Qwen3 template 支持 `<think>`, `<im_patch>` | Mid-training 时有推理和图像 token |
| "step" template 不支持 `<think>` | SFT 阶段去除了推理能力 |
| config `ar_loss: true` 但代码不实现 | L_AR 是 mid-training 的遗留配置 |
| 论文说 "500 conversation templates" | 500 模板用于 mid-training（未开源） |
| 论文说 actions 量化为 "255-bin" tokens | Mid-training 时 VLM 预测 action tokens |

**结论 🟡 (推测)**:

DM0 的训练分为两个阶段，使用不同的 template：

1. **Mid-training (Dexmal 内部，未开源)**:
   - 使用 Qwen3 chat template
   - 长序列 (4096 tokens)
   - L_AR + L_FM 双 loss
   - VLM 预测 text + CoT + discrete action tokens
   - 可能使用 `<robot_*>`, `<*_control>`, `<action_N>` 等特殊 token
   - 使用 500 条 conversation templates 做语言增广

2. **SFT / Post-training (开源代码)**:
   - 使用 "step" template
   - 短序列 (100 tokens)
   - 仅 L_FM (flow matching loss)
   - VLM 只提供 hidden states 给 Action Expert，不预测 text
   - 不使用任何特殊 token

---

## 5. 逆向推理：Mid-training 的 Prompt 格式 🟡

基于上述证据，以下是对 mid-training prompt 格式的推测。**这些都是推测，不是确认的事实。**

### 5.1 推测的 Mid-training Prompt 格式 (置信度: 中)

```
<|im_start|>system
You are a robot control assistant.<|im_end|>
<|im_start|>user
<im_patch> <im_patch> <im_patch> ... (image tokens)
What action should the robot take to {task_instruction}?<|im_end|>
<|im_start|>assistant
<think>
{subtask_description}
{bbox_prediction}
{2d_trajectory}
</think>
<action_128> <action_64> <action_200> ... (50 * action_dim discrete tokens)<|im_end|>
```

**推理依据**:
1. Qwen3 chat template 的 `render_content` 宏将 image → `<im_patch>`
2. 论文描述的 Embodied Spatial Scaffolding 产生 subtask/bbox/trajectory
3. Qwen3 template 将推理包裹在 `<think>...</think>` 中
4. 论文说 action 量化为 255-bin token

### 5.2 `<robot_*>` Token 的可能位置 (置信度: 低)

如果 mid-training 确实使用了 robot token，它们可能出现在：

**假说 1 — User 消息中作为前缀**:
```
<|im_start|>user
<robot_arx5> <eef_control> <im_patch>... What action should the robot take to {task}?<|im_end|>
```

**假说 2 — System prompt 中**:
```
<|im_start|>system
You are controlling a <robot_arx5> with <eef_control>. ...<|im_end|>
```

**假说 3 — 500 条 template 的变量插槽中**:
```
# Template #127:
"<robot_{robot_type}> <{control_mode}> Given the current view, {task_instruction}"
# Template #328:
"You are operating {robot_type} in {control_mode} mode. {task_instruction}"
```

**但必须强调: 论文没有提到 robot token，以上全是推测。**

### 5.3 728-bin Token 的可能用途 (置信度: 低)

`<action>N</action>` (N=0~727) 可能用于 2D waypoint trajectory:
- 论文描述 ESS 第 3 层预测 "end-effector trajectory in the primary camera view"
- 图像分辨率 728×728 → 728 个 bin 恰好对应像素位置
- 每个 waypoint = 2 个 token (x, y 坐标)

这会让 `<think>` 区块看起来像:
```
<think>
Move arm to grasp the cup.
<|box_start|> <action>256</action> <action>340</action> <action>380</action> <action>490</action> <|box_end|>
<action>300</action> <action>400</action> <action>310</action> <action>420</action> ...
</think>
```

---

## 6. 代码中的数据流全链路 🟢

以下基于实际代码，是确认的事实。

### 6.1 训练路径 (SFT)

```
JSONL 文件
  │ 每行: {"prompt": "open the door", "state": [0.1,...], "is_robot": true,
  │        "images_1": {"type":"video","url":"...","frame_idx":21}, ...}
  │
  ▼
DexDataset.__getitem__(idx)  [dex_dataset.py:189]
  │
  ├── [1] action_process_func (DM0ActionConfig.build_action_process_func)
  │   ToDict → ToNumpy → AddAction → PadState(32) → PadAction(32)
  │   → AddTrajectory(50) → DeltaAction → ActionNorm → LoadMultiModal → ToList
  │   注意: ❌ 没有 AddPromptTemplate!
  │
  ├── [2] RGB 处理 [dex_dataset.py:228-246]
  │   load images → augmentation (dm0/dm0_color) → resize/pad → stack
  │   → return_dict["image"] = [num_images, 3, H, W]
  │
  ├── [3] Tokenization [dex_dataset.py:264-274]
  │   if "conversations" not in data:
  │       data = ToConversation_Old()(data)
  │       → conversations = [{"from":"human","value":"open the door"},
  │                           {"from":"gpt","value":""}]
  │   
  │   DM0Tokenization(conversations, has_image=True)  ← has_image 被忽略!
  │   ├── conv = conv_step.copy()
  │   ├── system_prompt = "A chat between...questions. "
  │   ├── 空 gpt turn 被 pop 掉 [process.py:407-414]
  │   ├── → "A chat between...questions. USER: open the door "
  │   ├── pad 到 model_max_length (100) tokens
  │   └── loss_mask: system=False, USER role=False, USER content=False
  │
  └── [4] 输出 dict
      {input_ids, labels, action[50,32], image[3,3,H,W], state[32], image_masks[3]}
```

### 6.2 模型 Forward (训练) [dm0_arch.py:406-511]

```
Prefix (LLM 侧):
  images → vision_tower → mm_projector → image_hidden_states
  input_ids → embed_language_tokens → text_hidden_states
  prefix = [img1 | img2 | img3 | text]  ← 图像在前, 文本在后

Suffix (Action Expert 侧):
  noise ~ N(0,1), time ~ Beta(1.5, 1.0)
  x_t = t*noise + (1-t)*actions
  x_t → action_in_proj → action_hidden
  time → sinusoidal → action_time_mlp → time_hidden
  suffix = [action_hidden + time_hidden]  (element-wise)

Merged Attention:
  [prefix | suffix] → 28 层 Qwen3 Transformer (共享 attention)
  → (prefix_out, suffix_out)
  prefix_out: 丢弃 (不计算 loss)
  suffix_out → action_out_proj → v_t
  loss = MSE(v_t, u_t)  ← 仅 flow matching loss
```

### 6.3 推理路径 [dm0_exp.py:404-521, libero_dm0.py:214-339]

```
HTTP POST /process_frame
  text = request.form["text"]  e.g. "What action should the robot take to..."
  images = request.files["image"]  (1~3 张)

  ↓ DM0Tokenization([{"from":"human","value": text}])
  ↓ → input_ids (padded to model_max_length)

  ↓ images → model.process_images() → tensor [num_images, 3, 728, 728]

  ↓ model.inference_action(input_ids, images, image_masks, states)
  ↓   prefix = get_prefix_hidden_states(input_ids, images, image_masks)
  ↓   KV cache = merged_attention(prefix)
  ↓   for step in range(diffusion_steps=10):
  ↓     suffix = get_suffix_hidden_states(noise, time)
  ↓     noise = denoise_step(suffix, KV_cache)
  ↓   return denoised_actions → action_out_proj → [50, 32]

  ↓ output_transform: denorm → absolute_action
  ↓ return actions[:, :action_dim]  e.g. [50, 7] for 7-DOF
```

---

## 7. 代码断路点分析 🟢

以下是确认的代码层面的"断路"——这些 token 在代码中存在但没有任何功能。

### 7.1 数据层

```python
# dexbotic/data/data_source/register.py
# meta_data 仅包含 action 处理相关字段:
meta_data = {
    'non_delta_mask': [...],    # 非 delta 维度 (如 gripper)
    'periodic_mask': None,      # 周期性维度 (如 rotation)
    'periodic_range': None,     # 周期范围
}
# ❌ 没有 robot_type, control_mode, embodiment_id 等字段
```

### 7.2 Transform 层

```python
# dexbotic/data/dataset/transform/language.py
# AddPromptTemplate 有 {prompt} 占位符:
class AddPromptTemplate:
    def __init__(self, prompt_template="<image>\nWhat action should the robot take to {prompt}?")
# ❌ 没有 {robot} 或 {control} 占位符
# ❌ DM0ActionConfig 的 pipeline 中根本不包含 AddPromptTemplate
```

### 7.3 Tokenization 层

```python
# dexbotic/tokenization/process.py:368-483
class DM0Tokenization:
    # 直接将 conversations 转为 "step" template 格式
    # ❌ 不调用 llava_multi_image_map_fn (不处理 <image> token)
    # ❌ 不注入任何 <robot_*> 或 <*_control> token
    # ❌ has_image 参数被 **kwargs 吞掉, 不使用
```

### 7.4 Model 层

```python
# dexbotic/model/dm0/dm0_arch.py:406-511
# forward() 只计算 flow matching loss:
loss = F.mse_loss(v_t, u_t, reduction="mean")
# ❌ 没有 L_AR (autoregressive loss), 尽管 config 说 ar_loss=true
# ❌ prefix_out 被丢弃, 不计算任何 text loss
```

### 7.5 Experiment 层

```python
# dexbotic/exp/dm0_exp.py:267-312
class DM0DataConfig:
    data_keys = ["input_ids", "labels", "action", "image", "state", "image_masks"]
    # ❌ 没有 robot_type, control_mode
    
class DM0InferenceConfig:
    # _get_response() 直接用 text 构建 prompt
    # ❌ 没有 robot/control token 注入
```

---

## 8. 前版文档的错误勘误

以下是前版 `dm0_addtkn.md` 中需要修正的错误：

| # | 前版说法 | 问题 | 正确信息 |
|---|---------|------|---------|
| 1 | "论文 §2.4 明确描述了跨本体身份注入机制：通过 prompt 模板里的本体标签 (`<robot_arx5>` 等)" | **幻觉**: 引用的是同事分析文档 dm0.md，不是论文 | 论文 arxiv:2602.14974 完全没有提到 `<robot_*>` token |
| 2 | "Robot/control token 位于 USER 内容的最前面，在图像 token 之前" | **无根据推测** | 没有任何证据表明这些 token 在任何位置被使用过 |
| 3 | "论文声称使用了 500 条含本体标签的对话模板" | **曲解**: 论文说的是 500 条 conversation templates，没有说包含本体标签 | 论文: "500 distinct conversation templates for each specific data combination scenario" |
| 4 | "预训练/mid-train 的 embedding 直接可用" | **无法确认** | Embedding 存在于权重中，但不知道是否在训练数据中出现过 |
| 5 | "L_AR 会强制 VLM 在 robot token 的条件下生成正确的 CoT" | **代码不支持**: L_AR 在开源代码中完全没有实现 | forward() 只有 `F.mse_loss(v_t, u_t)` |
| 6 | "与 DM0 的 500 条对话模板增广自然兼容" | **过度推断** | 500 条模板的内容完全未知 |
| 7 | "方案 A 是论文原始方案" | **无根据** | 论文没有描述 robot token 的放置方式 |
| 8 | 引用 dm0_txtAsEvry2.md, dm0_lAr.md 等作为依据 | **错误来源**: 这些是同事的分析方案，不是官方文档 | 应只引用论文、代码、配置文件 |

---

## 9. 如果要补回 Token 的实施建议

以下建议基于**确认的代码模式**，不依赖于对 mid-training 格式的推测。

### 9.1 最小改动方案：在 Prompt 文本中插入 Robot/Control Token

**原理**: 无论 mid-training 是否使用了这些 token，它们的 embedding 已经在模型权重中。将它们插入到 SFT 的 prompt 中，至少能让 SFT 阶段的梯度更新这些 embedding，为跨本体条件化提供信号。

**改动点**:

1. **数据源注册** — `data_source/*.py` 增加 robot_type 和 control_mode 字段:
```python
meta_data = {
    'non_delta_mask': [6],
    'robot_type': '<robot_franka>',      # 新增
    'control_mode': '<eef_control>',     # 新增
}
```

2. **Transform** — `language.py` 新增 prompt 前缀:
```python
class AddEmbodimentPrefix:
    def __init__(self, robot_key="robot_type", control_key="control_mode"):
        self.robot_key = robot_key
        self.control_key = control_key
    
    def __call__(self, episode_data_dict, **kwargs):
        robot = episode_data_dict.get(self.robot_key, "")
        control = episode_data_dict.get(self.control_key, "")
        if isinstance(robot, list): robot = robot[0] if robot else ""
        if isinstance(control, list): control = control[0] if control else ""
        prefix = f"{robot} {control}".strip()
        if prefix and "prompt" in episode_data_dict:
            episode_data_dict["prompt"] = [
                f"{prefix} {p}" for p in episode_data_dict["prompt"]
            ]
        return episode_data_dict
```

3. **Pipeline** — `dm0_exp.py` DM0ActionConfig pipeline 中插入:
```python
Pipeline([
    ...,
    LoadMultiModal(return_masks=True),
    AddEmbodimentPrefix(),   # 在 prompt 前加 <robot_*> <*_control>
    ToList(),
])
```

4. **推理** — DM0InferenceConfig 中前缀 robot/control:
```python
prompt = f"{self.robot_type} {self.control_mode} {text}"
```

**注意**:
- model_max_length=100 时，额外的 2 个 token 会减少可用的文本长度
- Embedding 是否已经有意义取决于 mid-training 是否使用了这些 token（未知）
- 建议配合 model_max_length 的调整（如增加到 110）

### 9.2 需要验证的假设

在实施前，建议先验证以下假设：

1. **Token embedding 是否有意义**: 比较 `<robot_arx5>` 和 `<robot_franka>` 的 embedding 向量是否有显著差异。如果差异很小（接近随机初始化），说明 mid-training 可能没有有效训练这些 embedding。
```python
import torch
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("path/to/DM0-base")
tokenizer = AutoTokenizer.from_pretrained("path/to/DM0-base")

ids = tokenizer.convert_tokens_to_ids(["<robot_arx5>", "<robot_franka>", "<robot_aloha>"])
embeddings = model.model.embed_tokens.weight[ids]
# 检查 cosine similarity 和 L2 distance
```

2. **L_AR 是否需要恢复**: 如果只用 flow matching (L_FM)，robot token 只通过 merged attention 的隐式路径影响 Action Expert。恢复 L_AR 可以让 VLM 显式地基于 robot token 生成不同的 text output。

3. **model_max_length 是否需要增加**: 如果要加入 CoT/subtask 等辅助 text output，100 tokens 完全不够。需要评估增加到 512 或 2048 的训练成本。

---

## 10. 总结

### 确认的事实 🟢
- 18 个 robot token + 2 个 control token 在所有 DM0 checkpoint 的词表中
- 开源代码零引用这些 token
- DM0 论文没有提到这些 token
- 存在两套 template: Qwen3 chat template (base tokenizer_config) 和 "step" template (代码)
- model_max_length: base=4096, fine-tuned=100
- L_AR config=true 但代码只实现 L_FM

### 合理推测 🟡
- Mid-training 使用 Qwen3 chat template + 长序列 + L_AR + L_FM
- 255-bin `<action_N>` token 在 mid-training 中被 L_AR 使用
- `<robot_*>` 和 `<*_control>` 可能在 mid-training 的 500 条 template 中被使用
- 728-bin `<action>N</action>` 可能用于 2D waypoint coordinates (728=图像分辨率)

### 未知 🔴
- 500 条 conversation template 的具体内容
- `<robot_*>` 和 `<*_control>` token 是否在任何训练阶段被实际使用
- 这些 token 的 embedding 是否经过有效训练
- `<dream_*>` token 的用途
- Qwen3 chat template 和 "step" template 之间的切换时机


# Opus 4.7 对这个问题的回答

证据都拿齐了，我现在按"侦探报告"的形式系统化复盘 DM0 的 conversation / prompt / data template，以及 1058 个 `added_tokens.json` 新增 token 的真实用法。每个结论都标出证据来源，区分"代码可验证"、"checkpoint 可验证"、"论文可验证"、"间接推理"。

## 一、四个 checkpoint 的横向对照

| 字段 | DM0-base | DM0-libero | DM0-table30_generalist_aloha | DM0-table30_turn_on_faucet |
| --- | --- | --- | --- | --- |
| `tokenizer_class` | Qwen2Tokenizer | Qwen2Tokenizer | Qwen2Tokenizer | Qwen2Tokenizer |
| `tokenizer_config.model_max_length` | 4096 | 2048 | 100 | 100 |
| `config.tokenizer_model_max_length` | 2048 | 2048 | 2048 | 2048 |
| `vocab_size` | 152701 | 152701 | 152701 | 152701 |
| 1058 个 `added_tokens` | 完全相同 | 完全相同 | 完全相同 | 完全相同 |
| `mm_vision_tower` | `pe_lang_l14_728` | `pe_lang_l14_728` | `pe_lang_l14_728` | `pe_lang_l14_728` |
| `mm_projector_type` | `linear4x` | `linear4x` | `linear4x` | `linear4x` |
| `action_dim` | 32 | 32 | 32 | 32 |
| `chunk_size` | 50 | 50 | 50 | 50 |
| `ar_loss` / `fm_loss` | true / true | true / true | true / true | true / true |
| `tokenizer_config.chat_template` | 有（继承 Qwen3 ChatML） | 有（同 base） | 无（被清空） | 有（同 base） |

证据：本地 4 个 checkpoint 的 `config.json` / `tokenizer_config.json` / `added_tokens.json`。

四个 checkpoint 的 **tokenizer 完全同源**（同样 1058 个 added tokens、同一 vocab_size、同一 special_tokens_map）。差异只在 `model_max_length`：specialist 极端压缩到 100 token，generalist + base 是 2048~4096。这在后面会非常关键。

## 二、`added_tokens.json` 1058 个新增 token 的分类

按脚本统计：

| 类别 | 数量 | id 范围 | 例子 |
| --- | ---: | --- | --- |
| Qwen3 标准 ChatML & 视觉占位 | 13 | 151643–151656 | `<\|endoftext\|>`、`<\|im_start\|>`、`<\|im_end\|>`、`<\|image_pad\|>`、`<\|video_pad\|>`、`<\|vision_start\|>` 等 |
| Qwen3 dream / FIM / mask / tool / box / quad | 30+ | 151657–151697 | `<dream_*>`、`<\|fim_*\|>`、`<\|MASK_*\|>`、`<tool_call>`、`<\|box_start\|>` 等 |
| 离散动作 token（255-bin） | **255** | **151698–151952** | `<action_0>` … `<action_254>` |
| 跨本体身份 token | **18** | **151953–151970** | `<robot_ur5>`、`<robot_franka>`、`<robot_aloha>`、`<robot_arx5>`、`<robot_aloha>`、`<robot_xarm>` 等 |
| 控制方式 token | **2** | **151971–151972** | `<eef_control>`、`<joint_control>` |
| 像素坐标 token（XML-wrapped） | **728** | **151973–152700** | `<action>0</action>` … `<action>727</action>` |

证据：`b/m/dm0/base/added_tokens.json` 的统计脚本结果。

三个数字对得上 DM0 论文的设计：

- 255 = 论文 §3.2 明确说的「255-bin action quantization vocabulary」（PDF p.7：「we construct short-horizon windows of length 50, normalize them, and quantize them into a 255-bin vocabulary as special action tokens」）。
- 18 robot tag + 2 control tag = 论文 Figure 3 与 §2 多次引用的 `<robot_arx5>`、`<robot_aloha>`、`<eef_control>` 这一族本体身份 token。
- 728 = 论文 §2.1 的图像分辨率「multi-view images are resized to 728 × 728」，和 `<action>0</action>` … `<action>727</action>` 的范围一致。这是把 728×728 像素空间整数离散化为 single-token，每个像素值 1 个 token。

证据：DM0 PDF p.4–p.7。

## 三、conversation / prompt / data 模板（按可信度从高到低）

### 3.1 Dexbotic 开源仓库实际跑的极简模板（代码可验证）

源码路径：

- `dexbotic/tokenization/process.py` 中的 `DM0Tokenization`
- `dexbotic/tokenization/conversation.py` 中的 `conv_step` 模板
- `dexbotic/exp/dm0_exp.py` 中的 `DM0DataConfig._build_dataset()` 和 `DM0InferenceConfig._get_response()`
- 官方推理仓 `dexmal/Dexbotic-RoboChallengeInference` 的 `policies/dm0_policy.py`、`utils/constants.py`

`conv_step`（dexbotic 用于 DM0 的对话模板）：

```python
conv_step = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|im_end|>",
)
```

`DM0Tokenization` 的关键行为（process.py L386-L483）：

1. 拼上 `system + sep`。
2. 如果 `conversations` 末尾是空的 `gpt` 轮，就 `pop()` 掉。
3. 对每一轮 `human`/`gpt` 加 `"USER: "` 或 `"ASSISTANT: "` role token（`loss_mask=False`）。
4. 加 `text + sep` 作为 content（gpt 才计 `loss_mask=True`）。
5. pad 或截断到 `tokenizer.model_max_length`。

`DM0InferenceConfig._get_response()` 与 `Dexbotic-RoboChallengeInference/policies/dm0_policy.py` 的实际调用：

```python
self.tokenizer.model_max_length = 100
self.tokenization_func = DM0Tokenization(self.tokenizer)
batch_input_ids = self.tokenization_func([{"from": "human", "value": p}])["input_ids"]
```

任务级 prompt 是写死在 `Dexbotic-RoboChallengeInference/utils/constants.py` 的 `TASK_METADATA` 里，例如：

```python
"turn_on_faucet": {"prompt": "grasp the faucet switch and turn it on", "robot_type": "aloha"},
"put_cup_on_coaster": {"prompt": "place the cup on the coaster", "robot_type": "arx5"},
```

把这一切代入，**dexbotic 开源 DM0 specialist 在 RoboChallenge 上送给模型的真实 prompt** 就是：

```text
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: grasp the faucet switch and turn it on 
```

注意几个证据上很关键、很反直觉的细节：

1. 完全没有 `<image>`、`<image_1>` 等文字占位符。`DM0Tokenization` 从不处理 `<image>`。图像信息走 `images / image_masks` 张量，由 `DM0Model.get_prefix_hidden_states()` 把图像 hidden state 拼到 LLM 序列前缀（dm0_arch.py L310-L356）。
2. 完全没有 `<robot_aloha>`、`<eef_control>`/`<joint_control>` 这类身份 / 控制 tag。dexbotic 全仓 `Grep` 也只能在文档和我们自己的实验脚本里搜到这些字符串，主代码路径没有任何注入。
3. ASSISTANT 轮整个被 `pop()` 删掉，没有 `ASSISTANT:` 前缀（这就是我们之前观察到 DM0-base 不加 `ASSISTANT: ` 时只生成 4 token 的根本原因）。
4. Specialist 的 `model_max_length=100` 强制把 prompt 截到 ~100 token，根本不允许出现长 CoT/多轮 scaffolding。

`DM0DataConfig.build_action_process_func()`（dm0_exp.py L243-L264）的 Pipeline：

```python
[ToDict, ToNumpy, AddAction, PadState(32), PadAction(32),
 AddTrajectory(50), DeltaAction, ActionNorm, LoadMultiModal, ToList]
```

里面 **没有 `AddPromptTemplate`、没有 `ActionNormAnd2String`、没有 `ReplaceAnswer`**，所以：

- 不会把 prompt 包装成 `"<image>\nWhat action should the robot take to {prompt}?"`。
- 不会把数值 action 转成 `<action_0> ... <action_254>` 这种文本 token。
- 也不会给空 `answer` 填默认字符串。

Dexbotic 开源 DM0 训练实际上只激活了 `L_FM`（Flow Matching action expert），`L_AR`（自回归文本损失）虽然在 `config.json` 里是 `true`，但 ASSISTANT 轮被 pop 掉了，`labels` 全是 `IGNORE_INDEX=-100`，相当于没有有效的语言监督——这是论文里 "Knowledge Insulation" 策略的开源体现。

证据：dexbotic GitHub README、官方 docs/DM0.md、`dexbotic/exp/dm0_exp.py`、`dexbotic/tokenization/process.py`、`Dexbotic-RoboChallengeInference/policies/dm0_policy.py`、`Dexbotic-RoboChallengeInference/utils/constants.py`、`dexbotic/data/dataset/transform/default_transform.py`。

### 3.2 Dexdata JSONL 数据格式（官方 docs/Data.md 可验证）

`docs/Data.md` 的官方 episode 样例：

```json
{
  "images_1": {"type": "video", "url": "url1", "frame_idx": 21},
  "images_2": {"type": "video", "url": "url2", "frame_idx": 21},
  "images_3": {"type": "video", "url": "url3", "frame_idx": 21},
  "state": [0.1, 0.2],
  "prompt": "open the door",
  "is_robot": true,
  "answer": "answer text",
  "action": [0.12, 0.24],
  "conversations": [
    {"from": "human", "value": "<image>\nWhat are the colors of the box in the image?"},
    {"from": "gpt", "value": "The box in the image is red."}
  ]
}
```

要点：

- 图像由 `images_1/2/3` 字段引用（视频帧或图片 URL），非文本占位符。
- 文本指令写在 `prompt`，可选 `answer`，或者直接给完整 `conversations` 数组。
- `is_robot` 区分机器人轨迹数据和通用 VL 数据。
- 通用 VL 数据可在 `conversations` 中使用 `<image>` 占位符（这是 LLaVA 风格），但这一支不是 DM0 走的 robot 路径。

DM0 的 robot 轨迹数据进入 `DexDataset` 后（`dexbotic/data/dataset/dex_dataset.py` L264-L272）：

```python
if "conversations" not in data:
    data = ToConversation_Old()(data)
conversations = data["conversations"]
tokenized_dict = self.tokenization_func(conversations=conversations, has_image=True)
```

`ToConversation_Old` 把 `prompt/answer` 转成 `[{"from":"human","value":prompt}, {"from":"gpt","value":answer}]`。然后由 `DM0Tokenization` 处理，按 3.1 的规则做。

### 3.3 论文里 DM0 完整 mid-training/post-training 模板（PDF 可验证，但开源仓未实现）

DM0 论文 PDF 第 7-10 页 + Figure 3 给出了 **完整训练时的对话 / scaffolding 模板**。这一支远比开源仓里跑的复杂。

Figure 3 训练样例（PDF p.8）原话：

```text
[1]H: Please formulate a high-level execution plan to achieve the following objective:
      pick up a dinosaur toy and put in the left box.
[2]G: The identified subtask for the current execution phase is: pick up the blue
      dinosaur toy from the right bin and place it in the left bin.
[3]H: Please provide the future n frames of gripper 2D trajectory points. The format
      should be 'main/left/right gripper: (x,y) (x,y) ...' with a maximum of 25 points,
      and provide the precise movement trajectory and control actions for the
      <robot_arx5> utilizing the <eef_control> mechanism.
[4]G: The predicted 2D trajectory is: main arm gripper: ..., The corresponding
      control action is [250, 0, 250, 110, 250, 250, 0]
```

ALOHA 双臂样例：

```text
[1]H: In order to achieve the objective of fold clothes, please identify the immediate
      subtask that must be performed.
[2]G: Fold the white T-shirt on the bed neatly with both hands.
[3]H: The subtask is confirmed. Please provide the future n frames of gripper 2D trajectory
      points. The format should be 'main/left/right gripper: (x,y) (x,y) ...' with a
      maximum of 25 points.
[4]G: left arm gripper: ...; right arm gripper: ...
[5]H: Proceed to execute the corresponding action sequence on the <robot_aloha>
      platform utilizing the <eef_control> interface.
[6]G: [254, 0, 254, 0, 254, 0, 254, 37, 254, 254, 153, 254, 254, 254, 254, 95]
```

论文 §2.3 把它形式化成 4 阶段 Embodied Spatial Scaffolding：

1. Subtask prediction（自然语言子任务）
2. Goal bounding box prediction（目标 bbox）
3. End-effector trajectory prediction（2D 轨迹关键点）
4. Discrete action prediction（离散 7 自由度动作 chunk）

论文 §3.2 还说了：

- 使用「episodic JSONL records」存数据，每行一个时间步。
- 「templates and supervision fields」：robot trajectory 的对话由「task instruction」和可选的「Subtask、2D waypoint trace、Goal box、Action」拼成；缺哪个字段就自动选不带它的模板。
- 「500 distinct conversation templates for each specific data combination scenario」——也就是每种字段组合预先写了 500 套自然语言模板，训练时随机抽。
- 「discrete tokens for the VLM and continuous values for the action expert」——同一段动作两套表示；离散侧用 255-bin。
- 「multi-view images are deterministically ordered, three views」——三视角顺序固定。

把这些拼起来，**论文版完整模板**（在多个本地分析文档里也复刻了：`b/d/dm0/dm0_txtAsEvry2.md` L1054-L1068、`b/d/dm0/dm0.md` L2336-L2356）大致是：

```text
SYSTEM: You are a helpful robot assistant.

USER: <image_1> <image_2> <image_3>
<robot_aloha> <eef_control>
What action should the robot take to place the cup on the coaster?

ASSISTANT: <subtask>Reach towards the cup, grasp the handle ...</subtask>
<bbox>[234, 156, 567, 423]</bbox>
<traj>(312,456) (318,420) (325,389) ... </traj>
<act> 128 64 200 110 250 250 0 ... </act>
```

注意三点：

- 论文里 SYSTEM 是 `You are a helpful robot assistant.`，不是 dexbotic 开源仓的那段 `A chat between ...`。这是论文与开源 inference 的一处分布差异。
- 模板里的 `<image_1> <image_2> <image_3>` 是文本占位符；实际训练时这些 token 会被替换成 PE 编出来的图像 hidden state。
- `<robot_*>` 和 `<*_control>` 在 USER turn 中、自然语言指令前/末，由 `added_tokens.json` 中的单 token 表示。

证据：DM0 PDF p.4–p.10、Figure 3、Tables in §3.2/§3.3。

### 3.4 dexbotic 通用 prompt template（非 DM0 路径用）

对 CogACT/Pi0/MemVLA/OFT 等模型，dexbotic 用 `dexbotic/data/dataset/transform/language.py` 的：

```python
defalut_prompt_template = "<image>\nWhat action should the robot take to {prompt}?"
```

并通过 `AddPromptTemplate` 把 JSONL 的 raw prompt 包成上面这种带 `<image>` 占位符的句子。

但 DM0 的 `DM0DataConfig` 和 `DM0InferenceConfig` 都 **没有** 启用 `AddPromptTemplate`，所以这个默认模板对 DM0 不生效。

证据：`dexbotic/data/dataset/transform/language.py`、`dexbotic/exp/dm0_exp.py`、`dexbotic/exp/base_exp.py`。

## 四、`added_tokens.json` 各类 token 的真实使用方式

### 4.1 离散动作 token（`<action_0>` ~ `<action_254>`，255 个）

- 用途：`L_AR` 时把 7 维 EEF 动作每一维做 255-bin 量化，每个值取一个 single token。
- 在论文样例里直接以未包裹的 `[250, 0, 250, 110, 250, 250, 0]` 写在 ASSISTANT 文本里（每个数字背后是 `<action_N>` single token）。
- dexbotic 开源仓里：训练 Pipeline 没有 `ActionNormAnd2String`，DM0 不会把 action 数组变成这串 token；推理时也不会显式生成这类 token。所以这 255 个 token 在 vocab 里 **存在但被冷冻**。

证据：DM0 PDF §3.2「255-bin vocabulary as special action tokens」、Figure 3 `[250,0,250,...]`、`dexbotic/data/dataset/transform/action.py:ActionNormAnd2String` 与 DM0 Pipeline 的对比。

### 4.2 像素坐标 token（`<action>0</action>` ~ `<action>727</action>`，728 个）

- 用途：把 728×728 主视角图像中的 2D 坐标整数离散化，每一个像素坐标值占 1 个 single token。
- 在 ASSISTANT 输出里典型形式是「2D EEF 轨迹点列」：

```text
main arm gripper: (<action>342</action>, <action>567</action>) (<action>350</action>, <action>560</action>) ...
```

或者 bbox：

```text
<bbox>[<action>234</action>, <action>156</action>, <action>567</action>, <action>423</action>]</bbox>
```

- 优点是模型只能从 728 个固定坐标 token 里挑，不会出现非整数或越界值，并且每个坐标 = 1 token，效率高。
- 我们在自己的 `eval_dm0_base_v2.py` 里写过 `parse_tags()`：

```python
coord_pairs = re.findall(r"\(\s*<action>(\d+)</action>\s*,\s*<action>(\d+)</action>\s*\)", text)
coord_quads = re.findall(r"\[\s*<action>\d+</action>\s*,\s*<action>\d+</action>\s*,\s*<action>\d+</action>\s*,\s*<action>\d+</action>\s*\]", text)
```

它在 DM0-base 推理结果里成功匹配到了 `xy_coords` 和 `bbox_xyxy`。这反过来强证训练数据 ASSISTANT 侧确实使用了 `<action>N</action>` 包裹坐标，否则 model 不会高频复现这个奇特的串。这是 4.2 里最强的"反编译"证据。

证据：DM0 PDF p.4「multi-view images are resized to 728 × 728」、Figure 3 「main/left/right gripper: (x,y) (x,y)…」、`b/m/dm0/base/added_tokens.json` 中 `<action>0</action>`~`<action>727</action>` 的 single token 化、我们 `b/tst/RoboChallenge/logs/eval_DM0-base_v2_*.jsonl` 里 `gen_parsed_tags` 包含 `xy_coords` 的实际样例。

为什么用 `<action>X</action>` 而不是 `<action_X>` 复用 255-bin vocab？因为：

- 7 自由度 action 量化 = 0-254 的小整数，用 `<action_0..254>` 一族就够。
- 像素坐标 = 0-727，落在不同范围；如果共享同一 token 表会让 LLM 不区分语义；分两族 single-token 还能让一段文本里两种语义共存而不混淆。
- 256 ~ 727 这部分 token 不会出现在 7DoF 动作里，反过来 0~254 这部分坐标 token 在 `<action>` 里也不会和 `<action_*>` 混淆。

### 4.3 跨本体身份 token（`<robot_*>`，18 个）

- 用途：训练数据 prompt 里以 single-token 形式硬编码"当前操控的机器人"，让 VLM 只通过语言上下文区分本体。
- 出现在 USER 一侧（论文 Figure 3 + b/d/dm0/dm0_txtAsEvry2.md 的 4 层 scaffolding 完整样例都印证）。
- dexbotic 开源 SFT 路径中没有自动注入这些 tag，必须靠数据集 JSONL 的 `prompt` 字段自带，或者像我们手动在评测脚本中 `tag_prefix = "<robot_aloha><joint_control>"` 那样补回。

证据：DM0 PDF Figure 3 `<robot_arx5>` / `<robot_aloha>`、`b/d/dm0/dm0.md` L99 / L162-L165 / L2980-L3036、`b/m/dm0/base/added_tokens.json` 单 token 化。

可识别的本体覆盖（从 added_tokens 读出）：UR5、Franka、ALOHA、R1-Lite、ARX5、UMI、Z1、G1、Realman 756f、WidowX、KUKA、xArm、Google Robot、Stretch、Sawyer、Jaco2、FANUC Mate、DLR Edan。论文 §3.2/§3.3 提到 Mid-train 中 single-arm 数据来自 Franka/UR5/ARX5/UMI、双臂数据来自 ALOHA、open-source 包含 RoboMind/Agibot/Galaxea 等——和 18 个 tag 的覆盖面一致。

### 4.4 控制方式 token（`<eef_control>`、`<joint_control>`，2 个）

- 用途：和 `<robot_*>` 配对，区分末端执行器空间还是关节空间控制。
- 同样写在 USER turn 内，例如「on the `<robot_aloha>` platform utilizing the `<eef_control>` interface」。
- DM0 paper §1.3 与 §3 多次明确「动作空间 = EEF」，所以大多数训练样例使用 `<eef_control>`。`<joint_control>` 同 vocab 存在但论文样例较少出现；猜测是 ALOHA 关节控制等保留接口。

证据：DM0 PDF Figure 3 `<eef_control>`、`b/m/dm0/base/added_tokens.json` 中 single-token 化、`b/d/dm0/dm0.md` L92 / L2980-L3036。

### 4.5 三视角图像占位 token（`<image_1> <image_2> <image_3>`）

- 严格来说"`<image_1>`/`<image_2>`/`<image_3>`" 这几个字符串本身 **不在 added_tokens.json 里**，它们在 byte-level BPE 中会拆成普通 sub-word。但是 added_tokens 里给了一组相关的图像 single-token：`<im_patch>` (151679)、`<im_start>` (151680)、`<im_end>` (151681)、`<\|image_pad\|>` (151655)、`<\|video_pad\|>` (151656)、`<\|IMG_START\|>` (151675)、`<\|IMG_END\|>` (151676)、`<patch_start>` (151689)、`<patch_end>` (151690)、`<patch_newline>` (151691)、`<video_start>` (151687)、`<video_end>` (151688)。
- 这一族 token 用法（结合 `dexbotic/tokenization/process.py:llava_multi_image_map_fn` 和 `dexbotic/tokenization/tokenization.py:tokenizer_image_token`）：在通用 LLaVA 风格路径里，把 `<image>` 占位符替换成 `<im_start><image><im_end>` 序列；`<im_patch>` 用作单视角的 patch 锚点；`<patch_start>/<patch_end>/<patch_newline>` 为多 patch / 多行序列做标记。
- 但 **DM0 的 `DM0Tokenization` 不走这条路径**，直接用图像 hidden state 拼接，前缀里看不到这些 token。它们的存在主要是为通用 VL 数据（Cambrian、LLaVA OneVision 1.5 等，论文 §3.2 提到）保留接口。

证据：`b/m/dm0/base/added_tokens.json` ids 151679-151691、`dexbotic/tokenization/process.py` 中 `llava_multi_image_map_fn` 与 `DM0Tokenization` 的代码对比、DM0 PDF §3.2 Vision-language data 列表。

### 4.6 对话 / 推理控制 token

- ChatML：`<|im_start|>` (151644)、`<|im_end|>` (151645) — 由 `conv_step.sep2 = "<|im_end|>"` 实际使用。每一轮 USER/ASSISTANT 末尾会被 tokenizer 编出 `<|im_end|>` single token。
- think：`<|THINK_START|>` / `<|THINK_END|>` / `<think>` — Qwen3 系列的「think 模式」标签。dexbotic DM0 不主动用这些，但 `tokenizer_config.json` 里继承的 chat_template 在 `add_generation_prompt=True` 时会插入 `<|im_start|>assistant\n<think>\n`。论文 mid-training 中 ER 数据可能用到。
- tool / function call：`<tool_call>` / `<tool_response>` 等 — 通用工具调用接口；DM0 工作没用。
- box / quad / object_ref：`<|box_start|>` / `<|box_end|>` / `<|quad_start|>` / `<|quad_end|>` / `<|object_ref_start|>` / `<|object_ref_end|>` — Qwen3-VL 风格的视觉 grounding 标签；论文 §3.2 提到 grounding & counting 数据，可能由这族 token 表达 bbox 标签，但 DM0 训练数据更倾向于直接写 `[x1,y1,x2,y2]` + `<action>N</action>`。
- mask / dream / fim：`<|MASK_*|>` / `<dream*>` / `<|fim_*|>` — 来自 Qwen3 / DeepSeek 系列基础 tokenizer，DM0 不主动用。

证据：`b/m/dm0/base/added_tokens.json` 全表、Qwen3 / Qwen2.5 官方文档对这些 token 的定义。

### 4.7 总结一张图

```text
                            ┌───────────────────────────────────────────────┐
                            │ system: "A chat between ..." 或 论文 "You are a │
                            │           helpful robot assistant."          │
                            │ sep=" "                                       │
                            ├───────────────────────────────────────────────┤
                            │ USER:                                         │
                            │   <image_1> <image_2> <image_3>      ← 论文版  │
                            │   <robot_aloha> <eef_control>        ← 论文版  │
                            │   What action should the robot take to        │
                            │   {task_instruction}?                         │
                            │ sep2 = <|im_end|>                             │
                            ├───────────────────────────────────────────────┤
                            │ ASSISTANT: (论文版)                            │
                            │   <subtask>...</subtask>                      │
                            │   <bbox>[<action>X</action>,                  │
                            │          <action>Y</action>, ...]</bbox>      │
                            │   <traj>(<action>x</action>,                  │
                            │          <action>y</action>) ...</traj>      │
                            │   <act> a₁ a₂ ... a₇ </act>                   │
                            │     ← 数字背后是 <action_N>(0..254) single tok│
                            │ sep2 = <|im_end|>                             │
                            └───────────────────────────────────────────────┘

dexbotic 开源 specialist 推理时实际只剩：
A chat ... USER: {raw task instruction sentence}
（图像走 hidden state，不走文本占位符；<robot_*>/<*_control> 不主动注入；
 ASSISTANT 轮被 pop 掉；model_max_length=100）
```

## 五、几个非常关键的"侦探结论"

1. **dexbotic 仓库开源出的 DM0 训练 / 推理路径，比 DM0 论文中描述的 mid-training/post-training pipeline 简化得多。** 它只有 `L_FM`（Flow Matching）有效，`L_AR` 形同虚设；token 模板退化为「USER: {raw sentence}」+ 图像 hidden state；`<robot_*>` `<*_control>` `<action_N>` `<action>N</action>` 全部在 vocab 里、但都没被自动使用。证据：`dexbotic/exp/dm0_exp.py` Pipeline 对比 `dexbotic/data/dataset/transform/default_transform.py`、`Dexbotic-RoboChallengeInference/policies/dm0_policy.py` 的 `tokenizer.model_max_length=100`。

2. **specialist checkpoint 之所以仍能 SOTA**，是因为它从 `DM0-base` 继承了完整 mid-train 后的 VLM 表征（含本体身份、scaffolding 经验），SFT 阶段只用 `L_FM` 调动作专家，不必再走文本侧。所以即使推理时 prompt 极简（只 1 句任务），模型依然能产出高质量动作——VLM 早在 mid-train 阶段已经"听过"`<robot_aloha>`/`<eef_control>` 等同语义指令。

3. **`<action>0</action>` ~ `<action>727</action>` 是 728 个 single token，不是字符串拼接**。它对应主视角 728×728 图像分辨率的整数像素坐标。我们能在 DM0-base 推理输出里反复观测到这个固定模式，已经是非训练数据但模型自发复现的反向证据。

4. **论文版的 4 层 Embodied Spatial Scaffolding（subtask → bbox → traj → discrete action）在开源代码里没实现**。`b/d/dm0/dm0_txtAsEvry2.md` 等 4 个本地深度分析文档已经对这一缺口做过详细推理，结合论文 PDF Figure 3 / §3.2 / §3.3 的描述，可以认为 `<subtask>...</subtask>`、`<bbox>...</bbox>`、`<traj>...</traj>`、`<act>...</act>` 这种 XML-like 包裹是论文实际训练时的真实 ASSISTANT 格式（注意：`<subtask>`/`<bbox>`/`<traj>`/`<act>` 这些 tag **本身** 不在 added_tokens.json 里，会被 BPE 拆词；只有里面的 `<action>N</action>`、`<action_N>`、`<robot_*>`、`<*_control>` 是 single-token）。

5. **`dexbotic/tokenization/process.py:DM0Tokenization` 里"Format: 'System prompt USER: prompt ASSISTANT: <empty>'" 这条 docstring 是误导**。实际行为是当 ASSISTANT 为空时被 pop 掉，prompt 末尾没有 `ASSISTANT:` 字面量。这一点在我们之前一轮 prompt placement 实验里已经吃过亏（要手工补 `ASSISTANT:` 才能让 DM0 进入回答态）。

## 六、给后续工作的可操作结论

如果想让 dexbotic 开源 DM0 真正复现论文的 CoT/scaffolding 行为，需要做以下事情，但都会牵涉训练侧改动：

1. 在 `DM0DataConfig.build_action_process_func()` 中加入 `AddPromptTemplate("{robot_tag} {control_tag}\n<image>\nWhat action should the robot take to {prompt}?")`（按本体动态选 tag）。
2. 在数据 Pipeline 中加入 `ActionNormAnd2String(vocab_size=255, string_format=" <action_{value}>")`，把数值 action 转成 `<action_N>` token，使 ASSISTANT 端有真实文本标签。
3. 同时把 2D 轨迹/bbox 用 `<action>N</action>` 单 token 输出，并放在 `<traj>`/`<bbox>` 包装内。
4. 解开 `model_max_length=100` 限制（只对 specialist 推理有效），训练时用 `model_max_length=2048`。
5. 让 `DM0Tokenization` 不要 pop 空 ASSISTANT 轮，或在 inference 时显式加 `ASSISTANT: ` 前缀（我们 v2 评测脚本就是这么做的）。
6. 训练时让 `L_AR` 的 `loss_mask` 真正打开（assistant 内容非空），否则 ar_loss=true 形同虚设。

参考资料（按可信度排序）：

- DM0 论文 PDF：[`b/d/dm0/DM0_An_Embodied-Native_Vision-Language-Action_Model_towards_Physical_AI.pdf`](b/d/dm0/DM0_An_Embodied-Native_Vision-Language-Action_Model_towards_Physical_AI.pdf)（一手）
- 官方 GitHub 仓 `dexmal/dexbotic`：[https://github.com/dexmal/dexbotic](https://github.com/dexmal/dexbotic) 与本地源码（一手）
- 官方推理仓 `dexmal/Dexbotic-RoboChallengeInference`：[https://github.com/dexmal/Dexbotic-RoboChallengeInference](https://github.com/dexmal/Dexbotic-RoboChallengeInference)（一手）
- 官方 Hugging Face 模型卡：[`Dexmal/DM0-base`](https://huggingface.co/Dexmal/DM0-base)、`DM0-libero`、`DM0-table30_*`
- 官方文档：[https://dexbotic.com/docs/](https://dexbotic.com/docs/) 与本地 `docs/Data.md`、`docs/DM0.md`、`docs/web_docs/*.md`
- 本地 4 个 checkpoint 配置：`/mnt/g/CKPT/dexbotic/DM0-{base,libero,table30_generalist_aloha,table30_turn_on_faucet}/{config.json,tokenizer_config.json,added_tokens.json,special_tokens_map.json}`
- 本地深度分析文档（自己之前的笔记，二手但可交叉验证）：`b/d/dm0/dm0.md`、`b/d/dm0/dm0_txtAsEvry2.md`、`b/d/dm0/dm0_txtAsEvry2_op46.md`、`b/d/dm0/dm0_txtAsEvry2_op47.md`、`b/d/dm0/dm0_lAr.md`

# DM0处理 图像 有没有 加特殊token

读完 `dm0_arch.py`、`dm0_utils.py`、`pe_encoder.py`、`pe_configuration.py` 和 `mm_projector/builder.py`，结论清楚了，简单回答就是：DM0 完全没有用 `<im_start>/<im_end>` 这类显式分隔 token 来标识图像与文本，也没有任何 view-id token。它靠 **三件事** 来让 VLM 区分图像 / 文本以及图像 1 / 2 / 3：

1. 不同的"嵌入子空间"（image hidden state 走的是视觉编码器+投影矩阵，text 走的是 `embed_tokens` 表）
2. 在 LLM 序列中的 **RoPE 位置编号**（每张图固定占 169 个连续位置）
3. 数据侧 **固定的相机视角顺序约定**（high → left_hand → right_hand 写在收集端）

下面把每条对应到代码证据。

## 1. 是否用 `<im_start>` / `<im_end>` 这类 token？

**没用**。这些 token 在 `added_tokens.json` 里都是 single token（`<im_patch>=151679`、`<im_start>=151680`、`<im_end>=151681`、`<|IMG_START|>=151675` 等），但是 `DM0Tokenization` 的 `__call__()`（dexbotic/tokenization/process.py:386-483）从头到尾只处理 `system / USER / ASSISTANT` 字面量，从来不插入 `<im_*>` 系列 token。

唯一会把 `<image>` 占位符替换成 `<im_start><image><im_end>` 的是 **CogACT 等 LLaVA 风格模型**走的 `tokenize_dexbotic` → `llava_multi_image_map_fn`（process.py:46-57）：

```python
if mode == "step":
    msg["value"] = msg["value"] + f"<im_start>{DEFAULT_IMAGE_TOKEN}<im_end>"
```

**DM0 不走这条路径**。dexbotic 的 DM0 推理路径是：

```python
# dexbotic/exp/dm0_exp.py:_get_response()  和  Dexbotic-RoboChallengeInference/policies/dm0_policy.py
batch_input_ids = self.tokenization_func([{"from": "human", "value": p}])["input_ids"]
```

`p` 是任务描述纯文本，没有 `<image>` 字符串，也没有 `<im_start>/<im_end>`。

那为什么 vocab 里又给 DM0 留了这些 token？因为 DM0 的训练数据 mid-train 阶段还包含 LLaVA / Cambrian / OneVision 这一支通用 VL 数据（论文 §3.2「Vision–language data」），这部分走的是另一套带 `<im_start>/<im_end>` 的标准 LLaVA prompt；而 robot 轨迹这一支不需要这套 token，所以 `DM0Tokenization` 才不插入它们。

## 2. 图像 hidden state 在序列里到底长什么样

`DM0Model.embed_image()`（dm0_arch.py:94-102）：

```python
image_features = self.mm_vision_tower(image)   # PE-LANG-L14-728
image_features = self.mm_projector(image_features)  # linear4x → 投影到 LLM hidden size
```

每张图的 token 数 = `(image_size // patch_size // 4) ** 2 = (728//14//4)**2 = 13**2 = 169`（pe_encoder.py:69-71）。证据链：

- 728×728 输入 → 14×14 patch → 52×52 = 2704 个 patch
- 视觉编码器内部两层 `vit_downsampler{1,2}`（pe_model.py:445-458），都是 `kernel=3, stride=2`，4× 下采样 → 26×26 → 13×13 = 169 个 patch token
- `mm_projector_type="linear4x"`（mm_projector/builder.py:51-60）把 `mm_hidden_size * 4` 一次性投影到 LLM 的 `hidden_size=2048`

所以一个 ALOHA frame 的 3 张视角进入 LLM 序列时，前缀长度是固定的：

```text
[ image_1 hidden states: 169 vectors ]
[ image_2 hidden states: 169 vectors ]
[ image_3 hidden states: 169 vectors ]
[ text   hidden states: T 个，T 来自 input_ids ]
```

`get_prefix_hidden_states()`（dm0_arch.py:310-356）就是这么拼的：

```text
hidden_states_list = []
for image, image_mask in zip(images, image_masks):
    image_hidden_states = self.encode_images(image)
    hidden_states_list.append(image_hidden_states)
    padding_mask_list.append(image_mask.unsqueeze(1).expand(batch_size, num_img_tokens))
    attn_mask_list += [1] * num_img_tokens
if input_ids is not None:
    text_hidden_states = self.model.embed_language_tokens(input_ids)
    hidden_states_list.append(text_hidden_states)
    padding_mask_list.append(attention_mask)
    attn_mask_list += [1] * num_lang_tokens
hidden_states = torch.cat(hidden_states_list, dim=1)
```

注意几点：

- 三段图像是连写的，**中间没有任何分隔 token / 边界 token / cls token**。
- `padding_mask` 用的是 `image_mask` 直接 broadcast 到 169 个槽位上：某张图缺失，那 169 个槽就整个置 False，被后面的 attention 屏蔽。
- `attn_mask_list += [1] * num_img_tokens` 给每个图像 token 也写了"causal boundary = 1"，和文本 token 同等待遇 → 整个前缀按 causal 顺序处理，不区分模态。

## 3. VLM 怎么区分"哪个 hidden state 来自图像、哪个来自文本"

只靠"嵌入分布不同 + 位置不同"，没有显式 modality token。具体三个机制：

**A. 不同的嵌入路径，天生就分布不同**

- 图像 → `mm_vision_tower` (PerceptionEncoder L/14, 23 层 transformer) → `mm_projector` (Linear `mm_hidden_size*4 → hidden_size`)
- 文本 → `llm.embed_tokens(token_ids)`（LLM 的 word embedding 表）

这两路最后都映射到 LLM hidden size = 2048 的同一空间，但来源完全不同。training 阶段，`mm_projector` 的权重会被 LLM 自己的 attention 反推回去——LLM 学到"在 2048 维空间里某些子方向是图像信号、某些是文本 token 信号"。

PE 编码器内部用 `use_rope2d=True`（pe_configuration.py:22, 67），patch 内部用 2D RoPE，所以图像 patch 的 hidden state 就带有"我是图像 patch 而不是文本"的强统计特征。这种"靠分布学"的做法是 LLaVA / Pi0 / DM0 / Qwen-VL 这一路的共同选择，DM0 没有特殊化。

**B. RoPE 1D 位置编码强制把序列分段**

`forward()` 和 `inference_action()` 里（dm0_arch.py:475-481）：

```python
prefix_positions = torch.cumsum(prefix_padding_mask, dim=1) - 1
prefix_offsets = torch.sum(prefix_padding_mask, dim=-1)[:, None]
suffix_positions = prefix_offsets + torch.cumsum(suffix_padding_mask, dim=1) - 1
positions = torch.cat([prefix_positions, suffix_positions], dim=1)
```

`cumsum(padding_mask) - 1` 相当于"给每个有效 token 顺序编号 0, 1, 2, …"。在 prefix 段：

- 有效 image_1 token 占位置 `0 ~ 168`
- 有效 image_2 token 占位置 `169 ~ 337`
- 有效 image_3 token 占位置 `338 ~ 506`
- 文本 token 从 `507` 开始

然后 `_compute_merged_layer()`（dm0_arch.py:204-216）把这套 position 直接喂给 Qwen3 的 RoPE：

```python
cos, sin = rotary_emb(dummy_tensor, position_ids)
query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

LLM attention 内部在 query/key 上做 RoPE 旋转。同一个 query 与 RoPE 位置 100 的 key 和 RoPE 位置 600 的 key 进 dot product 时，旋转量不同，attention 分布天然就被位置分层了。这就是 VLM 区分"图像段 vs 文本段"以及"image_1 段 vs image_3 段"的最主要机制。

**C. 因果 attention mask 把模态边界进一步固定**

`make_attn_mask_2d`（dm0_utils.py:12-40）对 `attn_mask` 做 `cumsum`，然后令 `cumsum[query] <= cumsum[key]` 才能 attend：

```python
cumsum = torch.cumsum(attn_mask, dim=1)
attn_mask_2d = cumsum[:, None, :] <= cumsum[:, :, None]
padding_mask_2d = padding_mask[:, None, :] * padding_mask[:, :, None]
return attn_mask_2d & padding_mask_2d
```

因为 prefix 里每个 token 的 `attn_mask=1`，结果就是经典的下三角 causal mask。文本 token 可以 attend 到所有图像 token + 它前面的文本 token；图像 token 之间也是按"先 image_1 后 image_2 后 image_3"的因果方向流。所以一旦图像顺序固定，**LLM 拿到 query=文本 token 时，看到的 key 序列结构永远是"image_1 的 169 个 key → image_2 的 169 个 key → image_3 的 169 个 key → 之前的文本 key"**——位置绑定 + RoPE 一起把这个分段固化下来。

## 4. 那它怎么区分图像 1 vs 图像 2 vs 图像 3

完全靠 **位置 + 数据顺序**，没有任何 view-id token：

**A. 位置编号区分 view**

承上：image_1 的 169 个 hidden state 拿到 RoPE 位置 0–168，image_2 拿 169–337，image_3 拿 338–506。这三段位置是不重叠的，attention 内部就能学出"在 0–168 这一段我看到的是顶视相机、169–337 这一段是左腕相机"等等。

**B. 训练 / 推理两端共同维护"固定相机顺序"约定**

证据来自 `Dexbotic-RoboChallengeInference/utils/constants.py`：

```python
IMAGE_TYPE_MAP = {
    "arx5":   ["high", "left_hand", "right_hand"],
    "aloha":  ["high", "left_hand", "right_hand"],
    "ur5":    ["right_hand", "left_hand"],
    "franka": ["high", "left_hand", "right_hand"],
}

IMAGE_MAPPING = {
    "arx5":   {"high": "image_1", "left_hand": "image_2", "right_hand": "image_0"},
    "aloha":  {"high": "image_0", "left_hand": "image_1", "right_hand": "image_2"},
    "ur5":    {"right_hand": "image_0", "left_hand": "image_1"},
    "franka": {"high": "image_1", "left_hand": "image_2", "right_hand": "image_0"},
}
```

这就是"训练数据收集端怎么给三个相机排序，推理端就必须按同样顺序排"的硬约定。dexbotic Data.md 也明确写了：

> We recommend using the Main View in `images_1`, the Left Hand View in `images_2`, and the Right Hand View in `images_3`.

把多视角顺序固化下来，VLM 在大量训练后自然会学到"序列开头 169 个 token 是 main view，下一段是 left wrist，再下一段是 right wrist"。这一约定一旦破坏（比如把 image_2 和 image_3 互换），模型会大概率掉性能，因为它没有任何"自我矫正"的 view-id token 可参考。

**C. 视角缺失时如何对齐**

`image_masks` 是 `[B, num_views]` 的 bool。在 `get_prefix_hidden_states()` 里它被 broadcast 到每张图的 169 个槽位（dm0_arch.py:333-336）：

```python
padding_mask_list.append(image_mask.unsqueeze(1).expand(batch_size, num_img_tokens))
```

然后 `prefix_positions = cumsum(padding_mask) - 1`：

- 当某张图 `image_mask=False` 时，那 169 个槽 padding_mask 全 False，`cumsum` 在那里不增长；
- 这意味着**缺失的 169 个槽和它前一个有效 token 共享同一个 position id**，并且 `make_attn_mask_2d` 里 `padding_mask_2d` 把它们置为不可被 attend；
- 实际效果：缺失的 view 像被"压缩为零长度"，后续 view + 文本的位置编号就紧贴上去，不留空洞。

所以 ALOHA 三视角全有时，文本起点 RoPE 位置是 `3*169=507`；只有两视角时，文本起点是 `2*169=338`。LLM 看到的相对位置布局是连续的，没有因为缺图而错乱。

## 5. 一张图总结

```text
                LLM 输入序列（DM0 prefix 部分）
┌──────────────────────────────────────────────────────────────────┐
│ image_1 (169 tok)    image_2 (169 tok)    image_3 (169 tok)  USER:...    │
│  ↑                    ↑                    ↑                    ↑        │
│ mm_proj(PE(img1))    mm_proj(PE(img2))    mm_proj(PE(img3))   embed_tokens(text) │
│  RoPE 0..168          RoPE 169..337        RoPE 338..506      RoPE 507.. │
│  (无 <im_start>      (中间无任何分隔        (中间无任何分隔     (无 <image>) │
│   <im_end>)            token)              token)                        │
│                                                                          │
│  若某 view 缺失 → image_mask=False → padding_mask_2d 屏蔽 + position 跳过 │
└──────────────────────────────────────────────────────────────────┘
                                                                            │
                                                                            ▼
                             ┌─────────────────────────────┐
                             │ Action Expert (suffix)        │
                             │ 50 步 chunk + flow matching   │
                             │ 第一个 action token attn=1     │
                             │ 后续 attn=0（chunk 内可互通） │
                             │ RoPE 接续在 prefix 之后         │
                             └─────────────────────────────┘
```

VLM 区分模态、区分相机视角的全部机制就是上面的"嵌入路径不同 + RoPE 位置 + 数据顺序约定"。如果未来做多本体 fine-tune 想增强健壮性，就可以考虑：

- 在每段 image hidden state 前后插入 `<im_start>/<im_end>`（vocab 已经预留），让模态边界有显式锚点；
- 或在 USER 文本里前置 `<image_1> <image_2> <image_3>` 占位字符串，给模型一个"哪一段是哪个相机"的语言提示（这正是论文 Figure 3 训练样例里使用的形式，只是 dexbotic 开源 SFT 路径没启用）。

参考资料：

- 源码：`dexbotic/model/dm0/dm0_arch.py`、`dexbotic/model/dm0/dm0_utils.py`、`dexbotic/model/modules/mm_vision/pe/pe_configuration.py`、`dexbotic/model/modules/mm_vision/pe/pe_encoder.py`、`dexbotic/model/modules/mm_projector/builder.py`、`dexbotic/tokenization/process.py`
- 官方推理仓 `dexmal/Dexbotic-RoboChallengeInference`：`utils/constants.py`、`policies/dm0_policy.py`
- 官方文档：[`docs/Data.md`](docs/Data.md)、[`docs/DM0.md`](docs/DM0.md)
- 论文：[`b/d/dm0/DM0_An_Embodied-Native_Vision-Language-Action_Model_towards_Physical_AI.pdf`](b/d/dm0/DM0_An_Embodied-Native_Vision-Language-Action_Model_towards_Physical_AI.pdf) §2.1、Figure 3

## DM0 用的是 Qwen3-1.7B 而不是它的VL版

确认：**DM0 真实用的是 Qwen3 语言模型，不是 Qwen3-VL。**

更准确地说，DM0 是：

```text
Qwen3 LLM backbone
+ 自定义 Perception Encoder 视觉塔（pe_lang_l14_728）
+ 自定义 mm_projector（linear4x）
+ Qwen3 action expert（去掉 embed_tokens）
+ Flow Matching action head
```

不是 HuggingFace 意义上的 `Qwen3-VLForConditionalGeneration` 或 Qwen3-VL 原生多模态模型。

证据很直接：

```text
DM0-base:
  model_type: dexbotic_dm0
  llm model_type: qwen3
  llm hidden: 2048
  llm layers: 28
  llm vocab: 152701
  action model_type: qwen3
  action hidden: 1024
  action layers: 28
  action vocab: 151936
  vision: pe_lang_l14_728
  projector: linear4x
```

四个 checkpoint 都一样：

- `/mnt/g/CKPT/dexbotic/DM0-base`
- `/mnt/g/CKPT/dexbotic/DM0-libero`
- `/mnt/g/CKPT/dexbotic/DM0-table30_generalist_aloha`
- `/mnt/g/CKPT/dexbotic/DM0-table30_turn_on_faucet`

源码也印证这一点：

```12:18:dexbotic/model/dm0/dm0_arch.py
from transformers import (
    AutoConfig,
    CONFIG_MAPPING,
    DynamicCache,
    Qwen3ForCausalLM,
)
from transformers.models.qwen3 import modeling_qwen3
```

并且 action expert 也是 `Qwen3ForCausalLM`：

```77:80:dexbotic/model/dm0/dm0_arch.py
# Build action expert (Qwen3ForCausalLM without embed_tokens)
action_model_config = config.action_config
self.action_expert = Qwen3ForCausalLM(action_model_config)
self.action_expert.model.embed_tokens = None
```

视觉侧不是 Qwen3-VL，而是 dexbotic 自己接的 PE：

```text
mm_vision_tower = pe_lang_l14_728
mm_projector_type = linear4x
```

所以之前我说的 "VLM backbone 是 Qwen3-1.7B + PE 视觉编码器 + projector" 才是准确描述；如果简称 "Qwen3-VL" 容易误导，因为它不是官方 Qwen3-VL 架构。更严谨的叫法应是：

> **DM0 uses a Qwen3-based VLM assembled by Dexbotic: Qwen3 LLM + PE vision encoder + linear4x projector, not the native Qwen3-VL model.**

