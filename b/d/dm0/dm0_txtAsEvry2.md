# DM0 Embodied Spatial Scaffolding：4 层级辅助任务逐任务代码分析与实现方案

> 最后更新: 2026-05-16
> 前置阅读: [dm0.md](./dm0.md)（论文解读）、[dm0_txtAsEvry.md](./dm0_txtAsEvry.md)（跨架构 L_AR 全景）、[dm0_larCmpl.md](./dm0_larCmpl.md)（外部可比论文）

---

## 0. TL;DR

### 4 个辅助任务实现状态一览

| # | 辅助任务 | 论文要求 | 代码状态 | 基础设施就绪度 |
|---|---------|---------|---------|--------------|
| 1 | Subtask prediction | 预测细粒度子任务文本 | **未实现** | 20%（需新增数据标注 + 对话模板 + loss） |
| 2 | Goal bbox prediction | 预测目标物体 bbox | **未实现** | 10%（完全空白，需从零构建） |
| 3 | EEF trajectory prediction | 预测 2D EEF 轨迹点 | **未实现** | 40%（`AddTrajectory` 可复用，但缺文本化 + loss） |
| 4 | Discrete action prediction | 预测 255-bin 离散 action token | **未实现** | **80%**（`ActionNormAnd2String` + `lm_head` + `prefix_out` 全在） |

### 核心结论

1. **DM0 代码当前只有 L_FM（Flow Matching MSE loss），L_AR 分支完全空白** — `prefix_out` 在 `dm0_arch.py:486` 返回后被丢弃，`labels` 在 `dm0_arch.py:413` 接收但从未使用
2. **任务 4（离散动作预测）最容易实现** — 只需 ~15 行改动即可获得 L_AR loss
3. **任务 1-3 需要数据标注支持** — 当前 JSONL 数据格式中不包含 subtask/bbox/trajectory 文本字段
4. **HybridPi05 (`hybrid_pi05_arch.py:455-512`) 是所有 4 个任务的主参考实现** — text_loss + action_loss 双 Loss 的完整范本

### 与 dm0_txtAsEvry.md 的区别

| 维度 | dm0_txtAsEvry.md | 本文 (dm0_txtAsEvry2.md) |
|------|-----------------|------------------------|
| 视角 | 跨架构 L_AR 模式分类 | 逐任务深入分析 |
| 粒度 | 6 种模式 + 3 层方案 | 4 个辅助任务 × 各自的设计方案 |
| 外部参考 | 无 | 结合 dm0_larCmpl.md 中 14+ 篇论文 |
| 重点 | "怎么加 L_AR" | "4 个 scaffolding 任务各自怎么实现" |

---

## 1. 论文 vs 代码：Embodied Spatial Scaffolding 整体缺口

### 1.1 论文描述（dm0.md §2.9 / §5.5）

DM0 论文定义了 **L_total = λ·L_AR + L_FM (λ=1)**，其中 L_AR 监督以下 4 个层级的辅助任务：

```
[高层语义] Subtask prediction → "grasp the cup handle"
     ↓
[空间定位] Goal bbox prediction → [234, 156, 567, 423]
     ↓
[运动规划] EEF trajectory prediction → (x0,y0) (x1,y1) ... (xN,yN)
     ↓
[底层控制] Discrete action prediction → 128 64 200 110 250 250 0
```

所有 4 层都通过自回归 CE loss 统一监督 — "text-as-everything"。

### 1.2 代码实际状态

**DM0 的 forward 方法（`dm0_arch.py:406-511`）只计算 L_FM：**

```python
# dm0_arch.py:486 — prefix_out 返回但被丢弃
(prefix_out, suffix_out), _ = self._merged_attention_forward(...)

# dm0_arch.py:498-502 — 只有 flow matching loss
suffix_out_final = suffix_out[:, -self.model.config.chunk_size :]
v_t = self.model.action_out_proj(suffix_out_final)
action_loss = F.mse_loss(v_t, u_t, reduction="mean")
loss = action_loss  # ← 没有 text_loss，没有 L_AR
```

**关键缺口清单：**

| 缺口 | 具体位置 | 说明 |
|------|---------|------|
| `prefix_out` 未使用 | `dm0_arch.py:486` | 返回后直接丢弃 |
| `lm_head` 未调用 | `dm0_arch.py:140-142` | 仅为 `tie_weights` 兼容创建 |
| `labels` 被忽略 | `dm0_arch.py:413` | 签名中接收但函数体不使用 |
| `CausalLMOutputDexbotic` 字段空缺 | `dm0_arch.py:504-510` | 不传 `text_loss`/`action_loss` |
| assistant turn 被弹空 | `process.py:407-414` | 空 assistant → labels 全 IGNORE_INDEX |
| `use_special_tokens=False` | DM0 默认配置 | 不注册离散 action token |
| data_keys 无辅助字段 | `dm0_exp.py:271-278` | 无 subtask/bbox/trajectory/has_text |

### 1.3 "零成本"接入点

尽管 4 个任务都没实现，但 **基础设施层面 80% 的组件已经就绪**：

```
prefix_out (dm0_arch.py:486)
    → self.lm_head(prefix_out) (dm0_arch.py:140)
        → F.cross_entropy(logits, labels) 
            → CausalLMOutputDexbotic(text_loss=..., action_loss=...)
                → Trainer 自动发现 *_loss (trainer.py:172)
                    → wandb/tensorboard 自动日志 (trainer.py:185)
```

这条路径上每个组件都已存在，只是没有被连接起来。

---

## 2. 任务一：Subtask Prediction（子任务预测）

### 2.1 论文要求

DM0 论文 §2.3 描述：预测**细粒度子任务描述**，作为 Embodied Spatial Scaffolding 的最高层级。

**论文样例**（推断自 Table 5 的 CoT 预测示例）：
```
指令: "Place the cup on the coaster"
子任务预测: "Reach towards the cup. Grasp the cup handle. Lift the cup. 
             Move towards the coaster. Lower the cup onto the coaster. Release."
```

子任务文本由 VLM 自回归生成，通过 L_AR (CE loss) 监督。

### 2.2 代码现状

**关键词 "subtask" 搜索结果：**

仅在数据转换脚本中出现（如 `convert_so101_to_dexdata.py`、`convert_lerobot_to_dexdata.py`），作为 metadata 字段，**不在训练路径中**。

**DM0Tokenization (`process.py:368-483`)：**
- 只处理 human/gpt 对话轮次
- 空 assistant turn 被弹出（line 407-414）
- 无 subtask slot，无特殊标记

**DM0DataConfig (`dm0_exp.py:271-278`)：**
```python
data_keys: list[str] = field(
    default_factory=lambda: [
        "input_ids", "labels", "action",
        "image", "state", "image_masks",
    ]
)
```
无 `subtask` 字段。

### 2.3 跨库参考实现

**代码库内部：**
- **HybridPi05** (`hybrid_pi05_arch.py:455-479`)：text_loss 计算的完整范本
  ```python
  text_logits = self.lm_head(prefix_out)  # line 455
  token_loss = F.cross_entropy(pred_tokens.transpose(1, 2), target_tokens, reduction="none")  # line 462
  token_mask = torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)  # line 465
  ```
- **DM0Prog** (`dm0_prog_arch.py:93-95`)：progress 预测通道是辅助预测的参考架构，虽然只在推理阶段使用

**外部论文（dm0_larCmpl.md）：**
- **VLM2VLA (Actions as Language, ICLR 2026)**：将动作重新标注为自然语言文本（如 "move gripper left 2cm"），保留 VLM 85%+ VQA 能力。这是最接近 DM0 subtask prediction 理念的工作。
- **UniVLA (ICLR 2026)**：全模态离散化，文本和动作走统一自回归，LIBERO 95.5%

### 2.4 设计方案

#### 数据层

在 JSONL 中增加 `subtask` 字段：

```json
{
    "images_1": {"type": "video", "url": "...", "frame_idx": 42},
    "state": [0.1, 0.2, ...],
    "prompt": "Place the cup on the coaster",
    "subtask": "Grasp the cup handle and lift it up",
    "answer": ""
}
```

子任务标注方式（3 种来源）：
1. **人工标注**：对 500 条对话模板的子集进行子任务分解
2. **VLM 自动标注**：用 Qwen3 / GPT-4V 对轨迹视频生成子任务描述
3. **规则生成**：对简单任务（如 pick-place）用模板 `"reach {object}, grasp {object}, move to {target}, place {object}"`

#### 对话模板

在 assistant turn 中插入 subtask 文本：

```
USER: <image> What action should the robot take to place the cup on the coaster?
ASSISTANT: <subtask>Grasp the cup handle and lift it up.</subtask>
```

需要在 `conversation.py` 中注册 `<subtask>` / `</subtask>` 特殊标记。

#### Tokenization 改动

在 `DM0Tokenization.__call__()` 中：

```python
# process.py — 修改 1: 不弹空 assistant turn（当有 subtask 时）
conversations = list(conversations)
if (
    conversations
    and conversations[-1].get("from") == "gpt"
    and not conversations[-1].get("value")
    and not has_subtask  # ← 新增条件
):
    conversations.pop()
```

#### Forward 改动

```python
# dm0_arch.py — 在 line 502 之后插入
if labels is not None:
    text_logits = self.lm_head(prefix_out)
    # 对齐 HybridPi05 的 per-token 掩码逻辑
    shift_logits = text_logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    text_loss = token_loss
    loss = action_loss + text_loss
```

#### 工作量估计

| 组件 | 改动量 | 复杂度 |
|------|-------|--------|
| 数据标注 | 中（需 VLM 自动标注管线） | ★★★ |
| 对话模板 | ~10 行 | ★ |
| Tokenization | ~5 行 | ★ |
| Forward | ~10 行 | ★★ |
| **总计** | ~25 行代码 + 数据标注 | ★★★ |

---

## 3. 任务二：Goal Bounding Box Prediction（目标框预测）

### 3.1 论文要求

DM0 论文 §2.3 描述：预测**目标物体/区域的 bounding box**。

论文在自动驾驶数据中已使用了 `(category + metric depth + bbox in [0,1000])` 的格式。对机器人操作场景，预测的是目标物体在主视角图像中的 2D bbox。

**推断格式**：
```
Goal: "cup"
Bbox: [234, 156, 567, 423]  (x1, y1, x2, y2 in [0, 1000])
```

### 3.2 代码现状

**完全空白。**

在整个 `d:/SRC/Robot/dexbotic/` 代码库中搜索以下关键词：
- `bbox`: 仅在 augmentations.py docstring 中提到 albumentations 库支持 bbox — 未被 dexbotic 使用
- `bounding_box`, `bounding`: 无匹配
- `goal_bbox`, `grounding`: 无匹配
- `x1y1x2y2`, `xyxy`, `xywh`: 无匹配

无坐标格式化工具，无 bbox 数据加载，无 bbox loss 计算。

### 3.3 跨库参考实现

**代码库内部：**
- 无直接参考。但 `ActionNormAnd2String._action2bin()` (`action.py:387-391`) 的量化逻辑可以复用于 bbox 坐标量化

**外部论文（dm0_larCmpl.md）：**
- **DM0 论文自身**：驾驶数据使用 `bbox in [0,1000]` 格式 — 这就是论文推荐的 bbox 表达方式
- **OpenVLA**：grounding 数据使用归一化坐标
- **OmniSAT (ICLR 2026)**：B-Spline + Residual VQ 对坐标的压缩方法

### 3.4 设计方案

#### 坐标归一化策略

对齐论文驾驶数据格式，将像素坐标归一化到 **[0, 1000] 整数**：

```python
def normalize_bbox(bbox, img_width, img_height):
    """归一化 bbox 到 [0, 1000] 整数.
    
    Args:
        bbox: [x1, y1, x2, y2] 像素坐标
        img_width, img_height: 图像尺寸
    Returns:
        [x1, y1, x2, y2] in [0, 1000]
    """
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 / img_width * 1000)),
        int(round(y1 / img_height * 1000)),
        int(round(x2 / img_width * 1000)),
        int(round(y2 / img_height * 1000)),
    ]
```

[0, 1000] 的优势：
1. 与论文驾驶数据格式一致
2. 每个坐标值只需 1-4 个文本 token（"234" = 3 tokens）
3. 精度足够（1/1000 ≈ 0.7 像素 @728×728）

#### 数据层

JSONL 新增 `goal_bbox` 字段：

```json
{
    "prompt": "Place the cup on the coaster",
    "subtask": "Grasp the cup handle",
    "goal_bbox": [[234, 156, 567, 423]],
    "goal_object": ["cup"],
    "state": [0.1, 0.2, ...],
    "answer": ""
}
```

`goal_bbox` 是一个 list of lists，支持多目标（如双臂操作场景中有两个目标物体）。

标注方式：
1. **预训练检测模型**：用 Grounding-DINO / OWL-ViT 对每帧图像检测目标物体
2. **人工修正**：对检测结果做人工 QA
3. **仿真自动标注**：LIBERO / RoboTwin 等仿真环境可直接获取 GT bbox

#### 对话模板

```
USER: <image> What action should the robot take to place the cup on the coaster?
ASSISTANT: <subtask>Grasp the cup handle.</subtask> <bbox>[234, 156, 567, 423]</bbox>
```

#### 新增 Transform

```python
# transform/grounding.py — 新文件

class AddGoalBBox:
    """将 goal_bbox 字段格式化为文本并注入对话."""
    
    def __init__(self, coord_range=1000, img_size=728):
        self.coord_range = coord_range
        self.img_size = img_size
    
    def __call__(self, episode_data_dict, **kwargs):
        if 'goal_bbox' not in episode_data_dict:
            return episode_data_dict
        
        bboxes = episode_data_dict['goal_bbox']
        # 格式化为文本: "[234, 156, 567, 423]"
        bbox_strs = []
        for bbox in bboxes:
            normalized = self._normalize(bbox)
            bbox_strs.append(f"[{normalized[0]}, {normalized[1]}, {normalized[2]}, {normalized[3]}]")
        
        episode_data_dict['bbox_text'] = ' '.join(bbox_strs)
        return episode_data_dict
    
    def _normalize(self, bbox):
        """像素坐标 → [0, coord_range] 整数."""
        return [int(round(v / self.img_size * self.coord_range)) for v in bbox]
```

#### Forward 改动

无需额外改动 — bbox 文本已经作为 assistant turn 的一部分被 tokenize，走 §2.4 的统一 text_loss 路径。

#### 工作量估计

| 组件 | 改动量 | 复杂度 |
|------|-------|--------|
| 数据标注（检测模型 + 人工修正） | 大 | ★★★★ |
| `AddGoalBBox` transform | ~30 行新文件 | ★★ |
| 对话模板扩展 | ~5 行 | ★ |
| Forward | 0（走统一 text_loss） | — |
| **总计** | ~35 行代码 + 大量数据标注 | ★★★★ |

---

## 4. 任务三：EEF Trajectory Prediction（末端轨迹预测）

### 4.1 论文要求

DM0 论文 §2.3 描述：预测**主视角下未来若干帧的 EEF 2D 轨迹**。

这不是 action space 中的 3D EEF 位姿，而是**图像坐标系中的 2D 投影点**（类似于在摄像头画面上画出 EEF 的运动轨迹）。

**推断格式**（论文图示中可见）：
```
EEF trajectory (image coords): (312,456) (318,420) (325,389) (340,365) ... (412,234)
```

### 4.2 代码现状

**部分基础设施存在但未用于文本预测。**

**`AddTrajectory` (`action.py:156-226`)：**
```python
class AddTrajectory:
    """Create trajectories of length T from sequential actions.
    Output: (N, T, D) or flattened (N, T*D)
    Supports padding modes: 'last', 'zero'
    """
```
- 生成 action chunk 轨迹（`trajectory_length=50`），但这是 **action space 中的轨迹**（EEF 位姿的 delta），不是图像坐标系的 2D 投影
- 仅用于 FM 分支的 action 输入，不做文本预测

**DM0ActionConfig (`dm0_exp.py:244-264`)：**
```python
class DM0ActionConfig(ActionConfig):
    trajectory_length: int = field(default=50)
    # ... pipeline 中包含 AddTrajectory(trajectory_length=50, flatten=False)
```

**DM0Prog (`dm0_prog_arch.py:93-95`)：**
```python
self.progress_in_proj = nn.Linear(1, action_hidden)
self.progress_out_proj = nn.Linear(action_hidden, 1)
```
- Progress 是一个标量（0-1 进度百分比），在 suffix 中作为辅助 token 参与推理（仅推理阶段）
- 这是 DM0 代码中**唯一的辅助预测通道**，架构设计值得参考

### 4.3 跨库参考实现

**代码库内部：**
- `ActionNormAnd2String._action2bin()` (`action.py:387-391`)：量化逻辑可用于 2D 坐标量化
- `AddTrajectory` (`action.py:156-226`)：轨迹拼接和 padding 逻辑可复用

**外部论文（dm0_larCmpl.md）：**
- **UniVLA (ICLR 2026)**：全模态离散化（视觉 VQ + 动作 DCT），证明了轨迹数据可以走离散自回归
- **FAST tokenizer**：DCT 压缩连续信号为离散 token，压缩率 ~7x，**可直接应用于 2D 轨迹压缩**
- **NaVILA (`navila/loss.py:11-70`)**：`soft_cross_entropy` 对相邻 bin 使用高斯软标签，可改善坐标量化的精度
- **OmniSAT (ICLR 2026)**：B-Spline 编码统一不同轨迹长度 → 固定长度 token，压缩率 6.8-8.1x

### 4.4 设计方案

#### 2D 投影坐标获取

这是一个**数据标注问题**，不是模型问题。获取主视角下 EEF 2D 坐标有 3 种方式：

1. **相机内参投影**：已知 EEF 3D 位姿 + 相机内参 → 2D 投影
   ```python
   def project_eef_to_2d(eef_pos_3d, camera_intrinsics, camera_extrinsics):
       """3D EEF 位置 → 主视角 2D 像素坐标."""
       P = camera_intrinsics @ camera_extrinsics
       uv = P @ np.append(eef_pos_3d, 1.0)
       return uv[:2] / uv[2]
   ```
2. **仿真直接获取**：LIBERO / RoboTwin 可直接获取 EEF 在渲染图像中的 2D 坐标
3. **视觉检测**：用手部/夹爪检测模型在视频帧中定位 EEF

#### 格式选择

**方案 A（简单 bin 量化）— 推荐用于初期验证：**

与 bbox 格式一致，使用 [0, 1000] 归一化：

```
<traj>(312,456) (318,420) (325,389) (340,365) (412,234)</traj>
```

每个轨迹点 2 个坐标 × 3-4 token/坐标 = 6-8 token/点。
10 个关键帧 × 7 token ≈ 70 token（可接受）。

**方案 B（FAST DCT 压缩）— 推荐用于长轨迹：**

```python
from transformers import AutoProcessor
fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast")
# 将 2D 轨迹视为 2 维 "action"，用 FAST 压缩
traj_tokens = fast_tokenizer.encode(trajectory_2d)  # ~10-20 tokens vs 原始 70+ tokens
```

#### 新增 Transform

```python
# transform/trajectory.py — 新增或扩展

class AddEEFTrace2D:
    """从 action 数据中提取 2D EEF 轨迹并格式化为文本."""
    
    def __init__(self, num_keyframes=10, coord_range=1000, img_size=728):
        self.num_keyframes = num_keyframes
        self.coord_range = coord_range
        self.img_size = img_size
    
    def __call__(self, episode_data_dict, **kwargs):
        if 'eef_trace_2d' not in episode_data_dict:
            return episode_data_dict
        
        trace = episode_data_dict['eef_trace_2d']  # shape: (T, 2)
        
        # 均匀采样关键帧
        if len(trace) > self.num_keyframes:
            indices = np.linspace(0, len(trace)-1, self.num_keyframes, dtype=int)
            trace = trace[indices]
        
        # 归一化到 [0, coord_range]
        normalized = np.round(trace / self.img_size * self.coord_range).astype(int)
        normalized = np.clip(normalized, 0, self.coord_range)
        
        # 格式化为文本
        points = [f"({x},{y})" for x, y in normalized]
        episode_data_dict['traj_text'] = ' '.join(points)
        return episode_data_dict
```

#### 工作量估计

| 组件 | 改动量 | 复杂度 |
|------|-------|--------|
| 2D 投影标注管线 | 中（需相机内参或检测模型） | ★★★ |
| `AddEEFTrace2D` transform | ~40 行 | ★★ |
| 对话模板扩展 | ~5 行 | ★ |
| Forward | 0（走统一 text_loss） | — |
| **总计** | ~45 行代码 + 标注管线 | ★★★ |

---

## 5. 任务四：Discrete Action Prediction（离散动作预测）

### 5.1 论文要求

DM0 论文 §2.9 描述：
- 将连续 EEF 动作量化为 **255-bin 离散 token**
- 通过 **L_AR (自回归 CE loss)** 监督
- 与 **L_FM (Flow Matching MSE)** 同源监督同一段动作序列
- 联合训练公式：`L_total = λ·L_AR + L_FM, λ=1`

论文样例中的离散 action token 格式：`[250, 0, 250, 110, 250, 250, 0]`

### 5.2 代码现状（基础设施最完整的任务）

这是 4 个辅助任务中**基础设施最完整**的一个。所有核心组件都已存在，只是没有被连接。

#### 5.2.1 量化管线：ActionNormAnd2String (`action.py:281-398`)

**完整的 normalize → 量化 → 字符串化管线：**

```python
# action.py:387-391 — 连续 → 离散
def _action2bin(self, action, vocab_size) -> np.array:
    action = np.round((action + 1) / 2 * (vocab_size - 1))  # [-1,1] → [0, 254]
    action = np.clip(action, 0, vocab_size - 1)
    return action

# action.py:393-398 — 离散 → 字符串
def _bin2string(self, action, string_format) -> list[str]:
    action_str = [''.join([string_format.format(value=int(_))
                          for _ in action[i]]) for i in range(len(action))]
    return action_str
```

- 默认 `vocab_size=255`，`string_format=' {value}'`
- 输出格式示例：`" 128 255 64 0 127 200 1"`
- 每个 action 维度独立量化（与 OpenVLA 的方法完全一致）

#### 5.2.2 词表扩展：add_special_tokens (`base_exp.py:355-367`)

```python
# base_exp.py:355-367
def add_special_tokens(self, special_token_format, vocab_size, tokenizer, model):
    if not self.use_special_tokens:  # DM0 默认 False
        return tokenizer
    special_tokens = [special_token_format.format(i) for i in range(vocab_size)]
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer
```

DM0 默认 `use_special_tokens=False`，但**接口已经完备**，只需设为 True 即可。

#### 5.2.3 lm_head (`dm0_arch.py:140-142`)

```python
# dm0_arch.py:139-142
# Add lm_head for compatibility with parent class tie_weights
self.lm_head = nn.Linear(
    config.llm_config.hidden_size, config.llm_config.vocab_size, bias=False
)
```

已创建，可直接用于 `text_logits = self.lm_head(prefix_out)`。

#### 5.2.4 prefix_out 可用点 (`dm0_arch.py:486`)

```python
# dm0_arch.py:486
(prefix_out, suffix_out), _ = self._merged_attention_forward(...)
# prefix_out 在此可用，但当前被丢弃
```

cumsum 注意力掩码（`dm0_utils.py:37-38`）保证 prefix 不能 attend 到 suffix，所以 `prefix_out` 是**纯 VLM 编码**，不受 action expert 信息污染。

#### 5.2.5 OFT-Discrete 参考 (`oft_discrete_arch.py:168-191`)

OFT-Discrete 提供了 256-bin 并行分类的完整实现：

```python
# oft_discrete_arch.py:168 — lm_head 做并行分类
predicted_actions = self.lm_head(action_hidden_states)
# shape: (batch_size, chunk_size * action_dim, vocab_size)

# oft_discrete_arch.py:187-191 — cross-entropy loss
predicted_actions_flat = predicted_actions.reshape(-1, predicted_actions.size(-1))
discrete_action_labels_flat = discrete_action_labels.reshape(-1)
loss = nn.functional.cross_entropy(
    predicted_actions_flat, discrete_action_labels_flat, reduction="mean"
)
```

#### 5.2.6 HybridPi05 双 Loss 参考 (`hybrid_pi05_arch.py:455-512`)

HybridPi05 的 forward 是 DM0 实现双 Loss 的**最佳参考**：

```python
# hybrid_pi05_arch.py:455 — VLM 端文本预测
text_logits = self.lm_head(prefix_out)

# hybrid_pi05_arch.py:462-468 — per-token CE + IGNORE_INDEX 掩码
token_loss = F.cross_entropy(pred_tokens.transpose(1, 2), target_tokens, reduction="none")
token_mask = torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)
sample_loss = (token_loss * token_mask).sum(dim=-1) / torch.clamp(token_mask.sum(dim=-1), min=1.0)

# hybrid_pi05_arch.py:470-479 — per-sample has_text 掩码
has_text_mask = has_text.reshape(-1).to(sample_loss.device).float()
text_loss = (sample_loss * has_text_mask).sum() / (has_text_mask.sum() + 1e-6)

# hybrid_pi05_arch.py:506-512 — 双 Loss 组合
loss = text_loss + action_loss
```

### 5.3 外部论文参考（dm0_larCmpl.md）

| 论文 | 离散化方法 | 与 DM0 的关系 | 代码 |
|------|-----------|--------------|------|
| **OpenVLA** | 256-bin per-dim | 方法完全一致 | [github](https://github.com/openvla/openvla) |
| **π₀-FAST** | DCT + BPE（7x 压缩） | 可替代 256-bin 提升效率 | [github](https://github.com/Physical-Intelligence/openpi) |
| **VQ-VLA** (ICCV 2025) | Residual VQ-VAE | 学习型 tokenizer，长 horizon 更好 | [github](https://github.com/xiaoxiao0406/VQ-VLA) |
| **HybridVLA** | 离散 bin + 双 Loss | ★ 最接近 DM0 的 L_total 设计 | [github](https://github.com/PKU-HMI-Lab/Hybrid-VLA) |
| **UniVLA** (ICLR 2026) | VQ + DCT 全模态 | 验证了 text-as-everything 可行性 | [github](https://github.com/baaivision/UniVLA) |
| **OmniSAT** (ICLR 2026) | B-Spline + RVQ（8.1x 压缩） | 最高压缩率 | 待发布 |

**关键启示：**
- **HybridVLA** 用 `AR_DIFF_LOSS=true` 配置双 Loss，证明 AR + Diffusion 协作训练有效（模拟/真实分别超 SOTA 14%/19%）
- **OpenVLA-OFT** 的对比实验表明：并行解码 + 连续 action 比自回归 + 离散 action 效果更好（OpenVLA 的 256-bin 自回归被 OFT 的 L1 并行解码超过 20%+）。这意味着 DM0 的离散 action token 更多是作为**辅助监督**（强化 VLM 空间理解），而非取代 FM 作为主推理通道
- **FAST tokenizer** 可用 3 行代码接入，压缩率 7x，适合长 chunk（DM0 的 chunk_size=50）

### 5.4 设计方案

#### 方案 A：最小改动（~15 行）— 推荐用于第一步验证

**核心思路**：保留 L_FM 不变，在 forward 末尾加 `self.lm_head(prefix_out)` → CE loss。

**改动文件 1: `dm0_arch.py` forward 方法（line 502 之后）**

```python
# ===== 现有代码（不改） =====
action_loss = F.mse_loss(v_t, u_t, reduction="mean")

# ===== 新增代码（+12 行）=====
text_loss = None
if labels is not None:
    with torch.amp.autocast("cuda", dtype=torch.float32):
        text_logits = self.lm_head(prefix_out)
        # labels 形状: (batch_size, prefix_seq_len)
        # text_logits 形状: (batch_size, prefix_seq_len, vocab_size)
        shift_logits = text_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        text_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )

loss = action_loss + (text_loss if text_loss is not None else 0.0)

outputs = CausalLMOutputDexbotic(
    loss=loss,
    logits=v_t,
    past_key_values=past_key_values,
    hidden_states=None,
    attentions=None,
    text_loss=text_loss,        # ← 新增
    action_loss=action_loss,    # ← 新增
)
```

**改动文件 2: `process.py` DM0Tokenization（line 407-414）**

```python
# 修改: 当 assistant turn 有内容时不弹空
conversations = list(conversations)
if (
    conversations
    and conversations[-1].get("from") == "gpt"
    and not conversations[-1].get("value")
):
    # 仅当显式配置为保留空 assistant 时才弹出
    if not kwargs.get("keep_empty_assistant", False):
        conversations.pop()
```

**改动文件 3: 数据管线**

需要确保 assistant turn 中包含离散 action token 字符串。DM0 当前使用 `ActionNorm`（不字符串化），需改为 `ActionNormAnd2String` 或同时输出两者：

```python
# dm0_exp.py — DM0ActionConfig.build_action_process_func()
# 在 ActionNorm 之后追加 ActionNormAnd2String（仅生成 answer 字段）
ActionNormAnd2String(
    statistic_mapping=statistic_mapping,
    vocab_size=255,
    string_format=' {value}',
    add_answer=True,
)
```

**注意事项：**
- 当前 assistant turn 为空 → `labels` 全为 IGNORE_INDEX → text_loss = 0（无梯度）
- 必须在 assistant turn 中放入离散 action token 字符串，text_loss 才会生效
- `has_text` 掩码在方案 A 中可暂不使用（全 batch 都有 text）

#### 方案 B：完整离散 Token（~150 行）— 推荐用于正式训练

**核心改动：**

1. **启用词表扩展**：`DM0TokenizerConfig.use_special_tokens = True`
2. **数据管线双输出**：同时保留 `action`（连续给 FM）和 `answer`（离散字符串给 L_AR）
3. **Forward 双 Loss**：仿 HybridPi05 的 per-sample `has_text`/`has_action` 掩码

```python
# dm0_arch.py — 方案 B forward 核心逻辑

# 文本 Loss（L_AR）
text_loss = None
if labels is not None:
    text_logits = self.lm_head(prefix_out)
    text_len = input_ids.shape[1] if input_ids is not None else labels.shape[1]
    pred_tokens = text_logits[:, -text_len:-1]
    target_tokens = labels[:, 1:]
    
    token_loss = F.cross_entropy(
        pred_tokens.transpose(1, 2), target_tokens, reduction="none"
    )
    token_mask = torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)
    sample_loss = (token_loss * token_mask).sum(dim=-1) / torch.clamp(
        token_mask.sum(dim=-1), min=1.0
    )
    
    if has_text is not None:
        has_text_mask = has_text.reshape(-1).float()
    else:
        has_text_mask = torch.ones(sample_loss.shape[0], device=sample_loss.device)
    
    text_loss = (sample_loss * has_text_mask).sum() / (has_text_mask.sum() + 1e-6)

# 动作 Loss（L_FM）
action_loss = F.mse_loss(v_t, u_t, reduction="mean")

# 组合
loss = (text_loss if text_loss is not None else 0.0) + action_loss
```

**DM0DataConfig 扩展：**

```python
# dm0_exp.py
data_keys: list[str] = field(
    default_factory=lambda: [
        "input_ids", "labels", "action",
        "image", "state", "image_masks",
        "has_text", "has_action",  # ← 新增
    ]
)
```

#### 方案对比

| 维度 | 方案 A | 方案 B |
|------|--------|--------|
| 改动行数 | ~15 行 | ~150 行 |
| 词表扩展 | 否 | 是 |
| has_text/has_action 掩码 | 否 | 是 |
| 混合 batch 支持 | 否 | 是（纯文本 + 纯 action + 混合） |
| 验证价值 | 快速验证 L_AR 是否有效 | 完整的双 Loss 训练 |
| 推荐场景 | 概念验证（1 天） | 正式训练（1 周） |

---

## 6. 四任务联合实现方案

### 6.1 联合对话格式

将 4 个辅助任务组织在 assistant turn 中的**层次化 CoT 格式**：

```
USER: <image> What action should the robot take to place the cup on the coaster?

ASSISTANT: <subtask>Reach towards the cup, grasp the handle, lift the cup, 
move towards the coaster, lower the cup, release.</subtask>
<bbox>[234, 156, 567, 423]</bbox>
<traj>(312,456) (318,420) (325,389) (340,365) (412,234)</traj>
<act> 128 64 200 110 250 250 0</act>
```

**关键设计决策：**
- 所有辅助内容放在 **同一个 assistant turn** 中，走统一的 CE loss
- 使用 XML 标记 `<subtask>`, `<bbox>`, `<traj>`, `<act>` 作为分隔符
- 标记本身也参与 loss（让 VLM 学会在正确时机切换任务层级）
- 层级顺序固定：subtask → bbox → traj → act（从高到低）

### 6.2 多字段 Loss 计算

**统一 CE Loss 方案（推荐）：**

所有 4 个任务的文本在 assistant turn 中拼接，走**同一个** `F.cross_entropy`。这正是论文 "text-as-everything" 的核心思想 — 不需要 4 个独立的 loss function，只需 1 个 L_AR。

```python
# 不需要分别计算 subtask_loss + bbox_loss + traj_loss + action_loss
# 统一走 L_AR = CE(logits, labels) on assistant turn
text_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
loss = text_loss + action_loss  # L_total = L_AR + L_FM
```

**可选的分层 Loss 监控：**

虽然训练只用统一 CE，但可以在日志中分别监控各层的 loss，利用 XML 标记位置切分：

```python
# 日志层面的分层监控（不影响梯度）
with torch.no_grad():
    subtask_mask = ... # 通过 <subtask> token 位置定位
    bbox_mask = ...    # 通过 <bbox> token 位置定位
    traj_mask = ...    # 通过 <traj> token 位置定位
    act_mask = ...     # 通过 <act> token 位置定位
    
    outputs.subtask_loss = (token_loss * subtask_mask).sum() / subtask_mask.sum()
    outputs.bbox_loss = (token_loss * bbox_mask).sum() / bbox_mask.sum()
    outputs.traj_loss = (token_loss * traj_mask).sum() / traj_mask.sum()
    outputs.act_loss = (token_loss * act_mask).sum() / act_mask.sum()
```

由于 Trainer 自动发现 `*_loss` 字段（`trainer.py:172`），这些监控 loss 会自动出现在 wandb 日志中。

### 6.3 DM0DataConfig 扩展

```python
# dm0_exp.py — 完整扩展
@dataclass
class DM0ScaffoldingDataConfig(DM0DataConfig):
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids", "labels", "action",
            "image", "state", "image_masks",
            "has_text", "has_action",
            # 以下为可选字段（存在则使用，不存在则跳过）
            # "subtask", "goal_bbox", "eef_trace_2d"
        ]
    )
```

不需要为 subtask/bbox/traj 添加独立的 data_keys — 它们已经被嵌入到对话文本中，走 `input_ids` 和 `labels` 路径。

### 6.4 Collator 扩展

当前 collator (`collator.py:49-57`) 的 `mapping_keys` 已包含 `has_text` 和 `has_action`，无需额外改动。

### 6.5 训练策略：渐进式引入

| 阶段 | 新增任务 | 验证目标 | 工作量 |
|------|---------|---------|--------|
| **Phase 0** | 仅 L_FM（现状） | baseline | 0 |
| **Phase 1** | + 任务 4（离散 action token） | 验证 L_AR 是否改善 L_FM 效果 | 1 天 |
| **Phase 2** | + 任务 1（subtask prediction） | 验证子任务 CoT 是否改善泛化 | 1 周 |
| **Phase 3** | + 任务 3（EEF trajectory） | 验证空间轨迹监督的效果 | 1 周 |
| **Phase 4** | + 任务 2（goal bbox） | 完整 scaffolding | 2 周 |

**推荐优先级：4 → 1 → 3 → 2**

理由：
- 任务 4 基础设施最完整，改动最小，验证价值最高
- 任务 1 不需要额外标注工具（VLM 自动生成），且论文中 subtask prediction 在 CoT 链条最前面
- 任务 3 需要相机内参或视觉检测，但数据量需求适中
- 任务 2 需要目标检测模型 + 人工修正，数据标注成本最高

### 6.6 工作量汇总

| 组件 | 行数 | 难度 |
|------|------|------|
| `dm0_arch.py` forward 改动 | ~20 行 | ★★ |
| `process.py` tokenization 改动 | ~10 行 | ★ |
| `dm0_exp.py` config 改动 | ~15 行 | ★ |
| `AddGoalBBox` transform（新文件） | ~30 行 | ★★ |
| `AddEEFTrace2D` transform（新文件） | ~40 行 | ★★ |
| 对话模板扩展 | ~15 行 | ★ |
| 数据标注管线（subtask VLM 生成） | ~100 行 | ★★★ |
| 数据标注管线（bbox 检测 + 修正） | ~200 行 | ★★★★ |
| 数据标注管线（2D 投影） | ~50 行 | ★★★ |
| **代码总计** | ~130 行模型侧 + ~350 行数据侧 | — |

---

## 7. 外部论文方法对比与启示

### 7.1 按任务维度对比

#### 任务 1 — Subtask Prediction

| 论文 | 方法 | 效果 | 与 DM0 的可借鉴性 |
|------|------|------|------------------|
| **VLM2VLA** (ICLR 2026) | 将低层动作重标注为自然语言 | 保留 85%+ VQA 能力 | ★★★★★ 最接近 |
| **UniVLA** (ICLR 2026) | 全模态统一自回归 | LIBERO 95.5% | ★★★★ 验证了全自回归可行 |

**启示**：VLM2VLA 证明用自然语言（而非数值 token）表达动作/子任务可以**避免灾难性遗忘**。DM0 的 subtask prediction 应该用自然语言格式（"grasp the cup"），而非数值格式。

#### 任务 2 — Goal Bbox Prediction

| 论文 | 方法 | 效果 | 与 DM0 的可借鉴性 |
|------|------|------|------------------|
| **DM0 论文自身** | 驾驶数据 `bbox in [0,1000]` | — | ★★★★★ 格式已定义 |
| **OpenVLA** | grounding 数据归一化 | — | ★★★ 参考格式 |

**启示**：bbox 预测的瓶颈不在模型侧（VLM 天然支持文本格式的坐标输出），而在**数据标注侧**。需要构建 Grounding-DINO + 人工修正的标注管线。

#### 任务 3 — EEF Trajectory Prediction

| 论文 | 方法 | 效果 | 与 DM0 的可借鉴性 |
|------|------|------|------------------|
| **FAST** | DCT 压缩轨迹信号 | 7x 压缩 | ★★★★ 可用于长轨迹 |
| **OmniSAT** (ICLR 2026) | B-Spline + RVQ | 8.1x 压缩 | ★★★ 更高压缩率 |
| **NaVILA** | soft_cross_entropy 软标签 | 坐标精度提升 | ★★★ 可选优化 |

**启示**：
- 短轨迹（≤10 关键帧）直接用 [0,1000] 文本格式即可
- 长轨迹（50 帧）应考虑 FAST DCT 压缩，否则 token 数太多（50×2×4 = 400 tokens）

#### 任务 4 — Discrete Action Prediction

| 论文 | 方法 | 效果 | 与 DM0 的可借鉴性 |
|------|------|------|------------------|
| **HybridVLA** | 双 Loss (AR + Diffusion) | +14%/+19% | ★★★★★ 架构最接近 |
| **OpenVLA** | 256-bin + CE | baseline | ★★★★ 方法一致 |
| **π₀-FAST** | DCT + BPE | 训练 5x 快 | ★★★★ 更高效 tokenizer |
| **VQ-VLA** (ICCV 2025) | Residual VQ-VAE | 长 horizon 更好 | ★★★ 学习型 tokenizer |
| **OpenVLA-OFT** | 并行解码 + 连续 L1 | +20% vs 自回归 | ★★ 警示：自回归可能不是最优解码 |

**启示**：
1. **HybridVLA 验证了双 Loss 协作**：AR loss 和 diffusion loss 互相增强（论文原话 "reinforce each other"）
2. **OpenVLA-OFT 的警示**：离散 action token 的自回归解码效率和精度可能不如并行解码 + 连续回归。DM0 的设计（离散 token 做辅助监督，FM 做主推理）是正确的策略
3. **Action tokenizer 选型**：
   - 初期验证用 255-bin（与代码库 `ActionNormAnd2String` 完全兼容）
   - 正式训练考虑 FAST（HuggingFace 三行接入，压缩 7x）
   - 前沿方案用 OmniSAT B-Spline+RVQ（压缩 8.1x，但需自行实现）

### 7.2 综合推荐

| 决策点 | 推荐方案 | 理由 |
|--------|---------|------|
| Action tokenizer | 初期 255-bin → 后续 FAST | 255-bin 与代码库兼容；FAST 更高效 |
| 双 Loss 权重 | λ=1（AR 与 FM 等权） | 论文设定；HybridVLA 也用等权 |
| 子任务格式 | 自然语言 | VLM2VLA 证明避免灾难性遗忘 |
| Bbox 格式 | [0, 1000] 整数 | 对齐论文驾驶数据格式 |
| 轨迹格式 | 短用文本，长用 FAST DCT | 平衡 token 效率和精度 |
| 推理时 L_AR 角色 | 辅助/CoT，不替代 FM | OpenVLA-OFT 警示：FM 推理更优 |

---

## 附录 A: 代码引用索引

| 文件 | 行号 | 内容 | 用途 |
|------|------|------|------|
| `dexbotic/model/dm0/dm0_arch.py` | 140-142 | `self.lm_head = nn.Linear(...)` | VLM 输出层，可用于 text_logits |
| `dexbotic/model/dm0/dm0_arch.py` | 406-511 | `forward()` 方法 | 训练主入口 |
| `dexbotic/model/dm0/dm0_arch.py` | 413 | `labels: Optional[torch.LongTensor] = None` | 接收但未使用 |
| `dexbotic/model/dm0/dm0_arch.py` | 486 | `(prefix_out, suffix_out), _ = ...` | prefix_out 可用点 |
| `dexbotic/model/dm0/dm0_arch.py` | 500-502 | `action_loss = F.mse_loss(...)` / `loss = action_loss` | 当前只有 L_FM |
| `dexbotic/model/dm0/dm0_prog_arch.py` | 93-95 | `progress_in_proj`, `progress_out_proj` | 辅助预测通道参考 |
| `dexbotic/model/dm0/dm0_prog_arch.py` | 360-416 | `get_suffix_hidden_states()` | progress 嵌入逻辑 |
| `dexbotic/model/dm0/dm0_prog_arch.py` | 570-574 | `end_progress = self.model.progress_out_proj(...)` | 推理阶段 progress 预测 |
| `dexbotic/model/dm0/dm0_utils.py` | 37-38 | `cumsum` 注意力掩码 | 保证 prefix↛suffix |
| `dexbotic/model/pi05/hybrid_pi05_arch.py` | 455-512 | `text_logits = self.lm_head(prefix_out)` ... `loss = text_loss + action_loss` | ★主参考：双 Loss |
| `dexbotic/model/cogact/hybrid_cogact_arch.py` | 126-188 | 双 Loss (CE + diffusion) | 另一种双 Loss |
| `dexbotic/model/oft/oft_discrete_arch.py` | 168, 187-191 | 256-bin 并行分类 + CE | 离散 action 参考 |
| `dexbotic/model/navila/loss.py` | 11-70 | `soft_cross_entropy` | 软标签 loss |
| `dexbotic/model/dexbotic_arch.py` | 27-34 | `CausalLMOutputDexbotic` | `text_loss`, `action_loss` 字段 |
| `dexbotic/data/dataset/transform/action.py` | 156-226 | `AddTrajectory` | 轨迹拼接 |
| `dexbotic/data/dataset/transform/action.py` | 281-398 | `ActionNormAnd2String` | 255-bin 量化管线 |
| `dexbotic/data/dataset/transform/action.py` | 387-391 | `_action2bin()` | 连续→离散 |
| `dexbotic/tokenization/process.py` | 368-483 | `DM0Tokenization` | 当前 tokenization |
| `dexbotic/tokenization/process.py` | 404 | `ar_mask = [1] * len(tokens)` | ar_mask 已预留 |
| `dexbotic/tokenization/process.py` | 407-414 | empty assistant turn pop | 需条件禁用 |
| `dexbotic/data/collator.py` | 49-57 | `mapping_keys` | `has_text`/`has_action` 已就绪 |
| `dexbotic/exp/trainer.py` | 170-186 | `compute_loss` + `log` | `*_loss` 自动发现 |
| `dexbotic/exp/base_exp.py` | 355-367 | `add_special_tokens` | 词表扩展接口 |
| `dexbotic/exp/dm0_exp.py` | 244-264 | `DM0ActionConfig` | action pipeline |
| `dexbotic/exp/dm0_exp.py` | 268-312 | `DM0DataConfig` | data keys |
| `dexbotic/tokenization/conversation.py` | 197-270 | 对话模板 | 5 种模板 |

## 附录 B: JSONL 数据格式扩展示例

### 当前 DM0 JSONL 格式（无辅助任务）

```json
{
    "images_1": {"type": "video", "url": "episode_001.mp4", "frame_idx": 42},
    "images_2": {"type": "video", "url": "episode_001_wrist.mp4", "frame_idx": 42},
    "images_3": {"type": "image", "url": "static_cam.jpg"},
    "state": [0.12, -0.34, 0.56, 0.78, -0.12, 0.34, 1.0],
    "prompt": "Place the cup on the coaster",
    "is_robot": true,
    "answer": ""
}
```

### 扩展后 JSONL 格式（完整 4 层级 scaffolding）

```json
{
    "images_1": {"type": "video", "url": "episode_001.mp4", "frame_idx": 42},
    "images_2": {"type": "video", "url": "episode_001_wrist.mp4", "frame_idx": 42},
    "images_3": {"type": "image", "url": "static_cam.jpg"},
    "state": [0.12, -0.34, 0.56, 0.78, -0.12, 0.34, 1.0],
    "prompt": "Place the cup on the coaster",
    "is_robot": true,
    
    "subtask": "Reach towards the cup, grasp the handle, lift the cup, move to the coaster, lower and release.",
    "goal_bbox": [[234, 156, 567, 423]],
    "goal_object": ["cup"],
    "eef_trace_2d": [[312, 456], [318, 420], [325, 389], [340, 365], [412, 234]],
    
    "answer": "<subtask>Reach towards the cup, grasp the handle, lift the cup, move to the coaster, lower and release.</subtask> <bbox>[234, 156, 567, 423]</bbox> <traj>(312,456) (318,420) (325,389) (340,365) (412,234)</traj> <act> 128 64 200 110 250 250 0</act>"
}
```

字段说明：
- `subtask`: 自然语言子任务描述（人工或 VLM 自动标注）
- `goal_bbox`: 目标物体 bbox 列表（[0,1000] 归一化），支持多目标
- `goal_object`: 目标物体名称列表（与 bbox 一一对应）
- `eef_trace_2d`: EEF 2D 轨迹关键帧坐标（像素坐标）
- `answer`: 组装后的 assistant turn 文本（由 transform 管线自动生成）

### 渐进式数据格式

不要求所有样本都有全部 4 个字段。缺失字段 → 对应层级跳过：

```json
{"prompt": "...", "subtask": "...", "answer": "<subtask>...</subtask> <act>...</act>"}
```

```json
{"prompt": "...", "answer": "<act>...</act>"}
```

这与 `has_text` / `has_action` 掩码机制配合使用。

## 附录 C: 对话模板完整示例

### 场景 1：完整 4 层级 Scaffolding

```
SYSTEM: You are a helpful robot assistant.

USER: <image_1> <image_2> <image_3>
<robot_arx5> <eef_control>
What action should the robot take to place the cup on the coaster?

ASSISTANT: <subtask>Reach towards the cup, grasp the handle firmly, lift the cup 
above the table, move towards the coaster, carefully lower the cup onto the center 
of the coaster, and release the grip.</subtask>
<bbox>[234, 156, 567, 423]</bbox>
<traj>(312,456) (318,420) (325,389) (340,365) (355,340) (370,318) (385,296) 
(395,270) (405,248) (412,234)</traj>
<act> 128 64 200 110 250 250 0 128 60 195 108 248 250 0 130 55 190 105 245 250 0</act>
```

### 场景 2：仅 Subtask + Action（无 bbox/traj 标注）

```
ASSISTANT: <subtask>Reach and grasp the cup handle.</subtask>
<act> 128 64 200 110 250 250 0</act>
```

### 场景 3：仅 Action（Phase 1 最小改动）

```
ASSISTANT: <act> 128 64 200 110 250 250 0</act>
```

### 场景 4：纯 VL 对话（保持通识能力）

```
USER: <image_1> Describe what you see in this image.
ASSISTANT: I see a robotic arm positioned above a table with a cup and a coaster.
```

此场景下 `has_text=True, has_action=False`，仅计算 text_loss，不计算 action_loss。
