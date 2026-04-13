# 缺点
- 没考虑迁移后的loss曲线,lr曲线和训练效果要对齐原版,下一版要做到这种级别的对齐.   
- 没发现pdf文件,估计也有很多md没发现,下一版要手工指定一些或指定目录.

# 问题
- `这与普通 lerobot policy 常见的“直接监督 dataset.action chunk”有本质差异。` 怎么理解
- pi0.5生成的action是绝对值还是delta?  
- dm0有把数据集中的action转成delta action的代码吗?
- `DM0 的 norm_stats.json 是对“Pad 后、Trajectory 后、Delta 后”的目标空间做统计，而不是对 LeRobotDataset.action 原始字段做统计。`难道不是对训练输入数据的空间做统计吗? `设计 dm0_norm_stats.json sidecar`
- DM0对文本用的是什么tokenizer? 具体解释一下`DM0 当前的动作学习目标本质上是 Flow Matching MSE，不是语言建模损失`是什么意思?
- `state 并不直接作为模型条件输入参与前向，它主要用于 target 构造与推理后的绝对动作恢复。`,`state 用于 delta 构造和绝对恢复`
- `DM0 又比 PI05 多了一层 “anchor state -> absolute restore” 依赖`
- `状态相关接口经验可以参考 PI0`
- `DM0 仍有自己独立的 Qwen3 + 独立视觉塔 + Merged Attention + absolute restore`
- `构造 delta 目标`,
- `明确 image_masks 与空相机填充`,  `通过缺失图像 key / empty camera 支持`