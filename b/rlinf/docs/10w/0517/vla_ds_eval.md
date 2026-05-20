# VLAWAM_mdl_opti_op47_1.md 的测试集


| 名称 | 类型 | 简介 |
|---|---|---|
| **AgiBot World / AgiBot World 2026** | 真机数据集 | 1M+ 轨迹、217 任务，2026 版加入真实商业 / 家庭 / 通用场景、触觉、LiDAR、IMU、全身关节与数字孪生。 |
| **OXE-AugE** | OXE 增广数据集 | 将 16 个 OXE 数据集用 cross-painting 扩展到 9 个本体、4.4M trajectories，用于未见 robot-gripper 组合泛化。 |
| **UniHand-2.0** | 灵巧手 / 多本体数据集 | Being-H0.5 使用的数据，含 35K 小时人手数据、400M samples、30 本体。 |
| **EgoScale** | 人类第一视角视频数据 | GR00T N1.7 使用的 20,854 小时 human ego video，用于 dexterity scaling。 |
| **DreamDojo ego data** | 人类第一视角视频数据 | DreamDojo 使用的 44K 小时第一视角数据。 |
| **EgoDex** | 人类视频数据 | Ψ₀ 使用的 829 小时人类视频数据。 |
| **Humanoid Everyday** | 人形机器人数据集 | Ψ₀ 使用的 31 小时人形机器人数据。 |
| **Galaxea Open-World Dataset** | 真机多任务数据集 | GalaxeaVLA / OpenGalaxea 相关的 500 小时多任务数据。 |
| **TaF-Dataset** | 触觉 + 力矩数据集 | TaF-VLA 使用的 10M+ 同步触觉与 6 轴力矩数据。 |
| **Nav-AdaCoT** | 导航数据集 | VLingNav 使用的 2.9M 导航数据。 |
| **ABot-N0** | 导航专家轨迹 / 推理数据 | 16.9M 专家轨迹 + 5.0M 推理样本 + 7,802 个 3D 场景，覆盖 Point-Goal / Object-Goal / Instruction-Following / POI-Goal / Person-Following。 |

# embd_VLA_WAM_sota_chat_0516.md 的测试集

| 名称 | 类型 | 简介 |
|---|---|---|
| **AgiBot World Colosseo** | 真机人形 / 双臂数据平台 | AgiBot World 早期底层平台 / 论文，文档中作为百万级双臂人形数据与力控遥操来源单独提到。 |
| **Ego4D / EgoExo4D / Something-Something v2 (SSv2)** | 人类视频数据集 | 用于 LAPA 式 latent action 预训练；提供大量无动作标签的人类第一视角 / 外视角视频。 |
| **DexMimicGen** | 仿真合成数据平台 | NVIDIA 相关的大规模合成数据来源，适合从 source demos 扩展灵巧手 / 操作任务长尾数据。 |
| **GR00T-Sim** | 仿真合成数据平台 | NVIDIA GR00T 生态中的仿真数据源，用于人形 / 双臂任务、安全场景和长尾任务生成。 |
| **LingBot 20k h 双臂数据** | 双臂真机数据集 | LingBot-VLA 提到的 20k 小时双臂真机数据；文档中用来说明真机数据 scaling 仍未饱和。 |
| **ST-Human** | 4D 时空理解数据集 | ST-VLA 使用的 300k episodes 数据集，用于 4D 时空理解和人类动作/具身表征。 |
| **Robo4D-200k** | 4D 标注机器人数据集 | Kinema4D 使用的 20 万 episodes、4D 标注数据，含 URDF / 关节 / 末端 / 点云对齐信息。 |
| **ABot-PhysWorld 3M 物理标注片段** | 世界模型物理标注数据 | ABot-PhysWorld 使用的 3M 操作片段 + physics-aware annotations，用于减少物体穿模、反重力等物理违例。 |
| **LAION / ActivityNet / HowTo100M** | 通用图文 / 视频数据集 | 文档中作为维持 VLM 通识能力的互联网图文 / 视频数据来源；不是机器人专用 benchmark。 |