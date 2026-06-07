# config/backbone.toml

## 1. 文件职责

`config/backbone.toml` 保存 MonoDrive 统一序列 Transformer 主干的可提交配置。它声明主干读取的子配置路径、统一序列结构、Transformer 层数、注意力头数、视觉 RoPE、模态独立 FFN、身份嵌入、自车运动嵌入和精度策略。

该文件不保存本机私有绝对路径、临时输出路径、模型权重或训练缓存。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[modules]` | TOML table | 指向视觉嵌入、目标点嵌入、轨迹词表和检测头配置。 |
| `[architecture]` | TOML table | 统一主干的序列长度、层数、隐藏维度和模态独立 FFN 层索引。 |
| `[attention]` | TOML table | 注意力中使用 RoPE 的头数和 SDPA dropout。 |
| `[feed_forward]` | TOML table | FFN 第一层输出维度和激活类型。 |
| `[rope]` | TOML table | 视觉 3D RoPE 的基频、轴通道和位置归一化范围。 |
| `[identity]` | TOML table | Token 身份嵌入顺序。 |
| `[ego_motion]` | TOML table | 自车运动状态输入维度和数值变换。 |
| `[precision]` | TOML table | 主干和 SDPA 输入精度。 |
| `[initialization]` | TOML table | 主干新增嵌入层初始化标准差。 |

## 3. 关键类和函数

本文件没有 Python 类或函数。它由 `model/backbone.py` 中的 `load_backbone_config` 读取，并解析为 `BackboneConfig`。

## 4. 输入输出与 Shape

| 配置组 | Shape / 语义 | 说明 |
| --- | --- | --- |
| `[modules]` | 项目内相对路径 | 子配置路径会被解析到项目目录内。 |
| `[architecture]` | `[B, 2662, 384]` | 统一序列长度和隐藏维度。 |
| `[attention]` | 8 heads，其中前 6 heads 对视觉 Token 应用 RoPE | 非视觉 Token 不应用 RoPE。 |
| `[rope]` | `[H, W, T]` 位置坐标，范围 `[-1, 1]` | 与视觉 token 顺序 `[T, H, W]` 对齐。 |
| `[ego_motion]` | `[B, 3] -> [B, 384]` | 自车运动状态先 Symlog，再线性投影并加到轨迹 Token。 |

## 5. 关键实现逻辑

主干配置集中记录会影响统一序列结构的参数。`hidden_dim` 必须与视觉嵌入、目标点嵌入、轨迹词表和检测头配置中的隐藏维度一致。`expected_sequence_length` 必须等于视觉 Token、寄存器 Token、检测 Token、轨迹 Token 和目标导航点 Token 数之和。

`modal_ffn_layer_indices` 使用 0-based 索引。当前 `[1, 3, 5, 7, 9]` 对应设计文档中按 1 开始计数的第 2、4、6、8、10 层。

视觉 RoPE 的 `theta` 为 `100.0`。实现只对视觉 Token 的前 `rope_head_count` 个注意力头应用 RoPE；寄存器、检测、轨迹和目标导航点 Token 不构造零坐标，也不执行 RoPE。

## 6. 配置项

| 配置项 | 默认值来源 | 说明 |
| --- | --- | --- |
| `modules.vision_config_path` | 本文件 | 视觉嵌入配置路径。 |
| `modules.target_point_config_path` | 本文件 | 目标点嵌入配置路径。 |
| `modules.trajectory_vocab_config_path` | 本文件 | 轨迹词表配置路径。 |
| `modules.detection_head_config_path` | 本文件 | 检测头配置路径。 |
| `architecture.hidden_dim` | 本文件 | 统一主干隐藏维度。 |
| `architecture.layer_count` | 本文件 | Transformer 层数。 |
| `architecture.attention_head_count` | 本文件 | 注意力头数。 |
| `architecture.register_token_count` | 本文件 | 寄存器 Token 数。 |
| `architecture.expected_sequence_length` | 本文件 | 统一序列长度校验。 |
| `architecture.token_order` | 本文件 | Token 拼接顺序。 |
| `architecture.modal_ffn_layer_indices` | 本文件 | 使用模态独立 FFN 的 0-based 层索引。 |
| `architecture.rms_norm_eps` | 本文件 | RMSNorm 数值稳定项。 |
| `attention.rope_head_count` | 本文件 | 前多少个头对视觉 Token 应用 RoPE。 |
| `attention.attention_dropout` | 本文件 | SDPA dropout 概率。 |
| `feed_forward.ffn_layer1_output_dim` | 本文件 | FFN 第一层输出维度，要求等于 $4D$，SwiGLU 后为 $2D$。 |
| `feed_forward.ffn_activation` | 本文件 | FFN 激活类型。 |
| `rope.theta` | 本文件 | 3D RoPE 基频。 |
| `rope.axis_dims` | 本文件 | `[H, W, T]` 三轴 rotary 通道数。 |
| `rope.visual_position_order` | 本文件 | 视觉 RoPE 位置维度顺序。 |
| `rope.position_min/max` | 本文件 | 视觉坐标归一化范围。 |
| `identity.token_type_order` | 本文件 | 身份嵌入表顺序。 |
| `ego_motion.input_dim` | 本文件 | 自车运动输入维度。 |
| `ego_motion.vector_transform` | 本文件 | 自车运动数值变换。 |
| `precision.backbone_dtype` | 本文件 | 主干线性层和 FFN autocast 精度。 |
| `precision.attention_dtype` | 本文件 | SDPA 输入精度。 |
| `initialization.*_std` | 本文件 | 主干新增嵌入层初始化标准差。 |

## 7. 依赖关系

- 读取端：`model/backbone.py`。
- 子配置：`config/vision_embedding.toml`、`config/target_point_embedding.toml`、`config/trajectory_vocab.toml`、`config/detection_head.toml`。
- 使用端：训练主干、`visualization/backbone_feature_pca_viewer.py`。

## 8. 注意事项

- 子配置路径必须是项目内相对路径。
- 修改序列长度、Token 数或隐藏维度时，必须同步检查所有子配置。
- 修改 RoPE 头数、轴通道或基频时，必须同步更新 `model/backbone.py` 文档和主干可视化文档。
- 可视化脚本会在运行时覆盖精度为 FP32，但不会修改本配置文件。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增统一序列 Transformer 主干配置文档。 |
