# model/backbone.py

## 1. 文件职责

`model/backbone.py` 实现 MonoDrive 统一序列 Transformer 主干。它复用已有视觉嵌入、目标点嵌入、轨迹词表嵌入、检测查询、检测解码头和轨迹解码头，把输入组织为统一 Token 序列，执行 16 层 Pre-Norm Transformer，并输出检测和轨迹解码结果。第 1-12 层输入不包含目标点 Token，第 13 层输入前追加目标点 Token；检测解码使用第 12 层检测 Token，轨迹词表概率和残差使用第 16 层轨迹 Token。

该文件不实现 DINOv3、目标点嵌入、轨迹词表嵌入或检测头内部逻辑，也不在实现文件内重复写入 `config/backbone.toml` 中已有的结构默认值。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `BackboneConfig` | dataclass | 主干配置对象。 |
| `BackboneTokenSlices` | NamedTuple | 统一序列中各 Token 分段。 |
| `MonoDriveBackboneOutput` | NamedTuple | 主干前向输出。 |
| `MonoDriveBackbone` | class | 统一序列 Transformer 主干。 |
| `load_backbone_config` | function | 读取 `config/backbone.toml`。 |
| `override_backbone_precision` | function | 只替换主干和注意力精度字段。 |

## 3. 关键类和函数

### `BackboneConfig`

- 功能：保存主干结构、RoPE、FFN、身份嵌入、自车运动和精度配置。
- 输入：由 `load_backbone_config` 从 TOML 构造。
- 输出：供 `MonoDriveBackbone` 和 Transformer block 使用。
- 关键参数：`hidden_dim`、`layer_count`、`attention_head_count`、`rope_head_count`、`rope_theta`、`modal_ffn_layer_indices`。

### `BackboneTokenSlices`

- 功能：记录统一序列中视觉、寄存器、检测、Agent、Map、轨迹和 Goal Token 的切片。
- Shape：最终总长度为 `2614`，第 1-12 层不含 Goal Token 时长度为 `2612`。

### `MonoDriveBackbone`

- 功能：组装嵌入层、统一序列、Transformer 主干和解码头。
- 输入：`images`、`target_points`、`ego_motion`。
- 输出：`MonoDriveBackboneOutput`。
- Shape：
  - `images`: `[B, 8, 3, 288, 512]`。
  - `target_points`: `[B, 2]`，ego 坐标系米制 `[x, y]`。
  - `ego_motion`: `[B, 3]`，`[V_x, V_y, W]`。
  - `sequence_features`: `[B, 2614, 384]`。
  - 检测解码输入：第 12 层 `initial_detection_queries + detection_residual_projection(detection_features)`，shape 为 `[B, 48, 384]`。

### `VisualRoPESelfAttention`

- 功能：执行全序列 SDPA，并只对视觉 Token 的前 `rope_head_count` 个注意力头应用 3D RoPE。
- Shape：
  - 输入：第 1-12 层为 `[B, 2612, 384]`，第 13-16 层为 `[B, 2614, 384]`。
  - 视觉位置：`[2304, 3]`，最后一维为 `[H, W, T]`。
  - 输出：shape 与输入一致。

### `BackboneTransformerBlock`

- 功能：Pre-Norm Transformer Block。
- 实现：`RMSNorm -> SDPA -> 残差 -> RMSNorm -> FFN -> 残差`。
- 特殊逻辑：配置指定层使用 `ModalIndependentFeedForward`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `vision_tokens` | `[B, 2304, 384]` | 来自 `BackboneVisionEmbedding`。 |
| `register_tokens` | `[B, 4, 384]` | 主干新增可学习寄存器 Token。 |
| `detection_tokens` | `[B, 48, 384]` | 来自 `DetectionQueryEmbedding`，内部顺序为 Agent 后 Map。 |
| `initial_detection_queries` | `[B, 48, 384]` | 未加 agent/map 身份嵌入的检测查询，作为检测解码基线。 |
| `detection_decoder_features` | `[B, 48, 384]` | 第 12 层初始检测查询加骨干检测 Token 的线性残差，用于检测头解码。 |
| `trajectory_tokens` | `[B, 256, 384]` | 来自 `TrajectoryVocabularyEmbedding`。 |
| `goal_tokens` | `[B, 2, 384]` | 来自 `TargetPointEmbedding`。 |
| `pre_goal_sequence_features` | `[B, 2612, 384]` | 第 1-12 层不含目标点 Token 的序列。 |
| `sequence_features` | `[B, 2614, 384]` | 第 16 层最终统一序列。 |
| `visual_positions` | `[2304, 3]` | 视觉 RoPE 坐标，按 `[H, W, T]`。 |
| `detection_output.agent_class_logits` | `[B, 16, 4]` | 3 个 Agent 前景类加 none。 |
| `detection_output.map_class_logits` | `[B, 32, 4]` | 3 个 Map 前景类加 none。 |
| `trajectory_output.logits` | `[B, 256]` | 轨迹词表 logit。 |
| `trajectory_output.residuals` | `[B, 256, 6, 2]` | Symlog 空间轨迹残差。 |
| `layer_vision_features` | 每项 `[B, 2304, 384]` | 可选，每层输出后的视觉 Token。 |

## 5. 关键实现逻辑

主干先调用视觉嵌入层得到视觉 Token，并调用检测查询和轨迹词表嵌入生成第 1-12 层输入序列。检测查询会保留一份未加身份嵌入的 `initial_detection_queries` 作为解码基线；进入统一序列的检测 Token 仍会按 Agent 和 Map 分段添加身份嵌入。第 1-12 层只拼接视觉、寄存器、检测和轨迹 Token；第 13 层输入前再调用目标点嵌入，追加 2 个 Goal Token。

视觉位置坐标由 `VisionEmbeddingOutput.latent_grid_shape` 构造。视觉 token 展平顺序为 `[T, H, W]`，但传给 3D RoPE 的坐标最后一维按 `[H, W, T]`。每个轴都归一化到 `[-1, 1]`，并以 0 为中心。

注意力使用 PyTorch `scaled_dot_product_attention`。RoPE 只作用于视觉 Token 的前 6 个注意力头 Q/K；非视觉 Token 不使用 RoPE，也不使用零坐标替代。后 2 个注意力头对所有 Token 都只做内容匹配。

FFN 结构严格为 $(D \rightarrow 4D)_{\mathrm{Layer1}} \rightarrow \mathrm{SwiGLU}(4D \rightarrow 2D) \rightarrow (2D \rightarrow D)_{\mathrm{Layer2}}$。模态独立 FFN 作用于配置给出的 0-based 层 `[1, 3, 5, 7, 9]`。这些层将视觉相关 Token（视觉、寄存器）和驾驶相关 Token（检测、轨迹）分别送入独立 FFN 分支。新增的第 13-16 层不在 `modal_ffn_layer_indices` 内，均使用单路 FFN。

第 12 层输出后，主干保存检测 Token，并在关闭 autocast 的 FP32 上下文中通过 `detection_residual_projection`，再与 `initial_detection_queries` 相加后送入 `DetectionHeadDecoder`。该残差投影层的权重和偏置均零初始化，因此初始化时检测头看到的输入严格等于原始检测查询；训练后骨干只通过该投影层学习查询残差。第 13-16 层不再产生检测监督。

第 16 层输出后，`ego_motion` 先做 Symlog，再通过 FP32 线性层编码为 `[B, 384]`，并加到每个轨迹 Token 上。最终层只通过轨迹词表概率和残差参与直接监督。检测解码和轨迹解码继续调用已有 FP32 解码头。

## 6. 配置项

| 配置项 | 默认值来源 | 说明 |
| --- | --- | --- |
| `modules.*_config_path` | `config/backbone.toml` | 子模块配置路径。 |
| `architecture.hidden_dim` | `config/backbone.toml` | 统一隐藏维度。 |
| `architecture.layer_count` | `config/backbone.toml` | Transformer 层数。 |
| `architecture.attention_head_count` | `config/backbone.toml` | 注意力头数。 |
| `architecture.modal_ffn_layer_indices` | `config/backbone.toml` | 模态独立 FFN 的 0-based 层索引。 |
| `attention.rope_head_count` | `config/backbone.toml` | 使用 RoPE 的视觉注意力头数。 |
| `feed_forward.ffn_layer1_output_dim` | `config/backbone.toml` | FFN 第一层输出维度，要求等于 $4D$，SwiGLU 后为 $2D$。 |
| `rope.theta` | `config/backbone.toml` | RoPE 基频。 |
| `rope.axis_dims` | `config/backbone.toml` | `[H, W, T]` 三轴 rotary 通道数。 |
| `ego_motion.*` | `config/backbone.toml` | 自车运动嵌入口径。 |
| `precision.*` | `config/backbone.toml` | 主干和注意力精度。 |

检测残差投影层的零初始化是结构性约束，不在 `config/backbone.toml` 中开放为可调项；改成非零初始化会破坏“初始化时检测解码输入等于初始检测查询”的约定。

## 7. 依赖关系

- 上游：`data/b2d_dataset.py` 提供 `images`、`target_point`、`ego_motion`。
- 子模块：`model/vision_embedding.py`、`model/target_point_embedding.py`、`model/trajectory_vocab/trajectory_vocab.py`、`model/detection_head.py`、`model/rope_3d.py`、`model/swiglu.py`。
- 下游：训练流程、推理流程和 `visualization/backbone_feature_pca_viewer.py`。

## 8. 注意事项

- 只有视觉 Token 应用 RoPE；不要给非视觉 Token 添加零坐标 RoPE。
- `modal_ffn_layer_indices` 是 0-based 索引。
- 目标点 Token 只能在第 13 层输入前追加；不要让前 12 层看到 Goal Token。
- 检测头和检测 loss 使用第 12 层检测 Token；不要改回最终层检测 Token。
- 新增或修改 Token 分段、shape、精度或 RoPE 逻辑时，必须同步更新本文件文档、摘要文档、配置文档和 `doc/Code Doc/Index.md`。
- `return_layer_features=True` 会保留每层视觉 Token，用于可视化，训练中默认关闭以减少内存占用。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：记录 16 层两阶段主干、Agent 16 / Map 32 检测查询、第 12 层检测监督和第 13 层 Goal Token 注入。 |
| 2026-06-07 | 1os3_Codex | AI 完成：检测解码改为初始检测查询加零初始化线性残差。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增统一序列 Transformer 主干文档。 |
