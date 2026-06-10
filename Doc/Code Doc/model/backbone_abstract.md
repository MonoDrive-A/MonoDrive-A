# model/backbone.py 摘要

## 1. 文件基本功能

`model/backbone.py` 实现 MonoDrive 统一序列 Transformer 主干。它复用已有视觉嵌入、目标点嵌入、轨迹词表、检测查询和解码头，构造最终 2614 个 384 维 Token，执行 16 层 Pre-Norm Transformer，并输出检测和轨迹解码结果。第 1-12 层不输入目标点 Token；第 13 层输入前追加目标点 Token；检测头输入为第 12 层旁路累积 `Acc_{11}` cast 到 FP32。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `BackboneConfig` | dataclass | 主干配置对象。 |
| `BackboneTokenSlices` | NamedTuple | 统一序列分段。 |
| `MonoDriveBackboneOutput` | NamedTuple | 主干输出。 |
| `MonoDriveBackbone` | class | 完整统一主干。 |
| `load_backbone_config` | function | 读取 `config/backbone.toml`。 |
| `override_backbone_precision` | function | 覆盖主干和注意力精度。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 Shape | 输出 Shape |
| --- | --- | --- |
| `MonoDriveBackbone.forward` | `images: [B, 8, 3, 288, 512]`，`target_points: [B, 2]`，`ego_motion: [B, 3]` | `sequence_features: [B, 2614, 384]`，第 12 层检测旁路累积，第 16 层轨迹输出 |
| `load_backbone_config` | TOML 路径 | `BackboneConfig` |
| `override_backbone_precision` | `BackboneConfig` 和 dtype 名称 | 新的 `BackboneConfig` |

检测解码：`Acc_{11}: [B, 48, 384]`（骨干精度）cast 到 FP32 后送入 `DetectionHeadDecoder`。前 12 层每层先经逐层 `TokenRMSNorm`，再通过零初始化 $W_i$ 和 $E_i$ 做旁路残差与写回；`Query`（FP32）为累积种子。

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `BackboneConfig` | 通过 `load_backbone_config` 构造，默认值只以 `config/backbone.toml` 为准。 |
| `MonoDriveBackbone` | 输入图像值域应与视觉嵌入配置一致；目标点和自车运动使用 ego 坐标系米制物理量。 |
| `MonoDriveBackboneOutput` | 下游可使用检测和轨迹解码结果；可视化可读取 `layer_vision_features`。 |
| `override_backbone_precision` | 仅用于调试或可视化，dtype 支持 `float32` 和 `bfloat16`。 |

## 5. 最小使用示例

不在摘要中提供完整示例，因为实例化会加载 DINOv3 和轨迹词表。可使用 `visualization/backbone_feature_pca_viewer.py` 对 H5 样本做 FP32 诊断。

## 6. 维护注意事项

- RoPE 只作用于视觉 Token，基频从 `config/backbone.toml` 读取，当前为 `100.0`。
- 12 层检测残差 RMSNorm 与 12 层检测残差投影、12 层检测身份嵌入一一对应；残差投影和身份嵌入的初始权重必须为零，RMSNorm 权重默认为 1；训练中正常学习。
- 检测路径使用 per-layer 身份嵌入，不复用全局 `agent`/`map`。
- 第 1-12 层不能包含目标点 Token；第 13-16 层使用单路 FFN。
- 检测监督固定使用第 12 层旁路累积；最终层只直接监督轨迹词表概率和残差。
- 精度：仅 `DetectionQueryEmbedding` 与 `DetectionHeadDecoder` 为 FP32，骨干中间默认 BF16。
- FFN 结构必须保持为 $(D \rightarrow 4D)_{\mathrm{Layer1}} \rightarrow \mathrm{SwiGLU}(4D \rightarrow 2D) \rightarrow (2D \rightarrow D)_{\mathrm{Layer2}}$。
- 模态独立 FFN 层索引是 0-based，当前为 `[1, 3, 5, 7, 9]`。
- 修改 Token 数、shape、精度、RoPE 或解码口径时，同步更新完整文档、配置文档和 `doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-10 | 1os3_Cursor | AI 完成：同步检测旁路残差投影前逐层 TokenRMSNorm 摘要。 |
| 2026-06-10 | 1os3_Cursor | AI 完成：同步逐层旁路残差、per-layer 检测身份嵌入和 FP32/BF16 精度边界摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 16 层两阶段主干、Agent 16 / Map 32 检测查询和第 12 层检测监督摘要。 |
| 2026-06-07 | 1os3_Codex | AI 完成：记录检测查询加零初始化残差的解码口径。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增统一序列 Transformer 主干摘要文档。 |
