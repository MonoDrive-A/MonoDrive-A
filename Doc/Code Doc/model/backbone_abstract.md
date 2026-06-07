# model/backbone.py 摘要

## 1. 文件基本功能

`model/backbone.py` 实现 MonoDrive 统一序列 Transformer 主干。它复用已有视觉嵌入、目标点嵌入、轨迹词表、检测查询和解码头，构造 2662 个 384 维 Token，执行 12 层 Pre-Norm Transformer，并输出检测和轨迹解码结果。

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
| `MonoDriveBackbone.forward` | `images: [B, 8, 3, 288, 512]`，`target_points: [B, 2]`，`ego_motion: [B, 3]` | `sequence_features: [B, 2662, 384]`，检测输出，轨迹输出 |
| `load_backbone_config` | TOML 路径 | `BackboneConfig` |
| `override_backbone_precision` | `BackboneConfig` 和 dtype 名称 | 新的 `BackboneConfig` |

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
- FFN 结构必须保持为 $(D \rightarrow 4D)_{\mathrm{Layer1}} \rightarrow \mathrm{SwiGLU}(4D \rightarrow 2D) \rightarrow (2D \rightarrow D)_{\mathrm{Layer2}}$。
- 模态独立 FFN 层索引是 0-based，当前为 `[1, 3, 5, 7, 9]`。
- 修改 Token 数、shape、精度、RoPE 或解码口径时，同步更新完整文档、配置文档和 `doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增统一序列 Transformer 主干摘要文档。 |
