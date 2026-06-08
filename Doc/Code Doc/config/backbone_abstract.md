# config/backbone.toml 摘要

## 1. 文件基本功能

`config/backbone.toml` 保存统一序列 Transformer 主干配置，包括子配置路径、16 层主干、8 头注意力、仅视觉 Token 使用的 3D RoPE、模态独立 FFN 层索引、身份嵌入、自车运动嵌入和精度策略。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[modules]` | TOML table | 指向已有嵌入和解码配置。 |
| `[architecture]` | TOML table | 主干层数、序列长度和模态独立 FFN 层索引。 |
| `[attention]` | TOML table | RoPE head 数和 SDPA dropout。 |
| `[feed_forward]` | TOML table | FFN 第一层输出维度与激活。 |
| `[rope]` | TOML table | 视觉 3D RoPE 参数。 |
| `[identity]` | TOML table | Token 身份嵌入顺序。 |
| `[ego_motion]` | TOML table | 自车运动输入配置。 |
| `[precision]` | TOML table | 主干和注意力精度。 |

## 3. 输入输出 Shape 概览

| 配置组 | Shape / 语义 | 说明 |
| --- | --- | --- |
| `[architecture]` | `[B, 2614, 384]` | 最终统一序列；第 1-12 层输入不含 Goal Token 时为 `[B, 2612, 384]`。 |
| `[attention]` | 8 heads，前 6 heads 对视觉 Token 使用 RoPE | 非视觉 Token 不应用 RoPE。 |
| `[ego_motion]` | `[B, 3] -> [B, 384]` | 自车状态用于轨迹解码前加到轨迹 Token。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `load_backbone_config` | 读取本配置并校验路径、shape 和精度字段。 |
| `MonoDriveBackbone` | 使用本配置组装已有嵌入、主干和解码头。 |

## 5. 最小使用示例

本文件是配置文件，不单独运行。可通过 `visualization/backbone_feature_pca_viewer.py --config config/backbone.toml` 间接使用。

## 6. 维护注意事项

- `modal_ffn_layer_indices` 是 0-based 索引，当前 `[1, 3, 5, 7, 9]` 对应文档中按 1 开始计数的第 2、4、6、8、10 层；第 13-16 层使用单路 FFN。
- 第 1-12 层不输入目标点 Token，第 13 层输入前追加目标点 Token。
- `rope.theta` 为 `100.0`，且 RoPE 只作用于视觉 Token。
- 修改配置字段时，同步更新 `model/backbone.py`、本完整文档和 `doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 16 层、2614 序列长度和新增四层单路 FFN 摘要。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增统一主干配置摘要文档。 |
