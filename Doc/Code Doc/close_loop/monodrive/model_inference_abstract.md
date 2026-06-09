# close_loop/monodrive/model_inference.py 摘要

## 1. 文件基本功能

从 `MonoDriveBackboneOutput` 解码 256 路轨迹词表概率、Tanh 残差与物理轨迹，并返回 winner 与 top-k 候选。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrajectoryDecodeResult` | dataclass | 解码结果（probs、winner、top-k 轨迹）。 |
| `decode_trajectories` | function | 单次 batch=1 完整解码。 |
| `decode_winner_trajectory` | function | 按索引解码单条物理轨迹。 |
| `inverse_symlog` | function | Symlog → 米制。 |

## 3. Shape 概览

| 字段 | Shape |
| --- | --- |
| `probs` | `(V,)`，`V=256` |
| `winner_traj_phys` | `(6, 2)` |
| `top_trajs_phys` | `(top_k, 6, 2)` |

## 4. 使用规范

- `logits` 在函数内做 softmax；winner 默认 `argmax(probs)`。
- 残差组合：`vocab_symlog + residual * symlog_scale`，再 `inverse_symlog`。
- 与 `visualization/backbone_feature_pca_viewer.py` 解码逻辑保持一致。

## 5. 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | 新增 MonoDrive 闭环轨迹解码模块。 |
