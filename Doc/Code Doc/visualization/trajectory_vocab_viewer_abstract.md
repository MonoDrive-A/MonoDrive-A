# visualization/trajectory_vocab_viewer.py 摘要

## 1. 文件基本功能

读取轨迹词表 `.npz`，用 `trajectory_vocab_normalized` 和 `symlog_scale` 反求物理轨迹，与 `trajectory_vocab_m` 计算逐轨迹和全局 MSE，并导出蓝色原始物理轨迹、红色反求轨迹的逐条 PNG 叠图；也可额外导出所有轨迹叠加在同一 BEV 图中的全局叠图。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `TrajectoryVocabularyData` | dataclass | 封装物理词表、归一化词表、反求词表、MSE 和 metadata。 |
| `TrajectoryVocabularyViewConfig` | dataclass | 配置 BEV 范围、面板大小、列数和 MSE 阈值。 |
| `load_trajectory_vocabulary_npz` | function | 读取 `.npz`，反求物理轨迹并校验 MSE。 |
| `normalized_to_physical_trajectories` | function | 执行 `normalized * symlog_scale` 和 Symlog 反变换。 |
| `inverse_symlog` | function | 计算 Symlog 反变换。 |
| `render_trajectory_vocabulary` | function | 导出轨迹词表 PNG 叠图。 |
| `render_trajectory_vocabulary_overlay` | function | 导出所有词表轨迹叠加到同一 BEV 图的 PNG。 |

## 3. 输入输出 Shape 概览

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `trajectory_vocab_m` | `[V, 6, 2]` | `.npz` 中保存的原始物理轨迹。 |
| `trajectory_vocab_normalized` | `[V, 6, 2]` | `.npz` 中保存的归一化轨迹。 |
| `symlog_scale` | scalar | 全词表共享缩放系数。 |
| `reconstructed_trajectories` | `[V, 6, 2]` | 从归一化轨迹反求的物理轨迹。 |
| `per_trajectory_mse` | `[V]` | 每条词表轨迹的 MSE。 |
| 输出逐条 PNG | 可变 | 按词表索引网格排列的 BEV 叠图。 |
| 输出全局叠图 PNG | 可变 | 全部词表轨迹叠加到同一 BEV 面板。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `load_trajectory_vocabulary_npz(npz_path)` | `.npz` 必须包含 `trajectory_vocab_m`、`trajectory_vocab_normalized` 和 `symlog_scale`。 |
| `normalized_to_physical_trajectories` | 输入必须是 `[V, K, 2]`，`symlog_scale` 必须为有限正数。 |
| `render_trajectory_vocabulary` | 默认最多绘制前 32 条轨迹；可传入 `indices` 指定词表索引。 |
| `render_trajectory_vocabulary_overlay` | 始终绘制全部物理轨迹；可选同时绘制反归一化轨迹。 |
| `TrajectoryVocabularyViewConfig` | `max_mse` 用于阻止不一致词表继续出图。 |

## 5. 最小使用示例

```powershell
.\.venv\Scripts\python.exe -m visualization.trajectory_vocab_viewer `
  --npz data/preprocessed/trajectory_vocab_256.npz `
  --output visualization/outputs/trajectory_vocab_256.png
```

同时导出所有轨迹叠图：

```powershell
.\.venv\Scripts\python.exe -m visualization.trajectory_vocab_viewer `
  --npz data/preprocessed/trajectory_vocab_256.npz `
  --output visualization/outputs/trajectory_vocab_256.png `
  --overlay-output visualization/outputs/trajectory_vocab_256_overlay.png
```

只绘制指定索引：

```powershell
.\.venv\Scripts\python.exe -m visualization.trajectory_vocab_viewer `
  --npz data/preprocessed/trajectory_vocab_256.npz `
  --output visualization/outputs/trajectory_vocab_selected.png `
  --indices 0,1,2,10,20
```

## 6. 维护注意事项

- `.npz` 需要同时保存归一化轨迹、原始物理轨迹和缩放系数。
- MSE 校验在物理空间执行，默认全局阈值为 `1e-8`。
- 所有轨迹叠图默认绘制全部物理轨迹，不受逐条图的索引筛选限制。
- 输出图是诊断产物，应写入 `visualization/outputs/` 等 Git 忽略目录。
- 若词表字段名、轨迹维度或 Symlog 公式改变，必须同步更新本文件和完整文档。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-04 | 1os3_Codex | AI 完成：新增所有轨迹叠加在同一 BEV 图的可视化输出摘要。 |
| 2026-06-04 | 1os3_Codex | AI 完成：新增轨迹词表归一化反求校验与 PNG 叠图工具摘要。 |
