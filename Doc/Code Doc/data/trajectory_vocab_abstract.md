# data/trajectory_vocab.py 摘要

## 1. 文件基本功能

从预处理后的逐场景 H5 中读取 `labels/future_trajectory`，合并指定目录下所有场景的 ego 未来轨迹，并用 FTS 构建全局规划轨迹词表；第 0 条词表轨迹直接强制为全零静止轨迹，其余轨迹从全集样本中采样。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `TrajectoryVocabularyConfig` | dataclass | 定义 H5 输入、词表数量、轨迹字段路径和 FTS 分块大小。 |
| `TrajectoryVocabulary` | dataclass | 保存物理词表、Symlog 词表、归一化词表、来源索引和 metadata。 |
| `build_trajectory_vocabulary` | function | 执行跨场景 H5 轨迹读取、FTS 采样、Symlog 和共享缩放归一化。 |
| `load_future_trajectories` | function | 只读取多个 H5 的 `labels/future_trajectory` 字段，并跳过含 NaN/Inf 的单条轨迹。 |
| `sample_trajectory_vocabulary` | function | 使用全零静止中心初始化 FTS，并返回数据轨迹索引。 |
| `save_trajectory_vocabulary` | function | 将词表保存为 `.npz`。 |
| `symlog` | function | 计算 $Symlog(x)=Sign(x)\times \ln(|x|+1)$。 |

## 3. 输入输出 Shape 概览

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `labels/future_trajectory` | `[S, 6, 2]` | 单场景 H5 中的 ego 坐标系未来轨迹，单位 meter。 |
| 合并轨迹全集 | `[N, 6, 2]` | 指定 H5 集合的跨场景轨迹池。 |
| `trajectory_vocab_m` | `[256, 6, 2]` | 物理空间轨迹词表，第 0 条为全零静止轨迹。 |
| `trajectory_vocab_symlog` | `[256, 6, 2]` | Symlog 空间词表。 |
| `trajectory_vocab_normalized` | `[256, 6, 2]` | 使用单一 `symlog_scale` 归一化后的词表。 |
| `selected_source_h5_indices` | `[256]` | 词表来源 H5 索引，第 0 条为 `-1`。 |
| `selected_source_sample_indices` | `[256]` | 词表来源样本在源 H5 内的原始索引，第 0 条为 `-1`。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `TrajectoryVocabularyConfig(h5_paths)` | `h5_paths` 可以是 H5 文件、H5 目录或 H5 文件列表；目录模式只读取当前目录下的 `*.h5`。 |
| `build_trajectory_vocabulary(config)` | 第 0 条总是全零静止轨迹；不要传入已经分场景采样过的局部候选池。 |
| `load_future_trajectories(h5_paths)` | 单条轨迹中存在 NaN/Inf 时跳过该样本，返回的源样本索引仍指向 H5 原始索引。 |
| `sample_trajectory_vocabulary(trajectories)` | 输入必须是 `[N, K, D]`，且 `N >= num_trajectories - 1`。 |
| `save_trajectory_vocabulary(vocabulary, output_path)` | 默认不覆盖已有文件；输出 `.npz` 应放在 Git 忽略的数据目录。 |

## 5. 最小使用示例

```python
from data.trajectory_vocab import (
    TrajectoryVocabularyConfig,
    build_trajectory_vocabulary,
    save_trajectory_vocabulary,
)

config = TrajectoryVocabularyConfig(h5_paths="data/preprocessed")
vocabulary = build_trajectory_vocabulary(config)
save_trajectory_vocabulary(vocabulary, "data/preprocessed/trajectory_vocab_256.npz")
```

命令行示例：

```powershell
.\.venv\Scripts\python.exe -m data.trajectory_vocab --h5-dir data/preprocessed --output data/preprocessed/trajectory_vocab_256.npz
```

## 6. 维护注意事项

- 第 0 条词表轨迹必须强制为全零静止轨迹，不能改成从数据中查找最接近静止的样本。
- FTS 必须基于指定 H5 目录下所有逐场景文件的 `labels/future_trajectory` 全集。
- 含 NaN/Inf 的单条 future trajectory 会被跳过；过滤统计写入 `metadata_json`。
- FTS 距离在物理空间 ego 坐标系 meter 下计算；采样完成后才做 Symlog 和共享缩放归一化。
- 若修改未来点数、轨迹维度、词表数量或归一化口径，必须同步更新 `doc/Model.md` 和完整代码文档。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：将 Symlog 公式摘要从 `log` 修正为自然对数 `ln`。 |
| 2026-06-05 | 1os3_Codex | AI 完成：同步 NaN/Inf 轨迹逐样本跳过和 metadata 过滤统计说明。 |
| 2026-06-04 | 1os3_Codex | AI 完成：新增跨场景 H5 未来轨迹 FTS 词表采样工具摘要。 |
