# data/trajectory_vocab.py

## 1. 文件职责

`data/trajectory_vocab.py` 负责从预处理后的逐场景 H5 中构建规划轨迹词表。该文件只读取 `labels/future_trajectory` 字段，把指定 H5 目录下所有场景的自车未来轨迹合并为全集，再使用 FTS 采样得到全局共享词表。

该文件不负责生成 H5、不读取图像、不执行训练时 Dataset 随机目标点抽样，也不在单个场景内部单独生成词表。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrajectoryVocabularyConfig` | dataclass | 轨迹词表采样配置，定义 H5 输入、词表数量、轨迹字段路径和 FTS 分块大小。 |
| `TrajectoryVocabulary` | dataclass | 保存物理空间词表、Symlog 词表、归一化词表、来源索引和 metadata。 |
| `build_trajectory_vocabulary` | function | 从指定 H5 集合构建轨迹词表。 |
| `load_future_trajectories` | function | 读取多个 H5 的 `labels/future_trajectory` 并记录来源索引。 |
| `sample_trajectory_vocabulary` | function | 使用 FTS 从全集轨迹中采样 `num_trajectories - 1` 条数据轨迹。 |
| `save_trajectory_vocabulary` | function | 将词表保存为 `.npz`。 |
| `symlog` | function | 计算 $Symlog(x)=Sign(x)\times Log(|x|+1)$。 |
| `main` | function | 命令行入口。 |

## 3. 关键类和函数

### `TrajectoryVocabularyConfig`

- 功能：集中保存离线词表构建参数。
- 输入：`h5_paths`、`output_path`、`num_trajectories`、`trajectory_dataset`、`future_points`、`trajectory_dim`、`distance_batch_size`、`symlog_scale_eps`。
- 输出：供 `build_trajectory_vocabulary` 使用的配置对象。
- Shape：默认读取 `[S, 6, 2]`，输出词表 `[256, 6, 2]`。
- 关键参数：
  - `num_trajectories=256`：词表总数，第 0 条保留为静止轨迹。
  - `trajectory_dataset="labels/future_trajectory"`：H5 中未来 ego 轨迹字段。
  - `distance_batch_size=65536`：FTS 距离更新分块大小，避免一次构造过大的临时差值数组。

### `build_trajectory_vocabulary`

- 功能：执行完整词表构建流程。
- 输入：`TrajectoryVocabularyConfig`。
- 输出：`TrajectoryVocabulary`。
- Shape：
  - `trajectory_vocab_m`: `[V, 6, 2]`。
  - `trajectory_vocab_symlog`: `[V, 6, 2]`。
  - `trajectory_vocab_normalized`: `[V, 6, 2]`。
- 关键规则：
  - `trajectory_vocab_m[0]` 强制为全零静止轨迹。
  - `trajectory_vocab_m[1:]` 来自跨 H5 全集 FTS 采样。
  - FTS 距离在 ego 坐标系 meter 物理空间计算。
  - 采样完成后再做 Symlog 和共享单一缩放系数归一化。

### `load_future_trajectories`

- 功能：只读取每个 H5 的自车未来轨迹字段。
- 输入：H5 路径列表、字段路径、期望未来点数和维度。
- 输出：`(trajectories, source_h5_indices, source_sample_indices)`。
- 无效值处理：单条轨迹中任意坐标包含 NaN 或 Inf 时跳过该样本，其他样本继续参与全局 FTS。
- Shape：
  - `trajectories`: `[N, 6, 2] float32`。
  - `source_h5_indices`: `[N] int32`。
  - `source_sample_indices`: `[N] int64`，保留源 H5 内原始样本索引。

### `sample_trajectory_vocabulary`

- 功能：使用 FTS 选择数据轨迹索引。
- 输入：`trajectories`、`num_trajectories`、`distance_batch_size`。
- 输出：长度为 `num_trajectories - 1` 的全集轨迹索引。
- Shape：
  - 输入 `trajectories`: `[N, K, D]`。
  - 输出 `selected_indices`: `[V-1]`。
- 关键逻辑：调用方已经保留第 0 条全零静止轨迹，因此 FTS 初始化中心也使用全零轨迹。每一步选择到当前中心集合的最小 MSE 距离最大的样本。

### `save_trajectory_vocabulary`

- 功能：保存 `.npz` 词表文件。
- 输出字段：

| 字段 | Shape / 类型 | 说明 |
| --- | --- | --- |
| `trajectory_vocab_m` | `[V, 6, 2] float32` | ego 坐标系米制轨迹词表。 |
| `trajectory_vocab_symlog` | `[V, 6, 2] float32` | Symlog 变换后的词表。 |
| `trajectory_vocab_normalized` | `[V, 6, 2] float32` | 共享缩放系数归一化到 `[-1, 1]` 附近的词表。 |
| `symlog_scale` | scalar float32 | 全部轨迹、全部点、全部维度共享的缩放系数。 |
| `selected_source_h5_indices` | `[V] int32` | 词表来源 H5 索引；第 0 条静止轨迹为 `-1`。 |
| `selected_source_sample_indices` | `[V] int64` | 词表来源样本索引；第 0 条静止轨迹为 `-1`。 |
| `source_h5_paths` | `[M] str` | 源 H5 路径列表。 |
| `metadata_json` | str | 采样参数和数据来源 metadata。 |

`metadata_json` 中与无效样本过滤相关的字段包括：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `source_sample_count` | int | 过滤前读取到的 H5 轨迹样本总数。 |
| `valid_source_sample_count` | int | 过滤 NaN/Inf 后参与 FTS 的有效样本数。 |
| `skipped_invalid_sample_count` | int | 因包含 NaN 或 Inf 被跳过的样本总数。 |
| `skipped_invalid_by_h5` | list | 每个含无效样本的 H5 路径、跳过数量和源样本索引。 |

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| H5 输入字段 `labels/future_trajectory` | `[S, 6, 2]` | 单场景未来 3 秒、2Hz、ego 坐标系米制轨迹。 |
| 合并轨迹全集 | `[N, 6, 2]` | 指定 H5 集合中全部场景轨迹，$N=\sum_i S_i$。 |
| 物理词表 | `[256, 6, 2]` | 第 0 条为全零静止轨迹，后 255 条由 FTS 采样。 |
| Symlog 词表 | `[256, 6, 2]` | 对物理词表逐值应用 Symlog。 |
| 归一化词表 | `[256, 6, 2]` | 使用单一 `symlog_scale` 缩放。 |

FTS 使用物理空间 MSE 距离：

$$
d(a,b)=\frac{1}{KD}\sum_{k=1}^{K}\sum_{j=1}^{D}(a_{k,j}-b_{k,j})^2
$$

Symlog 公式为：

$$
Symlog(x)=Sign(x)\times Log(|x|+1)
$$

共享缩放系数为：

$$
s=\max_{v,k,j}|Symlog(T_{v,k,j})|
$$

归一化词表为：

$$
\hat{T}_{v,k,j}=\frac{Symlog(T_{v,k,j})}{s}
$$

若 $s$ 小于数值阈值，则使用 $s=1$，避免全静止极端输入导致除零。

## 5. 关键实现逻辑

`_resolve_h5_paths` 支持单文件、目录和文件列表。目录模式只读取当前目录下的 `*.h5`，不会递归进入 `map_cache` 等预处理副产物目录。

`load_future_trajectories` 逐个打开 H5，只读取 `labels/future_trajectory`。读取时校验字段存在、shape 为 `[S, 6, 2]`。若某条轨迹中任意坐标为 NaN 或 Inf，则跳过该样本并保留其他样本；每条有效轨迹同时记录来源 H5 索引和源文件内原始样本索引，便于后续复核词表来源。

内部 `_load_future_trajectories_with_report` 会额外统计过滤前样本数、有效样本数、跳过样本总数和逐 H5 跳过索引。`build_trajectory_vocabulary` 将这些统计写入 `metadata_json`，用于定位脏数据来源。

`sample_trajectory_vocabulary` 不返回静止轨迹本身，而是返回数据轨迹索引。第 0 条词表轨迹由 `build_trajectory_vocabulary` 直接写入全零数组。FTS 初始化时把全零轨迹作为已选中心，然后维护每条数据轨迹到已选中心集合的最小 MSE 距离；每轮选取最小距离最大的样本，并用新中心更新全集距离。距离更新按 `distance_batch_size` 分块，避免为大数据集一次分配过大的 `[N, 12]` 差值临时数组。

`build_trajectory_vocabulary` 在物理词表完成后，先执行 Symlog，再计算全词表共享的单一 `symlog_scale`，最后得到归一化词表。这个顺序对应 `doc/Model.md` 中“FTS 聚类得到轨迹词表，再对词表做 Symlog 变换，并将所有轨迹、所有维度共享同一个缩放系数”的约束。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `h5_paths` | 无 | 必填；可以是单个 H5、H5 目录或 H5 文件列表。 |
| `output_path` | `None` | 可选 `.npz` 输出路径。 |
| `num_trajectories` | `256` | 词表数量；第 0 条固定为全零静止轨迹。 |
| `trajectory_dataset` | `labels/future_trajectory` | H5 内未来 ego 轨迹字段。 |
| `future_points` | `6` | 未来轨迹点数。 |
| `trajectory_dim` | `2` | 每个轨迹点维度，当前为 ego XY。 |
| `distance_batch_size` | `65536` | FTS 距离更新分块大小。 |
| `symlog_scale_eps` | `1e-8` | 共享缩放系数除零保护阈值。 |

CLI 参数与上述配置对应，常用命令如下：

```powershell
.\.venv\Scripts\python.exe -m data.trajectory_vocab --h5-dir data/preprocessed --output data/preprocessed/trajectory_vocab_256.npz
```

## 7. 依赖关系

- 上游：`data/b2d_preprocess.py` 生成的逐场景 H5。
- 下游：规划轨迹查询初始化、轨迹词表概率监督、残差回归监督。
- 第三方依赖：`numpy`、`h5py`。

## 8. 注意事项

- 第 0 条词表轨迹必须直接强制为全零静止轨迹，不能从数据集中选择“最接近静止”的样本替代。
- FTS 候选池必须跨场景，读取整个指定 H5 目录下的逐场景 H5，而不是每个 H5 单独采样。
- `labels/future_trajectory` 中含 NaN/Inf 的单条轨迹会被跳过；如果过滤后有效轨迹数不足 `num_trajectories - 1`，FTS 仍会报样本数不足。
- FTS 距离在物理空间 ego 坐标系 meter 下计算；Symlog 和归一化只在采样完成后执行。
- 输入样本数必须至少为 `num_trajectories - 1`；否则无法为除静止轨迹外的词表项提供唯一数据样本。
- 输出 `.npz` 属于数据产物，应放在 `data/preprocessed/` 或其他 Git 忽略的数据目录，不应提交到 Git。
- 修改未来轨迹点数、坐标系、Symlog 口径或词表数量时，必须同步更新 `doc/Model.md` 和本代码文档。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-05 | 1os3_Codex | AI 完成：将 NaN/Inf 轨迹从整文件报错改为逐样本跳过，并在 metadata 中记录过滤统计。 |
| 2026-06-04 | 1os3_Codex | AI 完成：新增跨场景 H5 未来轨迹 FTS 词表采样工具，强制第 0 条为全零静止轨迹。 |
