# visualization/trajectory_vocab_viewer.py

## 1. 文件职责

`visualization/trajectory_vocab_viewer.py` 负责读取轨迹词表 `.npz`，从 `trajectory_vocab_normalized` 和 `symlog_scale` 反求物理空间轨迹，并与 `.npz` 中直接保存的 `trajectory_vocab_m` 计算 MSE，最后把两组轨迹叠加导出为 PNG 诊断图。

该文件不负责生成轨迹词表，不修改 `.npz`，不打开 GUI。默认逐条小面板图用蓝色绘制原始物理轨迹，用红色绘制由归一化轨迹反求的物理轨迹；可选所有轨迹叠图会把全部物理轨迹叠加到同一张 BEV 图中。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrajectoryVocabularyData` | dataclass | 封装物理词表、归一化词表、反求词表、MSE 和 metadata。 |
| `TrajectoryVocabularyViewConfig` | dataclass | 配置 BEV 绘制范围、面板大小、列数和 MSE 阈值。 |
| `load_trajectory_vocabulary_npz` | function | 读取 `.npz`，反求物理轨迹并校验 MSE。 |
| `normalized_to_physical_trajectories` | function | 用 `trajectory_vocab_normalized * symlog_scale` 反求物理轨迹。 |
| `inverse_symlog` | function | 计算 Symlog 反变换。 |
| `render_trajectory_vocabulary` | function | 把词表叠图导出为 PNG。 |
| `render_trajectory_vocabulary_overlay` | function | 把词表中的全部轨迹叠加到同一张 BEV 图。 |
| `main` | function | 命令行入口。 |

## 3. 关键类和函数

### `TrajectoryVocabularyData`

- 功能：保存词表可视化和校验所需数据。
- 输入：由 `load_trajectory_vocabulary_npz` 构造。
- 输出：供 `render_trajectory_vocabulary` 使用。
- Shape：
  - `physical_trajectories`: `[V, 6, 2]`。
  - `normalized_trajectories`: `[V, 6, 2]`。
  - `reconstructed_trajectories`: `[V, 6, 2]`。
  - `per_trajectory_mse`: `[V]`。

### `TrajectoryVocabularyViewConfig`

- 功能：定义 BEV 视野和 MSE 阈值。
- 坐标系：ego 坐标系，`x` 前向、`y` 左向，单位 meter。
- 关键参数：
  - `x_min/x_max/y_min/y_max`：BEV 坐标范围。
  - `panel_width/panel_height`：每个轨迹面板像素尺寸。
  - `columns`：网格列数。
  - `max_mse`：允许的全局 MSE 上限。

### `load_trajectory_vocabulary_npz`

- 功能：读取 `.npz` 并执行一致性校验。
- 输入：`.npz` 路径和 `max_mse`。
- 输出：`TrajectoryVocabularyData`。
- 必需字段：

| 字段 | Shape / 类型 | 说明 |
| --- | --- | --- |
| `trajectory_vocab_m` | `[V, 6, 2] float32` | 原始物理轨迹词表。 |
| `trajectory_vocab_normalized` | `[V, 6, 2] float32` | Symlog 后共享缩放归一化词表。 |
| `symlog_scale` | scalar float32 | 全词表共享缩放系数。 |

若 `.npz` 含有 `metadata_json`，函数会解析为字典并在图中显示采样算法、静止轨迹策略和坐标系。

### `normalized_to_physical_trajectories`

- 功能：执行归一化轨迹反变换。
- 输入：`normalized_trajectories` 和 `symlog_scale`。
- 输出：物理空间轨迹。
- Shape：输入输出均为 `[V, K, 2]`。

### `render_trajectory_vocabulary`

- 功能：导出 PNG 诊断图。
- 输入：`TrajectoryVocabularyData`、输出路径、可视化配置、可选词表索引列表。
- 输出：PNG 路径。
- 可视化：每个面板从 ego 原点连到 6 个未来点，蓝线为 `.npz` 中存储的物理轨迹，红线为反归一化轨迹；面板标题显示词表索引和该轨迹 MSE。

### `render_trajectory_vocabulary_overlay`

- 功能：把所有词表轨迹叠加到同一张 BEV 图。
- 输入：`TrajectoryVocabularyData`、输出路径、可视化配置、叠图尺寸、是否绘制反归一化轨迹。
- 输出：PNG 路径。
- 可视化：默认绘制所有 `trajectory_vocab_m`，第 0 条强制静止轨迹用黑色突出，其余轨迹用半透明蓝色；传入 `draw_reconstructed=True` 或 CLI `--overlay-reconstructed` 时，会同时用半透明红色绘制反归一化轨迹。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `trajectory_vocab_m` | `[V, 6, 2]` | 原始物理空间 ego 轨迹。 |
| `trajectory_vocab_normalized` | `[V, 6, 2]` | 归一化轨迹。 |
| `symlog_scale` | scalar | 共享缩放系数。 |
| `reconstructed_trajectories` | `[V, 6, 2]` | 由归一化轨迹反求得到的物理轨迹。 |
| `per_trajectory_mse` | `[V]` | 每条轨迹的物理空间 MSE。 |
| 输出逐条 PNG | `[H, W, 3]` | 网格叠图，尺寸由面板数量和配置决定。 |
| 输出全局叠图 PNG | `[H, W, 3]` | 所有词表轨迹叠加到一个 BEV 面板。 |

反变换流程为：

$$
Y_{v,k,j} = \hat{T}_{v,k,j} \times s
$$

$$
T'_{v,k,j}=Sign(Y_{v,k,j})\times (Exp(|Y_{v,k,j}|)-1)
$$

其中 $\hat{T}$ 为 `trajectory_vocab_normalized`，$s$ 为 `symlog_scale`，$T'$ 为反求物理轨迹。

MSE 计算为：

$$
MSE_v=\frac{1}{KD}\sum_{k=1}^{K}\sum_{j=1}^{D}(T'_{v,k,j}-T_{v,k,j})^2
$$

全局 MSE 为所有 `MSE_v` 的平均值。

## 5. 关键实现逻辑

`load_trajectory_vocabulary_npz` 先校验 `.npz` 是否包含 `trajectory_vocab_m`、`trajectory_vocab_normalized` 和 `symlog_scale`，并确认两个词表 shape 完全一致、最后一维为 2、数值均为有限值。随后使用 `normalized_to_physical_trajectories` 反求米制轨迹，并在物理空间计算逐轨迹 MSE 和全局 MSE。若全局 MSE 超过 `max_mse`，函数直接抛出 `ValueError`，避免继续使用不一致的词表。

`normalized_to_physical_trajectories` 先将归一化轨迹乘以共享缩放系数，得到 Symlog 空间轨迹，再调用 `inverse_symlog` 还原到物理空间。该流程与 `data/trajectory_vocab.py` 的保存逻辑对应。

`render_trajectory_vocabulary` 默认绘制前 32 条轨迹，也可通过 `--indices` 指定词表索引。每个小面板都有同一套 BEV 坐标映射，便于检查不同轨迹的形状、尺度和左右方向是否一致。

`render_trajectory_vocabulary_overlay` 不受 `--indices` 和 `--max-trajectories` 限制，会绘制词表中的全部轨迹。该图用于检查 FTS 词表整体覆盖范围、左右分布、是否存在异常离群轨迹，以及第 0 条静止轨迹是否位于 ego 原点。默认只画物理轨迹，避免反求轨迹与物理轨迹高度重合时造成视觉拥挤；需要复核反求几何时可开启 `--overlay-reconstructed`。

## 6. 配置项

| CLI 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--npz` | 无 | 必填，轨迹词表 `.npz` 文件。 |
| `--output` | 无 | 必填，输出 PNG 路径。 |
| `--overlay-output` | `None` | 可选，输出所有轨迹叠加在同一 BEV 图中的 PNG 路径。 |
| `--indices` | `None` | 逗号分隔的词表索引；默认从 0 开始绘制。 |
| `--max-trajectories` | `32` | 最多绘制的轨迹数量。 |
| `--columns` | `4` | 轨迹面板列数。 |
| `--max-mse` | `1e-8` | 允许的全局 MSE 上限。 |
| `--bev-x-min` | `-5` | BEV 前向最小距离。 |
| `--bev-x-max` | `70` | BEV 前向最大距离。 |
| `--bev-y-min` | `-35` | BEV 左向最小距离。 |
| `--bev-y-max` | `35` | BEV 左向最大距离。 |
| `--panel-width` | `240` | 单轨迹面板宽度。 |
| `--panel-height` | `240` | 单轨迹面板高度。 |
| `--overlay-width` | `900` | 所有轨迹叠图宽度。 |
| `--overlay-height` | `720` | 所有轨迹叠图高度。 |
| `--overlay-reconstructed` | `False` | 是否在全局叠图中同时绘制反归一化轨迹。 |

命令示例：

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

## 7. 依赖关系

- 上游：`data/trajectory_vocab.py` 生成的 `.npz`。
- 下游：人工检查词表归一化、反归一化和轨迹几何形状。
- 第三方依赖：`numpy`、`Pillow`。

## 8. 注意事项

- `.npz` 必须同时保存 `trajectory_vocab_m`、`trajectory_vocab_normalized` 和 `symlog_scale`。
- 该工具校验的是归一化轨迹是否能反求回 `.npz` 中保存的物理轨迹，不评估 FTS 采样质量。
- 全局叠图默认绘制所有物理轨迹，不受逐条网格图的 `--indices` 和 `--max-trajectories` 限制。
- 输出 PNG 属于诊断产物，应写入 `visualization/outputs/` 或其他 Git 忽略目录。
- 若未来词表点数或维度改变，必须同步更新 shape 校验和文档。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-04 | 1os3_Codex | AI 完成：新增所有轨迹叠加到同一 BEV 图的可视化输出。 |
| 2026-06-04 | 1os3_Codex | AI 完成：新增轨迹词表反归一化 MSE 校验与物理/反求轨迹叠图可视化工具。 |
