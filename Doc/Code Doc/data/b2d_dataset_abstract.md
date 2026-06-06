# data/b2d_dataset.py 摘要

## 1. 文件基本功能

读取 B2D 预处理 H5，并以 PyTorch `Dataset` 返回模型训练样本，包括 8 帧前视图像、未来轨迹、自车运动状态、未来目标候选随机抽样、控制量、Agent future、局部 Map 和交通元素标签；读取端缓存 H5 dataset 句柄以减少重复路径查找。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `B2DH5Dataset` | class | 读取单个 H5、H5 目录或 H5 文件列表。 |
| `B2DH5Dataset.close` | method | 显式关闭当前进程缓存的 H5 句柄。 |
| `SUPPORTED_SCHEMA_VERSIONS` | constant | 限制可读取的 H5 schema，当前为 `b2d_h5_v5`。 |

## 3. 输入输出 Shape 概览

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `images` | `[8, 3, 288, 512]` | 最近 8 帧前视 RGB。 |
| `future_trajectory` | `[6, 2]` | 未来 3 秒 2Hz ego 坐标系轨迹。 |
| `ego_motion` | `[3]` | `[Vx, Vy, W]`。 |
| `target_point` | `[2]` | 默认随机抽取的有效目标候选。 |
| `target_points` | `[32, 2]` | 未来 24-30m 可达目标候选池。 |
| `target_valid` | `[32]` | 目标候选 padding mask。 |
| `target_point_index` | scalar | 本次抽中的目标候选索引。 |
| `agent_boxes` | `[194, 10]` | `[x, y, l, w, h, yaw, v_x, v_y, a_x, a_y]`。 |
| `agent_valid` | `[194]` | Agent 有效 mask。 |
| `agent_future_trajectory` | `[194, 6, 2]` | Agent future 位移，原点为当前 Agent 中心，坐标轴沿当前 ego 坐标系。 |
| `agent_future_valid` | `[194, 6]` | Agent future 有效 mask。 |
| `map_points` | `[60, 100, 2]` | 局部 Map 元素。 |
| `map_valid` | `[60]` | Map 有效 mask。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `B2DH5Dataset(h5_paths)` | `h5_paths` 可以是文件、目录或列表；目录模式仅读取 `*.h5`。 |
| `schema_version` | 当前只支持 `b2d_h5_v5`；旧 H5 v4 的 Agent future 语义不同，必须重新预处理。 |
| `normalize_images` | 默认为 `True`，图像返回 `[0, 1]` 浮点张量。 |
| `random_target_point` | 默认 `True`，每次读取随机抽一个有效目标候选；确定性评估可关闭。 |
| `close()` | 调试或释放 Windows 文件锁时显式调用。 |

## 5. 最小使用示例

```python
from data.b2d_dataset import B2DH5Dataset

dataset = B2DH5Dataset("data/preprocessed")
sample = dataset[0]
print(sample["images"].shape)
dataset.close()
```

## 6. 维护注意事项

- 新增或重命名 H5 字段时必须同步更新 `__getitem__` 返回字典。
- 当前读取器只接受 `b2d_h5_v5`，防止旧 Agent future 绝对点标签被误当作位移标签使用。
- 多 worker 读取依赖懒加载 H5 句柄，不要在初始化阶段长期打开文件。
- 读取端缓存的是 H5 dataset 对象，不应改成整数组常驻内存缓存。
- 与 `data/b2d_preprocess.py` 的 H5 schema 保持一致。
- `target_point` 默认从 H5 的 `target_points/target_valid` 候选池抽取得到，训练日志应记录是否开启随机抽样。
- Agent future 与 Map 标签直接读取 H5；Dataset 不做投影、可见性过滤、位移转绝对点或 Map 重采样。
- Agent 速度和加速度必须保持为预处理差分结果，不读取 B2D 原始速度/加速度标注。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-05 | 1os3_Codex | AI 完成：同步 Agent future 位移读取语义。 |
| 2026-06-03 | 1os3_Codex | AI 完成：同步 H5 dataset 句柄缓存优化。 |
| 2026-06-03 | 1os3_Codex | AI 完成：同步 H5 v4 Agent future 与 Map 返回字段。 |
| 2026-06-02 | 1os3_Codex | AI 完成：同步目标候选池读取与随机抽样语义。 |
| 2026-06-02 | 1os3_Codex | AI 完成：同步 Agent 10D 平面运动标签摘要。 |
| 2026-06-02 | 1os3_Codex | AI 完成：新增 B2D H5 Dataset 摘要文档。 |
