# visualization/backbone_feature_pca_viewer.py

## 1. 文件职责

`visualization/backbone_feature_pca_viewer.py` 负责对统一序列 Transformer 主干做 FP32 诊断可视化。它读取 B2D H5 样本，加载 `config/backbone.toml`，临时把主干、注意力和视觉嵌入精度覆盖为 FP32，可选从 checkpoint 加载真实训练权重，然后直接调用 `MonoDriveBackbone`，收集每层 Transformer 输出后的视觉 Token，并将每层特征图 PCA 到 RGB 后导出 PNG。同时，它会把模型检测、地图和轨迹输出转换为 ego 米制 BEV 诊断图，默认最多检查 16 个 Agent 查询和 32 个 Map 查询，但只绘制最高概率类别不是 `none` 的查询；另有轨迹诊断栏显示 256 维词表概率统计和 top-k 轨迹的 residual 修正。

该文件不复制主干、DINOv3、RoPE、检测头或轨迹词表逻辑。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `ModelOutputVisualizationData` | dataclass | 模型输出 BEV 可视化所需的轨迹、轨迹词表概率、top-k residual 修正、Agent 和 Map 预测。 |
| `BackboneFeaturePCAVisualizationData` | dataclass | 渲染 PNG 所需的样本、shape、精度和 PCA 图。 |
| `run_backbone_feature_pca_sample` | function | 调用真实主干并收集每层视觉 Token PCA 数据。 |
| `render_backbone_feature_pca_sample` | function | 运行主干并保存 PNG。 |
| `render_visualization` | function | 把可视化数据渲染为 PIL 图像。 |
| `main` | function | 命令行入口。 |

## 3. 关键类和函数

### `BackboneFeaturePCAVisualizationData`

- 功能：保存主干 PCA 诊断图需要的元数据和统计张量。
- Shape：
  - `images`: `[8, H, W, 3]`。
  - `layer_pca_images`: `[16, 4, 288, 512, 3]`。
  - `layer_token_norms`: `[16, 4, 18, 32]`。
  - `model_outputs.trajectory_vocab_probabilities`: `[256]`。
  - `model_outputs.top_trajectory_points`: `[5, 6, 2]`。
  - `model_outputs.top_trajectory_residuals`: `[5, 6, 2]`。
  - `model_outputs.top_trajectory_corrections`: `[5, 6, 2]`。
  - `model_outputs.agent_boxes`: `[A, 6]`，`A <= 16`。
  - `model_outputs.map_points`: `[M, 100, 2]`，`M <= 32`。

### `ModelOutputVisualizationData`

- 功能：保存模型输出 BEV 面板需要的预测结果。
- 输入：由 `_summarize_model_outputs` 从 `MonoDriveBackboneOutput` 构造。
- Shape：
  - `trajectory_vocab_probabilities`: `[V]`，轨迹词表 softmax 概率。
  - `top_trajectory_points`: `[K, 6, 2]`，ego 坐标系米制 residual 修正后轨迹。
  - `top_trajectory_vocab_points`: `[K, 6, 2]`，ego 坐标系米制词表基准轨迹。
  - `top_trajectory_residuals`: `[K, 6, 2]`，模型输出的 top-k raw residual，来自 `tanh` 后的归一化空间。
  - `top_trajectory_corrections`: `[K, 6, 2]`，修正后轨迹相对词表基准轨迹的米制位移。
  - `agent_boxes`: `[A, 6]`，`[x, y, l, w, h, yaw]`。
  - `agent_none_scores`: `[A]`，完整类别 softmax 上的 `none` 概率。
  - `agent_future_points`: `[A, 6, 2]`，ego 坐标系米制 future。
  - `map_points`: `[M, 100, 2]`，ego 坐标系米制 Map 点。

### `run_backbone_feature_pca_sample`

- 功能：读取 H5 样本，加载配置，覆盖精度为 FP32，可选加载 checkpoint 权重，直接调用 `MonoDriveBackbone`。
- 输入：H5 路径、样本索引、主干配置路径、项目根目录、设备和可选 checkpoint 路径。
- 输出：`BackboneFeaturePCAVisualizationData`。

### `render_backbone_feature_pca_sample`

- 功能：运行主干并保存诊断 PNG。
- 输入：H5 路径、样本索引、配置路径、输出路径、项目根目录和设备。
- 输出：PNG 输出路径。

## 4. 输入输出与 Shape

| 名称 | Shape / 类型 | 说明 |
| --- | --- | --- |
| `sample["images"]` | `[8, 3, 288, 512] float32` | 由 `B2DH5Dataset` 返回，值域 `[0, 1]`。 |
| `sample["target_point"]` | `[2]` | ego 坐标系米制目标点。 |
| `sample["ego_motion"]` | `[3]` | `[V_x, V_y, W]`。 |
| `backbone_output.sequence_features` | `[1, 2614, 384]` | 统一主干最终输出。 |
| `backbone_output.layer_vision_features` | 16 项 `[1, 2304, 384]` | 每层视觉 Token。 |
| `layer_pca_images` | `[16, 4, 288, 512, 3]` | 每层每个 latent 时间片的 PCA RGB 图。 |
| `trajectory_vocab_probabilities` | `[256]` | 轨迹词表 softmax 概率，用于轨迹诊断栏的概率质量和熵统计。 |
| `top_trajectory_points` | `[5, 6, 2]` | 轨迹 top-k，`vocab_symlog + residual * symlog_scale` 后反 Symlog 到米制。 |
| `top_trajectory_residuals` | `[5, 6, 2]` | top-k 轨迹 raw residual，仍在模型 residual 输出空间。 |
| `top_trajectory_corrections` | `[5, 6, 2]` | top-k 轨迹 residual 生效后的米制修正量，即修正后轨迹减词表基准轨迹。 |
| `agent_boxes` | `[A, 6]` | Agent 查询，`A <= 16`；每个 query 先取包含 `none` 在内的类别 softmax argmax，若 argmax 为 `none` 则不绘制；绘制标签同时显示 argmax 类别概率和 `none` 概率。 |
| `map_points` | `[M, 100, 2]` | Map 查询，`M <= 32`；每个 query 先取包含 `none` 在内的类别 softmax argmax，若 argmax 为 `none` 则不绘制。 |
| 输出 PNG | image file | 主干诊断图。 |

## 5. 关键实现逻辑

脚本通过 `B2DH5Dataset` 读取真实 H5 样本，并保持 Dataset 图像归一化到 `[0, 1]`。加载主干配置后，使用 `override_backbone_precision` 把主干和注意力精度覆盖为 FP32，同时把视觉嵌入配置中的 DINOv3 和卷积精度覆盖为 FP32。其余结构配置保持来自配置文件。

模型调用路径是 `MonoDriveBackbone(..., return_layer_features=True)`。每层视觉 Token 根据 `latent_grid_shape` reshape 为 `[T, H, W, D]`，再转为 `[D, T, H, W]` 做通道 PCA。PCA 只用于诊断，不参与模型训练或推理逻辑。

模型输出 BEV 面板从同一次 `MonoDriveBackboneOutput` 中读取检测和轨迹输出。轨迹使用 Softmax 选择 top-k 词表项，把 `trajectory_vocab_symlog + predicted_residual * symlog_scale` 反 Symlog 到米制轨迹。右下角轨迹诊断栏额外显示完整词表概率数量、top-k 概率质量、归一化熵、每条 top 轨迹的概率条、raw residual 最大值、米制 residual correction 的 mean/max 以及最后一个 future 点的 correction。Agent 输出对每个 query 在包含 `none` 的完整类别 softmax 上取最高概率类别；若最高类别不是 `none`，且 argmax 概率不低于 `--agent-confidence-threshold`，才进入绘制候选并按该概率取 top-k，对 `x/y/v/future` 使用反 Symlog、对 `l/w/h` 使用 `expm1`，并用 `[sin_yaw, cos_yaw]` 反求 yaw。BEV 标签格式为 `class:argmax_prob none:none_prob`。Map 输出同样过滤 `none` query 和低于 `--map-confidence-threshold` 的 query 后再取 top-k，对点坐标反 Symlog 后绘制 polyline。`--agent-top-k` 和 `--map-top-k` 可用于临时减少绘制数量。

输出路径通过项目根目录校验，默认写入 `visualization/outputs/backbone_feature_pca/`。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `--h5` | 无 | 必填，预处理 H5 文件。 |
| `--sample-index` | 命令行默认 | 样本索引。 |
| `--config` | 命令行默认 | 主干配置路径。 |
| `--output` | 可选 | 单张 PNG 输出路径，必须位于项目内。 |
| `--checkpoint` | 可选 | 模型 checkpoint 路径，支持训练 payload 的 `model_state` 或直接保存的 state dict，必须位于项目内。 |
| `--output-dir` | 命令行默认 | 未指定 `--output` 时的输出目录，必须位于项目内。 |
| `--device` | 命令行默认 | 运行设备。 |
| `--trajectory-top-k` | `5` | 模型输出 BEV 面板绘制的轨迹 top-k 数量。 |
| `--agent-top-k` | `16` | 模型输出 BEV 面板最多绘制的非 `none` Agent 查询数量。 |
| `--map-top-k` | `32` | 模型输出 BEV 面板最多绘制的非 `none` Map 查询数量。 |
| `--agent-confidence-threshold` | `0.0` | Agent query 在完整类别 softmax 上的 argmax 概率下限；低于该阈值的非 `none` query 不绘制。 |
| `--map-confidence-threshold` | `0.0` | Map query 在完整类别 softmax 上的 argmax 概率下限；低于该阈值的非 `none` query 不绘制。 |

## 7. 依赖关系

- 上游：`data/b2d_dataset.py`。
- 核心实现：`model/backbone.py`。
- 配置：`config/backbone.toml` 以及其中引用的子配置。
- 输出目录：默认写入 `visualization/outputs/backbone_feature_pca/`，该目录位于项目内并被 `.gitignore` 忽略。
- 第三方依赖：`torch`、`numpy`、`PIL`、`transformers`。

## 8. 注意事项

- 本脚本固定以 FP32 调用主干和视觉嵌入，避免本机 BF16 过慢。
- 诊断图中的 PCA RGB 只用于观察特征空间结构，不代表注意力权重或 loss。
- 模型输出 BEV 面板只做诊断级反变换和排序展示，不替代正式推理后处理、NMS、Hungarian matching 或安全过滤。
- Detection query 的显示类别来自该 query 在包含 `none` 的完整类别 softmax 上的 argmax；argmax 为 `none` 或 argmax 概率低于对应置信度阈值的 query 会被过滤，不进入 BEV 绘制。已绘制的 Agent 标签会同时显示 argmax 类别概率和 `none` 概率。
- 轨迹诊断栏的 `raw|max` 是 top-k residual 张量的绝对值最大值；`meter mean/max` 和 `final delta` 是 residual 生效后相对词表基准轨迹的米制修正。
- `--checkpoint` 使用严格 `load_state_dict`，checkpoint 结构不匹配时应直接报错。
- CPU 上运行会加载 DINOv3 和完整 16 层主干，可能耗时较长。
- 修改主干输出、层特征收集或命令行参数时，必须同步更新摘要文档和 `doc/Code Doc/Index.md`。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-09 | 1os3_Composer | AI 完成：已绘制 Agent 标签同时显示 argmax 类别概率和 `none` 概率。 |
| 2026-06-09 | 1os3_Composer | AI 完成：新增 Agent/Map 检测 query 的 argmax 置信度阈值过滤，并暴露 `--agent-confidence-threshold` 与 `--map-confidence-threshold`。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 16 层主干、2614 序列长度和 Agent 16 / Map 32 默认展示数量。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增轨迹词表概率与 top-k residual 修正诊断栏，并记录 raw residual 与米制 correction。 |
| 2026-06-08 | 1os3_Codex | AI 完成：检测输出 BEV 面板过滤最高概率类别为 `none` 的 Agent/Map query。 |
| 2026-06-08 | 1os3_Codex | AI 完成：检测输出 BEV 面板改为每个 Agent/Map query 显示包含 `none` 在内的最高概率类别和概率。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增 `--checkpoint` 加载真实模型权重，并同步轨迹 residual 的 `symlog_scale` 反解口径。 |
| 2026-06-07 | 1os3_Codex | AI 完成：模型输出 BEV 默认展示完整 48 个 Agent 查询和 48 个 Map 查询，并暴露 top-k 参数。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增模型输出 BEV 面板，展示轨迹 top-k、Agent top-k 和 Map top-k 预测。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增统一主干每层视觉特征 FP32 PCA 可视化工具文档。 |
