# visualization/backbone_feature_pca_viewer.py

## 1. 文件职责

`visualization/backbone_feature_pca_viewer.py` 负责对统一序列 Transformer 主干做 FP32 诊断可视化。它读取 B2D H5 样本，加载 `config/backbone.toml`，临时把主干、注意力和视觉嵌入精度覆盖为 FP32，然后直接调用 `MonoDriveBackbone`，收集每层 Transformer 输出后的视觉 Token，并将每层特征图 PCA 到 RGB 后导出 PNG。

该文件不复制主干、DINOv3、RoPE、检测头或轨迹词表逻辑。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
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
  - `layer_pca_images`: `[12, 4, 288, 512, 3]`。
  - `layer_token_norms`: `[12, 4, 18, 32]`。

### `run_backbone_feature_pca_sample`

- 功能：读取 H5 样本，加载配置，覆盖精度为 FP32，直接调用 `MonoDriveBackbone`。
- 输入：H5 路径、样本索引、主干配置路径、项目根目录和设备。
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
| `backbone_output.sequence_features` | `[1, 2662, 384]` | 统一主干输出。 |
| `backbone_output.layer_vision_features` | 12 项 `[1, 2304, 384]` | 每层视觉 Token。 |
| `layer_pca_images` | `[12, 4, 288, 512, 3]` | 每层每个 latent 时间片的 PCA RGB 图。 |
| 输出 PNG | image file | 主干诊断图。 |

## 5. 关键实现逻辑

脚本通过 `B2DH5Dataset` 读取真实 H5 样本，并保持 Dataset 图像归一化到 `[0, 1]`。加载主干配置后，使用 `override_backbone_precision` 把主干和注意力精度覆盖为 FP32，同时把视觉嵌入配置中的 DINOv3 和卷积精度覆盖为 FP32。其余结构配置保持来自配置文件。

模型调用路径是 `MonoDriveBackbone(..., return_layer_features=True)`。每层视觉 Token 根据 `latent_grid_shape` reshape 为 `[T, H, W, D]`，再转为 `[D, T, H, W]` 做通道 PCA。PCA 只用于诊断，不参与模型训练或推理逻辑。

输出路径通过项目根目录校验，默认写入 `visualization/outputs/backbone_feature_pca/`。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `--h5` | 无 | 必填，预处理 H5 文件。 |
| `--sample-index` | 命令行默认 | 样本索引。 |
| `--config` | 命令行默认 | 主干配置路径。 |
| `--output` | 可选 | 单张 PNG 输出路径，必须位于项目内。 |
| `--output-dir` | 命令行默认 | 未指定 `--output` 时的输出目录，必须位于项目内。 |
| `--device` | 命令行默认 | 运行设备。 |

## 7. 依赖关系

- 上游：`data/b2d_dataset.py`。
- 核心实现：`model/backbone.py`。
- 配置：`config/backbone.toml` 以及其中引用的子配置。
- 输出目录：默认写入 `visualization/outputs/backbone_feature_pca/`，该目录位于项目内并被 `.gitignore` 忽略。
- 第三方依赖：`torch`、`numpy`、`PIL`、`transformers`。

## 8. 注意事项

- 本脚本固定以 FP32 调用主干和视觉嵌入，避免本机 BF16 过慢。
- 诊断图中的 PCA RGB 只用于观察特征空间结构，不代表注意力权重、loss 或物理空间预测。
- CPU 上运行会加载 DINOv3 和完整 12 层主干，可能耗时较长。
- 修改主干输出、层特征收集或命令行参数时，必须同步更新摘要文档和 `doc/Code Doc/Index.md`。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增统一主干每层视觉特征 FP32 PCA 可视化工具文档。 |
