# model/residual_block.py 摘要

## 1. 文件基本功能

`model/residual_block.py` 定义视觉编码和时空特征处理中使用的基础残差卷积模块，包括：

- 2D / 3D RMSNorm。
- 2D / 3D 标准瓶颈残差块。
- 先空间后时间的时空解耦 3D 残差块。
- ConvNeXt 风格 2D / 3D depthwise separable 残差块。

该文件只负责模块定义和本地自测，不负责配置读取、训练循环、权重加载、数据预处理或指标计算。

## 2. 公开接口速查

| 接口 | 类型 | 基本功能 | 输入 Shape | 输出 Shape |
| --- | --- | --- | --- | --- |
| `RMSNorm2d` | class | 2D 特征图通道 RMSNorm | `[N, C, H, W]` | `[N, C, H, W]` |
| `RMSNorm3d` | class | 3D 特征图通道 RMSNorm | `[N, C, D, H, W]` | `[N, C, D, H, W]` |
| `ResidualBlock` | class | 2D 瓶颈残差卷积 | `[N, C, H, W]` | `[N, C, H, W]` |
| `ResidualBlock3d` | class | 3D 瓶颈残差卷积 | `[N, C, D, H, W]` | `[N, C, D, H, W]` |
| `SpatioTemporalResidualBlock3d` | class | 时空解耦 3D 残差卷积 | `[N, C, D, H, W]` | `[N, C, D, H, W]` |
| `DepthwiseSeparableBlock` | class | ConvNeXt 风格 2D 残差块 | `[N, C, H, W]` | `[N, C, H, W]` |
| `DepthwiseSeparableBlock3d` | class | ConvNeXt 风格 3D 残差块 | `[N, C, D, H, W]` | `[N, C, D, H, W]` |

维度约定：

- `N`：batch size。
- `C`：通道数。
- `D`：3D 特征的深度维或时间维，具体语义由上游模块定义。
- `H`：特征图高度。
- `W`：特征图宽度。

## 3. 各接口使用规范

### `RMSNorm2d`

- 构造方式：`RMSNorm2d(normalized_shape: int, eps: float = 1e-6)`
- 使用场景：对 2D 卷积特征 `[N, C, H, W]` 做通道 RMSNorm。
- 参数要求：
  - `normalized_shape` 必须等于输入通道数 `C`。
  - `normalized_shape > 0`。
  - `eps > 0`。
- 输入要求：输入必须是 4D tensor，且第 2 维通道数与 `normalized_shape` 一致。
- 输出：shape 与输入一致。
- 注意事项：该模块不做均值中心化，只按 RMS 缩放。

### `RMSNorm3d`

- 构造方式：`RMSNorm3d(normalized_shape: int, eps: float = 1e-6)`
- 使用场景：对 3D 卷积特征 `[N, C, D, H, W]` 做通道 RMSNorm。
- 参数要求：
  - `normalized_shape` 必须等于输入通道数 `C`。
  - `normalized_shape > 0`。
  - `eps > 0`。
- 输入要求：输入必须是 5D tensor，且第 2 维通道数与 `normalized_shape` 一致。
- 输出：shape 与输入一致。
- 注意事项：`D` 可以是时间维或深度维，调用方必须保证语义一致。

### `ResidualBlock`

- 构造方式：`ResidualBlock(channels: int)`
- 使用场景：在 2D 特征图上做标准瓶颈残差卷积。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels >= 2`，否则 $\lfloor channels / 2 \rfloor$ 会得到无效中间通道数。
- 输入要求：`[N, C, H, W]`，其中 `C == channels`。
- 输出：`[N, C, H, W]`。
- 注意事项：卷积 stride 固定为 1，padding 保持空间分辨率不变，可直接残差相加。

### `ResidualBlock3d`

- 构造方式：`ResidualBlock3d(channels: int)`
- 使用场景：在 3D 特征图上做完整 `3x3x3` 瓶颈残差卷积。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels >= 2`。
- 输入要求：`[N, C, D, H, W]`，其中 `C == channels`。
- 输出：`[N, C, D, H, W]`。
- 注意事项：该模块联合建模 `D, H, W` 邻域，参数量和计算量高于时空解耦版本。

### `SpatioTemporalResidualBlock3d`

- 构造方式：`SpatioTemporalResidualBlock3d(channels: int)`
- 使用场景：需要 3D 卷积接口，但希望将空间建模和时间 / 深度建模解耦。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels >= 2`。
- 输入要求：`[N, C, D, H, W]`，其中 `C == channels`。
- 输出：`[N, C, D, H, W]`。
- 注意事项：
  - `(1, 3, 3)` 卷积只覆盖空间维。
  - `(3, 1, 1)` 卷积只覆盖 `D` 维。
  - 如果上游把 `D` 作为时间维，需要在上游文档中说明采样频率和时间顺序。

### `DepthwiseSeparableBlock`

- 构造方式：`DepthwiseSeparableBlock(channels: int, expansion_ratio: int = 2)`
- 使用场景：在 2D 特征图上使用 ConvNeXt 风格 depthwise separable 残差块。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels > 0`。
  - `expansion_ratio > 0`。
- 输入要求：`[N, C, H, W]`，其中 `C == channels`。
- 输出：`[N, C, H, W]`。
- 注意事项：
  - depthwise 卷积使用 `groups=channels`。
  - 内部扩展通道数为 `channels * expansion_ratio`。

### `DepthwiseSeparableBlock3d`

- 构造方式：`DepthwiseSeparableBlock3d(channels: int, expansion_ratio: int = 2)`
- 使用场景：在 3D 特征图上使用 ConvNeXt 风格 depthwise separable 残差块。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels > 0`。
  - `expansion_ratio > 0`。
- 输入要求：`[N, C, D, H, W]`，其中 `C == channels`。
- 输出：`[N, C, D, H, W]`。
- 注意事项：
  - depthwise 卷积使用 `groups=channels`。
  - `7x7x7` 卷积会带来较高显存和计算开销。

## 4. 最小使用示例

```python
import torch

from model.residual_block import ResidualBlock, SpatioTemporalResidualBlock3d


features_2d = torch.randn(2, 128, 32, 32)
block_2d = ResidualBlock(channels=128)
out_2d = block_2d(features_2d)
assert out_2d.shape == features_2d.shape

features_3d = torch.randn(2, 128, 4, 32, 32)
block_3d = SpatioTemporalResidualBlock3d(channels=128)
out_3d = block_3d(features_3d)
assert out_3d.shape == features_3d.shape
```

## 5. 维护注意事项

- 修改公开接口时，必须同步更新本摘要文档、完整文档 [residual_block.md](residual_block.md) 和目录页 [Index.md](../Index.md)。
- 修改通道数、卷积核、stride 或 padding 时，必须重新检查残差相加前后的 shape。
- 标准瓶颈块和时空解耦块依赖 $\lfloor channels / 2 \rfloor$，因此要求 `channels >= 2`。
- `D` 维可能表示时间或深度，具体语义必须由上游模块文档说明。
- 完整文档必须包含本摘要文档中的全部信息，并可在此基础上补充实现细节、依赖关系和维护记录。
