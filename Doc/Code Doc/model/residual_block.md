# model/residual_block.py

## 0. 摘要

### 0.1 文件基本功能

`model/residual_block.py` 定义视觉编码和时空特征处理中使用的基础残差卷积模块，包括：

- 2D / 3D RMSNorm。
- 2D / 3D 标准瓶颈残差块。
- 先空间后时间的时空解耦 3D 残差块。
- ConvNeXt 风格 2D / 3D depthwise separable 残差块。

该文件只负责模块定义和本地自测，不负责配置读取、训练循环、权重加载、数据预处理或指标计算。

### 0.2 公开接口速查

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

### 0.3 各接口使用规范

#### `RMSNorm2d`

- 构造方式：`RMSNorm2d(normalized_shape: int, eps: float = 1e-6)`
- 使用场景：对 2D 卷积特征 `[N, C, H, W]` 做通道 RMSNorm。
- 参数要求：
  - `normalized_shape` 必须等于输入通道数 `C`。
  - `normalized_shape > 0`。
  - `eps > 0`。
- 输入要求：输入必须是 4D tensor，且第 2 维通道数与 `normalized_shape` 一致。
- 输出：shape 与输入一致。
- 注意事项：该模块不做均值中心化，只按 RMS 缩放。

#### `RMSNorm3d`

- 构造方式：`RMSNorm3d(normalized_shape: int, eps: float = 1e-6)`
- 使用场景：对 3D 卷积特征 `[N, C, D, H, W]` 做通道 RMSNorm。
- 参数要求：
  - `normalized_shape` 必须等于输入通道数 `C`。
  - `normalized_shape > 0`。
  - `eps > 0`。
- 输入要求：输入必须是 5D tensor，且第 2 维通道数与 `normalized_shape` 一致。
- 输出：shape 与输入一致。
- 注意事项：`D` 可以是时间维或深度维，调用方必须保证语义一致。

#### `ResidualBlock`

- 构造方式：`ResidualBlock(channels: int)`
- 使用场景：在 2D 特征图上做标准瓶颈残差卷积。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels >= 2`，否则 $\lfloor channels / 2 \rfloor$ 会得到无效中间通道数。
- 输入要求：`[N, C, H, W]`，其中 `C == channels`。
- 输出：`[N, C, H, W]`。
- 注意事项：卷积 stride 固定为 1，padding 保持空间分辨率不变，可直接残差相加。

#### `ResidualBlock3d`

- 构造方式：`ResidualBlock3d(channels: int)`
- 使用场景：在 3D 特征图上做完整 `3x3x3` 瓶颈残差卷积。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels >= 2`。
- 输入要求：`[N, C, D, H, W]`，其中 `C == channels`。
- 输出：`[N, C, D, H, W]`。
- 注意事项：该模块联合建模 `D, H, W` 邻域，参数量和计算量高于时空解耦版本。

#### `SpatioTemporalResidualBlock3d`

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

#### `DepthwiseSeparableBlock`

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

#### `DepthwiseSeparableBlock3d`

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

### 0.4 最小使用示例

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

### 0.5 维护注意事项

- 修改公开接口时，必须同步更新本摘要文档、完整文档 [residual_block.md](residual_block.md) 和目录页 [Index.md](../Index.md)。
- 修改通道数、卷积核、stride 或 padding 时，必须重新检查残差相加前后的 shape。
- 标准瓶颈块和时空解耦块依赖 $\lfloor channels / 2 \rfloor$，因此要求 `channels >= 2`。
- `D` 维可能表示时间或深度，具体语义必须由上游模块文档说明。
- 完整文档必须包含摘要文档中的全部信息，并可在此基础上补充实现细节、依赖关系和维护记录。

## 1. 文件职责

`model/residual_block.py` 定义视觉编码和时空特征处理中使用的基础卷积残差模块。该文件提供 2D / 3D RMSNorm、标准瓶颈残差块、时空解耦 3D 残差块，以及 ConvNeXt 风格的 depthwise separable 残差块。

该文件只负责模块结构定义和一个本地 shape / 参数量自测入口，不负责训练流程、配置加载、权重加载或数据处理。模块通过 `__all__` 显式声明公开接口。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `RMSNorm2d` | class | 面向 2D 特征图 `[N, C, H, W]` 的 RMSNorm。 |
| `RMSNorm3d` | class | 面向 3D 特征图 `[N, C, D, H, W]` 的 RMSNorm。 |
| `ResidualBlock` | class | 2D 瓶颈残差卷积块，保持输入输出 shape 不变。 |
| `ResidualBlock3d` | class | 3D 瓶颈残差卷积块，保持输入输出 shape 不变。 |
| `SpatioTemporalResidualBlock3d` | class | 时空解耦 3D 残差块，先做空间卷积，再做时间卷积。 |
| `DepthwiseSeparableBlock` | class | ConvNeXt 风格 2D depthwise separable 残差块。 |
| `DepthwiseSeparableBlock3d` | class | ConvNeXt 风格 3D depthwise separable 残差块。 |

## 3. 接口使用规范

### `RMSNorm2d`

- 构造方式：`RMSNorm2d(normalized_shape: int, eps: float = 1e-6)`
- 使用场景：对 2D 卷积特征 `[N, C, H, W]` 做通道 RMSNorm。
- 参数要求：
  - `normalized_shape` 必须等于输入通道数 `C`。
  - `normalized_shape > 0`。
  - `eps > 0`。
- 输入要求：输入必须是 4D tensor，且第 2 维通道数与 `normalized_shape` 一致。
- 输出约定：输出 shape 与输入一致。
- 注意事项：该模块不做均值中心化，只按 RMS 缩放。

### `RMSNorm3d`

- 构造方式：`RMSNorm3d(normalized_shape: int, eps: float = 1e-6)`
- 使用场景：对 3D 卷积特征 `[N, C, D, H, W]` 做通道 RMSNorm。
- 参数要求：
  - `normalized_shape` 必须等于输入通道数 `C`。
  - `normalized_shape > 0`。
  - `eps > 0`。
- 输入要求：输入必须是 5D tensor，且第 2 维通道数与 `normalized_shape` 一致。
- 输出约定：输出 shape 与输入一致。
- 注意事项：`D` 可以是时间维或深度维，调用方必须保证语义一致。

### `ResidualBlock`

- 构造方式：`ResidualBlock(channels: int)`
- 使用场景：在 2D 特征图上做标准瓶颈残差卷积。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels >= 2`，否则 $\lfloor channels / 2 \rfloor$ 会得到无效中间通道数。
- 输入要求：`[N, C, H, W]`，其中 `C == channels`。
- 输出约定：`[N, C, H, W]`。
- 注意事项：卷积 stride 固定为 1，padding 保持空间分辨率不变，可直接残差相加。

### `ResidualBlock3d`

- 构造方式：`ResidualBlock3d(channels: int)`
- 使用场景：在 3D 特征图上做完整 `3x3x3` 瓶颈残差卷积。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels >= 2`。
- 输入要求：`[N, C, D, H, W]`，其中 `C == channels`。
- 输出约定：`[N, C, D, H, W]`。
- 注意事项：该模块联合建模 `D, H, W` 邻域，参数量和计算量高于时空解耦版本。

### `SpatioTemporalResidualBlock3d`

- 构造方式：`SpatioTemporalResidualBlock3d(channels: int)`
- 使用场景：需要 3D 卷积接口，但希望将空间建模和时间 / 深度建模解耦。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels >= 2`。
- 输入要求：`[N, C, D, H, W]`，其中 `C == channels`。
- 输出约定：`[N, C, D, H, W]`。
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
- 输出约定：`[N, C, H, W]`。
- 注意事项：
  - depthwise 卷积使用 `groups=channels`。
  - 内部扩展通道数为 $channels \times expansion\_ratio$。

### `DepthwiseSeparableBlock3d`

- 构造方式：`DepthwiseSeparableBlock3d(channels: int, expansion_ratio: int = 2)`
- 使用场景：在 3D 特征图上使用 ConvNeXt 风格 depthwise separable 残差块。
- 参数要求：
  - `channels` 必须等于输入和输出通道数 `C`。
  - `channels > 0`。
  - `expansion_ratio > 0`。
- 输入要求：`[N, C, D, H, W]`，其中 `C == channels`。
- 输出约定：`[N, C, D, H, W]`。
- 注意事项：
  - depthwise 卷积使用 `groups=channels`。
  - `7x7x7` 卷积会带来较高显存和计算开销。

## 4. 关键类和函数

### `RMSNorm2d`

- 功能：对 2D 特征图按通道维计算 RMS，并使用可学习通道权重重新缩放。
- 输入：`x: torch.Tensor`
- 输出：归一化后的 `torch.Tensor`
- Shape：`[N, C, H, W] -> [N, C, H, W]`
- 关键参数：
  - `normalized_shape: int`：通道数 `C`。
  - `eps: float = 1e-6`：数值稳定项，避免除零。
- 参数校验：
  - `normalized_shape` 必须为正整数。
  - `eps` 必须为正数。
- 实现说明：`x.pow(2).mean(1, keepdim=True).sqrt()` 在通道维上计算 RMS，`self.weight[:, None, None]` 广播到空间维。

### `RMSNorm3d`

- 功能：对 3D 特征图按通道维计算 RMS，并使用可学习通道权重重新缩放。
- 输入：`x: torch.Tensor`
- 输出：归一化后的 `torch.Tensor`
- Shape：`[N, C, D, H, W] -> [N, C, D, H, W]`
- 关键参数：
  - `normalized_shape: int`：通道数 `C`。
  - `eps: float = 1e-6`：数值稳定项，避免除零。
- 参数校验：
  - `normalized_shape` 必须为正整数。
  - `eps` 必须为正数。
- 实现说明：与 `RMSNorm2d` 相同，但 `self.weight[:, None, None, None]` 广播到深度、高度和宽度维。

### `ResidualBlock`

- 功能：2D 标准瓶颈残差块，用于空间特征提取。
- 输入：`x: torch.Tensor`
- 输出：`out + identity`
- Shape：`[N, C, H, W] -> [N, C, H, W]`
- 关键参数：
  - `channels: int`：输入和输出通道数。
- 参数校验：
  - `channels >= 2`，避免 $\lfloor channels / 2 \rfloor$ 得到 0 个中间通道。
- 主路径：
  1. `RMSNorm2d(C)`
  2. `1x1 Conv2d`：$C \rightarrow C/2$
  3. `3x3 Conv2d`：$C/2 \rightarrow C/2$，`padding=1`
  4. `GELU`
  5. `1x1 Conv2d`：$C/2 \rightarrow C$
  6. 与输入残差相加

### `ResidualBlock3d`

- 功能：3D 标准瓶颈残差块，用于联合建模深度 / 时间维和空间维。
- 输入：`x: torch.Tensor`
- 输出：`out + identity`
- Shape：`[N, C, D, H, W] -> [N, C, D, H, W]`
- 关键参数：
  - `channels: int`：输入和输出通道数。
- 参数校验：
  - `channels >= 2`，避免 $\lfloor channels / 2 \rfloor$ 得到 0 个中间通道。
- 主路径：
  1. `RMSNorm3d(C)`
  2. `1x1x1 Conv3d`：$C \rightarrow C/2$
  3. `3x3x3 Conv3d`：$C/2 \rightarrow C/2$，`padding=1`
  4. `GELU`
  5. `1x1x1 Conv3d`：$C/2 \rightarrow C$
  6. 与输入残差相加

### `SpatioTemporalResidualBlock3d`

- 功能：时空解耦 3D 残差块，用 3D 卷积分别完成空间建模和时间建模。
- 输入：`x: torch.Tensor`
- 输出：`out + identity`
- Shape：`[N, C, D, H, W] -> [N, C, D, H, W]`
- 关键参数：
  - `channels: int`：输入和输出通道数。
- 参数校验：
  - `channels >= 2`，避免 $\lfloor channels / 2 \rfloor$ 得到 0 个中间通道。
- 主路径：
  1. `RMSNorm3d(C)`
  2. `1x1x1 Conv3d`：$C \rightarrow C/2$
  3. `(1, 3, 3) Conv3d`：$C/2 \rightarrow C/2$，`padding=(0, 1, 1)`
  4. `(3, 1, 1) Conv3d`：$C/2 \rightarrow C/2$，`padding=(1, 0, 0)`
  5. `GELU`
  6. `1x1x1 Conv3d`：$C/2 \rightarrow C$
  7. 与输入残差相加
- 设计意图：空间卷积只覆盖 `H, W`，时间卷积只覆盖 `D`，在保留 3D 卷积实现形式的同时降低参数量和计算量。

### `DepthwiseSeparableBlock`

- 功能：ConvNeXt 风格 2D 残差块，先用 depthwise 卷积提取空间特征，再用 1x1 卷积进行通道混合。
- 输入：`x: torch.Tensor`
- 输出：`out + identity`
- Shape：`[N, C, H, W] -> [N, C, H, W]`
- 关键参数：
  - `channels: int`：输入和输出通道数。
  - `expansion_ratio: int = 2`：通道扩展比例，内部通道数为 `C * expansion_ratio`。
- 参数校验：
  - `channels > 0`。
  - `expansion_ratio > 0`。
- 主路径：
  1. `7x7 Depthwise Conv2d(C -> C, groups=C, padding=3)`
  2. `RMSNorm2d(C)`
  3. `1x1 Conv2d(C -> C * expansion_ratio)`
  4. `GELU`
  5. `1x1 Conv2d(C * expansion_ratio -> C)`
  6. 与输入残差相加

### `DepthwiseSeparableBlock3d`

- 功能：ConvNeXt 风格 3D depthwise separable 残差块。
- 输入：`x: torch.Tensor`
- 输出：`out + identity`
- Shape：`[N, C, D, H, W] -> [N, C, D, H, W]`
- 关键参数：
  - `channels: int`：输入和输出通道数。
  - `expansion_ratio: int = 2`：通道扩展比例，内部通道数为 `C * expansion_ratio`。
- 参数校验：
  - `channels > 0`。
  - `expansion_ratio > 0`。
- 主路径：
  1. `7x7x7 Depthwise Conv3d(C -> C, groups=C, padding=3)`
  2. `RMSNorm3d(C)`
  3. `1x1x1 Conv3d(C -> C * expansion_ratio)`
  4. `GELU`
  5. `1x1x1 Conv3d(C * expansion_ratio -> C)`
  6. 与输入残差相加

## 5. 输入输出与 Shape

| 模块 | 输入 Shape | 输出 Shape | 说明 |
| --- | --- | --- | --- |
| `RMSNorm2d` | `[N, C, H, W]` | `[N, C, H, W]` | 按通道维计算 RMS。 |
| `RMSNorm3d` | `[N, C, D, H, W]` | `[N, C, D, H, W]` | 按通道维计算 RMS。 |
| `ResidualBlock` | `[N, C, H, W]` | `[N, C, H, W]` | 2D 瓶颈残差块。 |
| `ResidualBlock3d` | `[N, C, D, H, W]` | `[N, C, D, H, W]` | 3D 瓶颈残差块。 |
| `SpatioTemporalResidualBlock3d` | `[N, C, D, H, W]` | `[N, C, D, H, W]` | 空间和时间卷积分离。 |
| `DepthwiseSeparableBlock` | `[N, C, H, W]` | `[N, C, H, W]` | 2D ConvNeXt 风格块。 |
| `DepthwiseSeparableBlock3d` | `[N, C, D, H, W]` | `[N, C, D, H, W]` | 3D ConvNeXt 风格块。 |

维度约定：

- `N`：batch size。
- `C`：通道数。
- `D`：3D 特征的深度维或时间维，具体语义由上游模块决定。
- `H`：特征图高度。
- `W`：特征图宽度。

## 6. 关键实现逻辑

### RMSNorm

`RMSNorm2d` 和 `RMSNorm3d` 都只在通道维上计算均方根，不做均值中心化。计算流程为：

1. 计算 `rms = sqrt(mean(x^2, dim=channel))`。
2. 使用 `x / (rms + eps)` 归一化。
3. 使用可学习参数 `weight` 按通道缩放。

这种实现比 LayerNorm 少一步均值中心化，适合卷积特征图中较轻量的归一化需求。

### 标准残差块

`ResidualBlock` 和 `ResidualBlock3d` 使用瓶颈结构：先把通道从 $C$ 压缩到 $C/2$，在中间通道上做卷积，再恢复到 $C$。卷积 padding 设置保证空间维或 3D 维度不变，因此可以直接和 `identity` 相加。

### 时空解耦残差块

`SpatioTemporalResidualBlock3d` 使用两个 3D 卷积近似完整 `3x3x3` 卷积：

- `(1, 3, 3)` 只建模空间邻域。
- `(3, 1, 1)` 只建模时间 / 深度邻域。

该结构保留 3D 卷积接口，参数量约为完整 `3x3x3` 中间卷积的：

$$
\frac{1 \times 3 \times 3 + 3 \times 1 \times 1}{3 \times 3 \times 3} = \frac{12}{27}
$$

约为 44.4%。

### Depthwise Separable 残差块

`DepthwiseSeparableBlock` 和 `DepthwiseSeparableBlock3d` 采用 ConvNeXt 风格：

1. depthwise 卷积负责局部空间或时空特征提取。
2. RMSNorm 负责归一化。
3. 两个 `1x1` 或 `1x1x1` 卷积负责通道扩展、激活和压缩。
4. 输出与输入做残差相加。

## 7. 配置项

该文件本身不读取配置文件，所有可配置项都来自类构造函数。

| 参数 | 所属模块 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `normalized_shape` | `RMSNorm2d`, `RMSNorm3d` | 无 | 归一化通道数，通常等于 `C`。 |
| `eps` | `RMSNorm2d`, `RMSNorm3d` | `1e-6` | 防止除零的数值稳定项。 |
| `channels` | 所有残差块 | 无 | 输入和输出通道数。 |
| `expansion_ratio` | `DepthwiseSeparableBlock`, `DepthwiseSeparableBlock3d` | `2` | ConvNeXt 风格块的通道扩展比例。 |

## 8. 依赖关系

- 第三方依赖：
  - `torch`
  - `torch.nn`
- 上游：
  - 视觉编码器、时序编码器或其他需要卷积残差块的模型模块。
- 下游：
  - 模型 backbone、时空特征压缩模块、可视化或训练脚本中的模块实例化逻辑。

## 9. 注意事项

- $\lfloor channels / 2 \rfloor$：标准瓶颈块和时空解耦块使用整除得到中间通道数。若 `channels` 为奇数，中间通道会向下取整，仍可运行，但通道压缩比例不再精确等于 $1/2$。
- 残差相加要求输入输出 shape 完全一致。修改 kernel、stride、padding 或通道数时必须重新检查 shape。
- `DepthwiseSeparableBlock` 使用 `groups=channels`，因此 `in_channels` 和 `out_channels` 都必须等于 `channels`。
- RMSNorm 的 `eps` 不建议设为 0，否则全零输入或极小幅值输入可能产生数值问题。
- 3D 模块的 `D` 维在不同上游中可能代表时间或深度，使用时必须在上游文档中明确语义。
- `__main__` 中的自测会创建 `channels=768`、`D=16`、`H=W=32` 的随机输入，运行 3D block 时显存占用较高；低显存环境可调小这些参数。

## 10. 本地自测

该文件包含 `if __name__ == "__main__":` 自测入口，可从仓库根目录运行：

```powershell
python model/residual_block.py
```

自测覆盖：

- `ResidualBlock`
- `DepthwiseSeparableBlock`
- `ResidualBlock3d`
- `SpatioTemporalResidualBlock3d`
- `DepthwiseSeparableBlock3d`

自测会打印输入 shape、输出 shape 和参数量，用于确认模块能前向运行并保持 shape 不变。

## 11. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-01 | 1os3_Codex | 根据 `model/residual_block.py` 当前实现补全代码文档示例。 |
| 2026-06-01 | 1os3_Codex | 同步源码规范化改动：中文 docstring、公开接口声明、参数校验和依赖清理。 |
| 2026-06-01 | 1os3_Codex | 增补详细摘要内容，并将摘要全部纳入完整文档。 |
| 2026-06-01 | 1os3_Codex | 增加独立接口使用规范章节，并将数学表达统一为 LaTeX。 |
| 2026-06-01 | 1os3_Codex | 补充 AI 变更署名规范，并同步维护记录格式。 |
