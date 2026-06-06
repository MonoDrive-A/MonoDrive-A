"""视觉编码器使用的残差卷积模块。

本文件提供 2D / 3D RMSNorm、标准瓶颈残差块、时空解耦 3D 残差块，
以及 ConvNeXt 风格的 depthwise separable 残差块。
"""

import torch
import torch.nn as nn


__all__ = [
    "RMSNorm2d",
    "RMSNorm3d",
    "ResidualBlock",
    "ResidualBlock3d",
    "SpatioTemporalResidualBlock3d",
    "DepthwiseSeparableBlock",
    "DepthwiseSeparableBlock3d",
]


class RMSNorm2d(nn.Module):
    """适用于 2D 特征图的 RMSNorm。

    RMSNorm 只做均方根归一化，不做均值中心化。相比 LayerNorm，
    该实现计算更轻量，适合卷积特征图中的通道归一化。

    Args:
        normalized_shape: 输入特征的通道数。
        eps: 数值稳定项，避免除零。

    Shape:
        输入: [N, C, H, W]
        输出: [N, C, H, W]
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError(f"normalized_shape 必须为正整数，实际为 {normalized_shape}。")
        if eps <= 0:
            raise ValueError(f"eps 必须为正数，实际为 {eps}。")

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 2D 特征图执行 RMSNorm。"""
        rms = x.pow(2).mean(1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        x = self.weight[:, None, None] * x
        return x


class RMSNorm3d(nn.Module):
    """适用于 3D 特征图的 RMSNorm。

    RMSNorm 只做均方根归一化，不做均值中心化。相比 LayerNorm，
    该实现计算更轻量，适合 3D 卷积特征图中的通道归一化。

    Args:
        normalized_shape: 输入特征的通道数。
        eps: 数值稳定项，避免除零。

    Shape:
        输入: [N, C, D, H, W]
        输出: [N, C, D, H, W]
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError(f"normalized_shape 必须为正整数，实际为 {normalized_shape}。")
        if eps <= 0:
            raise ValueError(f"eps 必须为正数，实际为 {eps}。")

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 3D 特征图执行 RMSNorm。"""
        rms = x.pow(2).mean(1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        x = self.weight[:, None, None, None] * x
        return x


class ResidualBlock(nn.Module):
    """2D 瓶颈残差卷积块。

    结构:
        RMSNorm -> 1x1 Conv(C->C/2) -> 3x3 Conv(C/2->C/2)
        -> GELU -> 1x1 Conv(C/2->C) + 残差连接

    Args:
        channels: 输入和输出通道数。

    Shape:
        输入: [N, C, H, W]
        输出: [N, C, H, W]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError(f"channels 必须不小于 2，实际为 {channels}。")

        mid_channels = channels // 2

        self.norm = RMSNorm2d(channels)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 2D 瓶颈残差卷积。"""
        identity = x

        out = self.norm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)

        return out + identity


class ResidualBlock3d(nn.Module):
    """3D 瓶颈残差卷积块。

    结构:
        RMSNorm -> 1x1x1 Conv(C->C/2) -> 3x3x3 Conv(C/2->C/2)
        -> GELU -> 1x1x1 Conv(C/2->C) + 残差连接

    Args:
        channels: 输入和输出通道数。

    Shape:
        输入: [N, C, D, H, W]
        输出: [N, C, D, H, W]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError(f"channels 必须不小于 2，实际为 {channels}。")

        mid_channels = channels // 2

        self.norm = RMSNorm3d(channels)
        self.conv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv3d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 3D 瓶颈残差卷积。"""
        identity = x

        out = self.norm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)

        return out + identity


class SpatioTemporalResidualBlock3d(nn.Module):
    """时空解耦 3D 残差块。

    该模块先用 (1, 3, 3) 卷积建模空间邻域，再用 (3, 1, 1) 卷积建模
    时间或深度邻域。所有卷积都使用 nn.Conv3d，空维度的卷积核大小设为 1。

    Args:
        channels: 输入和输出通道数。

    Shape:
        输入: [N, C, D, H, W]
        输出: [N, C, D, H, W]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError(f"channels 必须不小于 2，实际为 {channels}。")

        mid_channels = channels // 2

        self.norm = RMSNorm3d(channels)
        self.conv_in = nn.Conv3d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_spatial = nn.Conv3d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )
        self.conv_temporal = nn.Conv3d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
        )
        self.act = nn.GELU()
        self.conv_out = nn.Conv3d(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行时空解耦 3D 残差卷积。"""
        identity = x

        out = self.norm(x)
        out = self.conv_in(out)
        out = self.conv_spatial(out)
        out = self.conv_temporal(out)
        out = self.act(out)
        out = self.conv_out(out)

        return out + identity


class DepthwiseSeparableBlock(nn.Module):
    """ConvNeXt 风格 2D depthwise separable 残差块。

    结构:
        7x7 DepthwiseConv(C->C) -> RMSNorm -> 1x1 Conv(C->2C)
        -> GELU -> 1x1 Conv(2C->C) + 残差连接

    Args:
        channels: 输入和输出通道数。
        expansion_ratio: 通道扩展比例。

    Shape:
        输入: [N, C, H, W]
        输出: [N, C, H, W]
    """

    def __init__(self, channels: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels 必须为正整数，实际为 {channels}。")
        if expansion_ratio <= 0:
            raise ValueError(f"expansion_ratio 必须为正整数，实际为 {expansion_ratio}。")

        expanded_channels = channels * expansion_ratio

        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=channels,
            bias=False,
        )
        self.norm = RMSNorm2d(channels)
        self.pwconv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(
            in_channels=expanded_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 ConvNeXt 风格 2D 残差卷积。"""
        identity = x

        out = self.depthwise_conv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        return out + identity


class DepthwiseSeparableBlock3d(nn.Module):
    """ConvNeXt 风格 3D depthwise separable 残差块。

    结构:
        7x7x7 DepthwiseConv(C->C) -> RMSNorm -> 1x1x1 Conv(C->2C)
        -> GELU -> 1x1x1 Conv(2C->C) + 残差连接

    Args:
        channels: 输入和输出通道数。
        expansion_ratio: 通道扩展比例。

    Shape:
        输入: [N, C, D, H, W]
        输出: [N, C, D, H, W]
    """

    def __init__(self, channels: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels 必须为正整数，实际为 {channels}。")
        if expansion_ratio <= 0:
            raise ValueError(f"expansion_ratio 必须为正整数，实际为 {expansion_ratio}。")

        expanded_channels = channels * expansion_ratio

        self.depthwise_conv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=channels,
        )
        self.norm = RMSNorm3d(channels)
        self.pwconv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(
            in_channels=expanded_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 ConvNeXt 风格 3D 残差卷积。"""
        identity = x

        out = self.depthwise_conv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        return out + identity


if __name__ == "__main__":
    batch_size = 2
    channels = 768
    height, width = 32, 32
    depth = 16

    print("测试 ResidualBlock:")
    residual_block = ResidualBlock(channels=channels)
    x = torch.randn(batch_size, channels, height, width)
    output = residual_block(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in residual_block.parameters()):,}")

    print("\n测试 DepthwiseSeparableBlock:")
    dw_block = DepthwiseSeparableBlock(channels=channels)
    output = dw_block(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in dw_block.parameters()):,}")

    print("\n测试 ResidualBlock3d:")
    residual_block_3d = ResidualBlock3d(channels=channels)
    x_3d = torch.randn(batch_size, channels, depth, height, width)
    output = residual_block_3d(x_3d)
    print(f"输入形状: {x_3d.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in residual_block_3d.parameters()):,}")

    print("\n测试 SpatioTemporalResidualBlock3d:")
    st_block_3d = SpatioTemporalResidualBlock3d(channels=channels)
    output = st_block_3d(x_3d)
    print(f"输入形状: {x_3d.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in st_block_3d.parameters()):,}")

    print("\n测试 DepthwiseSeparableBlock3d:")
    dw_block_3d = DepthwiseSeparableBlock3d(channels=channels)
    output = dw_block_3d(x_3d)
    print(f"输入形状: {x_3d.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in dw_block_3d.parameters()):,}")
