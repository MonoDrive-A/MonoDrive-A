# config/trajectory_vocab.toml 摘要

## 1. 文件基本功能

集中保存模型侧轨迹词表配置，包括 `.npz` 路径和字段名、词表 shape、384 维轨迹特征、高频编码、SwiGLU 中间维度，以及解码层初始 logit 和残差输出。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `[vocabulary]` | TOML table | 指定词表路径、字段名和 `[256, 6, 2]` shape。 |
| `[embedding]` | TOML table | 指定高频编码、SwiGLU 嵌入层和 `hidden_dim=384`。 |
| `[decoder]` | TOML table | 指定 logit 初始输出 1、残差经 Tanh 后初始输出 0。 |

## 3. 输入输出 Shape 概览

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `trajectory_vocab_normalized` | `[256, 6, 2]` | 模型嵌入层使用的词表字段。 |
| 高频编码 | `[256, 1536]` | 每步按 `[phi_y(y), phi_x(x)]` 拼接；频带为 $2\pi / 10^{i/64}$。 |
| 轨迹查询特征 | `[256, 384]` | 嵌入层输出。 |
| 解码输入 | `[B, 256, 384]` | Transformer 后轨迹 token。 |
| `logits` | `[B, 256]` | 未激活轨迹词表 logit。 |
| `residuals` | `[B, 256, 6, 2]` | Tanh 后 Symlog 残差。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `[vocabulary]` | `path` 必须是项目内相对路径；模型嵌入层只使用 `normalized_key` 指向的字段。 |
| `[embedding]` | `hidden_dim` 同时约束轨迹查询和解码输入维度；高频编码默认使用 `frequency_base=10.0` 和 `frequency_scale=2π`。 |
| `[decoder]` | `residual_output_init_value` 必须在 `(-1, 1)` 内，当前 `residual_activation` 仅支持 `tanh`。 |

## 5. 最小使用示例

在项目根目录执行：

```python
from model.trajectory_vocab import load_trajectory_vocab_config, load_trajectory_vocabulary

config = load_trajectory_vocab_config("config/trajectory_vocab.toml")
vocabulary = load_trajectory_vocabulary(config)
```

## 6. 维护注意事项

- 新增可提交配置统一使用 TOML。
- 不要在实现文件中重复写入本配置文件已有默认值。
- 修改词表数量、点数、维度、高频编码或解码初始化时，必须同步更新 `model/trajectory_vocab/trajectory_vocab.py` 的代码文档。
- `trajectory_vocab_normalized` 是模型嵌入层输入，不能改为物理空间字段。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：同步高频编码配置公式和默认值。 |
| 2026-06-06 | 1os3_Codex | AI 完成：新增模型侧轨迹词表配置摘要。 |
