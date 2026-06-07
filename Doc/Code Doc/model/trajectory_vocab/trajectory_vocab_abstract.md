# model/trajectory_vocab/trajectory_vocab.py 摘要

## 1. 文件基本功能

提供模型侧轨迹词表加载、嵌入和解码能力。模块位于 `model/trajectory_vocab/`，与 `trajectory_vocab_256.npz` 同目录；从 TOML 配置读取参数，从 `.npz` 加载词表，使用已归一化字段生成 `[256, 384]` 轨迹查询特征，并将 `[B, 256, 384]` 轨迹 token 解码为轨迹词表 logit 和 Tanh 残差。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `TrajectoryVocabModelConfig` | dataclass | 保存配置文件中的词表、嵌入和解码参数。 |
| `TrajectoryVocabData` | dataclass | 保存加载后的物理词表、Symlog 词表、归一化词表和缩放系数。 |
| `TrajectoryDecoderOutput` | NamedTuple | 保存 `logits` 和 `residuals`。 |
| `TrajectoryVocabularyEmbedding` | class | 将 `trajectory_vocab_normalized` 编码为 384 维轨迹查询。 |
| `TrajectoryVocabularyDecoder` | class | 单层线性层输出 logit 和 Tanh 残差。 |
| `load_trajectory_vocab_config` | function | 读取 `config/trajectory_vocab.toml`。 |
| `load_trajectory_vocabulary` | function | 加载并校验 `.npz` 词表。 |

## 3. 输入输出 Shape 概览

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `trajectory_vocab_normalized` | `[256, 6, 2]` | 嵌入层使用的已归一化词表字段。 |
| 高频编码 | `[256, 1536]` | 每步按 `[phi_y(y), phi_x(x)]` 拼接；每坐标使用 $2\pi / 10^{i/64}$ 的 64 频 sin/cos 编码。 |
| `trajectory_queries` | `[256, 384]` | 轨迹查询特征。 |
| `trajectory_features` | `[B, 256, 384]` | 解码层输入。 |
| `logits` | `[B, 256]` | 未激活轨迹词表 logit，初始输出为 1。 |
| `residuals` | `[B, 256, 6, 2]` | 经 Tanh 后的 Symlog 残差，初始输出为 0。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `load_trajectory_vocab_config(config_path)` | `config_path` 指向 TOML 文件；词表路径必须是项目内相对路径。 |
| `load_trajectory_vocabulary(config)` | `.npz` 必须包含配置指定字段，三个词表字段 shape 必须为 `[256, 6, 2]`。 |
| `TrajectoryVocabularyEmbedding(config, vocabulary)` | 嵌入层只使用 `vocabulary.trajectory_vocab_normalized`；最后一维按 ego `[x, y]` 解释，输出 `[256, 384]`。 |
| `TrajectoryVocabularyDecoder(config)` | 输入必须为 `[B, 256, 384]`；logits 不做激活，residuals 经过 Tanh。 |

## 5. 最小使用示例

在项目根目录执行：

```python
from model.trajectory_vocab import (
    TrajectoryVocabularyDecoder,
    TrajectoryVocabularyEmbedding,
    load_trajectory_vocab_config,
    load_trajectory_vocabulary,
)

config = load_trajectory_vocab_config("config/trajectory_vocab.toml")
vocabulary = load_trajectory_vocabulary(config)
embedding = TrajectoryVocabularyEmbedding(config, vocabulary)
trajectory_queries = embedding()

decoder = TrajectoryVocabularyDecoder(config)
output = decoder(trajectory_queries.unsqueeze(0))
```

## 6. 维护注意事项

- 配置默认值只放在 `config/trajectory_vocab.toml`，不要在实现文件重复写默认值。
- 修改词表数量、未来点数、轨迹维度或 `hidden_dim` 时，必须同步检查 shape 校验、解码输出和代码文档。
- 解码层初始化要求 logits 初始输出 1，Tanh 残差初始输出 0。
- 高频编码和嵌入层输入必须来自 `.npz` 的已归一化词表字段；编码频带为 $2\pi / 10^{i/64}$，每步按 `[phi_y(y), phi_x(x)]` 拼接。
- SwiGLU 激活来自公共 `model/swiglu.py`，不要在本文件重新实现私有激活层。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：同步轨迹词表高频编码公式和 y/x 拼接顺序。 |
| 2026-06-06 | 1os3_Codex | AI 完成：记录轨迹词表嵌入层改为复用公共 SwiGLU。 |
| 2026-06-06 | 1os3_Codex | AI 完成：将摘要文档移动到镜像目录 `doc/Code Doc/model/trajectory_vocab/`。 |
| 2026-06-06 | 1os3_Codex | AI 完成：新增模型侧轨迹词表模块摘要。 |
