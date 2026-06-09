# close_loop/monodrive/model_inference.py

## 1. 文件职责

封装 MonoDrive 轨迹词表推理后处理：softmax、词表 Symlog 与 Tanh 残差合成、反 Symlog 到 ego-local 米制轨迹，以及 top-k 候选提取。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrajectoryDecodeResult` | dataclass | 含全词表 `probs`、winner 与 top-k 数组。 |
| `decode_trajectories` | function | 主解码入口。 |
| `decode_winner_trajectory` | function | 强制 winner 索引时的单轨解码。 |
| `inverse_symlog` | function | $\mathrm{sign}(x)\cdot(e^{|x|}-1)$。 |

## 3. 实现逻辑

1. 读取 `trajectory_output.logits[0]` → softmax 得 `(V,)`。
2. 读取 `residuals[0]` 形状 `[V, 6, 2]`。
3. 从 `model.vocabulary` 取 `trajectory_vocab_symlog` 与 `symlog_scale`。
4. top-k：`torch.topk(probs)` 索引词表与残差，反 Symlog 得 `(top_k, 6, 2)`。
5. winner：`argmax(probs)`，必要时单独解码。

## 4. 依赖关系

- `model.backbone.MonoDriveBackbone`, `MonoDriveBackboneOutput`
- `torch`, `numpy`

## 5. 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | 新增文件。 |
