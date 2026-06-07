## MonoDrive

### 1. 基本信息

1. 模型输入最近 8 帧前视单目 RGB 图像，并将时序特征压缩为 4 帧 latent。
2. 输入图像分辨率为 $[H, W]=[288, 512]$，视觉 Token 总数为 2304。
3. 轨迹规划采用“离散轨迹词表 + 残差回归”的形式：
   - 使用 FTS 聚类从数据集中得到 256 条轨迹；
   - 至少保留 1 条轨迹表示“静止”；
   - 轨迹采样频率为 2Hz，每条轨迹包含 6 个轨迹点；
   - 模型在预测轨迹词表概率的同时，回归残差来修正被选中的轨迹。
4. 模型采用统一序列建模，并使用全矢量化的场景表征。
5. 主干网络包含 12 层 Transformer。
6. 模型只使用前向单目图像。
7. 项目面向大众化自动驾驶场景，目标是在 24GB 显存内完成训练，并尽量做到 1 天内可训、架构先进。
8. 数据或实验设置采用 B2D。

### 2. 视觉编码器

1. 视觉编码器使用冻结的 DINOv3-ViT-B。
2. 仅使用 DINOv3 输出的图像 Patch 序列。
3. DINOv3 后接 4 层 3D 卷积模块 `SpatioTemporalResidualBlock3d`，用于处理时序信息。
4. 在第 3 层和第 4 层 3D 卷积之间插入一个 $[2, 1, 1]$ 卷积，用于压缩时间维度。
5. 最后使用 1 层 $1 \times 1$ 卷积，将通道维度降到 384。

### 3. Transformer Block 和位置编码

1. Transformer Block 使用 Pre-Norm 结构，归一化层采用 RMSNorm。
2. FFN 结构为：
   $$
   (D \rightarrow 4D)_{Layer1} \rightarrow SwiGLU(4D \rightarrow 2D) \rightarrow (2D \rightarrow D)_{Layer2}
   $$
3. 注意力计算使用 PyTorch 优化后的 SDPA。
4. 注意力头数为 8。
5. 视觉 Token 使用 3D RoPE 位置编码，编码维度为 $[H, W, T]$。
6. 3 个维度的位置坐标都以 0 为中心，并归一化到 $[-1, 1]$。
7. RoPE 基频为 $\theta=100$。
8. 仅前 6 个注意力头使用位置编码，后 2 个注意力头只做内容匹配。
9. 在第 $Index=2, 4, 6, 8, 10$ 层使用模态独立 FFN：
   - 视觉相关 Token：视觉 Token、寄存器 Token；
   - 驾驶相关 Token：检测查询、轨迹查询、目标导航点；
   - 两类 Token 分别进入独立的 FFN 分支。

### 4. 序列组织、感知和规划

#### 4.1 Token 序列

模型输入的 Token 序列由以下部分组成，总长度为 2662：

| Token 类型 | 数量 |
| --- | ---: |
| 视觉 Token | 2304 |
| 寄存器 Token | 4 |
| 检测查询 | 96 |
| 轨迹查询 | 256 |
| 目标导航点 Token | 2 |

每种 Token 都有独立的身份嵌入，包括：轨迹、视觉、寄存器、Agent、Map、Goal。

#### 4.2 感知任务

感知采用全矢量化形式。96 个检测查询进一步划分为：

| 查询类型 | 数量 |
| --- | ---: |
| Agent | 48 |
| Map | 48 |

Agent 查询负责预测动态目标，输出内容包括：

1. Agent Box 的平面位置、尺寸、朝向、平面速度和平面加速度等运动状态；
2. Agent 未来 3 秒位移轨迹；
3. 每个 Agent 预测 4 个运动 Mode；
4. 初始化时，Agent 查询覆盖前方 120 度空间。

Agent 检测类别包括：

```python
["car", "bicycle", "pedestrian"]
```

Map 查询负责预测局部道路几何结构：

1. 每个 Map Token 表示一条局部矢量地图元素；
2. 每条地图元素统一重采样为 100 个均匀分布的点；
3. 模型需要回归车道线点，并预测地图元素类别。

Map 检测类别包括：

```python
["lane_divider", "road_edge", "centerline"]
```

对于没有方向语义的地图元素，例如 `lane_divider` 和 `road_edge`，点序正反等价。训练时自动选择正向或反向中误差更小的一种进行监督。

对于具有行驶方向语义的 `centerline`，可根据任务需要保留方向监督。

模型不检测红绿灯、Stop 标志或 CrossWalk，也不检测 `motorcycle` 类别。

Agent 和 Map 检测任务都包含“无”类别，并使用匈牙利匹配。

#### 4.3 规划任务

轨迹词表的编码流程如下：

1. 对聚类得到的轨迹词表做 Symlog 变换；
2. 将所有轨迹、所有维度共享同一个缩放系数，统一归一化到 $[-1, 1]$；
3. 对归一化后的轨迹使用每维 64 频高频编码；
4. 使用线性层将编码结果映射为轨迹查询。

高频编码按每个未来时间步分别处理 ego 坐标，坐标约定为 `x` 前向、`y` 左向：

$$
q_n
=
\operatorname{MLP}
\left(
\operatorname{Concat}_{t=1}^{T}
\left[
\phi_y(y_{n,t}),
\phi_x(x_{n,t})
\right]
\right)
$$

其中单坐标编码为：

$$
\phi_x(x)
=
\left[
\sin\left(\frac{2\pi x}{10^{0/64}}\right),
\cos\left(\frac{2\pi x}{10^{0/64}}\right),
\dots,
\sin\left(\frac{2\pi x}{10^{63/64}}\right),
\cos\left(\frac{2\pi x}{10^{63/64}}\right)
\right]
$$

`y` 坐标使用同一组频带构造 $\phi_y(y)$。

统一缩放系数很重要，否则不同维度或不同轨迹之间会出现尺度混乱。

目标点编码流程如下：

1. 目标点来自预处理后的未来可达目标候选池 `labels/target_points`，训练时从有效候选中随机抽取 1 个作为 `target_point`；
2. 候选池由当前帧之后的 ego 真实轨迹构造，默认搜索到场景结束，选择距离当前 ego 原点 24-30m 的全部未来点；
3. 若未来轨迹中没有点落入 24-30m，则选择搜索范围内的最远未来点作为兜底目标，避免使用下一帧等过近目标；
4. 将抽取到的目标点映射到 $18 \times 16$ 栅格；
5. 栅格覆盖范围为车辆前方 32m、后方 4m、左右各 32m；
6. 计算目标点到每个栅格位置的米制向量；
7. 对米制向量逐坐标做 Symlog 变换，使卷积输入位于 Symlog 空间；
8. 使用 1 层 $1 \times 1$ 和 1 层 $3 \times 3$ 和 1 层 $2 \times 2$ 卷积处理并下采样；
9. 中间通道维度为 16，下采样后的空间尺寸为 $[9, 8]$；
10. 将结果展平后，通过线性层投影为 2 个目标导航点 Token。

不直接信任 B2D annotation 中的 `x_command_near/y_command_near`、`x_command_far/y_command_far` 或 `x_target/y_target` 作为训练目标点。这些字段可以作为命令或路线元数据保留，但目标点监督必须来自未来实际 ego 轨迹；默认 24-30m 范围与前方 32m 栅格覆盖保持一致。

自车状态不进入 Transformer 主干。

在输出轨迹词表概率前，先编码当前车辆运动状态 $[V_x, V_y, W]$，再与每个轨迹查询相加。相加后的结果输入线性层，用于同时预测：

1. 轨迹词表概率；
2. 对选中轨迹的残差修正。

### 5. 训练和 Loss

#### 5.1 数值空间

所有进入模型的物理量都先做 Symlog 归一化：

$$
Symlog(x)=Sign(x) \times Log(|x|+1)
$$

模型的预测也在 Symlog 空间或概率空间中完成，不直接在物理空间做监督。

以下计算仍然在物理空间完成：

1. Winner 选择；
2. 危险轨迹判断；
3. 用于构造概率标签的 MSE；
4. 匈牙利匹配。

特殊输出约定：

1. 角度直接预测 $[sin(\theta), cos(\theta)]$；
2. Box 的长、宽、高预测 $Log(X+1)$。

#### 5.2 轨迹词表监督

先根据每条词表轨迹与 GT 轨迹之间的 MSE 构造 soft label：

1. 计算每条轨迹与 GT 的 MSE；
2. 对 MSE 取倒数；
3. 将倒数归一化，使最大值为 10；
4. 经过 Softmax 后作为轨迹词表标签；
5. 如果某条轨迹在未来 3 秒内会与其他 Agent 碰撞，则将该轨迹的 logit 强制设为 $-1e9$。

危险轨迹判定必须覆盖未来 3 秒的全部 2Hz 轨迹点。对候选轨迹第 $k$ 个未来点做碰撞判断时，只能使用该未来时刻对应帧的实际 Agent 标签；不能使用当前帧 Agent 标签外推或静态复用来判断未来是否碰撞。也就是说，若当前帧为 $t$，未来监督帧为 `[t+5, t+10, t+15, t+20, t+25, t+30]`，则候选轨迹第 $k$ 点必须与对应 `t + 5k` 帧中的 Agent 标注进行碰撞判定，其中 $k \in [1, 6]$。

公式为：

$$
P_{GT}
=
\frac{
Softmax \left(
Logit_{pre_i}
=
10
\times
\frac{
\frac{1}{MSE_i+\varepsilon}
}{
\max_j\left(\frac{1}{MSE_j+\varepsilon}\right)
},
\quad
Logit_{Traj_{Danger}}=-10^9
\right)
}{
\max_j
\left[
Softmax \left(
Logit_{pre_j}
=
10
\times
\frac{
\frac{1}{MSE_j+\varepsilon}
}{
\max_k\left(\frac{1}{MSE_k+\varepsilon}\right)
},
\quad
Logit_{Traj_{Danger}}=-10^9
\right)
\right]
}
$$

模型输出同样通过 Softmax 归一化。

残差监督只作用在 Winner 轨迹上：

1. 计算 GT 轨迹的 Symlog；
2. 计算 Winner 词表轨迹的 Symlog；
3. 求逐坐标残差；
4. 使用 MSE 监督模型预测的逐坐标 Symlog 残差。

#### 5.3 Agent 轨迹监督

Agent 未来轨迹预测采用 Winner Only 机制。Agent future 的监督目标不是全局 ego 坐标下的绝对轨迹点，而是以当前帧该 Agent 中心为原点的未来位移：

$$
\Delta p^a_k = p^a_{t+k} - p^a_t
$$

其中 $p^a_t$ 和 $p^a_{t+k}$ 都先转换到当前帧 ego 坐标系，坐标轴仍为当前 ego 的 `x` 前向、`y` 左向；只是在数值上减去当前 Agent 中心。因此 Agent future 不是 Agent 自身朝向坐标系，也不随 Agent yaw 旋转。

对于每个 Agent 的多个 Mode：

1. 根据物理空间误差选择 Winner Mode；
2. Winner Mode 的分类标签设为 1；
3. 其他 Mode 作为负样本，标签设为 0；
4. Mode 概率使用交叉熵监督；
5. Winner Mode 的连续轨迹坐标使用 Symlog 空间 MSE 监督。

#### 5.4 检测查询初始化和可见性过滤

检测查询使用特殊初始化，使各类别的初始预测值按照类别特性均匀分布在相机可见区域内。

检测框可见性过滤流程如下：

1. 使用 annotation 中 `sensors.CAM_FRONT.world2cam` 将 world 坐标 3D 顶点变换到前视相机坐标系；
2. 将 CARLA 相机坐标 `[camera_x, camera_y, camera_z]` 转为针孔投影坐标 `[right, down, forward] = [camera_y, -camera_z, camera_x]`；
3. 使用 annotation 中 `sensors.CAM_FRONT.intrinsic` 投影到原始前视图像平面，图像尺寸来自 `image_size_x/image_size_y`；
4. 至少保留 2 个顶点落在前视图像内且 `forward > 0.1m` 的检测框；
5. 若前视 8-bit 深度图存在，则只把深度图对应 2x2 邻域中存在非 255 有效表面的顶点计为有深度支撑；
6. 若缺少有效 3D 角点、缺少相机内外参或投影失败，则该检测框不进入可见 Agent 标签。

这里的深度图只作为有效表面支撑信号，不做米制深度遮挡比较；B2D 样例中的前视深度图为 8-bit PNG，不能在当前实现中无损恢复真实米制深度。

单帧可见性条件满足后，还必须按同一 Agent ID 在 8 帧历史输入窗口内统计可见帧数；默认要求包含当前帧在内的历史窗口中至少 2 帧满足上述单帧可见性条件，才把该 Agent 计入当前样本标签。当前帧中心仍必须位于前向 32m、左右各 32m 范围内。该过滤必须尽量在数据预处理阶段完成；H5 读取端和训练 Dataset 不应再承担投影、深度可见性判断或范围裁剪。

Map 标签也必须在预处理阶段完成局部裁剪和重采样。每条局部 Map 元素统一重采样为 100 个 ego 坐标系 XY 点，只保留前向 32m、左右各 32m 范围内且能使用同一套 `CAM_FRONT.world2cam/intrinsic` 投影到前视图像中的局部元素；缺少相机内外参或投影失败时不应默认视为可见。

模型不包含 Traffic Element 检测头，因此不对红绿灯或 Stop 标志计算检测监督。

#### 5.5 优化器和精度

优化器使用 AdamW：

1. 初始学习率为 $1e-5$；
2. Warmup 为 5000 step；
3. 峰值学习率为 $1e-4$；
4. 最后 5000 step 使用余弦退火。

Loss 初始化时需要保证各项量级一致，可通过数值归一化实现。

精度策略如下：

1. DINOv3 冻结，并在 `no_grad` 下使用 BF16 前向；
2. 3D Conv 和 Transformer 主干使用 BF16 前向与反向；
3. 其他未明确指定 BF16 的部分均使用 FP32。

轨迹词表模块整体强制使用 FP32，包括词表 buffer、高频编码、轨迹查询嵌入 MLP、轨迹词表解码线性头和 Tanh 残差激活；即使外层使用 BF16 autocast 或父模型整体转为 BF16，该模块也应在内部恢复并输出 FP32。

目标点嵌入层整体强制使用 FP32，包括栅格中心 buffer、目标点向量场、3 层卷积、展平后的线性投影和输出目标导航点 Token；即使外层使用 BF16 autocast 或父模型整体转为 BF16，该模块也应在内部恢复并输出 FP32。

使用 FP32 的部分包括但可能不限于：

1. 输入嵌入层；
2. 输出层；
3. Hungarian matching；
4. 轨迹 MSE；
5. 碰撞判断；
6. soft label 构造；
7. 可见性判断；
8. 高频编码；
9. 轨迹词表查询嵌入与解码头；
10. 目标点嵌入层；
11. Loss 监督；
12. 标签构建。

#### 5.6 速度和加速度字段

所有 `v` 和 `a` 字段都必须基于采样轨迹差分计算，不信任数据集直接提供的任何速度或加速度标注。

速度使用平面二分量表示：

$$
v_x=\frac{x_{t+\Delta t}-x_{t-\Delta t}}{2\Delta t}, \quad
v_y=\frac{y_{t+\Delta t}-y_{t-\Delta t}}{2\Delta t}
$$

加速度同样使用平面二分量表示：

$$
a_x=\frac{x_{t+\Delta t}-2x_t+x_{t-\Delta t}}{\Delta t^2}, \quad
a_y=\frac{y_{t+\Delta t}-2y_t+y_{t-\Delta t}}{\Delta t^2}
$$

边界帧或目标缺少同一 `id` 的相邻轨迹时，允许使用单边差分计算速度；加速度缺少前后两侧轨迹时应置零或在 loss mask 中忽略，不能回退到数据集标注。

### 6. 数据集预处理与采样

#### 6.1 原始目录兼容

B2D 原始场景按包含 `anno/` 和 `camera/rgb_front/` 的目录识别。预处理程序必须兼容以下两种结构：

| 结构 | 示例 |
| --- | --- |
| 一级场景目录 | `datasets/SceneName/anno` |
| 二级场景目录 | `datasets/SceneName/SceneName/anno` |

预处理输出按逐场景 H5 储存，默认输出目录为 `data/preprocessed/`，该目录不提交到 Git。

#### 6.2 输入下采样和滑窗

原始 B2D 数据集帧率为 10Hz，模型输入帧率为 5Hz。预处理时每 2 个原始帧取 1 个输入帧：

$$
stride_{input} = \frac{10Hz}{5Hz} = 2
$$

每个训练样本使用最近 8 个 5Hz 前视单目 RGB 帧。若当前原始帧为 `t`，默认历史输入帧为：

```python
[t - 14, t - 12, t - 10, t - 8, t - 6, t - 4, t - 2, t]
```

默认滑窗步长为 1 个 5Hz 模型帧，即 2 个 10Hz 原始帧，时间间隔为 0.2 秒。

#### 6.3 规划标签采样

规划标签保持未来 3 秒、2Hz、6 个轨迹点。若当前原始帧为 `t`，未来监督帧为：

```python
[t + 5, t + 10, t + 15, t + 20, t + 25, t + 30]
```

未来点数为：

$$
K = 3s \times 2Hz = 6
$$

未来轨迹和导航目标统一转换到当前 ego 坐标系，坐标约定为 `x` 前向、`y` 左向，单位 meter。B2D annotation 中 `theta` 转为 world yaw 时使用 $yaw = \theta - \pi/2$，该关系应以 `CAM_FRONT.world2cam` 的前向轴为校验基准；同一 yaw 必须用于轨迹、目标点、Agent 中心、Agent yaw、速度、加速度和 Map 点的 ego 变换。自车当前运动状态 `[V_x, V_y, W]` 由相邻帧轨迹差分计算，不直接使用数据集速度或加速度标注。

未来规划轨迹默认不做平滑，建议保留原始差分轨迹。实测即使 1 次三点核平滑也可能造成轨迹几何失真，尤其会削弱急弯、急停、避让和路口局部行为，因此除非有明确实验记录和可视化复核，不建议开启该机制。若为了消融实验显式开启，使用的三点核为：

$$
p'_k = 0.25p_{k-1} + 0.5p_k + 0.25p_{k+1}
$$

其中当前 ego 原点作为 $p_0$ 参与第一个未来点平滑，最后一个 3 秒端点保持不变。平滑不能改变轨迹采样频率、点数、坐标系或未来碰撞判定使用的实际帧 Agent 标签；开启后必须记录实验口径，并用可视化检查目标点和未来轨迹是否已经偏离真实行为。

#### 6.4 H5 训练字段

逐场景 H5 至少包含以下训练字段：

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `frames/rgb_front` | `[F, 288, 512, 3]` | 去重储存的 5Hz 前视 RGB。 |
| `samples/input_frame_indices` | `[S, 8]` | 每个样本的历史输入帧索引。 |
| `labels/future_trajectory` | `[S, 6, 2]` | ego 坐标系未来规划轨迹。 |
| `labels/ego_motion` | `[S, 3]` | `[V_x, V_y, W]`。 |
| `labels/target_point` | `[S, 2]` | ego 坐标系默认目标点，为候选池首个有效点或最远点兜底，主要用于兼容和可视化。 |
| `labels/target_points` | `[S, 32, 2]` | ego 坐标系未来可达目标候选点，默认选取距离当前 ego 24-30m 的未来轨迹点。 |
| `labels/target_valid` | `[S, 32]` | 目标候选 padding mask；训练读取时随机抽取一个有效候选作为 `target_point`。 |
| `labels/agent_boxes` | `[S, 194, 10]` | padded Agent 运动标签：`[x, y, l, w, h, yaw, v_x, v_y, a_x, a_y]`。 |
| `labels/agent_future_trajectory` | `[S, 194, 6, 2]` | 每个有效 Agent 的未来 3 秒 2Hz 位移，以当前 Agent 中心为原点，坐标轴沿当前 ego 坐标系。 |
| `labels/agent_future_valid` | `[S, 194, 6]` | Agent 未来轨迹逐点有效 mask。 |
| `labels/map_points` | `[S, 60, 100, 2]` | 已裁剪到局部可见范围内的 Map 元素，统一重采样为 100 点。 |
| `labels/map_classes` | `[S, 60]` | Map 类别，padding 为 `-1`。 |
| `labels/map_valid` | `[S, 60]` | Map padding mask。 |

### 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-06 | 1os3_Codex | AI 完成：将模型检测查询改为 96 个，其中 Agent 和 Map 各 48 个，并移除 CrossWalk、红绿灯、Stop 标志和 motorcycle 检测。 |
| 2026-06-05 | 1os3_Codex | AI 完成：将 Agent future 监督口径改为以当前 Agent 为原点、坐标轴沿当前 ego 的未来位移。 |
| 2026-06-04 | 1os3_Codex | AI 完成：将 Agent 可见性准入改为 8 帧历史窗口内至少 2 帧满足单帧可见性条件。 |
| 2026-06-04 | 1os3_Codex | AI 完成：修正 B2D `theta` 到 ego yaw 的换算口径，强调轨迹、Agent 与 Map 必须使用同一 ego 坐标变换。 |
| 2026-06-03 | 1os3_Codex | AI 完成：明确可见性过滤使用 `CAM_FRONT.world2cam/intrinsic`，并收紧投影失败时的 Agent/Map 可见性口径。 |
| 2026-06-03 | 1os3_Codex | AI 完成：将未来轨迹平滑改为默认关闭，并强调通常不建议开启。 |
| 2026-06-03 | 1os3_Codex | AI 完成：强调 Agent/Map 可见性与范围裁剪应在预处理阶段完成，并补充 Agent 未来轨迹与 Map H5 v4 字段。 |
| 2026-06-02 | 1os3_Codex | AI 完成：将目标点机制改为未来 24-30m 可达候选池，训练随机抽取候选；若无候选则使用最远未来点兜底。 |
| 2026-06-02 | 1os3_Codex | AI 完成：记录未来轨迹轻量平滑约束。 |
| 2026-06-02 | 1os3_Codex | AI 完成：强调危险轨迹判定必须使用未来每帧实际 Agent 标签，不能用当前帧 Agent 标签判定未来碰撞。 |
| 2026-06-02 | 1os3_Codex | AI 完成：声明所有速度和加速度标注必须由轨迹差分构造，并将 Agent 标签改为无 `z` 的平面运动状态。 |
| 2026-06-02 | 1os3_Codex | AI 完成：补充 B2D 数据集预处理、滑窗采样、规划标签和 H5 字段约定。 |
