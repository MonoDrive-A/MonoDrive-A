# close_loop/monodrive/camera_config.py

## 1. 文件职责

集中保存 B2D ``CAM_FRONT`` 相机内外参常量，供 Carla 闭环 spawn 前视 RGB 相机、MP4 投影与文档引用。数值来自 ``datasets/scenes/*/anno/*.json.gz`` 中 ``sensors.CAM_FRONT``。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `B2D_CAMERA_FOV_DEG` | float | 水平 FOV，默认 70°。 |
| `B2D_CAMERA_WIDTH` / `B2D_CAMERA_HEIGHT` | int | 训练采集 1600×900。 |
| `B2D_CAMERA_HW` | tuple | `(H, W) = (900, 1600)`。 |
| `B2D_CAM2EGO_XYZ` | tuple | cam2ego 平移 `(0.8, 0.0, 1.6)` m。 |
| `B2D_CAMERA_FX_1600` 等 | float | 1600×900 针孔内参样例值。 |
| `pinhole_intrinsics` | function | 由 FOV 与分辨率计算 `(fx, fy, cx, cy)`。 |
| `scale_b2d_intrinsics` | function | 将 1600×900 内参线性缩放到其它分辨率。 |

## 3. 坐标系

- B2D ``cam2ego``：ego ``x`` 前 / ``y`` 左 / ``z`` 上。
- Carla 车体：``x`` 前 / ``y`` 右 / ``z`` 上；``cam2ego`` 的 ``y=0`` 时平移可直接用于 ``attach_to`` 传感器。

## 4. 依赖关系

- 标准库 ``math``；不依赖 ``carla``。

## 5. 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | 新增；对齐 datasets B2D CAM_FRONT FOV 70° 与 cam2ego。 |
