# close_loop/monodrive/camera_config.py（摘要）

## 文件职责

保存 B2D ``CAM_FRONT`` 相机 FOV、1600×900 内参与 cam2ego 外参常量，并提供针孔内参计算 helper。

## 公开接口

| 名称 | 说明 |
| --- | --- |
| `B2D_CAMERA_FOV_DEG` | 70° 水平 FOV。 |
| `B2D_CAM2EGO_XYZ` | `(0.8, 0.0, 1.6)` m 车体安装位。 |
| `pinhole_intrinsics` | `(width, height, fov_deg) -> (fx, fy, cx, cy)`。 |

## 使用约束

- 闭环默认 FOV 与 cam2ego 必须与本模块常量一致，除非显式覆盖 CLI。
- 降低渲染分辨率时保持 FOV 不变即可维持与训练相同的视场；``--camera-full-res`` 同时匹配像素级内参。

## 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | AI 完成：新增 B2D 相机参数模块。 |
