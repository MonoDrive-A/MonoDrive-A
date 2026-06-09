# close_loop/monodrive/run_closed_loop.py 摘要

## 1. 文件基本功能

Carla 闭环 CLI：连接仿真、生成路线与交通流、挂载相机、驱动 `MonoDriveAgent` 主循环，并可选录制 MP4 / world.debug 可视化。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `parse_args` | function | argparse 定义。 |
| `main` | function | 主入口，返回进程 exit code。 |

## 3. 运行方式

```powershell
python -m close_loop.monodrive.run_closed_loop --help
```

Carla 相关模块在 `main()` 内延迟导入，`--help` 不依赖 `carla` 包。

## 4. 使用规范

- 须先启动 Carla 服务端并启用同步模式（脚本内配置 `fixed_delta_seconds=0.125`）。
- `--checkpoint` 指向 `train.checkpointing` 产物。
- `--list-maps` 仅查询地图后退出。

## 5. 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | 自 `run_jepa.py` 重命名并切换 MonoDriveAgent。 |
