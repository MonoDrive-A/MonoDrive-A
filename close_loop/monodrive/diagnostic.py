"""闭环单步推理诊断 dump。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("monodrive_diag")


def _frames_buf_to_uint8_mosaic(
    frames: torch.Tensor,
    ncols: int = 8,
) -> np.ndarray:
    """``(T, 3, H, W)`` 或 ``(3, T, H, W)`` float [0,1] → uint8 拼图。"""
    x = frames.detach().cpu().clamp(0, 1)
    if x.ndim != 4:
        raise ValueError(f"frames 期望 4 维，实际 {tuple(x.shape)}")

    if int(x.shape[1]) == 3:
        time_first = True
        t_count = int(x.shape[0])
        h, w = int(x.shape[2]), int(x.shape[3])
    elif int(x.shape[0]) == 3:
        time_first = False
        t_count = int(x.shape[1])
        h, w = int(x.shape[2]), int(x.shape[3])
    else:
        raise ValueError(f"无法识别 frames 布局：{tuple(x.shape)}")

    rows = []
    for r0 in range(0, t_count, ncols):
        row_imgs = []
        for i in range(r0, min(r0 + ncols, t_count)):
            if time_first:
                rgb = (x[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            else:
                rgb = (x[:, i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            row_imgs.append(rgb)
        while len(row_imgs) < ncols:
            row_imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(rows, axis=0)


def _safe_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    return plt


def dump_replan_snapshot(
    out_dir: Path,
    tick: int,
    frames_past: torch.Tensor,
    ego_motion: torch.Tensor,
    target_point: torch.Tensor,
    trajs_phys: np.ndarray,
    probs: np.ndarray,
    winner_idx: int,
    goal_local_xy: np.ndarray,
    v_kmh: float,
    goal_d_m: float,
    goal_refreshed: bool,
    extra_text: str = "",
    save_npz: bool = True,
) -> Path:
    """落盘一张诊断 PNG（+ 可选 npz），返回 PNG 路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt = _safe_matplotlib()

    motion = ego_motion.detach().cpu().float().numpy()
    target = target_point.detach().cpu().float().numpy()
    if motion.shape != (3,):
        raise ValueError(f"ego_motion 形状非 (3,)：{motion.shape}")
    if target.shape != (2,):
        raise ValueError(f"target_point 形状非 (2,)：{target.shape}")

    mosaic = _frames_buf_to_uint8_mosaic(frames_past, ncols=8)
    n_traj = int(probs.shape[0])

    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.8, 0.8, 1.2, 2.8], hspace=0.42)

    win_prob = float(probs[winner_idx]) if 0 <= winner_idx < n_traj else 0.0
    fig.suptitle(
        f"MonoDrive closed-loop replan tick={tick} | win={winner_idx} "
        f"p[win]={win_prob:.3f} "
        f"| v={v_kmh:.1f} km/h | goal_d={goal_d_m:.1f} m"
        + (" | goal↻" if goal_refreshed else ""),
        fontsize=13, y=0.995,
    )

    ax_m = fig.add_subplot(gs[0, 0])
    ax_m.imshow(mosaic, aspect="auto")
    ax_m.set_title("frames_past 8 frames (oldest left → newest right)", fontsize=9)
    ax_m.axis("off")

    ax_t = fig.add_subplot(gs[1, 0])
    ax_t.axis("off")
    tbl = ax_t.table(
        cellText=[
            ["vx", f"{motion[0]:+.3f}"],
            ["vy", f"{motion[1]:+.3f}"],
            ["w", f"{motion[2]:+.3f}"],
            ["target_x", f"{target[0]:+.3f}"],
            ["target_y", f"{target[1]:+.3f}"],
        ],
        colLabels=["field", "value"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)
    ax_t.set_title(
        f"ego_motion (vx, vy, w) + target_point  |  ||target||={np.linalg.norm(target):.2f} m",
        fontsize=10,
    )

    ax_p = fig.add_subplot(gs[2, 0])
    xs = np.arange(n_traj)
    ax_p.bar(xs, probs, width=0.6, color="#4C72B0", label="prob (top-k)")
    ax_p.set_xticks(xs)
    ax_p.set_xlabel("trajectory index (top-k)")
    ax_p.set_ylim(0, max(1.02, float(probs.max()) * 1.05 if n_traj else 1.02))
    ax_p.legend(loc="upper right", fontsize=8)

    ax_x = fig.add_subplot(gs[3, 0])
    winner_in_top = int(np.where(np.arange(n_traj) == winner_idx)[0].size > 0)
    for k in range(n_traj):
        xy = trajs_phys[k, :, :2]
        is_winner = (k == winner_idx) or (
            not winner_in_top and k == 0 and winner_idx >= n_traj
        )
        if is_winner or k == np.argmax(probs):
            ax_x.plot(xy[:, 0], xy[:, 1], color="#C44E52", lw=3.0, zorder=5,
                      label=f"winner #{winner_idx} p={win_prob:.3f}")
        else:
            ax_x.plot(xy[:, 0], xy[:, 1], color="#7f7f7f",
                      alpha=0.35 + 0.5 * float(probs[k]), lw=1.0, zorder=3)
    ax_x.scatter([0], [0], c="#222", marker="x", s=80, zorder=6, label="ego(now)")
    ax_x.scatter([goal_local_xy[0]], [goal_local_xy[1]],
                 c="#FF7F0E", marker="*", s=180,
                 edgecolors="k", linewidths=0.5, zorder=7,
                 label=f"target ({goal_local_xy[0]:.1f},{goal_local_xy[1]:.1f})")
    ax_x.set_aspect("equal", adjustable="datalim")
    ax_x.grid(True, alpha=0.3)
    ax_x.legend(loc="upper right", fontsize=8)

    if extra_text:
        fig.text(0.01, 0.005, extra_text, fontsize=8, color="#444")

    png_path = out_dir / f"replan_tick{tick:05d}_win{winner_idx:03d}.png"
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    if save_npz:
        npz_path = out_dir / f"replan_tick{tick:05d}.npz"
        np.savez_compressed(
            npz_path,
            tick=np.int64(tick),
            winner=np.int64(winner_idx),
            frames_past=frames_past.detach().cpu().numpy().astype(np.float32),
            ego_motion=motion.astype(np.float32),
            target_point=target.astype(np.float32),
            trajs_phys=trajs_phys.astype(np.float32),
            probs=probs.astype(np.float32),
            goal_local_xy=goal_local_xy.astype(np.float32),
            v_kmh=np.float32(v_kmh),
            goal_d_m=np.float32(goal_d_m),
        )

    logger.info("dumped replan snapshot %s", png_path)
    return png_path


def dump_openloop_snapshot(
    out_dir: Path,
    tick: int,
    frames_past: torch.Tensor,
    ego_motion: torch.Tensor,
    target_point: torch.Tensor,
    v_kmh: float,
    goal_d_m: float,
    goal_refreshed: bool,
) -> Path:
    """落盘一个 ``.pt`` snapshot，便于与开环 H5 样本字段对照。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_past_cpu = frames_past.detach().cpu().float().contiguous()
    if frames_past_cpu.ndim == 4 and int(frames_past_cpu.shape[1]) == 3:
        _, _, height, width = frames_past_cpu.shape
    else:
        _, _, height, width = frames_past_cpu.shape

    motion_cpu = ego_motion.detach().cpu().float().contiguous()
    target_cpu = target_point.detach().cpu().float().contiguous()
    if motion_cpu.shape != (3,):
        raise ValueError(f"ego_motion 形状应为 (3,)，实际 {tuple(motion_cpu.shape)}")
    if target_cpu.shape != (2,):
        raise ValueError(f"target_point 形状应为 (2,)，实际 {tuple(target_cpu.shape)}")

    payload = {
        "images": frames_past_cpu,
        "ego_motion": motion_cpu,
        "target_point": target_cpu,
        "future_trajectory": torch.zeros(6, 2, dtype=torch.float32),
        "tick": int(tick),
        "v_kmh": float(v_kmh),
        "goal_d_m": float(goal_d_m),
        "goal_refreshed": bool(goal_refreshed),
        "source": "carla_closed_loop",
        "image_hw": (int(height), int(width)),
    }
    pt_path = out_dir / f"snapshot_tick{tick:05d}.pt"
    torch.save(payload, pt_path)
    logger.info("dumped open-loop-compatible snapshot %s", pt_path)
    return pt_path
