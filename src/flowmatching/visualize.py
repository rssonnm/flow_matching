"""
visualize.py — Visualization & Animation Utilities for Flow Matching
=====================================================================

Static plots:
  1. Scatter plots: real vs generated samples
  2. Vector field: quiver plot của v_θ(·, t) tại thời điểm t
  3. Trajectory: đường đi của samples từ noise → data
  4. Training curve: loss theo epoch
  5. Flow evolution: density snapshots tại nhiều thời điểm t

GIF Animations:
  6. animate_flow_evolution — density morphing N(0,I) → p_data
  7. animate_trajectories  — sample paths being traced over time
  8. animate_vector_field  — vector field evolving with t
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend cho server/headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Style Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Publication-quality matplotlib defaults
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.size": 10,
    "font.family": "sans-serif",
})

# Color palette
COLORS = {
    "real": "#58a6ff",       # Blue
    "generated": "#f97316",  # Orange
    "trajectory": "#a78bfa", # Purple
    "field": "#34d399",      # Green
    "accent": "#f472b6",     # Pink
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """Tensor → numpy, detach + move to CPU."""
    return x.detach().cpu().numpy()


def _save_fig(fig: plt.Figure, path: str, tight: bool = True):
    """Save figure with tight layout."""
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  💾 Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sample Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_samples(
    real: torch.Tensor,
    generated: torch.Tensor,
    title: str = "Flow Matching Samples",
    save_path: Optional[str] = None,
    max_points: int = 4000,
) -> plt.Figure:
    """
    Scatter plot so sánh real data vs generated samples.

    Parameters
    ----------
    real : Tensor, shape (N, 2)
    generated : Tensor, shape (M, 2)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    real_np = _to_numpy(real[:max_points])
    gen_np = _to_numpy(generated[:max_points])

    # Real data
    axes[0].scatter(
        real_np[:, 0], real_np[:, 1],
        c=COLORS["real"], s=2, alpha=0.5, edgecolors="none"
    )
    axes[0].set_title("Real Data", fontsize=13, fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.2)

    # Generated data
    axes[1].scatter(
        gen_np[:, 0], gen_np[:, 1],
        c=COLORS["generated"], s=2, alpha=0.5, edgecolors="none"
    )
    axes[1].set_title("Generated Samples", fontsize=13, fontweight="bold")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.2)

    # Overlay
    axes[2].scatter(
        real_np[:, 0], real_np[:, 1],
        c=COLORS["real"], s=2, alpha=0.3, edgecolors="none", label="Real"
    )
    axes[2].scatter(
        gen_np[:, 0], gen_np[:, 1],
        c=COLORS["generated"], s=2, alpha=0.3, edgecolors="none", label="Generated"
    )
    axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(fontsize=10, loc="upper right", framealpha=0.5)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    if save_path:
        _save_fig(fig, save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vector Field
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_vector_field(
    model: nn.Module,
    t_values: List[float] = None,
    grid_range: Tuple[float, float] = (-3.0, 3.0),
    grid_size: int = 25,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Quiver plot của learned vector field v_θ(x, t) tại các thời điểm t.

    Vẽ grid of arrows cho thấy hướng và cường độ của flow tại mỗi điểm.

    Parameters
    ----------
    model : nn.Module
    t_values : list[float]
        Các thời điểm t để vẽ.
    grid_range : (float, float)
        Phạm vi grid.
    grid_size : int
        Số điểm mỗi chiều.
    """
    if t_values is None:
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    model.eval()
    n_plots = len(t_values)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Create grid
    x_range = torch.linspace(grid_range[0], grid_range[1], grid_size)
    y_range = torch.linspace(grid_range[0], grid_range[1], grid_size)
    xx, yy = torch.meshgrid(x_range, y_range, indexing="xy")
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)

    for ax, t_val in zip(axes, t_values):
        t_tensor = torch.tensor(t_val, device=device).expand(grid_points.shape[0])
        v = model(grid_points, t_tensor)

        v_np = _to_numpy(v)
        xx_np = _to_numpy(xx)
        yy_np = _to_numpy(yy)

        # Magnitude for coloring
        vx = v_np[:, 0].reshape(grid_size, grid_size)
        vy = v_np[:, 1].reshape(grid_size, grid_size)
        magnitude = np.sqrt(vx**2 + vy**2)

        ax.quiver(
            xx_np, yy_np, vx, vy,
            magnitude,
            cmap="cool",
            alpha=0.8,
            scale=None,
        )

        ax.set_title(f"t = {t_val:.2f}", fontsize=12, fontweight="bold")
        ax.set_xlim(grid_range)
        ax.set_ylim(grid_range)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Learned Vector Field v_θ(x, t)",
        fontsize=14, fontweight="bold", y=1.02
    )

    if save_path:
        _save_fig(fig, save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trajectory Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectory(
    trajectory: torch.Tensor,
    n_show: int = 200,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ đường đi (trajectory) của samples từ noise đến data.

    Parameters
    ----------
    trajectory : Tensor, shape (T+1, B, 2)
        Toàn bộ trajectory từ ODE solver.
    n_show : int
        Số trajectories hiển thị.
    """
    traj_np = _to_numpy(trajectory[:, :n_show, :])  # (T+1, n_show, 2)
    T = traj_np.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Vẽ mỗi trajectory dưới dạng line với gradient màu theo t
    for i in range(min(n_show, traj_np.shape[1])):
        points = traj_np[:, i, :]  # (T+1, 2)
        segments = np.stack(
            [points[:-1], points[1:]], axis=1
        )  # (T, 2, 2)

        # Color gradient: tím (t=0/noise) → cam (t=1/data)
        colors = plt.cm.plasma(np.linspace(0, 1, T - 1))

        lc = LineCollection(segments, colors=colors, alpha=0.3, linewidth=0.5)
        ax.add_collection(lc)

    # Start points (noise) — nhỏ, mờ
    ax.scatter(
        traj_np[0, :, 0], traj_np[0, :, 1],
        c=COLORS["trajectory"], s=3, alpha=0.3, label="t=0 (noise)"
    )

    # End points (generated) — đậm hơn
    ax.scatter(
        traj_np[-1, :, 0], traj_np[-1, :, 1],
        c=COLORS["generated"], s=5, alpha=0.6, label="t=1 (generated)"
    )

    ax.set_title(
        "ODE Trajectories: Noise → Data",
        fontsize=14, fontweight="bold"
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.5)

    # Auto-scale
    all_x = traj_np[:, :, 0].flatten()
    all_y = traj_np[:, :, 1].flatten()
    margin = 0.5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    if save_path:
        _save_fig(fig, save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(
    losses: List[float],
    lrs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training loss (và learning rate nếu có).

    Parameters
    ----------
    losses : list[float]
        Loss mỗi epoch.
    lrs : list[float], optional
        Learning rate mỗi epoch.
    """
    n_subplots = 2 if lrs else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 4))

    if n_subplots == 1:
        axes = [axes]

    # Loss curve
    axes[0].plot(losses, color=COLORS["accent"], linewidth=1.5, alpha=0.9)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("CFM Loss", fontsize=11)
    axes[0].set_title("Training Loss", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.2)
    axes[0].set_yscale("log")

    # LR curve
    if lrs:
        axes[1].plot(lrs, color=COLORS["field"], linewidth=1.5, alpha=0.9)
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Learning Rate", fontsize=11)
        axes[1].set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.2)

    if save_path:
        _save_fig(fig, save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Flow Evolution (Density Snapshots)
# ─────────────────────────────────────────────────────────────────────────────

def plot_flow_evolution(
    trajectory: torch.Tensor,
    n_snapshots: int = 8,
    max_points: int = 2000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grid of density snapshots at different time steps.

    Hiển thị cách phân phối biến đổi từ N(0,I) → p_data.

    Parameters
    ----------
    trajectory : Tensor, shape (T+1, B, 2)
    n_snapshots : int
        Số snapshots để vẽ.
    """
    T_total = trajectory.shape[0]
    indices = np.linspace(0, T_total - 1, n_snapshots, dtype=int)

    n_cols = min(4, n_snapshots)
    n_rows = (n_snapshots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_snapshots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (ax_idx, t_idx) in enumerate(zip(
        range(n_snapshots), indices
    )):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row, col]

        points = _to_numpy(trajectory[t_idx, :max_points])
        t_val = t_idx / max(T_total - 1, 1)

        # Color by local density (KDE-like via scatter alpha)
        color = plt.cm.plasma(t_val)
        ax.scatter(
            points[:, 0], points[:, 1],
            c=[color], s=2, alpha=0.4, edgecolors="none"
        )

        ax.set_title(f"t = {t_val:.2f}", fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

        # Consistent axis limits
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)

    # Hide empty subplots
    for idx in range(n_snapshots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Flow Evolution: N(0,I) → p_data",
        fontsize=14, fontweight="bold", y=1.02
    )

    if save_path:
        _save_fig(fig, save_path)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# GIF ANIMATIONS
# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# 6. Animate Flow Evolution
# ─────────────────────────────────────────────────────────────────────────────

def animate_flow_evolution(
    trajectory: torch.Tensor,
    n_frames: int = 60,
    max_points: int = 2000,
    fps: int = 20,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    GIF animation: phân phối morphing từ N(0,I) → p_data.

    Mỗi frame hiển thị scatter plot tại thời điểm t.
    Màu thay đổi liên tục theo t (plasma colormap).

    Parameters
    ----------
    trajectory : Tensor, shape (T+1, B, 2)
    n_frames : int
        Số frames trong GIF.
    max_points : int
        Số điểm tối đa hiển thị.
    fps : int
        Frames per second.
    save_path : str, optional
        Đường dẫn lưu .gif.
    """
    traj_np = _to_numpy(trajectory[:, :max_points, :])  # (T+1, N, 2)
    T_total = traj_np.shape[0]
    frame_indices = np.linspace(0, T_total - 1, n_frames, dtype=int)

    # Tính axis limits từ toàn bộ trajectory
    all_x = traj_np[:, :, 0].flatten()
    all_y = traj_np[:, :, 1].flatten()
    margin = 0.5
    xlim = (np.percentile(all_x, 1) - margin, np.percentile(all_x, 99) + margin)
    ylim = (np.percentile(all_y, 1) - margin, np.percentile(all_y, 99) + margin)

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter([], [], s=3, alpha=0.5, edgecolors="none")
    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=14, fontweight="bold", verticalalignment="top",
        color="#58a6ff",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#30363d", alpha=0.8)
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_title("Flow Evolution: N(0,I) → p_data", fontsize=14, fontweight="bold")

    def update(frame_idx):
        t_idx = frame_indices[frame_idx]
        t_val = t_idx / max(T_total - 1, 1)
        points = traj_np[t_idx]

        scatter.set_offsets(points)
        color = plt.cm.plasma(t_val)
        scatter.set_color(color)
        time_text.set_text(f"t = {t_val:.3f}")
        return scatter, time_text

    anim = FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=True
    )

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        plt.close(fig)
        print(f"  🎬 Saved: {save_path}")

    return anim


# ─────────────────────────────────────────────────────────────────────────────
# 7. Animate Trajectories
# ─────────────────────────────────────────────────────────────────────────────

def animate_trajectories(
    trajectory: torch.Tensor,
    n_show: int = 300,
    n_frames: int = 60,
    fps: int = 20,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    GIF animation: sample paths được vẽ dần từ noise → data.

    Mỗi frame thêm một đoạn mới cho tất cả trajectories,
    tạo hiệu ứng "drawing" từ t=0 → t=1.

    Parameters
    ----------
    trajectory : Tensor, shape (T+1, B, 2)
    n_show : int
        Số trajectories hiển thị.
    n_frames : int
    fps : int
    save_path : str, optional
    """
    traj_np = _to_numpy(trajectory[:, :n_show, :])  # (T+1, n_show, 2)
    T_total = traj_np.shape[0]
    frame_indices = np.linspace(0, T_total - 1, n_frames, dtype=int)

    # Axis limits
    all_x = traj_np[:, :, 0].flatten()
    all_y = traj_np[:, :, 1].flatten()
    margin = 0.5
    xlim = (np.percentile(all_x, 1) - margin, np.percentile(all_x, 99) + margin)
    ylim = (np.percentile(all_y, 1) - margin, np.percentile(all_y, 99) + margin)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_title("ODE Trajectories: Noise → Data", fontsize=14, fontweight="bold")

    # Pre-draw faded start points
    ax.scatter(
        traj_np[0, :, 0], traj_np[0, :, 1],
        c=COLORS["trajectory"], s=3, alpha=0.15, zorder=1
    )

    # Current positions scatter (animated)
    current_scatter = ax.scatter(
        [], [], c=COLORS["generated"], s=8, alpha=0.7, zorder=3,
        edgecolors="none"
    )

    # Line collections list — accumulated over frames
    line_collections = []

    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=14, fontweight="bold", verticalalignment="top",
        color="#f97316",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#30363d", alpha=0.8)
    )

    def update(frame_idx):
        t_idx = frame_indices[frame_idx]
        t_val = t_idx / max(T_total - 1, 1)

        # Add trail segments from previous frame to current
        if frame_idx > 0:
            prev_idx = frame_indices[frame_idx - 1]
            # Draw segments between prev and current position
            starts = traj_np[prev_idx]  # (n_show, 2)
            ends = traj_np[t_idx]       # (n_show, 2)
            segments = np.stack([starts, ends], axis=1)  # (n_show, 2, 2)

            color = plt.cm.plasma(t_val)
            lc = LineCollection(
                segments, colors=[color], alpha=0.25, linewidth=0.6, zorder=2
            )
            ax.add_collection(lc)
            line_collections.append(lc)

        # Update current positions
        current_scatter.set_offsets(traj_np[t_idx])
        time_text.set_text(f"t = {t_val:.3f}")

        return [current_scatter, time_text] + line_collections

    anim = FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=False  # blit=False vì thêm artists dynamically
    )

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        plt.close(fig)
        print(f"  🎬 Saved: {save_path}")

    return anim


# ─────────────────────────────────────────────────────────────────────────────
# 8. Animate Vector Field
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def animate_vector_field(
    model: nn.Module,
    n_frames: int = 60,
    grid_range: Tuple[float, float] = (-3.0, 3.0),
    grid_size: int = 20,
    fps: int = 20,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    GIF animation: vector field v_θ(x, t) thay đổi khi t: 0 → 1.

    Quiver plot animated, cho thấy cách flow field evolves.

    Parameters
    ----------
    model : nn.Module
    n_frames : int
    grid_range : (float, float)
    grid_size : int
    fps : int
    device : torch.device
    save_path : str, optional
    """
    model.eval()

    # Grid points
    x_range = torch.linspace(grid_range[0], grid_range[1], grid_size)
    y_range = torch.linspace(grid_range[0], grid_range[1], grid_size)
    xx, yy = torch.meshgrid(x_range, y_range, indexing="xy")
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    xx_np = _to_numpy(xx)
    yy_np = _to_numpy(yy)

    # Pre-compute all frames' velocity fields
    t_values = np.linspace(0, 1, n_frames)
    all_vx = []
    all_vy = []
    all_mag = []
    for t_val in t_values:
        t_tensor = torch.tensor(t_val, device=device, dtype=torch.float32)
        t_batch = t_tensor.expand(grid_points.shape[0])
        v = model(grid_points, t_batch)
        v_np = _to_numpy(v)
        vx = v_np[:, 0].reshape(grid_size, grid_size)
        vy = v_np[:, 1].reshape(grid_size, grid_size)
        mag = np.sqrt(vx**2 + vy**2)
        all_vx.append(vx)
        all_vy.append(vy)
        all_mag.append(mag)

    # Tìm global max magnitude cho consistent scaling
    global_max = max(m.max() for m in all_mag)
    scale = global_max * 15  # quiver scale factor

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(grid_range)
    ax.set_ylim(grid_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_title("Learned Vector Field v_θ(x, t)", fontsize=14, fontweight="bold")

    # Initial quiver
    quiver = ax.quiver(
        xx_np, yy_np, all_vx[0], all_vy[0],
        all_mag[0], cmap="cool", alpha=0.8, scale=scale
    )

    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=14, fontweight="bold", verticalalignment="top",
        color="#34d399",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#30363d", alpha=0.8)
    )

    def update(frame_idx):
        quiver.set_UVC(all_vx[frame_idx], all_vy[frame_idx], all_mag[frame_idx])
        time_text.set_text(f"t = {t_values[frame_idx]:.3f}")
        return quiver, time_text

    anim = FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=True
    )

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        plt.close(fig)
        print(f"  🎬 Saved: {save_path}")

    return anim


# ═════════════════════════════════════════════════════════════════════════════
# ALL-IN-ONE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def plot_all(
    real_data: torch.Tensor,
    generated: torch.Tensor,
    trajectory: torch.Tensor,
    losses: List[float],
    lrs: List[float],
    model: nn.Module,
    device: torch.device,
    output_dir: str = "outputs",
    dataset_name: str = "dataset",
):
    """
    Generate all static plots AND GIF animations, save to output_dir.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefix = dataset_name

    # ── Static Plots ──
    print("\n📊 Generating static plots...")

    plot_samples(
        real_data, generated,
        title=f"Flow Matching — {dataset_name}",
        save_path=str(out / f"{prefix}_samples.png"),
    )

    plot_vector_field(
        model, device=device,
        save_path=str(out / f"{prefix}_vector_field.png"),
    )

    plot_trajectory(
        trajectory,
        save_path=str(out / f"{prefix}_trajectories.png"),
    )

    plot_training_curve(
        losses, lrs,
        save_path=str(out / f"{prefix}_training_curve.png"),
    )

    plot_flow_evolution(
        trajectory,
        save_path=str(out / f"{prefix}_flow_evolution.png"),
    )

    # ── GIF Animations ──
    print("\n🎬 Generating GIF animations...")

    animate_flow_evolution(
        trajectory,
        save_path=str(out / f"{prefix}_flow_evolution.gif"),
    )

    animate_trajectories(
        trajectory,
        save_path=str(out / f"{prefix}_trajectories.gif"),
    )

    animate_vector_field(
        model, device=device,
        save_path=str(out / f"{prefix}_vector_field.gif"),
    )

    print(f"\n✅ All plots & animations saved to: {out}/")
