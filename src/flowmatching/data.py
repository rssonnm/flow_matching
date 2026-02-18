"""
data.py — 2D Toy Datasets for Flow Matching
============================================

Các dataset 2D dùng để test và visualize Flow Matching.
Mỗi dataset sinh ra point cloud với các cấu trúc hình học khác nhau,
từ đơn giản (Gaussian) đến phức tạp (spirals, checkerboard).

Tất cả dữ liệu được normalize về [-1, 1]² cho training ổn định.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Generators
# ─────────────────────────────────────────────────────────────────────────────

def make_moons(
    n_samples: int = 10000,
    noise: float = 0.05,
) -> torch.Tensor:
    """
    Two half-moons (sklearn-style nhưng implement thuần).

    Mỗi moon là nửa đường tròn:
        Moon 1: (cos(θ), sin(θ))           θ ∈ [0, π]
        Moon 2: (1 - cos(θ), 1 - sin(θ) - 0.5)  θ ∈ [0, π]

    Parameters
    ----------
    n_samples : int
        Tổng số điểm (chia đều 2 moons).
    noise : float
        Standard deviation của Gaussian noise.

    Returns
    -------
    data : Tensor, shape (n_samples, 2)
    """
    n_half = n_samples // 2

    # Moon 1: nửa trên
    theta1 = torch.linspace(0, math.pi, n_half)
    x1 = torch.cos(theta1)
    y1 = torch.sin(theta1)

    # Moon 2: nửa dưới, shift sang phải và xuống
    theta2 = torch.linspace(0, math.pi, n_samples - n_half)
    x2 = 1.0 - torch.cos(theta2)
    y2 = 1.0 - torch.sin(theta2) - 0.5

    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    data = torch.stack([x, y], dim=-1)

    # Thêm Gaussian noise
    data += noise * torch.randn_like(data)

    # Normalize về zero-mean, unit-variance
    data = (data - data.mean(dim=0)) / data.std(dim=0)

    # Shuffle
    perm = torch.randperm(n_samples)
    return data[perm]


def make_circles(
    n_samples: int = 10000,
    noise: float = 0.03,
    factor: float = 0.5,
) -> torch.Tensor:
    """
    Hai đường tròn đồng tâm.

    Vòng ngoài: (cos(θ), sin(θ))        bán kính = 1
    Vòng trong: (f·cos(θ), f·sin(θ))    bán kính = factor

    Parameters
    ----------
    n_samples : int
    noise : float
    factor : float
        Tỷ lệ bán kính vòng trong / vòng ngoài.

    Returns
    -------
    data : Tensor, shape (n_samples, 2)
    """
    n_half = n_samples // 2

    # Vòng ngoài
    theta1 = 2 * math.pi * torch.rand(n_half)
    outer = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=-1)

    # Vòng trong
    theta2 = 2 * math.pi * torch.rand(n_samples - n_half)
    inner = factor * torch.stack([torch.cos(theta2), torch.sin(theta2)], dim=-1)

    data = torch.cat([outer, inner], dim=0)
    data += noise * torch.randn_like(data)

    perm = torch.randperm(n_samples)
    return data[perm]


def make_8gaussians(
    n_samples: int = 10000,
    std: float = 0.1,
    radius: float = 2.0,
) -> torch.Tensor:
    """
    8 Gaussians xếp thành vòng tròn.

    Tâm của 8 Gaussians đặt tại:
        μ_k = radius · (cos(2πk/8), sin(2πk/8)),   k = 0,...,7

    Mỗi Gaussian có covariance σ²I (isotropic).

    Đây là benchmark kinh điển cho generative models vì:
    - Multi-modal: model phải capture đúng 8 modes
    - Mode collapse dễ bị phát hiện
    - Dễ visualize

    Returns
    -------
    data : Tensor, shape (n_samples, 2)
    """
    n_per_mode = n_samples // 8
    centers = []
    for k in range(8):
        angle = 2 * math.pi * k / 8
        centers.append([radius * math.cos(angle), radius * math.sin(angle)])

    all_points = []
    for i, center in enumerate(centers):
        n = n_per_mode if i < 7 else (n_samples - 7 * n_per_mode)
        points = std * torch.randn(n, 2) + torch.tensor(center)
        all_points.append(points)

    data = torch.cat(all_points, dim=0)

    # Normalize
    data = data / (radius + 3 * std)

    perm = torch.randperm(n_samples)
    return data[perm]


def make_2spirals(
    n_samples: int = 10000,
    noise: float = 0.05,
    n_turns: float = 2.0,
) -> torch.Tensor:
    """
    Hai spiral (xoắn ốc) Archimedes đối xứng.

    Spiral Archimedes:  r = a + bθ
    Ở đây dùng:        r = θ / (2π · n_turns)

    Spiral 1:  (r·cos(θ), r·sin(θ))
    Spiral 2:  (-r·cos(θ), -r·sin(θ))    (đối xứng gốc)

    Returns
    -------
    data : Tensor, shape (n_samples, 2)
    """
    n_half = n_samples // 2

    # θ tăng tuyến tính
    theta = torch.linspace(0, 2 * math.pi * n_turns, n_half)
    r = theta / (2 * math.pi * n_turns)

    # Spiral 1
    x1 = r * torch.cos(theta)
    y1 = r * torch.sin(theta)
    spiral1 = torch.stack([x1, y1], dim=-1)

    # Spiral 2 (đối xứng)
    spiral2 = -spiral1.clone()

    data = torch.cat([spiral1, spiral2], dim=0)

    # Thêm noise
    data += noise * torch.randn_like(data)

    # Normalize
    data = (data - data.mean(dim=0)) / data.std(dim=0) * 0.5

    perm = torch.randperm(n_samples)
    return data[perm]


def make_checkerboard(
    n_samples: int = 10000,
    grid_size: int = 4,
) -> torch.Tensor:
    """
    Checkerboard pattern — uniform sampling trên các ô đen.

    Chia không gian [-1,1]² thành grid_size × grid_size ô.
    Sample uniform trên các ô (i,j) thỏa (i+j) % 2 == 0 (ô "đen").

    Returns
    -------
    data : Tensor, shape (n_samples, 2)
    """
    cell_size = 2.0 / grid_size

    # Tìm các ô "đen"
    black_cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                x_min = -1.0 + i * cell_size
                y_min = -1.0 + j * cell_size
                black_cells.append((x_min, y_min))

    n_cells = len(black_cells)
    n_per_cell = n_samples // n_cells

    all_points = []
    for idx, (x_min, y_min) in enumerate(black_cells):
        n = n_per_cell if idx < n_cells - 1 else (n_samples - (n_cells - 1) * n_per_cell)
        x = x_min + cell_size * torch.rand(n)
        y = y_min + cell_size * torch.rand(n)
        all_points.append(torch.stack([x, y], dim=-1))

    data = torch.cat(all_points, dim=0)

    perm = torch.randperm(n_samples)
    return data[perm]


def make_swissroll(
    n_samples: int = 10000,
    noise: float = 0.05,
) -> torch.Tensor:
    """
    Swiss roll 2D projection.

    Parametric curve:
        θ ~ Uniform[1.5π, 4.5π]
        x = θ · cos(θ)
        y = θ · sin(θ)

    Đây là manifold 1D embed trong 2D, tạo hình cuộn.

    Returns
    -------
    data : Tensor, shape (n_samples, 2)
    """
    theta = 1.5 * math.pi + 3.0 * math.pi * torch.rand(n_samples)

    x = theta * torch.cos(theta)
    y = theta * torch.sin(theta)

    data = torch.stack([x, y], dim=-1)
    data += noise * torch.randn_like(data)

    # Normalize
    data = (data - data.mean(dim=0)) / data.std(dim=0) * 0.5

    perm = torch.randperm(n_samples)
    return data[perm]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Registry & DataLoader
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "moons": make_moons,
    "circles": make_circles,
    "8gaussians": make_8gaussians,
    "2spirals": make_2spirals,
    "checkerboard": make_checkerboard,
    "swissroll": make_swissroll,
}


def get_dataset(
    name: str,
    n_samples: int = 10000,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    """
    Lấy dataset theo tên.

    Parameters
    ----------
    name : str
        Tên dataset: 'moons', 'circles', '8gaussians', '2spirals',
        'checkerboard', 'swissroll'.
    n_samples : int
        Số samples.
    device : torch.device
    dtype : torch.dtype
    **kwargs
        Tham số bổ sung cho từng dataset.

    Returns
    -------
    data : Tensor, shape (n_samples, 2), trên device đã chỉ định.
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}"
        )
    data = DATASETS[name](n_samples=n_samples, **kwargs)
    return data.to(device=device, dtype=dtype)


def get_dataloader(
    data: torch.Tensor,
    batch_size: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    """
    Tạo DataLoader từ tensor data.

    Note: Trên MPS, nên dùng pin_memory=False vì data đã ở trên device.
    """
    dataset = TensorDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=False,
    )
