"""
config.py — Device Detection & Hyperparameters for Flow Matching
================================================================

Tự động phát hiện Apple MPS (Metal Performance Shaders) trên M-series chip,
CUDA trên GPU NVIDIA, hoặc fallback về CPU.

Tất cả hyperparameters được quản lý qua dataclass để dễ thay đổi và serialize.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Device Detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Ưu tiên:  MPS (Apple Silicon)  →  CUDA  →  CPU

    Trên MacBook Air M4, MPS backend cho phép PyTorch offload tính toán
    sang Apple GPU, nhanh hơn CPU ~3-5x cho tensor operations.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")



@dataclass
class FlowMatchingConfig:
    """
    Toàn bộ hyperparameters cho Flow Matching.

    Attributes
    ----------
    data_dim : int
        Số chiều dữ liệu (mặc định 2 cho toy datasets).
    hidden_dims : List[int]
        Kích thước các hidden layers trong velocity network.
        [256]*4 đủ mạnh cho 2D, tăng lên [512]*6 cho dữ liệu phức tạp hơn.
    time_embed_dim : int
        Số chiều của sinusoidal time embedding.
        128 cung cấp đủ tần số Fourier để encode t ∈ [0,1].
    sigma_min : float
        σ_min trong interpolant:  ψ_t(x|x₁) = (1-(1-σ_min)t)x₀ + tx₁.
        Giá trị nhỏ (1e-4) đảm bảo ψ₁ ≈ x₁ (gần như exact matching).
    lr : float
        Learning rate cho AdamW optimizer.
    weight_decay : float
        L2 regularization strength.
    batch_size : int
        Batch size cho training.  512 phù hợp cho 2D toy datasets trên MPS.
    epochs : int
        Số epochs training.
    n_train_samples : int
        Số samples trong training dataset.
    n_ode_steps : int
        Số bước trong ODE solver khi sampling.
        100 bước Euler cho kết quả đủ tốt; RK4 cần ít hơn (~50).
    ode_method : str
        Phương pháp ODE solver: 'euler', 'rk4', hoặc 'dopri5'.
    n_gen_samples : int
        Số samples sinh ra khi evaluate.
    seed : int
        Random seed cho reproducibility.
    dtype : torch.dtype
        Kiểu dữ liệu. float32 tương thích tốt nhất với MPS.
    """
    # Data
    data_dim: int = 2

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    time_embed_dim: int = 128

    # Flow matching
    sigma_min: float = 1e-4

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Training
    batch_size: int = 512
    epochs: int = 100
    n_train_samples: int = 20_000

    # Sampling / ODE
    n_ode_steps: int = 100
    ode_method: str = "rk4"
    n_gen_samples: int = 4096

    # Reproducibility
    seed: int = 42
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.device = get_device()
