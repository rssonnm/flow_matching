"""
sampler.py — ODE-Based Sampling for Flow Matching
==================================================

Sau khi train v_θ(x, t), sinh samples bằng cách giải ODE:

    dx/dt = v_θ(x, t),    x(0) ~ N(0, I),    t ∈ [0, 1]

Nghiệm x(1) chính là sample từ learned distribution.

Tại sao ODE chứ không phải SDE?
---------------------------------
Flow Matching sinh ra deterministic ODE (Probability Flow ODE),
không có noise term. So với diffusion models (SDE), điều này cho phép:
1. Exact likelihood computation thông qua Change of Variables formula
2. Deterministic sampling (reproducible với cùng seed)
3. Ít bước hơn vì không cần fight against noise

Trajectory
----------
Toàn bộ path {x(t)}_{t=0}^{1} cho thấy cách learned flow biến đổi
noise thành data, rất hữu ích cho visualization và debugging.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .utils import ode_solve


@torch.no_grad()
def sample(
    model: nn.Module,
    n_samples: int,
    data_dim: int = 2,
    n_steps: int = 100,
    method: str = "rk4",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sinh samples từ learned flow.

    Algorithm
    ---------
    1. x₀ ~ N(0, I)  ∈ ℝ^{n_samples × D}
    2. Giải ODE:  dx/dt = v_θ(x, t)  từ t=0 → t=1
    3. Return x(1)

    Parameters
    ----------
    model : nn.Module
        Trained velocity network.
    n_samples : int
        Số samples cần sinh.
    data_dim : int
        Dimension của data space.
    n_steps : int
        Số bước ODE solver.
    method : str
        'euler', 'rk4', hoặc 'dopri5'.
    device : torch.device
    dtype : torch.dtype

    Returns
    -------
    samples : Tensor, shape (n_samples, data_dim)
        Generated samples.
    """
    model.eval()

    # x₀ ~ N(0, I)
    x0 = torch.randn(n_samples, data_dim, device=device, dtype=dtype)

    # Wrapper: v_fn(x, t) — model cần t dạng (B,) tensor
    def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(x.shape[0])
        return model(x, t_batch)

    # Giải ODE: dx/dt = v_θ(x, t), t: 0 → 1
    samples = ode_solve(
        v_fn, x0,
        t_span=(0.0, 1.0),
        n_steps=n_steps,
        method=method,
        return_trajectory=False,
    )

    return samples


@torch.no_grad()
def sample_trajectory(
    model: nn.Module,
    n_samples: int,
    data_dim: int = 2,
    n_steps: int = 100,
    method: str = "rk4",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sinh samples VÀ trả về toàn bộ trajectory.

    Trajectory cho phép visualize cách ODE flow biến đổi Gaussian → data.

    Returns
    -------
    trajectory : Tensor, shape (n_steps+1, n_samples, data_dim)
        trajectory[0]  = x(0) — noise
        trajectory[-1] = x(1) — generated samples
        trajectory[k]  = x(k/n_steps) — intermediate states
    """
    model.eval()

    x0 = torch.randn(n_samples, data_dim, device=device, dtype=dtype)

    def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(x.shape[0])
        return model(x, t_batch)

    trajectory = ode_solve(
        v_fn, x0,
        t_span=(0.0, 1.0),
        n_steps=n_steps,
        method=method,
        return_trajectory=True,
    )

    return trajectory


@torch.no_grad()
def compute_log_likelihood(
    model: nn.Module,
    x: torch.Tensor,
    n_steps: int = 100,
    method: str = "rk4",
) -> torch.Tensor:
    """
    Ước lượng log-likelihood bằng Change of Variables formula.

    Cho ODE dx/dt = v_θ(x,t), CNF Change of Variables:

        log p₁(x₁) = log p₀(x₀) - ∫₀¹ Tr(∂v_θ/∂x) dt

    trong đó:
        p₀ = N(0, I)  →  log p₀(x₀) = -D/2 · log(2π) - ½||x₀||²
        Tr(∂v_θ/∂x) = divergence = Σᵢ ∂vᵢ/∂xᵢ

    Tính divergence bằng Hutchinson's trace estimator:
        Tr(J) ≈ εᵀ · J · ε,    ε ~ N(0, I)
    
    Đây là unbiased estimator với variance O(D).

    Parameters
    ----------
    model : nn.Module
    x : Tensor, shape (B, D)
        Điểm cần tính log-likelihood.
    n_steps : int
    method : str

    Returns
    -------
    log_prob : Tensor, shape (B,)
        Log-likelihood estimates.

    Note
    ----
    Cần backprop qua model → chạy chậm hơn sampling thuần.
    Chỉ dùng cho evaluation, không cho training.
    """
    device = x.device
    dtype = x.dtype
    batch_size, data_dim = x.shape

    # Giải ODE ngược: t=1 → t=0 (từ data về noise)
    # Đồng thời tích lũy divergence

    dt = -1.0 / n_steps  # đi ngược
    x_t = x.clone().requires_grad_(True)
    log_det = torch.zeros(batch_size, device=device, dtype=dtype)

    for i in range(n_steps):
        t_val = 1.0 + i * dt  # 1.0 → 0.0
        t_tensor = torch.tensor(t_val, device=device, dtype=dtype)
        t_batch = t_tensor.expand(batch_size)

        # Enable grad cho Hutchinson estimator
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            v = model(x_t, t_batch)

            # Hutchinson trace estimator
            epsilon = torch.randn_like(x_t)
            # ε^T · (∂v/∂x) bằng vjp
            vjp = torch.autograd.grad(
                outputs=v,
                inputs=x_t,
                grad_outputs=epsilon,
                create_graph=False,
                retain_graph=False,
            )[0]
            # Tr(J) ≈ ε^T · J · ε = (vjp · ε).sum(dim=-1)
            div_estimate = (vjp * epsilon).sum(dim=-1)

        # Euler step ngược
        x_t = x_t.detach() + dt * v.detach()
        log_det += dt * div_estimate.detach()  # tích lũy ∫ div dt (dt < 0)

    # x_t bây giờ ≈ x₀ (noise)
    # log p₀(x₀) = -D/2 · log(2π) - ½||x₀||²
    log_p0 = -0.5 * data_dim * torch.log(
        torch.tensor(2 * 3.141592653589793, device=device, dtype=dtype)
    ) - 0.5 * (x_t ** 2).sum(dim=-1)

    # log p₁(x) = log p₀(x₀) - ∫ div dt
    # Ở đây log_det đã có dấu âm (vì dt < 0 và ta đi ngược)
    log_prob = log_p0 - log_det

    return log_prob
