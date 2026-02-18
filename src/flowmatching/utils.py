"""
math_utils.py — Pure-Math ODE Solvers, Interpolants & Embeddings
================================================================

Tất cả được implement thuần từ công thức toán học, không dùng black-box.

ODE Solvers
-----------
Để sinh samples, ta cần giải ODE:   dx/dt = v_θ(x, t)
từ t=0 (noise) đến t=1 (data).

Ba phương pháp:
  1. Euler         — O(h)   — nhanh, cần nhiều bước
  2. Runge-Kutta 4 — O(h⁴)  — cân bằng tốc độ/chính xác
  3. Dormand-Prince (RK45) — adaptive step — chính xác nhất

Interpolants
------------
Conditional probability path (Optimal Transport):
  ψ_t(x₀ | x₁) = (1 - (1-σ_min)t) · x₀  +  t · x₁

  Tại t=0:  ψ₀ = x₀        (noise)
  Tại t=1:  ψ₁ ≈ x₁        (data, nếu σ_min ≈ 0)

Conditional vector field:
  u_t(x₀ | x₁) = dψ_t/dt = x₁ - (1-σ_min) · x₀

Sinusoidal Embedding
--------------------
Encode scalar t ∈ [0,1] thành vector cao chiều bằng Fourier features,
cho phép network học cả low-frequency và high-frequency patterns theo thời gian.
"""

from __future__ import annotations

import math
import torch
from typing import Callable, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Time Sampling
# ─────────────────────────────────────────────────────────────────────────────

def logit_normal_sample(
    shape: torch.Size,
    mean: float = 0.0,
    std: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample time t từ Logit-Normal distribution.

    Algorithm:
        s ~ N(mean, std²)
        t = sigmoid(s) = 1 / (1 + exp(-s))

    Motivation (Stable Diffusion 3, Esser et al. 2024):
    Uniform sampling t ~ U[0,1] đặt quá nhiều trọng số vào các vùng t ≈ 0 và t ≈ 1
    nơi bài toán đơn giản (pure noise hoặc near-data). Logit-Normal concentrat
    samples vào giữa (t ≈ 0.5), nơi dynamics phức tạp nhất cần học nhiều hơn.

    Parameters
    ----------
    shape : torch.Size
    mean : float
        Shift distribution về phía t=0 (mean < 0) hoặc t=1 (mean > 0).
    std : float
        Độ rộng. Std nhỏ → tập trung quanh logistic(mean).

    Returns
    -------
    t : Tensor, shape 'shape'
        Values in (0, 1).
    """
    s = torch.randn(shape, device=device, dtype=dtype) * std + mean
    return torch.sigmoid(s)


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal Time Embedding
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Biến đổi scalar time t thành vector embedding bằng Fourier features.

    Công thức (giống Transformer positional encoding):
        emb[2k]   = sin(t · ω_k)
        emb[2k+1] = cos(t · ω_k)
    với:
        ω_k = exp(-k · ln(10000) / (dim/2))

    Parameters
    ----------
    t : Tensor, shape (B,) hoặc (B,1)
        Giá trị thời gian trong [0, 1].
    dim : int
        Số chiều embedding (phải chẵn).

    Returns
    -------
    Tensor, shape (B, dim)
        Time embedding vectors.

    Mathematical Justification
    --------------------------
    Fourier features tạo ra một tập basis functions {sin(ω_k t), cos(ω_k t)}
    trải đều trên dải tần số từ thấp (k=0) đến cao (k=dim/2-1).
    Điều này cho phép network phân biệt mọi giá trị t ∈ [0,1] thông qua
    projection lên không gian cao chiều — bản chất là kernel trick.
    """
    assert dim % 2 == 0, f"Embedding dim phải chẵn, got {dim}"

    if t.dim() == 1:
        t = t.unsqueeze(-1)  # (B,) → (B, 1)

    half_dim = dim // 2

    # ω_k = exp(-k · ln(10000) / (dim/2))
    # Tạo dải tần số logarithmic từ 1 đến 1/10000
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, device=t.device, dtype=t.dtype)
        / half_dim
    )  # shape: (dim/2,)

    # t · ω_k  →  (B, 1) × (dim/2,) → (B, dim/2)
    args = t * freq.unsqueeze(0)

    # Concatenate sin và cos: (B, dim)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    return embedding


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Probability Path (Optimal Transport Interpolant)
# ─────────────────────────────────────────────────────────────────────────────

def linear_interpolant(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    """
    Interpolant tuyến tính (Optimal Transport conditional path):

        ψ_t(x₀ | x₁)  =  (1 - (1 - σ_min) · t) · x₀  +  t · x₁

    Đây là đường geodesic thẳng trong không gian Wasserstein W₂ giữa
    phân phối nguồn q₀ = N(0,I) và phân phối đích q₁ = p_data.

    Parameters
    ----------
    x0 : Tensor, shape (B, D)
        Samples từ source distribution (noise).
    x1 : Tensor, shape (B, D)
        Samples từ target distribution (data).
    t  : Tensor, shape (B, 1) hoặc (B,)
        Thời gian trong [0, 1].
    sigma_min : float
        Hệ số co nhỏ nhất, đảm bảo tính non-degenerate tại t=1.

    Returns
    -------
    x_t : Tensor, shape (B, D)
        Điểm trên probability path tại thời điểm t.

    Derivation
    ----------
    Với OT plan π(x₀, x₁) = q₀(x₀) · δ(x₁ - T(x₀)), transport map T
    đưa x₀ → x₁ theo đường thẳng. Interpolant là:

        ψ_t = (1 - αt) · x₀ + αt · x₁

    với α(t) = t cho linear schedule.  Thêm σ_min để tránh collapse:

        ψ_t = (1 - (1-σ_min)t) · x₀ + t · x₁
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # (B,) → (B, 1)

    # Hệ số cho noise: giảm từ 1 → σ_min khi t: 0 → 1
    alpha_t = 1.0 - (1.0 - sigma_min) * t

    # ψ_t = α_t · x₀ + t · x₁
    x_t = alpha_t * x0 + t * x1

    return x_t


def conditional_vector_field(
    x0: torch.Tensor,
    x1: torch.Tensor,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    """
    Vector field điều kiện (target cho training):

        u_t(x₀ | x₁)  =  dψ_t/dt  =  x₁ - (1 - σ_min) · x₀

    Đây chính là "velocity" mà model cần học để tái tạo.

    Lưu ý: u_t KHÔNG phụ thuộc vào t — đây là tính chất đặc biệt của
    linear interpolant, khiến target luôn nhất quán bất kể thời điểm.

    Parameters
    ----------
    x0 : Tensor, shape (B, D)
        Noise samples.
    x1 : Tensor, shape (B, D)
        Data samples.
    sigma_min : float
        Hệ số sigma_min.

    Returns
    -------
    u_t : Tensor, shape (B, D)
        Conditional vector field.
    """
    return x1 - (1.0 - sigma_min) * x0


# ─────────────────────────────────────────────────────────────────────────────
# Variance-Preserving (VP) Trigonometric Interpolant
# ─────────────────────────────────────────────────────────────────────────────

def trigonometric_interpolant(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolant lượng giác bảo toàn phương sai (Albergo & Vanden-Eijnden, 2023).

        ψ_t(x₀ | x₁) = cos(πt/2) · x₀ + sin(πt/2) · x₁

    Khác với linear interpolant, đường dẫn này nằm trên "cầu" (sphere) trong
    không gian sample. Nó đảm bảo:
        ||ψ_t||² = ||x₀||² = ||x₁||²  (nếu x₀, x₁ cùng norm)

    Giúp training ổn định hơn vì variance của signal không bị "thắt cổ chai"
    ở giữa t=0.5 như linear interpolant.

    Returns
    -------
    x_t : Tensor
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)

    cos_t = torch.cos(math.pi / 2 * t)
    sin_t = torch.sin(math.pi / 2 * t)

    return cos_t * x0 + sin_t * x1


def trigonometric_vector_field(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Target vector field cho trigonometric interpolant.

        u_t = dψ_t/dt = (π/2) · [-sin(πt/2) · x₀ + cos(πt/2) · x₁]

    Lưu ý: u_t phụ thuộc vào t, khác với linear case!
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)

    sin_t = torch.sin(math.pi / 2 * t)
    cos_t = torch.cos(math.pi / 2 * t)

    return (math.pi / 2) * (-sin_t * x0 + cos_t * x1)


# ─────────────────────────────────────────────────────────────────────────────
# ODE Solvers (Pure PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def euler_step(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Euler method (bậc 1):

        x_{n+1} = x_n + dt · v(x_n, t_n)

    Đây là xấp xỉ Taylor bậc 1 của nghiệm ODE.
    Sai số cục bộ O(dt²), sai số tích lũy O(dt).

    Parameters
    ----------
    v_fn : callable
        Velocity field v_θ(x, t).
    x : Tensor, shape (B, D)
        Trạng thái hiện tại.
    t : Tensor, scalar
        Thời điểm hiện tại.
    dt : float
        Bước thời gian.

    Returns
    -------
    x_next : Tensor, shape (B, D)
    """
    return x + dt * v_fn(x, t)


def rk4_step(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Runge-Kutta bậc 4 (RK4):

        k₁ = v(x,        t       )
        k₂ = v(x + ½dt·k₁, t + ½dt)
        k₃ = v(x + ½dt·k₂, t + ½dt)
        k₄ = v(x +  dt·k₃, t + dt  )

        x_{n+1} = x_n + (dt/6)(k₁ + 2k₂ + 2k₃ + k₄)

    Sai số cục bộ O(dt⁵), sai số tích lũy O(dt⁴).
    Đòi hỏi 4 evaluations per step nhưng chính xác hơn Euler rất nhiều.

    Derivation
    ----------
    RK4 xấp xỉ Taylor expansion đến bậc 4 mà KHÔNG cần tính đạo hàm
    bậc cao.  Thay vào đó, dùng nhiều evaluations tại các điểm trung gian.
    Các hệ số (1/6, 1/3, 1/3, 1/6) được chọn để cancel error terms
    đến O(dt⁴) (matching conditions của Butcher tableau).
    """
    half_dt = 0.5 * dt

    k1 = v_fn(x, t)
    k2 = v_fn(x + half_dt * k1, t + half_dt)
    k3 = v_fn(x + half_dt * k2, t + half_dt)
    k4 = v_fn(x + dt * k3, t + dt)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def midpoint_step(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Explicit Midpoint Method (Runge-Kutta 2nd order).

        k₁ = v(x, t)
        k₂ = v(x + ½dt·k₁, t + ½dt)
        x_{n+1} = x + dt · k₂

    Ưu điểm:
    - Chỉ cần 2 evaluations/step (so với 4 của RK4).
    - Chính xác O(dt²), tốt hơn Euler O(dt).
    - Rất phù hợp cho Flow Matching vì trajectories thường gần thẳng,
      không cần bậc quá cao như RK4 nhưng Euler thì quá tệ.
    """
    half_dt = 0.5 * dt
    k1 = v_fn(x, t)
    k2 = v_fn(x + half_dt * k1, t + half_dt)
    return x + dt * k2


def dopri5_step(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dormand-Prince RK45 — một bước (embedded pair).

    Tính cả nghiệm bậc 5 (chính xác) và bậc 4 (ước lượng sai số).
    Sai số ước lượng = |y5 - y4| dùng cho adaptive step control.

    Butcher Tableau (Dormand-Prince):
    ┌───────────────────────────────────────────┐
    │ 0     |                                   │
    │ 1/5   | 1/5                               │
    │ 3/10  | 3/40       9/40                    │
    │ 4/5   | 44/45     -56/15      32/9         │
    │ 8/9   | 19372/6561 -25360/2187 64448/6561  │
    │       |            -212/729                │
    │ 1     | 9017/3168  -355/33    46732/5247   │
    │       |             49/176    -5103/18656  │
    │ 1     | 35/384      0        500/1113      │
    │       |             125/192  -2187/6784    │
    │       |             11/84                  │
    └───────────────────────────────────────────┘

    Returns
    -------
    y5 : Tensor — nghiệm bậc 5
    error : Tensor — |y5 - y4| cho adaptive stepping
    """
    # Butcher tableau coefficients
    k1 = v_fn(x, t)
    k2 = v_fn(x + dt * (1/5 * k1), t + dt/5)
    k3 = v_fn(x + dt * (3/40 * k1 + 9/40 * k2), t + 3*dt/10)
    k4 = v_fn(x + dt * (44/45 * k1 - 56/15 * k2 + 32/9 * k3), t + 4*dt/5)
    k5 = v_fn(
        x + dt * (19372/6561 * k1 - 25360/2187 * k2
                  + 64448/6561 * k3 - 212/729 * k4),
        t + 8*dt/9
    )
    k6 = v_fn(
        x + dt * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3
                  + 49/176 * k4 - 5103/18656 * k5),
        t + dt
    )

    # Bậc 5 solution (dùng làm kết quả chính)
    y5 = x + dt * (35/384 * k1 + 500/1113 * k3
                   + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)

    # Bậc 4 solution (cho error estimation)
    k7 = v_fn(y5, t + dt)
    y4 = x + dt * (5179/57600 * k1 + 7571/16695 * k3
                   + 393/640 * k4 - 92097/339200 * k5
                   + 187/2100 * k6 + 1/40 * k7)

    error = torch.abs(y5 - y4)

    return y5, error


# ─────────────────────────────────────────────────────────────────────────────
# General ODE Solver
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ode_solve(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t_span: Tuple[float, float] = (0.0, 1.0),
    n_steps: int = 100,
    method: str = "rk4",
    return_trajectory: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> torch.Tensor:
    """
    Giải ODE:  dx/dt = v_fn(x, t)  từ t_start đến t_end.

    Parameters
    ----------
    v_fn : callable
        Vector field v_θ(x, t).  Input: (x: [B,D], t: scalar Tensor) → [B,D].
    x0 : Tensor, shape (B, D)
        Điều kiện ban đầu.
    t_span : (float, float)
        (t_start, t_end).
    n_steps : int
        Số bước (cho fixed-step methods).
    method : str
        'euler', 'rk4', hoặc 'dopri5'.
    return_trajectory : bool
        Nếu True, trả về toàn bộ trajectory (T+1, B, D).
    atol, rtol : float
        Tolerances cho adaptive method (dopri5).

    Returns
    -------
    Nếu return_trajectory=False:
        x_final : Tensor, shape (B, D)
    Nếu return_trajectory=True:
        trajectory : Tensor, shape (T+1, B, D)
    """
    t_start, t_end = t_span
    device = x0.device
    dtype = x0.dtype

    if method == "dopri5":
        return _ode_solve_adaptive(
            v_fn, x0, t_start, t_end,
            return_trajectory=return_trajectory,
            atol=atol, rtol=rtol,
        )

    # ── Fixed-step methods ──
    dt = (t_end - t_start) / n_steps

    if method == "euler":
        step_fn = euler_step
    elif method == "midpoint":
        step_fn = midpoint_step
    else:
        step_fn = rk4_step

    x = x0.clone()
    trajectory = [x.clone()] if return_trajectory else None

    for i in range(n_steps):
        t_current = torch.tensor(
            t_start + i * dt, device=device, dtype=dtype
        )
        x = step_fn(v_fn, x, t_current, dt)

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=0)  # (T+1, B, D)
    return x


def _ode_solve_adaptive(
    v_fn: Callable,
    x0: torch.Tensor,
    t_start: float,
    t_end: float,
    return_trajectory: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_steps: int = 1000,
    safety: float = 0.9,
    min_dt: float = 1e-6,
    max_dt: float = 0.5,
) -> torch.Tensor:
    """
    Adaptive-step ODE solver dùng Dormand-Prince RK45.

    Step size control:
        dt_new = safety · dt · (tol / err)^(1/5)

    với:
        tol = atol + rtol · |x|
        err = max(|y5 - y4|)

    Công thức này đến từ lý thuyết sai số bậc p=5 của RK45:
    nếu sai số hiện tại = C·dt⁵, ta muốn sai số mới ≤ tol,
    nên dt_new = dt · (tol/err)^(1/5).
    """
    device = x0.device
    dtype = x0.dtype

    x = x0.clone()
    t = t_start
    dt = (t_end - t_start) / 10.0  # initial step guess

    trajectory = [x.clone()] if return_trajectory else None

    for _ in range(max_steps):
        if t >= t_end - 1e-10:
            break

        # Đảm bảo không vượt quá t_end
        dt = min(dt, t_end - t)
        dt = max(dt, min_dt)

        t_tensor = torch.tensor(t, device=device, dtype=dtype)
        x_new, error = dopri5_step(v_fn, x, t_tensor, dt)

        # Error norm (per-element tolerance)
        tol = atol + rtol * torch.abs(x)
        err_ratio = (error / tol).max().item()

        if err_ratio <= 1.0:
            # Accept step
            x = x_new
            t += dt
            if return_trajectory:
                trajectory.append(x.clone())

        # Adjust step size: dt_new = safety · dt · (1/err_ratio)^(1/5)
        if err_ratio > 0:
            dt *= safety * (1.0 / max(err_ratio, 1e-10)) ** 0.2
        else:
            dt *= 2.0  # err = 0, double step

        dt = max(min(dt, max_dt), min_dt)

    if return_trajectory:
        return torch.stack(trajectory, dim=0)
    return x
