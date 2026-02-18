"""
flow_matching.py — Core Flow Matching Training Logic
=====================================================

Triển khai thuật toán Conditional Flow Matching (CFM) từ:
    Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
    Tong et al., "Improving and Generalizing Flow-Based Generative Models" (2023)

Thuật toán CFM
--------------
Mục tiêu: Học vector field v_θ(x, t) sao cho flow φ_t generated bởi v_θ
biến đổi noise distribution q₀ = N(0,I) thành data distribution q₁ = p_data.

Loss function (Conditional Flow Matching objective):

    L_CFM(θ) = E_{t~U[0,1], x₁~p_data, x₀~N(0,I)} || v_θ(ψ_t(x₀|x₁), t) - u_t(x₀|x₁) ||²

trong đó:
    ψ_t(x₀|x₁) = (1-(1-σ_min)t)·x₀ + t·x₁     (interpolant)
    u_t(x₀|x₁) = x₁ - (1-σ_min)·x₀              (target velocity)

Tại sao CFM hoạt động?
-----------------------
Theorem (Lipman 2023): Nếu p_t là marginal probability path induced bởi
conditional paths ψ_t, và u_t(x) là vector field tương ứng, thì:

    ∇_θ L_FM(θ) = ∇_θ L_CFM(θ)

Nghĩa là: minimize CFM loss (tractable) tương đương với minimize FM loss
(intractable vì cần biết u_t marginal).

Ưu điểm so với Diffusion Models
---------------------------------
1. Simulation-free training (không cần giải ODE trong training)
2. Straight-line interpolant → ODE paths thẳng hơn → ít bước hơn khi sampling
3. Kết nối tự nhiên với Optimal Transport
"""

from __future__ import annotations

import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from scipy.optimize import linear_sum_assignment
from .utils import (
    linear_interpolant, conditional_vector_field,
    trigonometric_interpolant, trigonometric_vector_field,
    logit_normal_sample
)


# ─────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average (EMA)
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Update: θ_ema = decay * θ_ema + (1 - decay) * θ_current
    
    Tại sao?
    - Giảm noise từ SGD batches.
    - Flat minima selection → generalization tốt hơn.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (
                        self.decay * self.shadow[name] 
                        + (1.0 - self.decay) * param.data
                    )
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        """Load EMA weights vào model để evaluation/sampling."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        """Restore original weights từ backup."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup[name])



class FlowMatching:
    """
    Conditional Flow Matching trainer.

    Thuộc tính chính
    ----------------
    sigma_min : float
        Tham số σ_min trong interpolant.
    device : torch.device
    dtype : torch.dtype
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.sigma_min = sigma_min
        self.device = device
        self.dtype = dtype

    def ot_coupling(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mini-batch Optimal Transport Coupling.
        
        Tìm permutation π sao cho tổng khoảng cách ||x0_i - x1_π(i)||² là nhỏ nhất.
        Giải bằng Hungarian algorithm (scipy.optimize.linear_sum_assignment).
        
        Effect:
        - Straightens trajectories (paths không bị chéo nhau).
        - Giảm variance của gradient estimator.
        """
        # 1. Compute Cost Matrix: C_ij = ||x0_i - x1_j||²
        # (B, 1, D) - (1, B, D) → (B, B, D) → norm → (B, B)
        # Efficient way: ||x||² + ||y||² - 2<x,y>
        with torch.no_grad():
            x0_flat = x0.flatten(start_dim=1)
            x1_flat = x1.flatten(start_dim=1)
            
            # Euclidean distance squared
            cost_matrix = torch.cdist(x0_flat, x1_flat, p=2) ** 2
            cost_matrix_np = cost_matrix.cpu().numpy()
            
            # 2. Convert squared distances to linear assignment problem
            # Trả về row_ind, col_ind
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
            
            # 3. Reorder x1 theo optimal permutation
            # col_ind[i] là index của x1 ghép với x0[i]
            x1_aligned = x1[col_ind]
            
            return x0, x1_aligned

    def compute_loss(
        self,
        model: nn.Module,
        x1_batch: torch.Tensor,
        time_sampling: str = "uniform",  # 'uniform' or 'logit_normal'
        path_type: str = "ot_linear",    # 'ot_linear' or 'vp_trig'
        use_ot: bool = False,            # Mini-batch OT coupling
    ) -> torch.Tensor:
        """
        Tính CFM loss với các improvements.
        """
        batch_size = x1_batch.shape[0]

        # Step 1: Sample t
        if time_sampling == "logit_normal":
            t = logit_normal_sample(
                (batch_size, 1), 
                mean=0.0, std=1.0, 
                device=self.device, dtype=self.dtype
            )
        else:
            t = torch.rand(
                batch_size, 1,
                device=self.device, dtype=self.dtype
            )

        # Step 2: Sample x₀ ~ N(0, I)
        x0 = torch.randn_like(x1_batch)
        
        # OT Coupling (Improvement #3)
        if use_ot:
            x0, x1_batch = self.ot_coupling(x0, x1_batch)

        # Step 3 & 4: Interpolate & Target Velocity
        if path_type == "vp_trig":
            # Improvement #2: Trigonometric (Variance Preserving)
            x_t = trigonometric_interpolant(x0, x1_batch, t)
            target = trigonometric_vector_field(x0, x1_batch, t)
        else:
            # Standard Linear OT path
            x_t = linear_interpolant(x0, x1_batch, t, self.sigma_min)
            target = conditional_vector_field(x0, x1_batch, self.sigma_min)

        # Step 5: Model prediction
        predicted = model(x_t, t.squeeze(-1))

        # Step 6: MSE loss
        loss = torch.mean((predicted - target) ** 2)

        return loss

    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_steps: int = 500,
        log_interval: int = 10,
        verbose: bool = True,
        use_ema: bool = True,
        time_sampling: str = "logit_normal",
        path_type: str = "ot_linear",  # 'vp_trig' hay 'ot_linear'
        use_ot: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Training loop hoàn chỉnh.

        Optimizer: AdamW
        ----------------
        AdamW = Adam + decoupled weight decay.
            θ_{t+1} = θ_t - lr · (m̂_t / (√v̂_t + ε) + λ · θ_t)

        So với vanilla Adam, AdamW regularize đúng hơn vì weight decay
        không bị scale bởi adaptive learning rate.

        Scheduler: Cosine Annealing with Warmup
        ----------------------------------------
        Phase 1 (warmup, 0 → warmup_steps):
            lr(step) = lr_base · step / warmup_steps

        Phase 2 (cosine decay, warmup_steps → total_steps):
            lr(step) = lr_min + ½(lr_base - lr_min)(1 + cos(π · progress))

        Warmup giúp ổn định gradient ở đầu training (khi model chưa tốt).
        Cosine decay giúp converge chặt hơn ở cuối.

        Parameters
        ----------
        model : nn.Module
        dataloader : DataLoader
        epochs : int
        lr : float
        weight_decay : float
        warmup_steps : int
        log_interval : int — in loss mỗi N epochs
        verbose : bool

        Returns
        -------
        history : dict
            {'loss': [...], 'lr': [...], 'epoch_time': [...]}
        """
        model.train()
        model.to(self.device)

        # ── Optimizer: AdamW ──
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # ── Scheduler: Cosine with linear warmup ──
        total_steps = epochs * len(dataloader)

        def lr_lambda(step: int) -> float:
            """
            Warmup + Cosine Annealing.

            Returns multiplier cho base lr.
            """
            if step < warmup_steps:
                # Linear warmup: 0 → 1
                return step / max(warmup_steps, 1)
            else:
                # Cosine decay: 1 → 0
                progress = (step - warmup_steps) / max(
                    total_steps - warmup_steps, 1
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ── EMA Init ──
        ema = EMA(model) if use_ema else None

        # ── Training History ──
        history = {
            "loss": [],
            "lr": [],
            "epoch_time": [],
        }

        # ── Training Loop ──
        global_step = 0

        for epoch in range(epochs):
            epoch_losses = []
            t_start = time.time()

            for batch in dataloader:
                x1 = batch[0]  # TensorDataset wraps in tuple

                # Đảm bảo data trên đúng device
                if x1.device != self.device:
                    x1 = x1.to(self.device)

                # Forward + Loss
                loss = self.compute_loss(
                    model, x1, 
                    time_sampling=time_sampling,
                    path_type=path_type,
                    use_ot=use_ot,
                )

                # Backward + Clip + Step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping — ngăn exploding gradients
                # Max norm = 1.0 là convention cho generative models
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                
                # Update EMA
                if ema:
                    ema.update(model)

                epoch_losses.append(loss.item())
                global_step += 1

            # ── Epoch Statistics ──
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t_start

            history["loss"].append(avg_loss)
            history["lr"].append(current_lr)
            history["epoch_time"].append(epoch_time)

            if verbose and (epoch + 1) % log_interval == 0:
                print(
                    f"Epoch {epoch+1:4d}/{epochs} | "
                    f"Loss: {avg_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.2f}s"
                )

        # ── EMA Finalize ──
        if ema:
            # Apply EMA weights to model for inference/saving
            ema.apply_shadow(model)
            if verbose:
                print("  ✅ Applied EMA weights to model.")

        # ── Summary ──
        total_time = sum(history["epoch_time"])
        if verbose:
            print(f"\n✓ Training complete in {total_time:.1f}s")
            print(f"  Final loss: {history['loss'][-1]:.6f}")
            print(f"  Best loss:  {min(history['loss']):.6f}")

        return history
