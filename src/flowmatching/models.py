"""
model.py — Time-Conditioned Velocity Network
=============================================

Neural network v_θ(x, t) xấp xỉ vector field cho Flow Matching.

Kiến trúc
---------
Input:  x ∈ ℝ^D  (data)  +  t ∈ [0,1]  (time)
Output: v ∈ ℝ^D  (velocity / vector field)

Pipeline:
  1. t → Sinusoidal Embedding → ℝ^{d_emb}
  2. [x, emb(t)] → MLP with skip connections → v

Skip Connections
----------------
Sử dụng additive skip connections mỗi 2 layers:

    h_{l+2} = σ(W_{l+2} · σ(W_{l+1} · h_l + b_{l+1}) + b_{l+2}) + h_l

Điều này giúp:
  - Gradient flow tốt hơn (giảm vanishing gradient)
  - Network có thể học identity mapping (nếu cần)
  - Ổn định hơn khi tăng depth

Activation: SiLU (Sigmoid Linear Unit)
---------------------------------------
    SiLU(x) = x · σ(x) = x / (1 + e^{-x})

Smooth, non-monotonic, self-gated. Đã chứng minh tốt hơn ReLU cho
continuous tasks (Ramachandran et al., 2017).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import List

from .utils import sinusoidal_embedding


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Block 2 tầng với skip connection:

        output = SiLU(Linear(SiLU(Linear(x)))) + x

    Khi input_dim ≠ output_dim, dùng linear projection cho skip.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

        # Skip projection nếu dimension thay đổi
        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Layer Normalization (AdaLN)
# ─────────────────────────────────────────────────────────────────────────────

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (Peebles & Xie, 2023).

    y = (1 + scale) · norm(x) + shift

    Scale & shift được regressed từ time embedding.
    Giúp network "modulate" activations dựa trên thời gian.
    """

    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, elementwise_affine=False, eps=eps)

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        # scale, shift: (B, num_features)
        # x: (B, num_features)
        return self.norm(x) * (1 + scale) + shift


class AdaLNResidualBlock(nn.Module):
    """
    Residual Block với AdaLN conditioning.

    Structure:
      x → AdaLN(t) → Linear → SiLU → Linear → + → x
           ↑
      time_emb
    """

    def __init__(self, in_dim: int, out_dim: int, time_emb_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

        # 2 set coefficients: (scale, shift) cho mỗi block
        # Tuy nhiên để đơn giản và hiệu quả cho MLP đơn giản, ta dùng
        # cơ chế gating đơn giản hơn:
        #   h = MLP(x) + time_mlp(t)
        #
        # Nhưng để đúng chuẩn AdaLN, ta cần regress scale/shift từ time_emb.
        self.ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_dim * 2)  # Output [scale, shift]
        )

        # Skip projection
        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()

        # Main layers
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim)
        t_emb: (B, time_emb_dim)
        """
        # Regress adaptive parameters
        # (B, out_dim * 2)
        params = self.ada_lin(t_emb)
        scale, shift = params.chunk(2, dim=1)

        # Standard Pre-Norm ResBlock with AdaLN modulation after norm?
        # DiT uses: x + Gating * Block(AdaLN(x))
        # Với simple MLP, ta dùng cơ chế modulation trực tiếp lên hidden features.

        # 1. Project x (nếu in!=out)
        h_skip = self.skip(x)

        # 2. Main path
        # Linear 1
        h = self.linear1(x)
        h = self.act(h)

        # AdaLN modulation tại middle layer
        # h = h * (1 + scale) + shift
        h = self.norm(h) * (1 + scale) + shift

        # Linear 2
        h = self.linear2(h)
        h = self.act(h)

        return h_skip + h



class VelocityNetwork(nn.Module):
    """
    Time-conditioned MLP cho velocity field v_θ(x, t).

    Architecture:
        1. t  →  sinusoidal_embedding(t, d_emb)  →  MLP_time  →  t_feat
        2. [x, t_feat]  →  ResidualBlock stack  →  Linear  →  v

    Parameters
    ----------
    data_dim : int
        Dimension của dữ liệu (2 cho toy datasets).
    hidden_dims : list[int]
        Kích thước các hidden layers. Ví dụ [256, 256, 256, 256].
    time_embed_dim : int
        Dimension của sinusoidal time embedding (trước MLP_time).
    """

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dims: List[int] = None,
        time_embed_dim: int = 128,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256]

        self.data_dim = data_dim
        self.time_embed_dim = time_embed_dim

        # ── Time embedding network ──
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # ── Main network ──
        # Input now just x, time injected via AdaLN
        input_dim = data_dim

        self.blocks = nn.ModuleList()
        prev_dim = input_dim

        for i in range(0, len(hidden_dims), 2):
            out_dim = hidden_dims[i]
            # Use AdaLN block
            self.blocks.append(
                AdaLNResidualBlock(prev_dim, out_dim, time_embed_dim)
            )
            prev_dim = out_dim

        # Output projection
        self.final_norm = nn.LayerNorm(prev_dim)
        self.output_proj = nn.Linear(prev_dim, data_dim, bias=False)  # Zero initialization implemented below?

        # ── Weight Initialization ──
        self._init_weights()

        # Zero-init final layer (typical for flow matching/diffusion)
        nn.init.zeros_(self.output_proj.weight)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (x, t) → v_θ(x, t).
        """
        # Handle scalar t
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # 1. Time embedding
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_embed(t_emb)

        # 2. Backbone with AdaLN
        h = x
        for block in self.blocks:
            h = block(h, t_emb)

        # 3. Output
        h = self.final_norm(h)
        v = self.output_proj(h)

        return v
