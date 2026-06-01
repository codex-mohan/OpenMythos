"""
Muon optimizer — Momentum Orthogonalized by Newton-Schulz.

Reference: Keller Jordan et al. (2024), "Muon: An optimizer for matrix-valued
parameters trained with momentum and Newton-Schulz orthogonalization."

Key properties vs AdamW:
  - No adaptive per-parameter learning rates → lower memory, simpler dynamics
  - Newton-Schulz orthogonalizes the momentum before applying it, which helps
    with training stability for large matrix parameters
  - Pair with μP (maximal update parameterization) for best results
  - Typical training sees 1.5-2× faster convergence than AdamW at the same LR

Usage:
    muon_params  = [p for p in model.parameters() if p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    optimizer = Muon(
        muon_params, lr=0.02, momentum=0.95, weight_decay=0.1,
        adamw_params=adamw_params, adamw_lr=0.004,
    )

The split is conventional: Muon handles 2D+ weight matrices, AdamW handles
1D biases and norms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer

# Maximum leading dimension for Newton-Schulz orthogonalization.
# G@G.T allocates an M×M intermediate.  M ≤ 4096 keeps peak memory < 70 MB.
# Larger matrices (embeddings, huge MoE experts) fall back to plain momentum.
_NS_MAX_LEADING_DIM = 4096


def _newton_schulz_5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to orthogonalize a matrix G so that G @ G.T ≈ I.

    Starting from G normalized by its Frobenius norm:
        G = G / ‖G‖_F

    Then iterate (step times):
        a = G @ G.T
        b = a @ G
        c = b - G
        G = (3/2)·G - (1/2)·c

    After 5 steps this produces a near-orthogonal matrix (up to machine precision
    for well-conditioned inputs).

    Args:
        G:     square or rectangular matrix of shape (M, N)
        steps: number of Newton-Schulz iterations (5 is standard)

    Returns:
        Orthogonalized matrix of the same shape as G
    """
    assert G.ndim == 2, f"Newton-Schulz expects 2D input, got {G.ndim}D"
    G = G / (G.norm() + 1e-8)
    for _ in range(steps):
        a = G @ G.T
        G = 1.5 * G - 0.5 * (a @ G)
    return G


class Muon(Optimizer):
    """
    Muon optimizer — momentum with Newton-Schulz orthogonalization.

    Handles two parameter groups internally:
      - muon_params:  2D+ weight matrices → Muon update (orthogonalized momentum)
      - adamw_params: 1D biases/norms      → AdamW update

    Args:
        muon_params  : iterable of parameters for Muon (typically nn.Linear.weight, etc.)
        lr           : learning rate for Muon parameters
        momentum     : momentum coefficient (default 0.95)
        nesterov     : use Nesterov-style momentum (default True)
        ns_steps     : Newton-Schulz iteration count (default 5)
        weight_decay : weight decay coefficient (applied decoupled, like AdamW)
        adamw_params : optional iterable of 1D parameters to optimize with AdamW
        adamw_lr     : learning rate for AdamW group (default lr/5, since biases
                       typically want smaller LRs than weights)
        adamw_betas  : beta coefficients for AdamW group
        adamw_eps    : epsilon for AdamW group
    """

    def __init__(
        self,
        muon_params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_params=None,
        adamw_lr: float | None = None,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        if adamw_lr is None:
            adamw_lr = lr / 5.0

        param_groups = [
            {
                "params": list(muon_params),
                "lr": lr,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
                "weight_decay": weight_decay,
                "optimizer": "muon",
            }
        ]

        if adamw_params is not None:
            adamw_list = list(adamw_params)
            if adamw_list:
                param_groups.append(
                    {
                        "params": adamw_list,
                        "lr": adamw_lr,
                        "betas": adamw_betas,
                        "eps": adamw_eps,
                        "weight_decay": 0.0,  # biases typically don't get weight decay
                        "optimizer": "adamw",
                    }
                )

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            opt_type = group.get("optimizer", "muon")

            if opt_type == "muon":
                self._step_muon(group)
            elif opt_type == "adamw":
                self._step_adamw(group)

        return loss

    def _step_muon(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if wd > 0:
                p.mul_(1.0 - lr * wd)

            # --- momentum buffer ---
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)

            if nesterov:
                update = grad.add(buf, alpha=momentum)
            else:
                update = buf

            # Orthogonalize the update for 2D+ parameters with bounded leading dim.
            # Embeddings and large MoE experts skip NS and fall back to momentum SGD.
            if update.ndim >= 2:
                original_shape = update.shape
                update_2d = update.view(original_shape[0], -1)
                m, n = update_2d.shape

                # Put the smaller dimension first so G@G.T is small
                if m > n:
                    update_2d = update_2d.T
                    m, n = n, m
                    transposed = True
                else:
                    transposed = False

                if m <= _NS_MAX_LEADING_DIM:
                    update_2d = _newton_schulz_5(update_2d, steps=ns_steps)

                if transposed:
                    update_2d = update_2d.T
                update = update_2d.view(original_shape)

            p.add_(update, alpha=-lr)

    def _step_adamw(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group.get("weight_decay", 0.0)

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if wd > 0:
                p.mul_(1.0 - lr * wd)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(grad)
                state["exp_avg_sq"] = torch.zeros_like(grad)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]

            denom = exp_avg_sq.sqrt().add_(eps)
            step_size = lr / bias_correction1
            p.addcdiv_(exp_avg, denom, value=-step_size)


def split_muon_adamw(
    model: nn.Module,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_wd: float = 0.1,
    adamw_lr: float | None = None,
) -> Muon:
    """
    Convenience: split model parameters into Muon (2D+) and AdamW (1D) groups
    and return a configured Muon optimizer.

    Args:
        model       : the model whose parameters to optimize
        muon_lr     : learning rate for Muon (matrix parameters)
        muon_momentum: momentum for Muon
        muon_wd     : weight decay for Muon parameters
        adamw_lr    : learning rate for AdamW (1D biases/norms); defaults to muon_lr/5

    Returns:
        Configured Muon optimizer
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return Muon(
        muon_params=muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=muon_wd,
        adamw_params=adamw_params,
        adamw_lr=adamw_lr,
    )
