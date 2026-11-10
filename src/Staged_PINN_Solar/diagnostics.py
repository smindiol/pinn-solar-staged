"""Conditioning diagnostics of E1, all dimensionless.

(a) the maximum share one term contributes to the gradient, and log10 of the largest to the smallest
per-term gradient norm; (b) lambda_max of the Hessian of the total loss by power iteration with
Hessian-vector products divided by the mean trace estimated by Hutchinson (tr / P), which does not
change under a global rescaling of the loss. Power iteration converges to the eigenvalue of largest
magnitude; with a non-convex loss that is the quantity the protocol's threat-to-validity note warns about.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r03.md",
    "sections": ["4", "6"],
    "equations": ["11"],
    "experiments": ["E1"],
    "invariants": [
        "power_iteration_finds_largest_eigenvalue_of_quadratic",
        "hutchinson_trace_exact_on_diagonal",
        "hutchinson_trace_converges_on_dense_quadratic",
        "ratio_invariant_to_global_rescaling",
        "gradient_share_is_one_for_single_term",
        "log10_norm_ratio",
    ],
}

HVP = Callable[[torch.Tensor], torch.Tensor]


def power_iteration(hvp: HVP, dim: int, iters: int = 20, seed: int = 0) -> float:
    """Rayleigh quotient of the dominant eigenvector after `iters` Hessian-vector products."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g, dtype=torch.float64)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        w = hvp(v)
        lam = float(v @ w)
        n = float(w.norm())
        if n == 0.0:
            return 0.0
        v = w / n
    return lam


def hutchinson_trace(hvp: HVP, dim: int, probes: int = 10, seed: int = 0) -> float:
    """Mean of z^T H z over Rademacher probes z (an unbiased estimate of tr H)."""
    g = torch.Generator().manual_seed(seed + 1)
    acc = 0.0
    for _ in range(probes):
        z = (torch.randint(0, 2, (dim,), generator=g) * 2 - 1).to(torch.float64)
        acc += float(z @ hvp(z))
    return acc / probes


def curvature_ratio(hvp: HVP, dim: int, power_iters: int = 20, probes: int = 10, seed: int = 0) -> float:
    """lambda_max / (trace / P): invariant to a global rescaling of the loss."""
    lam = power_iteration(hvp, dim, power_iters, seed)
    mean_trace = hutchinson_trace(hvp, dim, probes, seed) / dim
    return lam / mean_trace if mean_trace != 0.0 else float("nan")


def _weighted_norms(grads: dict, lam: dict) -> dict:
    return {k: float(lam.get(k, 1.0)) * float(np.linalg.norm(g)) for k, g in grads.items()}


def gradient_shares(grads: dict, lam: dict) -> dict:
    """share_i = ||lambda_i grad L_i|| / sum_k ||lambda_k grad L_k||."""
    norms = _weighted_norms(grads, lam)
    total = sum(norms.values())
    return {k: (v / total if total > 0 else float("nan")) for k, v in norms.items()}


def log10_norm_ratio(grads: dict, lam: dict) -> float:
    """log10 of the largest over the smallest per-term gradient norm (weighted by lambda)."""
    norms = [v for v in _weighted_norms(grads, lam).values()]
    lo = min(norms)
    return float("inf") if lo == 0.0 else float(np.log10(max(norms) / lo))
