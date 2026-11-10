"""Level 2 invariants of diagnostics.py (conditioning diagnostics of E1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import diagnostics as D


def _quadratic(eigs, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(len(eigs), len(eigs))))
    A = torch.tensor(q @ np.diag(eigs) @ q.T, dtype=torch.float64)
    return A, (lambda v: A @ v)


def test_power_iteration_finds_largest_eigenvalue_of_quadratic() -> None:
    """lambda_max by power iteration with Hessian-vector products (Hessian of 1/2 x^T A x is A)."""
    A, hvp = _quadratic([5.0, 1.0, 0.5, 0.1, 0.01])
    lam = D.power_iteration(hvp, 5, iters=200, seed=0)
    assert lam == pytest.approx(5.0, rel=1e-6)


def test_hutchinson_trace_exact_on_diagonal() -> None:
    """Rademacher probes: z^T A z = tr A exactly for a diagonal A."""
    A = torch.diag(torch.tensor([3.0, 2.0, 1.0, 4.0], dtype=torch.float64))
    tr = D.hutchinson_trace(lambda v: A @ v, 4, probes=5, seed=1)
    assert tr == pytest.approx(10.0, rel=1e-12)


def test_hutchinson_trace_converges_on_dense_quadratic() -> None:
    A, hvp = _quadratic([5.0, 3.0, 1.0, 0.5, 0.2, 0.1], seed=2)
    tr = D.hutchinson_trace(hvp, 6, probes=4000, seed=3)
    assert tr == pytest.approx(float(torch.trace(A)), rel=0.05)


def test_ratio_invariant_to_global_rescaling() -> None:
    """lambda_max / (trace / P) does not change when the whole loss is multiplied by a constant."""
    A, hvp = _quadratic([5.0, 1.0, 0.5, 0.1], seed=4)
    r1 = D.curvature_ratio(hvp, 4, power_iters=300, probes=64, seed=0)
    r2 = D.curvature_ratio(lambda v: 1e6 * hvp(v), 4, power_iters=300, probes=64, seed=0)
    assert r2 == pytest.approx(r1, rel=1e-9)


def test_gradient_share_is_one_for_single_term() -> None:
    """The maximum share one term contributes to the gradient: 1.0 when one term carries all of it."""
    g = {"poisson": np.array([1.0, -2.0]), "n": np.zeros(2), "p": np.zeros(2)}
    s = D.gradient_shares(g, {"poisson": 1.0, "n": 1.0, "p": 1.0})
    assert max(s.values()) == pytest.approx(1.0) and sum(s.values()) == pytest.approx(1.0)


def test_log10_norm_ratio() -> None:
    g = {"a": np.array([100.0, 0.0]), "b": np.array([0.0, 1.0])}
    assert D.log10_norm_ratio(g, {"a": 1.0, "b": 1.0}) == pytest.approx(2.0)
    assert D.log10_norm_ratio(g, {"a": 1.0, "b": 100.0}) == pytest.approx(0.0)
