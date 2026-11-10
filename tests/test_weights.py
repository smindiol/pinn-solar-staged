"""Level 2 invariants of weights.py (Eqs. 12-13 and the outer-loop hook of section 5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import weights as W


def _grads(seed=42):
    rng = np.random.default_rng(seed)
    return {"poisson": rng.normal(size=50) * 3.0, "n": rng.normal(size=50) * 1e-4,
            "p": rng.normal(size=50) * 1e3}


def test_lambda_hat_balances_mean_gradient_magnitudes() -> None:
    """r03 section 5, Eq. (12): lambda_hat_i = max|grad L_Poisson| / mean|grad L_i|, hence
    mean|lambda_hat_i grad L_i| = max|grad L_Poisson| for every i other than Poisson."""
    g = _grads()
    lh = W.lambda_hat(g, reference="poisson")
    assert set(lh) == {"n", "p"}
    target = np.max(np.abs(g["poisson"]))
    for k in lh:
        assert np.mean(np.abs(lh[k] * g[k])) == pytest.approx(target, rel=1e-12)


def test_moving_average_update_is_convex_and_has_the_fixed_point() -> None:
    """r03 section 5, Eq. (13): lambda <- (1-alpha) lambda + alpha lambda_hat is a convex
    combination (alpha = 0 keeps, alpha = 1 replaces) and leaves lambda = lambda_hat unchanged."""
    lam, lh = {"n": 2.0, "p": 5.0}, {"n": 10.0, "p": 1.0}
    assert W.ema_update(lam, lh, 0.0) == lam
    assert W.ema_update(lam, lh, 1.0) == lh
    mid = W.ema_update(lam, lh, 0.25)
    assert mid["n"] == pytest.approx(0.75 * 2.0 + 0.25 * 10.0)
    assert W.ema_update(lh, lh, 0.3) == pytest.approx(lh)


def test_poisson_weight_is_the_fixed_reference() -> None:
    """r03 section 5: Poisson is the reference term of Eq. (12) (i in I \\ {Poisson}), so its
    weight is never updated and stays at 1."""
    bw = W.BlockWeights(names=("poisson", "n", "p"), alpha=0.5)
    bw.begin_block()
    bw.end_block(_grads())
    assert bw.values["poisson"] == 1.0
    assert bw.values["n"] != 1.0


def test_weights_change_only_between_blocks() -> None:
    """r03 section 5: lambda is updated only between blocks of iterations, never inside one,
    because a quasi-Newton method needs a fixed objective while it accumulates memory."""
    bw = W.BlockWeights(names=("poisson", "n", "p"), alpha=0.5)
    bw.begin_block()
    inside = dict(bw.values)
    with pytest.raises(RuntimeError, match="inside a block"):
        bw.update_inside_block(_grads())
    assert bw.values == inside
    bw.end_block(_grads())
    assert bw.values != inside
    assert bw.generation == 1


def test_update_notifies_the_optimizer_to_reset_its_memory() -> None:
    """r03 section 5: the memory of the quasi-Newton method is reset at every lambda update;
    BlockWeights calls the registered callbacks exactly once per update."""
    calls = []
    bw = W.BlockWeights(names=("poisson", "n", "p"), alpha=0.5, on_update=[lambda: calls.append(1)])
    for _ in range(3):
        bw.begin_block()
        bw.end_block(_grads())
    assert len(calls) == 3


def test_vanishing_term_gradient_is_refused() -> None:
    """r03 section 5, Eq. (12): the ratio is undefined when mean|grad L_i| = 0; the weights are
    refused instead of inventing a value."""
    g = _grads()
    g["n"] = np.zeros(50)
    with pytest.raises(ValueError, match="mean gradient magnitude"):
        W.lambda_hat(g, reference="poisson")


def test_bayesian_outer_loop_hook_returns_the_best_proposal() -> None:
    """r03 section 5, alternative: weights chosen by an outer loop. The hook asks a proposer
    for candidate weights, evaluates each (a full inner training in practice) and returns the
    lowest objective; with a deterministic proposer and objective the argmin is known."""
    candidates = iter([{"n": 1.0}, {"n": 3.0}, {"n": 2.0}])
    best, history = W.outer_loop_weights(lambda hist: next(candidates),
                                         lambda lam: (lam["n"] - 2.1) ** 2, n_trials=3)
    assert best == {"n": 2.0}
    assert len(history) == 3


def test_term_gradients_match_autograd() -> None:
    """Eq. (12) needs grad_theta L_i per term: the helper returns flat gradients equal to autograd."""
    theta = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64, requires_grad=True)
    L = {"poisson": (theta ** 2).sum(), "n": theta[0] * theta[1]}
    g = W.term_gradients(L, [theta])
    np.testing.assert_allclose(g["poisson"], 2.0 * theta.detach().numpy())
    np.testing.assert_allclose(g["n"], np.array([-2.0, 1.0, 0.0]))
