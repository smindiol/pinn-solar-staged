"""Level 2 invariants of optimizers.py (Eqs. 14-17)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from Staged_PINN_Solar.optimizers import LBFGS, SOAP

D = torch.float64


def _quadratic(n=6, seed=42):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=D))
    A = Q @ torch.diag(torch.logspace(0, 3, n, dtype=D)) @ Q.T
    A = 0.5 * (A + A.T)
    return A, (lambda x: (0.5 * x @ A @ x, A @ x))


def test_lbfgs_inverse_hessian_satisfies_the_secant_condition() -> None:
    """r02 section 6, Eq. (15): after storing the pair (s_k, y_k), the limited-memory
    approximation maps the newest gradient difference onto the step, H_{k+1} y_k = s_k."""
    A, _ = _quadratic()
    g = torch.Generator().manual_seed(7)
    opt = LBFGS(memory=5)
    for _ in range(4):
        s = torch.randn(6, generator=g, dtype=D)
        y = A @ s
        opt.update(s, y)
        torch.testing.assert_close(opt.apply_inverse_hessian(y), s, rtol=1e-10, atol=1e-12)


def test_lbfgs_memory_is_bounded_and_reset_clears_it() -> None:
    """r02 section 5-6: the memory keeps the last m pairs and is reset when lambda changes."""
    A, _ = _quadratic()
    g = torch.Generator().manual_seed(7)
    opt = LBFGS(memory=3)
    for _ in range(6):
        s = torch.randn(6, generator=g, dtype=D)
        opt.update(s, A @ s)
    assert opt.history_size == 3
    opt.reset()
    assert opt.history_size == 0
    v = torch.randn(6, generator=g, dtype=D)
    torch.testing.assert_close(opt.apply_inverse_hessian(v), v)  # H_0 = identity again


def test_lbfgs_step_decreases_the_objective_monotonically() -> None:
    """r02 section 6, Eq. (14): theta_{k+1} = theta_k - a_k H_k grad with a_k from a line
    search; the objective must not increase along the iterations."""
    _, fun = _quadratic()
    opt = LBFGS(memory=5)
    x, info = opt.minimize(torch.ones(6, dtype=D), fun, max_iter=30, gtol=1e-12)
    values = info["history"]
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:]))
    assert values[-1] < 1e-6 * values[0]


def test_lbfgs_reset_on_objective_change_keeps_descending() -> None:
    """r02 section 5: lambda changes between blocks and the memory is reset; the second block
    started from the first block's endpoint must still decrease the *new* objective."""
    A, fun1 = _quadratic()
    fun2 = lambda x: (0.5 * x @ (2.0 * A) @ x + x.sum(), 2.0 * A @ x + 1.0)
    opt = LBFGS(memory=5)
    x, _ = opt.minimize(torch.ones(6, dtype=D), fun1, max_iter=10)
    opt.reset()
    f0 = fun2(x)[0].item()
    x2, info = opt.minimize(x, fun2, max_iter=20)
    assert fun2(x2)[0].item() < f0


def test_soap_covariances_are_ema_of_gram_matrices() -> None:
    """r02 section 6, Eq. (16): P_L and P_R are exponential moving averages of G G^T and
    G^T G; they are symmetric positive semi-definite."""
    g = torch.Generator().manual_seed(42)
    opt = SOAP(lr=1e-2, shampoo_beta=0.9, precondition_frequency=10 ** 9)
    W = torch.randn(4, 3, generator=g, dtype=D)
    G1, G2 = (torch.randn(4, 3, generator=g, dtype=D) for _ in range(2))
    W = opt.step(W, G1)
    W = opt.step(W, G2)
    PL = 0.9 * (0.1 * G1 @ G1.T) + 0.1 * G2 @ G2.T
    PR = 0.9 * (0.1 * G1.T @ G1) + 0.1 * G2.T @ G2
    torch.testing.assert_close(opt.state.P_L, PL)
    torch.testing.assert_close(opt.state.P_R, PR)
    assert torch.linalg.eigvalsh(opt.state.P_L).min() > -1e-12


def test_soap_rotation_matrices_are_orthogonal() -> None:
    """r02 section 6, Eq. (17): Q_L, Q_R are eigenvector matrices of the symmetric P_L, P_R,
    hence orthogonal: Q^T Q = I."""
    g = torch.Generator().manual_seed(42)
    opt = SOAP(lr=1e-2, precondition_frequency=1)
    W = torch.randn(5, 4, generator=g, dtype=D)
    for _ in range(3):
        W = opt.step(W, torch.randn(5, 4, generator=g, dtype=D))
    I5, I4 = torch.eye(5, dtype=D), torch.eye(4, dtype=D)
    torch.testing.assert_close(opt.state.Q_L.T @ opt.state.Q_L, I5, atol=1e-10, rtol=0)
    torch.testing.assert_close(opt.state.Q_R.T @ opt.state.Q_R, I4, atol=1e-10, rtol=0)


def test_soap_update_matches_equation_17_in_the_identity_basis() -> None:
    """r02 section 6, Eq. (17): with Q_L = Q_R = I the update is the bias-corrected Adam step
    with epsilon *inside* the root: dW = -eta m_hat / sqrt(v_hat + eps_s). One step from zero
    moments gives m_hat = G and v_hat = G^2."""
    g = torch.Generator().manual_seed(42)
    W = torch.randn(3, 2, generator=g, dtype=D)
    G = torch.randn(3, 2, generator=g, dtype=D)
    eta, eps = 0.05, 1e-8
    opt = SOAP(lr=eta, eps=eps, precondition_frequency=10 ** 9)
    W1 = opt.step(W.clone(), G)
    torch.testing.assert_close(W1, W - eta * G / torch.sqrt(G ** 2 + eps))


def test_soap_rotated_update_is_equivariant() -> None:
    """r02 section 6, Eq. (17): dW = -eta Q_L (m'/sqrt(v'+eps)) Q_R^T. With Q fixed to a given
    orthogonal pair the step equals that formula evaluated by hand."""
    g = torch.Generator().manual_seed(1)
    W = torch.randn(3, 2, generator=g, dtype=D)
    G = torch.randn(3, 2, generator=g, dtype=D)
    QL, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=D))
    QR, _ = torch.linalg.qr(torch.randn(2, 2, generator=g, dtype=D))
    eta, eps = 0.05, 1e-8
    opt = SOAP(lr=eta, eps=eps, precondition_frequency=10 ** 9)
    opt.state.Q_L, opt.state.Q_R = QL, QR
    W1 = opt.step(W.clone(), G)
    Gp = QL.T @ G @ QR
    expected = W - eta * QL @ (Gp / torch.sqrt(Gp ** 2 + eps)) @ QR.T
    torch.testing.assert_close(W1, expected)
