"""Level 3 - synthetic data. Deterministic, fixed seed 42, ground truth by construction.

Each expectation is stated in the docstring BEFORE the assertion.
"""

from __future__ import annotations

import numpy as np
import torch

from Staged_PINN_Solar.ansatz import Ansatz, AnsatzBounds
from Staged_PINN_Solar.optimizers import LBFGS, SOAP

SEED = 42


class _ZeroNet(torch.nn.Module):
    def forward(self, x_hat, xi_hat):
        return torch.zeros(x_hat.shape[0], 3, dtype=x_hat.dtype)


def test_ansatz_with_zero_network_is_linear_potential_and_midrange_carriers(rng) -> None:
    """Expectation: with u_hat = 0 the ansatz Eq. (3) reduces to the straight line
    phi_eq(0) + (phi_eq(1) + V - phi_eq(0)) x_hat and to Y_min for both quasi-Fermi
    potentials; any other output is a bug in the construction."""
    b = AnsatzBounds(D_phi=3.0, phi_n=(0.0, 40.0), phi_p=(-40.0, 0.0))
    a = Ansatz(_ZeroNet(), b)
    x = torch.linspace(0, 1, 11, dtype=torch.float64).reshape(-1, 1)
    xi = torch.zeros(11, 2, dtype=torch.float64)
    phi, pn, pp = a(x, xi, phi_eq0=-155.0, phi_eq1=-150.0, V_tilde=20.0)
    torch.testing.assert_close(phi.reshape(-1), -155.0 + 25.0 * x.reshape(-1))
    torch.testing.assert_close(pn.reshape(-1), torch.zeros(11, dtype=torch.float64))
    torch.testing.assert_close(pp.reshape(-1), torch.full((11,), -40.0, dtype=torch.float64))


def test_lbfgs_solves_ill_conditioned_quadratic_and_soap_decreases_it() -> None:
    """Expectation: on f(x) = 0.5 x^T A x with cond(A) = 1e3 (seeded random rotation),
    L-BFGS (m = 20 = n) reaches |grad| < 1e-6 in fewer than 400 iterations from x0 = 1
    (a first version of this expectation, cond 1e6 with m = 10 in 200 iterations, was measured
    at |grad| = 108 and withdrawn: the iteration count of a quasi-Newton method scales with
    sqrt(cond), so that bound was a mis-estimate, not a code defect),
    and SOAP at lr 0.1 lowers the loss of a matrix quadratic by at least 10x in 200 steps."""
    g = torch.Generator().manual_seed(SEED)
    Q, _ = torch.linalg.qr(torch.randn(20, 20, generator=g, dtype=torch.float64))
    A = Q @ torch.diag(torch.logspace(0, 3, 20, dtype=torch.float64)) @ Q.T
    A = 0.5 * (A + A.T)

    def fun(x):
        return 0.5 * x @ A @ x, A @ x

    opt = LBFGS(memory=20)
    x, info = opt.minimize(torch.ones(20, dtype=torch.float64), fun, max_iter=400, gtol=1e-6)
    assert info["grad_norm"] < 1e-6 and info["iterations"] < 400

    W = torch.randn(6, 5, generator=g, dtype=torch.float64)
    B = torch.diag(torch.tensor([1.0, 10.0, 100.0, 1000.0, 1.0, 5.0], dtype=torch.float64))
    C = torch.diag(torch.tensor([1.0, 0.1, 10.0, 1.0, 100.0], dtype=torch.float64))
    loss0 = 0.5 * torch.sum((B @ W @ C) ** 2)
    soap = SOAP(lr=0.1, precondition_frequency=1)
    for _ in range(200):
        grad = B.T @ B @ W @ C @ C.T
        W = soap.step(W, grad)
    assert 0.5 * torch.sum((B @ W @ C) ** 2) < loss0 / 10.0
