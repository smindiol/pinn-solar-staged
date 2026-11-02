"""Level 2 invariants of residuals.py (Eqs. 5-7 and 9)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import residuals as R


def test_current_balance_is_sum_of_continuity_residuals() -> None:
    """r02 section 3, Eq. (7): R_J = R_n + R_p = delta d(J_n + J_p)/dx_hat,
    because G and R_total enter R_n and R_p with opposite signs."""
    rng = np.random.default_rng(42)
    dJn, dJp, G, Rr = (rng.normal(size=9) for _ in range(4))
    d = 3.2e-4
    r_n = R.residual(R.continuity_n_terms(dJn, G, Rr, d))
    r_p = R.residual(R.continuity_p_terms(dJp, G, Rr, d))
    r_j = R.residual(R.current_balance_terms(dJn, dJp, d))
    np.testing.assert_allclose(r_n + r_p, r_j, rtol=0, atol=1e-15)
    np.testing.assert_allclose(r_j, d * (dJn + dJp), rtol=1e-14)


def test_current_density_definition() -> None:
    """r02 section 3, Eq. (6): J = delta mu rho d(phi_q)/dx_hat for both carriers."""
    d, mu, rho, dphi = 2e-3, 3.0, 0.5, -7.0
    assert R.current_density(mu, rho, dphi, d) == pytest.approx(d * mu * rho * dphi, rel=1e-15)


def test_poisson_terms_sign_convention() -> None:
    """r02 section 3, Eq. (5): R_Poisson = delta^2 (eps phi')' - (n - p - Ndop);
    the four summands are exactly these four signed terms."""
    flux_dx, n, p, Nd, d = 2.0, 0.3, 0.1, 0.05, 0.1
    t = R.poisson_terms(flux_dx, n, p, Nd, d)
    assert len(t) == 4
    assert R.residual(t) == pytest.approx(d ** 2 * flux_dx - (n - p - Nd), rel=1e-14)
    assert R.scale(t) == pytest.approx(d ** 2 * abs(flux_dx) + n + p + Nd, rel=1e-14)


def test_contact_residuals_vanish_at_equilibrium() -> None:
    """r02 section 3, Eq. (9): in equilibrium J = 0 and n = n_eq, p = p_eq at
    both contacts, so all four Robin residuals are zero for any S."""
    z = np.zeros(2)
    n = np.array([0.3, 0.7])
    p = np.array([0.2, 0.9])
    S_n, S_p = np.array([5.0, 1.0]), np.array([2.0, 8.0])
    tn = R.contact_n_terms(z, n, n, S_n)
    tp = R.contact_p_terms(z, p, p, S_p)
    np.testing.assert_allclose(R.residual(tn), 0.0, atol=1e-15)
    np.testing.assert_allclose(R.residual(tp), 0.0, atol=1e-15)


def test_contact_residual_signs() -> None:
    """r02 section 3, Eq. (9): B_n0 = J_n(0) - S_n0 (n(0)-n_eq(0)), B_n1 = J_n(1) +
    S_n1 (n(1)-n_eq(1)), B_p0 = J_p(0) + S_p0 (p(0)-p_eq(0)), B_p1 = J_p(1) - S_p1 (p(1)-p_eq(1))."""
    J = np.array([1.5, -2.0])
    n, neq = np.array([4.0, 5.0]), np.array([1.0, 2.0])
    S = np.array([0.5, 0.25])
    bn = R.residual(R.contact_n_terms(J, n, neq, S))
    assert bn[0] == pytest.approx(J[0] - S[0] * (n[0] - neq[0]))
    assert bn[1] == pytest.approx(J[1] + S[1] * (n[1] - neq[1]))
    bp = R.residual(R.contact_p_terms(J, n, neq, S))
    assert bp[0] == pytest.approx(J[0] + S[0] * (n[0] - neq[0]))
    assert bp[1] == pytest.approx(J[1] - S[1] * (n[1] - neq[1]))


def test_staggered_derivative_is_second_order_on_nonuniform_grid() -> None:
    """Numerical stand-in for autograd on the dPV nodes: the conservative
    staggered derivative (c f')' converges at second order on a smoothly graded grid."""
    errs = []
    for N in (200, 400):
        s = np.linspace(0.0, 1.0, N)
        x = s + 0.3 * s * (1.0 - s)
        c = 1.0 + x
        f = np.sin(2.0 * x)
        got = R.staggered_flux_derivative(x, f, c)
        exact = 2.0 * np.cos(2.0 * x) + (1.0 + x) * (-4.0 * np.sin(2.0 * x))
        errs.append(np.max(np.abs(got - exact[1:-1])))
    assert errs[1] < errs[0] / 3.0


def test_autograd_derivative_matches_analytic() -> None:
    """Eq. (5)-(7) derivatives come from autograd on the network; check the
    helper against d/dx (x^3) = 3 x^2 and a second derivative."""
    x = torch.linspace(0.1, 0.9, 9, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    y = x ** 3
    g = R.autograd_derivative(y, x)
    torch.testing.assert_close(g, 3.0 * x.detach() ** 2)
    g2 = R.autograd_derivative(g, x)
    torch.testing.assert_close(g2, 6.0 * x.detach())
