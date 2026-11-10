"""Residuals of the coupled Poisson / drift-diffusion system (Eqs. 5-7, 9).

Each residual is returned as the tuple of its signed summands T_iq, because
Eq. (10) scales a residual by the sum of the absolute values of its summands
(`scale`) and Eq. (11) uses the residual itself (`residual`, the plain sum).

Two derivative providers serve the same residual definitions: autograd on
the network (`autograd_derivative`) and a conservative staggered finite
difference on the dPV nodes (`staggered_flux_derivative`), the numerical
stand-in used to evaluate the residuals on the reference solution.
"""

from __future__ import annotations

import numpy as np
import torch

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r03.md",
    "sections": ["3"],
    "equations": ["5", "6", "7", "9"],
    "invariants": [
        "current_balance_is_sum_of_continuity_residuals",
        "current_density_definition",
        "poisson_terms_sign_convention",
        "contact_residuals_vanish_at_equilibrium",
        "contact_residual_signs",
        "contact_summand_convention_changes_only_the_scale",
        "grouped_and_expanded_domain_residuals_agree",
        "staggered_derivative_is_second_order_on_nonuniform_grid",
        "autograd_derivative_matches_analytic",
    ],
}


def _on(ref, values):
    """`values` (numpy) as a tensor of the dtype and device of `ref` when `ref` is a tensor, else unchanged.

    Keeps the per-contact constants (signs, surface velocities) on the backend of the densities they
    multiply, so the same functions serve the numpy reference and the PINN on any device."""
    if isinstance(ref, torch.Tensor):
        return torch.as_tensor(values, dtype=ref.dtype, device=ref.device)
    return values


def _abs(x):
    return torch.abs(x) if isinstance(x, torch.Tensor) else np.abs(x)


def residual(terms):
    """R_i = sum_q T_iq."""
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def scale(terms):
    """sum_q |T_iq|, the quantity squared and averaged in Eq. (10)."""
    out = _abs(terms[0])
    for t in terms[1:]:
        out = out + _abs(t)
    return out


def poisson_terms(flux_dx, n, p, Ndop, delta):
    """Eq. (5): R = delta^2 d(eps dphi/dx)/dx - (n - p - Ndop); `flux_dx` is the derivative."""
    return (delta ** 2 * flux_dx, -n, p, Ndop + 0.0 * n)


def poisson_terms_grouped(flux_dx, n, p, Ndop, delta):
    """Eq. (5) with the parenthesis (n - p - Ndop) kept as ONE summand."""
    return (delta ** 2 * flux_dx, -(n - p - Ndop))


def continuity_n_terms_grouped(dJn_dx, G, R_total, delta):
    """Eq. (7) with the parenthesis (G - R_total) kept as ONE summand."""
    return (delta * dJn_dx, G - R_total)


def continuity_p_terms_grouped(dJp_dx, G, R_total, delta):
    """Eq. (7) with the parenthesis (G - R_total) kept as ONE summand."""
    return (delta * dJp_dx, -(G - R_total))


def current_density(mu, density, dphi_q_dx, delta):
    """Eq. (6): J = delta mu rho d(phi_q)/dx_hat (rho = n with phi_n, rho = p with phi_p)."""
    return delta * mu * density * dphi_q_dx


def continuity_n_terms(dJn_dx, G, R_total, delta):
    """Eq. (7): R_n = delta dJ_n/dx + (G - R_total)."""
    return (delta * dJn_dx, G + 0.0 * dJn_dx, -R_total)


def continuity_p_terms(dJp_dx, G, R_total, delta):
    """Eq. (7): R_p = delta dJ_p/dx - (G - R_total)."""
    return (delta * dJp_dx, -G + 0.0 * dJp_dx, R_total)


def current_balance_terms(dJn_dx, dJp_dx, delta):
    """Eq. (7): R_J = delta d(J_n + J_p)/dx."""
    return (delta * dJn_dx, delta * dJp_dx)


def contact_n_terms(J_n, n, n_eq, S_n):
    """Eq. (9), electrons, expanded summands; last axis is (x_hat = 0, x_hat = 1).

    B_n0 = J_n(0) - S_n0 n(0) + S_n0 n_eq(0); B_n1 = J_n(1) + S_n1 n(1) - S_n1 n_eq(1).
    Eq. (10) says "the summand q of the residual" without saying whether a parenthesis of
    Eqs. (5), (7), (9) is one summand or several; the expanded reading (three summands per
    contact here) is the one that reproduces the magnitudes of the proposal's own table
    (see `contact_n_terms_grouped` for the parenthesis-as-written reading).
    """
    sign, S_n = _on(n, np.array([-1.0, 1.0])), _on(n, S_n)
    return (J_n, sign * S_n * n, -sign * S_n * n_eq)


def contact_p_terms(J_p, p, p_eq, S_p):
    """Eq. (9), holes, expanded summands; last axis is (x_hat = 0, x_hat = 1).

    B_p0 = J_p(0) + S_p0 p(0) - S_p0 p_eq(0); B_p1 = J_p(1) - S_p1 p(1) + S_p1 p_eq(1).
    """
    sign, S_p = _on(p, np.array([1.0, -1.0])), _on(p, S_p)
    return (J_p, sign * S_p * p, -sign * S_p * p_eq)


def contact_n_terms_grouped(J_n, n, n_eq, S_n):
    """Eq. (9), electrons, with the parenthesis (n - n_eq) kept as ONE summand: (J_n, +-S_n (n - n_eq))."""
    sign, S_n = _on(n, np.array([-1.0, 1.0])), _on(n, S_n)
    return (J_n, sign * S_n * (n - n_eq))


def contact_p_terms_grouped(J_p, p, p_eq, S_p):
    """Eq. (9), holes, with the parenthesis (p - p_eq) kept as ONE summand: (J_p, +-S_p (p - p_eq))."""
    sign, S_p = _on(p, np.array([1.0, -1.0])), _on(p, S_p)
    return (J_p, sign * S_p * (p - p_eq))


def autograd_derivative(y, x):
    """dy/dx of a scalar-per-row field with respect to the input column x (keeps the graph)."""
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def staggered_flux_derivative(x, f, coeff):
    """(coeff f')' at the interior nodes of a (possibly graded) grid.

    The flux c f' is formed at the half nodes with the arithmetic mean of c
    and divided back by the mean of the two adjacent cell widths, so the
    scheme is conservative and second order. Returns N-2 values.
    """
    x = np.asarray(x)
    h = np.diff(x)
    c_mid = 0.5 * (coeff[..., 1:] + coeff[..., :-1])
    flux = c_mid * np.diff(f, axis=-1) / h
    return (flux[..., 1:] - flux[..., :-1]) / (0.5 * (h[1:] + h[:-1]))
