"""Level 2 invariants of carriers.py (Eqs. 4 and 8)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from Staged_PINN_Solar.carriers import (
    Material, electron_density, equilibrium_densities, hole_density,
    intrinsic_density_squared, recombination)


def _material(n: int = 7, **over) -> Material:
    kw = dict(Nc=np.full(n, 8e-2), Nv=np.full(n, 1.8), chi=np.full(n, 150.87),
              Eg=np.full(n, 58.0), eps_r=np.full(n, 9.4), mu_n=np.full(n, 100.0),
              mu_p=np.full(n, 100.0), tau_n=np.full(n, 1.8e2), tau_p=np.full(n, 1.8e2),
              Et=np.zeros(n), Br=np.zeros(n), Ndop=np.linspace(-1e-2, 1e-2, n))
    kw.update(over)
    return Material(**kw)


def test_equilibrium_np_product_is_ni_squared() -> None:
    """r02 section 3, Eq. (4): with phi_n = phi_p = 0 the Boltzmann densities
    satisfy n p = n_i^2 = Nc Nv exp(-Eg) for every phi."""
    m = _material()
    phi = np.linspace(-160.0, -150.0, 7)
    n, p = equilibrium_densities(phi, m)
    np.testing.assert_allclose(n * p, intrinsic_density_squared(m), rtol=1e-12)
    np.testing.assert_allclose(n * p, m.Nc * m.Nv * np.exp(-m.Eg), rtol=1e-12)


def test_quasi_fermi_shifts_scale_densities() -> None:
    """r02 section 3, Eq. (4): n is proportional to exp(phi_n) and p to
    exp(-phi_p) at fixed phi."""
    m = _material()
    phi = np.linspace(-160.0, -150.0, 7)
    a = 3.7
    n0, p0 = electron_density(phi, 0.0 * phi, m), hole_density(phi, 0.0 * phi, m)
    n1, p1 = electron_density(phi, a + 0.0 * phi, m), hole_density(phi, a + 0.0 * phi, m)
    np.testing.assert_allclose(n1 / n0, np.exp(a), rtol=1e-12)
    np.testing.assert_allclose(p1 / p0, np.exp(-a), rtol=1e-12)


def test_recombination_zero_at_equilibrium_and_signed() -> None:
    """r02 section 3, Eq. (8): R_total vanishes when n p = n_i^2 (phi_n = phi_p = 0)
    and has the sign of (n p - n_i^2), i.e. of phi_n - phi_p."""
    m = _material()
    phi = np.linspace(-160.0, -150.0, 7)
    z = 0.0 * phi
    n, p = equilibrium_densities(phi, m)
    np.testing.assert_allclose(recombination(n, p, m), 0.0, atol=1e-30)
    n2, p2 = electron_density(phi, z + 5.0, m), hole_density(phi, z - 5.0, m)
    assert np.all(recombination(n2, p2, m) > 0.0)
    n3, p3 = electron_density(phi, z - 5.0, m), hole_density(phi, z + 5.0, m)
    assert np.all(recombination(n3, p3, m) < 0.0)


def test_recombination_srh_closed_form_and_radiative_term() -> None:
    """r02 section 3, Eq. (8): with Et = 0 and tau_n = tau_p = tau,
    R = (n p - n_i^2) [Br + 1/(tau (n + p + 2 n_i))]."""
    tau, Br = 1.8e2, 0.4
    m = _material(tau_n=np.full(7, tau), tau_p=np.full(7, tau), Br=np.full(7, Br))
    phi = np.linspace(-160.0, -150.0, 7)
    z = 0.0 * phi
    n, p = electron_density(phi, z + 4.0, m), hole_density(phi, z - 4.0, m)
    ni = np.sqrt(intrinsic_density_squared(m))
    expected = (n * p - ni ** 2) * (Br + 1.0 / (tau * (n + p + 2.0 * ni)))
    np.testing.assert_allclose(recombination(n, p, m), expected, rtol=1e-12)


def test_torch_and_numpy_agree() -> None:
    """The same Eq. (4)/(8) code path serves autograd (torch) and the
    reference (numpy): both backends must return the same numbers."""
    m = _material()
    phi = np.linspace(-160.0, -150.0, 7)
    ph_n, ph_p = 0.0 * phi + 2.0, 0.0 * phi - 2.0
    n_np, p_np = electron_density(phi, ph_n, m), hole_density(phi, ph_p, m)
    r_np = recombination(n_np, p_np, m)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    n_t, p_t = electron_density(t(phi), t(ph_n), m), hole_density(t(phi), t(ph_p), m)
    r_t = recombination(n_t, p_t, m)
    assert isinstance(n_t, torch.Tensor)
    np.testing.assert_allclose(n_t.numpy(), n_np, rtol=1e-12)
    np.testing.assert_allclose(p_t.numpy(), p_np, rtol=1e-12)
    np.testing.assert_allclose(r_t.numpy(), r_np, rtol=1e-10)
