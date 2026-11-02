"""Level 2 invariants of scaling.py (Eqs. 1-2)."""

from __future__ import annotations

import numpy as np
import pytest

from Staged_PINN_Solar.scaling import (
    EPS0_F_PER_CM, K_BOLTZMANN, Q_ELEMENTARY, Coordinates, Scales)


@pytest.mark.parametrize("C0,mu0", [(1e17, 1.0), (1e19, 1.0), (3e18, 25.0)])
def test_debye_scale_identities(C0, mu0) -> None:
    """r02 section 2, Eq. (1): G0 = C0/t0 = C0 mu0 kB T/(q L_D^2) and
    J0 = q C0 L_D/t0 = kB T C0 mu0 / L_D, v0 = L_D/t0."""
    s = Scales(C0=C0, mu0=mu0)
    kT = K_BOLTZMANN * s.T
    assert s.G0 == pytest.approx(C0 * mu0 * kT / (Q_ELEMENTARY * s.L_D ** 2), rel=1e-12)
    assert s.J0 == pytest.approx(kT * C0 * mu0 / s.L_D, rel=1e-12)
    assert s.v0 == pytest.approx(s.L_D / s.t0, rel=1e-14)
    assert s.V_th == pytest.approx(0.025852, abs=2e-6)
    assert s.L_D ** 2 == pytest.approx(EPS0_F_PER_CM * s.V_th / (Q_ELEMENTARY * C0), rel=1e-12)


def test_second_derivative_chain_rule() -> None:
    """r02 section 2, Eq. (2): d^2/dx_D^2 = delta^2 d^2/dx_hat^2 (checked on
    f = sin(3 x_hat) by central differences in both coordinates)."""
    c = Coordinates(Scales(), L_total_cm=2.0e-4)
    xh = np.linspace(0.2, 0.8, 4001)
    f = np.sin(3.0 * xh)
    h = xh[1] - xh[0]
    d2_hat = (f[2:] - 2 * f[1:-1] + f[:-2]) / h ** 2
    xd = c.to_debye(xh)
    hd = xd[1] - xd[0]
    d2_debye = (f[2:] - 2 * f[1:-1] + f[:-2]) / hd ** 2
    np.testing.assert_allclose(d2_debye, c.d2_dxD2(d2_hat), rtol=1e-9)
    exact = -9.0 * np.sin(3.0 * xh[1:-1])
    np.testing.assert_allclose(d2_hat, exact, atol=2e-6)


def test_normalized_coordinate_range() -> None:
    """r02 section 2, Eq. (2): x_hat = x/L_total maps [0, L_total] onto [0, 1]
    and the map is invertible."""
    L = 3.1e-4
    c = Coordinates(Scales(), L_total_cm=L)
    x_cm = np.linspace(0.0, L, 11)
    xh = c.to_hat(x_cm / c.scales.L_D)
    assert xh[0] == 0.0
    assert xh[-1] == pytest.approx(1.0, rel=1e-14)
    np.testing.assert_allclose(c.to_debye(xh) * c.scales.L_D, x_cm, rtol=1e-12, atol=1e-18)
