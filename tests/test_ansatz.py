"""Level 2 invariants of ansatz.py (Eq. 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import data
from Staged_PINN_Solar.ansatz import Ansatz, AnsatzBounds, normalize_xi

BOUNDS = AnsatzBounds(D_phi=2.5, phi_n=(0.0, 40.0), phi_p=(-40.0, 0.0))


class _RandomNet(torch.nn.Module):
    """An architecture stand-in: any smooth map into the three raw outputs."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(3, 3, generator=g, dtype=torch.float64))

    def forward(self, x_hat, xi_hat):
        z = torch.cat([x_hat, xi_hat], dim=1)[:, :3]
        raw = torch.tanh(z @ self.w.T) * 7.0
        return torch.stack([raw[:, 0], torch.sigmoid(raw[:, 1]), torch.sigmoid(raw[:, 2])], dim=1)


def _call(seed, x):
    a = Ansatz(_RandomNet(seed), BOUNDS)
    xi = torch.full((x.shape[0], 2), 0.4, dtype=torch.float64)
    return a(x, xi, phi_eq0=-155.07, phi_eq1=-150.2, V_tilde=12.0)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_contacts_are_exact_for_any_network(seed) -> None:
    """r02 section 2, Eq. (3): phi(0) = phi_eq(0) and phi(1) = phi_eq(1) + V_tilde by
    construction, for every parameter value of the network."""
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    phi, _, _ = _call(seed, x)
    assert phi[0].item() == pytest.approx(-155.07, abs=1e-12)
    assert phi[1].item() == pytest.approx(-150.2 + 12.0, abs=1e-12)


def test_interior_correction_is_bounded_by_d_phi() -> None:
    """r02 section 2, Eq. (3): the correction 4 x (1-x) D_phi u_phi has |.| <= D_phi |u_phi|
    since 4 x (1-x) <= 1 on [0, 1]."""
    x = torch.linspace(0, 1, 101, dtype=torch.float64).reshape(-1, 1)
    net = _RandomNet(42)
    a = Ansatz(net, BOUNDS)
    xi = torch.full((101, 2), 0.4, dtype=torch.float64)
    phi, _, _ = a(x, xi, phi_eq0=-155.07, phi_eq1=-150.2, V_tilde=12.0)
    line = -155.07 + (-150.2 + 12.0 + 155.07) * x
    u = net(x, xi)[:, 0:1]
    assert torch.all((phi - line).abs() <= BOUNDS.D_phi * u.abs() + 1e-12)


def test_carrier_outputs_stay_inside_their_bounds() -> None:
    """r02 section 2, Eq. (3): Y_k = Y_min + (Y_max - Y_min) u_k with u_k in [0, 1] keeps
    phi_n in [0, 40] and phi_p in [-40, 0] for every input."""
    x = torch.rand(500, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(42))
    _, pn, pp = _call(42, x)
    assert pn.min() >= BOUNDS.phi_n[0] and pn.max() <= BOUNDS.phi_n[1]
    assert pp.min() >= BOUNDS.phi_p[0] and pp.max() <= BOUNDS.phi_p[1]


def test_normalized_parameters_match_the_dataset_u() -> None:
    """r02 section 2, Eq. (3): xi_hat = (g(xi) - g_min)/(g_max - g_min) with g = log10 for the
    dopings and the identity for thicknesses; it must reproduce the `u` stored with every dPV device."""
    inv = data.inventory()
    for ref in (inv[0], inv[2100], inv[3000]):
        dev = data.load_device(ref)
        got = normalize_xi(dev.xi, dev.xi_specs)
        np.testing.assert_allclose(got, dev.u, atol=1e-12)
