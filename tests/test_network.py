"""Level 2 invariants of network.py (the two architectures of the protocol)."""

from __future__ import annotations

import pytest
import torch

from Staged_PINN_Solar import network as N


def _inputs(b=5, d=4):
    g = torch.Generator().manual_seed(0)
    return torch.rand(b, 1, generator=g, dtype=torch.float64), torch.rand(b, d, generator=g, dtype=torch.float64)


def test_architecture_a_is_one_net_for_three_outputs() -> None:
    net = N.build("A", n_xi=4, width=16, depth=2, seed=0)
    assert len(net.nets) == 1 and net.nets[0][-1].out_features == 3
    assert net(*_inputs()).shape == (5, 3)


def test_architecture_b_is_one_net_per_field() -> None:
    net = N.build("B", n_xi=4, width=16, depth=2, seed=0)
    assert len(net.nets) == 3 and all(m[-1].out_features == 1 for m in net.nets)
    assert net(*_inputs()).shape == (5, 3)


def test_protocol_size_is_256_by_4() -> None:
    net = N.build("A", n_xi=4, width=256, depth=4, seed=0)
    linears = [m for m in net.nets[0].modules() if isinstance(m, torch.nn.Linear)]
    assert [l.out_features for l in linears] == [256, 256, 256, 256, 3]


def test_outputs_are_in_the_ranges_the_ansatz_assumes() -> None:
    """Eq. 3: u_phi in [-1, 1] (so the interior correction is bounded by D_phi), u_n, u_p in [0, 1]."""
    for arch in "AB":
        net = N.build(arch, 4, 16, 2, seed=1)
        x, xi = _inputs(200)
        out = net(x * 5, xi * 5)
        assert out[:, 0].abs().max() <= 1.0 and out[:, 1:].min() >= 0.0 and out[:, 1:].max() <= 1.0


def test_activation_is_tanh_times_gaussian() -> None:
    z = torch.linspace(-3, 3, 13, dtype=torch.float64)
    assert torch.allclose(N.tanh_gauss(z), torch.tanh(z) * torch.exp(-z ** 2 / 2))


def test_same_seed_gives_same_initialization_and_seeds_differ() -> None:
    a1, a2, a3 = (N.build("A", 4, 16, 2, seed=s) for s in (0, 0, 1))
    assert all(torch.equal(p, q) for p, q in zip(a1.parameters(), a2.parameters()))
    assert not all(torch.equal(p, q) for p, q in zip(a1.parameters(), a3.parameters()))


def test_float64() -> None:
    assert all(p.dtype == torch.float64 for p in N.build("B", 4, 16, 2, 0).parameters())


def test_equilibrium_start_gives_zero_quasi_fermi_and_linear_potential() -> None:
    """At the start phi_n = phi_p = 0 exactly (the V = 0 solution) and u_phi = 0 (phi = phi_lin), for any input."""
    from Staged_PINN_Solar.ansatz import Ansatz, AnsatzBounds

    bounds = AnsatzBounds(D_phi=50.0, phi_n=(-3.0, 44.0), phi_p=(-45.0, 2.0))
    for arch in "AB":
        net = N.equilibrium_start(N.build(arch, 4, 16, 2, seed=3), bounds)
        x, xi = _inputs(64)
        phi, phi_n, phi_p = Ansatz(net, bounds)(x, xi, torch.zeros_like(x) - 20.0, torch.zeros_like(x) + 20.0, 0.0)
        assert torch.allclose(phi_n, torch.zeros_like(phi_n), atol=1e-12)
        assert torch.allclose(phi_p, torch.zeros_like(phi_p), atol=1e-12)
        assert torch.allclose(phi, -20.0 + 40.0 * x, atol=1e-12)
        # trainable: the zeroed output rows still receive gradient
        (phi.sum() + phi_n.sum() + phi_p.sum()).backward()
        assert all(m[-1].weight.grad.abs().sum() > 0 for m in net.nets)


def test_equilibrium_start_refuses_a_box_without_zero() -> None:
    from Staged_PINN_Solar.ansatz import AnsatzBounds

    with pytest.raises(ValueError):
        N.equilibrium_start(N.build("A", 4, 16, 2, seed=0), AnsatzBounds(1.0, (0.5, 4.0), (-4.0, 1.0)))
