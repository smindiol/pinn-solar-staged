"""Level 2 - invariants of the data reader (data.py).

The invariants of the other modules live in their own files
(test_scaling.py, test_carriers.py, test_residuals.py, test_loss.py,
test_ansatz.py, test_weights.py, test_optimizers.py); every id declared in a
module's `__provenance__` has a `test_<id>` in this package of test files.
"""

from __future__ import annotations

import numpy as np
import pytest

from Staged_PINN_Solar import data
from Staged_PINN_Solar.scaling import Scales


@pytest.fixture(scope="module")
def inventory():
    return data.inventory()


def test_inventory_is_4096_ok_devices_from_e1a_and_e1b(inventory) -> None:
    """r02 section 1 / handoff map: the training set is dPV E1A+E1B, 2048 Sobol
    devices each, 4096 in total, all with state 'ok'; the partial E1B_b001 of
    session s1 (814/1075 done) is NOT used, the complete one of s4 is."""
    assert len(inventory) == 4096
    assert len({d.uid for d in inventory}) == 4096
    assert {d.family_stage for d in inventory} == {"E1A", "E1B"}
    assert sum(d.family_stage == "E1A" for d in inventory) == 2048
    assert sum(d.family_stage == "E1B" for d in inventory) == 2048
    sources = {d.source.parent.name for d in inventory if d.block == 1 and d.family_stage == "E1B"}
    assert sources == {"s4"}


def test_split_is_device_level_80_10_10_and_seeded(inventory) -> None:
    """r02 section 1 / premises: device-level 80/10/10 split with seed 42:
    disjoint, exhaustive, 3277/410/409, reproducible, and seed-dependent."""
    a = data.split_devices(inventory, seed=42)
    b = data.split_devices(inventory, seed=42)
    c = data.split_devices(inventory, seed=43)
    assert (len(a.train), len(a.val), len(a.test)) == (3277, 410, 409)
    uids = [d.uid for d in a.train + a.val + a.test]
    assert len(set(uids)) == 4096 == len(uids)
    assert [d.uid for d in a.train] == [d.uid for d in b.train]
    assert [d.uid for d in a.train] != [d.uid for d in c.train]


def test_loaded_device_respects_dpv_contract(inventory) -> None:
    """dpv-dataset/README and ansatz Eq. (3): phi(0) = phi_eq(0), phi(1) = phi_eq(1) + V/V_th
    at every stored voltage; phi_n(0) ~ 0 and phi_p(1) ~ -V/V_th; stored x spans L_total in
    units of L_D (C0 = 1e19), so delta = 1/x[-1]."""
    s = Scales()
    dev = data.load_device(inventory[0])
    V = dev.V / s.V_th
    np.testing.assert_allclose(dev.phi[:, 0], dev.phi_eq[0], atol=1e-9)
    np.testing.assert_allclose(dev.phi[:, -1], dev.phi_eq[-1] + V, atol=1e-6)
    assert dev.phi.shape == dev.phi_n.shape == dev.phi_p.shape == (len(dev.V), len(dev.x_debye))
    assert abs(dev.phi_n[0, 0]) < 1e-5
    L_total = dev.xi["L_n"] + dev.xi["L_p"]
    assert dev.x_debye[-1] * s.L_D == pytest.approx(L_total, rel=1e-6)
    assert dev.delta(s) == pytest.approx(1.0 / dev.x_debye[-1], rel=1e-12)


def test_device_material_doping_step_and_contacts(inventory) -> None:
    """dPV ex1 (dpvgen._material_e1): n-side donors Nd_n then p-side acceptors -Na_p, single
    material; contacts S = (Snl, Snr, Spl, Spr) = (1e7, 0, 0, 1e7) cm/s."""
    s = Scales()
    dev = data.load_device(inventory[5])
    m = data.device_material(dev, s)
    assert m.Ndop[0] == pytest.approx(dev.xi["Nd_n"] / s.C0, rel=1e-12)
    assert m.Ndop[-1] == pytest.approx(-dev.xi["Na_p"] / s.C0, rel=1e-12)
    assert np.count_nonzero(np.diff(np.sign(m.Ndop))) == 1
    c = data.device_contacts(s)
    assert (c.S_n0, c.S_n1, c.S_p0, c.S_p1) == (1e7 / s.v0, 0.0, 0.0, 1e7 / s.v0)


def test_reference_solution_closes_the_equations(inventory) -> None:
    """Measured convention check (Eqs. 4-7 on the dPV reference): the dimensionless Poisson
    residual of the stored solution is below 1e-9 of its summands and the dPV contact currents
    satisfy Eq. (9) -- which fixes C0 = 1e19 and mu0 = 1 as dPV's own scales."""
    from Staged_PINN_Solar import loss
    s = Scales()
    dev = data.load_device(inventory[0])
    t = loss.reference_terms(dev, s)
    rel = np.abs(loss.residual(t["poisson"])) / loss.scale(t["poisson"])
    assert rel.max() < 1e-9
    assert np.abs(loss.residual(t["bcn"])).max() < 1e-6 * np.abs(t["bcn"][0]).max()
    assert np.abs(loss.residual(t["bcp"])).max() < 1e-6 * np.abs(t["bcp"][0]).max()


def test_reference_continuity_closes_with_the_dpv_operator(inventory) -> None:
    """Measured convention check (Eqs. 6-7): evaluated with dPV's own interface currents the
    continuity residuals of the stored solution are below 1e-3 of their summands at every node
    (sign conventions of Eq. 7 and G units verified); the continuous-form reading of Eq. (6)
    on the same nodes is looser, and that gap is the point of the 'currents' switch."""
    from Staged_PINN_Solar import loss
    s = Scales()
    dev = data.load_device(inventory[0])
    t = loss.reference_terms(dev, s, currents="dpv")
    for name in ("n", "p"):
        rel = np.abs(loss.residual(t[name])) / loss.scale(t[name])
        assert np.nanmax(rel) < 1e-3, name
    c = loss.reference_terms(dev, s, currents="continuous")
    rel_c = np.abs(loss.residual(c["n"])) / loss.scale(c["n"])
    assert np.nanmax(rel_c) > 0.1
