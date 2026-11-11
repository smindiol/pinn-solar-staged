"""Level 2 invariants of loss.py (Eqs. 10-11)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import loss


def _terms(rng, c=1.0):
    return {"poisson": tuple(c * rng.normal(size=(3, 40)) for _ in range(4)),
            "n": tuple(1e-11 * c * rng.normal(size=(3, 40)) for _ in range(3))}


def test_sigma_squared_dominates_residual_squared() -> None:
    """r03 section 4, Eqs. (10)-(11): |sum_q T_q| <= sum_q |T_q|, hence f_i <= sigma_i^2 and
    rho_i = f_i / sigma_i^2 lies in [0, 1] for every term and every sample."""
    rng = np.random.default_rng(42)
    st = loss.ReferenceStatistics()
    st.add(_terms(rng))
    rho = st.rho()
    assert set(rho) == {"poisson", "n"}
    assert all(0.0 <= r <= 1.0 for r in rho.values())


def test_loss_at_reference_equals_rho() -> None:
    """r03 section 4, Eq. (11): L_i(theta) = <R_i^2>_theta / sigma_i^2, so a network that
    reproduces the reference (R_i = the reference residual) has L_i = rho_i exactly."""
    rng = np.random.default_rng(42)
    terms = _terms(rng)
    st = loss.ReferenceStatistics()
    st.add(terms)
    L = loss.normalized_losses({k: loss.residual(v) for k, v in terms.items()}, st.sigma2())
    for k in L:
        assert L[k] == pytest.approx(st.rho()[k], rel=1e-12)


def test_normalization_removes_the_residual_scale() -> None:
    """r03 section 4, Eq. (11): scaling a residual's summands by c scales sigma_i^2 by c^2 and
    leaves L_i unchanged, even when two terms differ by 21 orders of magnitude."""
    rng = np.random.default_rng(42)
    base = _terms(rng)
    big = {k: tuple(1e-10 * t for t in v) if k == "n" else v for k, v in base.items()}
    s0, s1 = loss.ReferenceStatistics(), loss.ReferenceStatistics()
    s0.add(base)
    s1.add(big)
    assert s1.sigma2()["n"] == pytest.approx(1e-20 * s0.sigma2()["n"], rel=1e-12)
    assert s1.sigma2()["poisson"] == pytest.approx(s0.sigma2()["poisson"], rel=1e-14)
    L0 = loss.normalized_losses({k: loss.residual(v) for k, v in base.items()}, s0.sigma2())
    L1 = loss.normalized_losses({k: loss.residual(v) for k, v in big.items()}, s1.sigma2())
    assert L1["n"] == pytest.approx(L0["n"], rel=1e-12)


def test_total_loss_is_the_weighted_sum() -> None:
    """r03 section 4, Eq. (11): L(theta; lambda) = sum_i lambda_i L_i(theta)."""
    L = {"poisson": 2.0, "n": 0.5, "p": 4.0}
    lam = {"poisson": 1.0, "n": 10.0, "p": 0.25}
    assert loss.total_loss(L, lam) == pytest.approx(2.0 + 5.0 + 1.0)
    Lt = {k: torch.tensor(v, requires_grad=True) for k, v in L.items()}
    tot = loss.total_loss(Lt, lam)
    tot.backward()
    assert Lt["n"].grad.item() == pytest.approx(10.0)


def test_streaming_statistics_equal_batch_statistics() -> None:
    """r03 section 4, Eq. (10): <.>_ref is a pooled mean over nodes, devices and voltages, so
    accumulating in chunks must give the same sigma_i^2 and f_i as one batch."""
    rng = np.random.default_rng(42)
    parts = [_terms(rng) for _ in range(3)]
    stream = loss.ReferenceStatistics()
    for p in parts:
        stream.add(p)
    whole = loss.ReferenceStatistics()
    whole.add({k: tuple(np.concatenate([p[k][q] for p in parts], axis=0) for q in range(len(parts[0][k])))
               for k in parts[0]})
    for k in whole.sigma2():
        assert stream.sigma2()[k] == pytest.approx(whole.sigma2()[k], rel=1e-12)
        assert stream.f()[k] == pytest.approx(whole.f()[k], rel=1e-12)


def test_reference_sigma_squared_reproduces_the_proposals_table() -> None:
    """r03 section 4, Eq. (10) and its table (E1A, 400 devices, seed 0): on 100 training devices of
    the 4096-device set the five sigma_i^2 stay within a factor 1.5 of the tabulated values
    (Poisson 2.52e-3, BCn 0.204, BCp 0.334, n = p = 1.70e-22), and the spread between the largest
    and smallest is within 1 order of the '21 orders' of section 1."""
    from Staged_PINN_Solar import data
    from Staged_PINN_Solar.scaling import Scales
    split = data.split_devices(data.inventory(), seed=42)
    st = loss.measure_reference_statistics(split.train[:100], Scales())
    s2 = st.sigma2()
    table = {"poisson": 2.52e-3, "bcn": 0.204, "bcp": 0.334, "n": 1.70e-22, "p": 1.70e-22}
    for k, v in table.items():
        assert v / 1.5 < s2[k] < v * 1.5, (k, s2[k], v)
    spread = np.log10(max(s2.values()) / min(s2.values()))
    assert 20.0 < spread < 22.0


def _swept_pooled(reading: str) -> dict:
    from measurement import pooled
    from sweep import SWEEP_SIZE

    return pooled(range(SWEEP_SIZE), reading)


def test_sigma_squared_elementary_terms_fixed() -> None:
    """r03 Eq. (10): T_iq are the separately signed terms of R_i. Read that way, the pooled
    sigma_i^2 of 200 swept devices reproduce the proposal's table within 30% and span >= 20 orders."""
    import math

    from measurement import PROPOSAL_SIGMA2, TERMS

    sigma2 = _swept_pooled("expanded")["sigma2"]
    for t in TERMS:
        assert abs(sigma2[t] / PROPOSAL_SIGMA2[t] - 1.0) < 0.30, t
    assert math.log10(max(sigma2.values())) - math.log10(min(sigma2.values())) >= 20.0


def test_reference_floor_below_unity_with_stated_operator() -> None:
    """r03 Section 4: with the discrete operator of dPV (its interface currents) every rho_i is far
    below one on every swept device."""
    from measurement import TERMS, measure
    from sweep import SWEEP_SIZE

    worst = max(measure(i)["expanded"][t]["f"] / measure(i)["expanded"][t]["sigma2"]
                for i in range(SWEEP_SIZE) for t in TERMS)
    assert worst < 1e-3


def test_relative_residual_is_bounded_by_one() -> None:
    """Eq. (10) of r04: |R_hat_i| < 1 at every node, whatever the magnitude of the summands."""
    from Staged_PINN_Solar import loss as L
    rng = np.random.default_rng(0)
    big = (rng.standard_normal((4, 50)) * 1e-7, rng.standard_normal((4, 50)) * 1e-9)
    tiny = (rng.standard_normal((4, 50)) * 1e-22, rng.standard_normal((4, 50)) * 1e-22)
    eps = {"n": 1e-11, "p": 1e-11}
    for name, t in (("n", big), ("p", tiny)):
        r = L.residual(t) / (L.scale(t) + eps[name])
        assert np.all(np.abs(r) < 1.0)
    out = L.relative_losses({"n": big, "p": tiny}, eps)
    assert 0.0 <= out["n"] < 1.0 and 0.0 <= out["p"] < 1.0


def test_relative_loss_floor_is_the_root_of_sigma_squared() -> None:
    from Staged_PINN_Solar import loss as L
    assert L.residual_floor({"n": 1.70e-22, "poisson": 2.52e-3}) == pytest.approx({"n": np.sqrt(1.70e-22), "poisson": np.sqrt(2.52e-3)})


def test_relative_loss_weighs_dark_and_bright_nodes_alike() -> None:
    """The defect of r03 measured in the first wave: a global sigma^2 lets the bright nodes (summands ~1e-7)
    swamp the dark ones (~1e-22). The same relative error at both kinds of node gives the same relative loss;
    the global normalization gives losses 1e30 apart."""
    from Staged_PINN_Solar import loss as L
    bright = (np.full((1, 10), 1e-7), np.full((1, 10), -1e-7 * 0.9))      # 10 % relative error
    dark = (np.full((1, 10), 1e-22), np.full((1, 10), -1e-22 * 0.9))
    eps = {"b": 1e-30, "d": 1e-30}
    rel = L.relative_losses({"b": bright, "d": dark}, eps)
    assert rel["b"] == pytest.approx(rel["d"], rel=1e-6)
    glob = L.normalized_losses({"b": L.residual(bright), "d": L.residual(dark)}, {"b": 1.7e-22, "d": 1.7e-22})
    assert glob["b"] / glob["d"] > 1e29
