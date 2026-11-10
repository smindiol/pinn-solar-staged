"""Level 2 invariants of pce.py (Eqs. 13-14: terminal current, calcPmax, J-V summary, pce_check)."""

from __future__ import annotations

import numpy as np
import pytest

from Staged_PINN_Solar import data, pce
from Staged_PINN_Solar.scaling import Scales


def test_calcpmax_exact_on_quadratic_power() -> None:
    """Eq. 14: dPV's spline is linear on its first piece (a = 0) and quadratic with continuous slope after it,
    so a power curve of that form is reproduced exactly: P = s V on [0, v1], then P = s V + a (V - v1)^2 with a < 0
    has its maximum at V = v1 - s / (2 a), of value s V* + a (V* - v1)^2."""
    s_, a_, v1 = 3.0, -2.0, 0.2
    v = np.linspace(0.0, 1.4, 8)
    power = np.where(v <= v1, s_ * v, s_ * v + a_ * (v - v1) ** 2)
    j = np.divide(power, v, out=np.full_like(v, s_), where=v > 0)
    pmax, vmax = pce.calc_pmax(v, j)
    v_star = v1 - s_ / (2 * a_)
    assert vmax == pytest.approx(v_star, rel=1e-9)
    assert pmax == pytest.approx(s_ * v_star + a_ * (v_star - v1) ** 2, rel=1e-9)


def test_sweep_is_cut_where_dpv_stops() -> None:
    """r10 Eq. 14: the curve stops at the first k >= 2 with J_{k-1} J_k <= 0 or J_k < 0."""
    j = np.array([20.0, 19.0, 15.0, 4.0, -3.0, -9.0])
    assert pce.sweep_length(j) == 5
    assert pce.sweep_length(np.array([5.0, 4.0, 3.0, 2.0])) == 4


def test_jv_summary_of_a_known_diode_curve() -> None:
    """J(V) = Jsc (1 - (exp(V/Vt) - 1)/(exp(Voc/Vt) - 1)): PCE = Pmax / P_in, FF = Pmax/(Voc Jsc)."""
    v = 0.05 * np.arange(0, 21)
    jsc, voc = 0.02, 0.9                                     # A/cm^2, V
    j = jsc * (1.0 - (np.exp(v / 0.0259) - 1.0) / (np.exp(voc / 0.0259) - 1.0))
    s = pce.jv_summary(v, j, p_in_std=900.0)
    assert s["n_voltages"] == int(np.argmax(j < 0)) + 1 or s["n_voltages"] <= len(v)
    assert 0.0 < s["pce"] < 0.2 and s["pce"] == pytest.approx(s["pmax"] / 900.0)
    assert s["jsc"] == pytest.approx(jsc * 1e4)
    assert 0.5 < s["ff"] < 1.0 and s["voc"] == pytest.approx(voc, abs=0.05)


def test_relative_error_percent() -> None:
    assert pce.relative_error_percent(0.18, 0.20) == pytest.approx(10.0)
    assert np.isnan(pce.relative_error_percent(float("nan"), 0.2))


needs_dataset = pytest.mark.skipif(not (data.default_root() / "s1").exists(),
                                   reason="dPV reference set not available")


@needs_dataset
def test_stored_pce_reproduced_on_dataset_devices() -> None:
    """Prerequisite of the protocol: the evaluator reproduces Device.pce from the stored currents."""
    refs = data.inventory()[:6] + data.inventory()[-6:]
    rec = pce.check_devices(refs, Scales(), tolerance=1e-6)
    assert rec["n_devices"] == 12 and rec["passed"] is True
    assert max(r["rel_error"] for r in rec["devices"].values()) < 1e-6


def test_pce_check_records_tolerance_and_verdict() -> None:
    """The output carries the explicit tolerance and a pass/fail that can be False."""
    class _Dev:                                   # one curve whose stored pce disagrees with its currents
        pass
    scales = Scales()
    n = 30
    V = 0.05 * np.arange(6)
    jd = np.array([20.0, 19.0, 15.0, 4.0, -3.0, -9.0]) * 1e-3 / scales.J0
    dev = _Dev()
    dev.V, dev.J_n, dev.J_p = V, np.tile(jd[:, None], (1, n)), np.zeros((6, n))
    dev.pce, dev.p_in_std = 0.5, 900.0
    dev.ref = type("R", (), {"uid": "x"})()
    rec = pce.check_device_objects([dev], scales, tolerance=1e-6)
    assert rec["tolerance"] == 1e-6 and rec["passed"] is False
    assert rec["devices"]["x"]["rel_error"] > 1e-6
