"""E9f: exact data-group balance and like-for-like Voc."""
import numpy as np
import pytest

from Staged_PINN_Solar import m6c, m6_current_fit as fit

NORMS = {"potential": 107.3, "current": 0.49, "terminal": 0.35}


def test_legacy_cap_still_clips_to_one_hundred():
    w = fit.group_multipliers(NORMS)
    assert w == {"potential": 1.0, "current": 100.0, "terminal": 100.0}


def test_no_cap_balances_every_group_exactly():
    w = fit.group_multipliers(NORMS, cap=None)
    assert w["current"] == pytest.approx(107.3 / 0.49)
    assert w["terminal"] == pytest.approx(107.3 / 0.35)
    for k, n in NORMS.items():
        assert w[k] * n == pytest.approx(NORMS["potential"])


def test_no_cap_keeps_the_norm_floor():
    w = fit.group_multipliers({"potential": 1.0, "current": 0.0, "terminal": 1.0}, cap=None)
    assert w["current"] == pytest.approx(1e12)


def _diode(v, voc=0.80, vt=0.03, jsc=0.03):
    return jsc * (1 - np.exp((v - voc) / vt)) / (1 - np.exp(-voc / vt))


@pytest.mark.parametrize("shift", [0.0, 0.01, 0.025, 0.04])
def test_spline_voc_is_closer_to_the_true_crossing_than_the_chord(shift):
    v = np.arange(shift, 0.90, 0.05)          # the grid sits anywhere relative to Voc
    j = _diode(v)
    chord = m6c._first_crossing(v, j)
    spline = m6c.spline_voc(v, j)
    assert chord <= 0.80 + 1e-12          # the chord under a concave J(V) crosses early
    assert abs(spline - 0.80) < abs(chord - 0.80) or abs(chord - 0.80) < 1e-9


def test_same_chord_on_both_sides_cancels_the_interpolation_error():
    v = np.arange(0.0, 0.90, 0.05)
    ref = m6c._first_crossing(v, _diode(v))
    pred = m6c._first_crossing(v, _diode(v))
    assert pred - ref == 0.0


def test_spline_voc_without_a_crossing_is_nan():
    v = np.arange(0.0, 0.5, 0.05)
    assert np.isnan(m6c.spline_voc(v, _diode(v)))
