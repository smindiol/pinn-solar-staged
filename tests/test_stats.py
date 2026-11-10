"""Level 2 invariants of stats.py (the common decision rule of the protocol)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from Staged_PINN_Solar import stats as S


def _pair(seed=7, n=60, shift=0.5):
    rng = np.random.default_rng(seed)
    a = rng.gamma(2.0, 1.0, n)
    return a, a + shift + 0.2 * rng.normal(size=n)


def test_wilcoxon_matches_scipy() -> None:
    """Protocol, common decision rule: paired two-sided Wilcoxon over test devices."""
    a, b = _pair()
    assert S.paired_wilcoxon(a, b) == pytest.approx(sps.wilcoxon(a, b, alternative="two-sided").pvalue, rel=1e-12)


def test_holm_matches_hand_computation() -> None:
    """Holm step-down: sorted p_(k) * (m - k + 1), made monotone, capped at 1."""
    p = [0.01, 0.04, 0.03, 0.20]
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> 0.09 (monotone), 0.20*1=0.20
    assert S.holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert S.holm([0.5]) == [0.5]
    assert S.holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def test_holm_never_below_raw_p_and_never_above_one() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=9)
    adj = np.asarray(S.holm(p))
    assert np.all(adj >= p - 1e-15) and np.all(adj <= 1.0)


def test_bootstrap_ci_of_median_difference_brackets_true_shift() -> None:
    """Paired bootstrap 95 % CI of the median of per-device differences (10000 resamples)."""
    a, b = _pair(shift=0.5, n=400)
    lo, hi = S.bootstrap_median_diff_ci(a, b, resamples=10000, level=0.95, seed=0)
    med = float(np.median(a - b))
    assert lo <= med <= hi
    assert hi < 0.0                       # a is lower than b by about 0.5
    assert S.bootstrap_median_diff_ci(a, b, 10000, 0.95, 0) == (lo, hi)   # seeded


def test_decision_requires_holm_and_ci_with_same_sign() -> None:
    """A comparison is declared for an arm only if Holm p < 0.05 AND the CI excludes 0 on that side."""
    assert S.verdict(p_holm=0.01, ci=(-0.4, -0.1)) == "A"      # diff = A - B < 0 -> A has lower error
    assert S.verdict(p_holm=0.01, ci=(0.1, 0.4)) == "B"
    assert S.verdict(p_holm=0.01, ci=(-0.1, 0.4)) == "none"    # CI includes 0
    assert S.verdict(p_holm=0.20, ci=(-0.4, -0.1)) == "none"   # Holm not significant


def test_per_device_error_is_averaged_over_seeds() -> None:
    """Each test device contributes ONE value per arm: its error averaged over the seeds."""
    runs = {0: {"d1": 1.0, "d2": 4.0}, 1: {"d1": 3.0, "d2": 8.0}}
    uids, mean = S.seed_average(runs)
    assert uids == ["d1", "d2"]
    assert list(mean) == [2.0, 6.0]


def test_compare_arms_holm_within_family() -> None:
    rng = np.random.default_rng(3)
    base = rng.gamma(2.0, 1.0, 200)
    fam = {"c1": (base, base + 0.4 + 0.1 * rng.normal(size=200)),
           "c2": (base, base + 0.0 * rng.normal(size=200) + rng.normal(size=200))}
    out = S.compare_family(fam, resamples=500, seed=1)
    assert set(out) == {"c1", "c2"}
    raw = {k: S.paired_wilcoxon(*v) for k, v in fam.items()}
    holm = S.holm(list(raw.values()))
    for (k, r), h in zip(raw.items(), holm):
        assert out[k]["p"] == pytest.approx(r) and out[k]["p_holm"] == pytest.approx(h)
    assert out["c1"]["winner"] == "A"
