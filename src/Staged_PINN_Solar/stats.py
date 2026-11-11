"""The common decision rule of the protocol (experiments v03), applied to per-device PCE errors.

For each arm and test device the relative PCE error is averaged over the 5 seeds (`seed_average`);
two arms are compared with a paired two-sided Wilcoxon over the test devices; Holm is applied to the
p-values of one experiment's family (never to intervals); the paired bootstrap 95 % CI of the median
per-device difference (10000 resamples) is the effect size and fixes its sign. A comparison is
declared for an arm only if its Holm p < 0.05 and the CI excludes 0 with that sign.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

__provenance__ = {
    "revision": "experiments-protocolo-experimental-de-la-pinn-por-etapas-sobre-dpv-v03.md",
    "sections": ["E1", "E2", "E3", "E4", "E5"],
    "experiments": ["E1", "E2", "E3", "E4", "E5"],
    "invariants": [
        "wilcoxon_matches_scipy",
        "holm_matches_hand_computation",
        "holm_never_below_raw_p_and_never_above_one",
        "bootstrap_ci_of_median_difference_brackets_true_shift",
        "decision_requires_holm_and_ci_with_same_sign",
        "per_device_error_is_averaged_over_seeds",
        "compare_arms_holm_within_family",
    ],
}


def seed_average(runs: dict) -> tuple:
    """`runs`: seed -> {uid: error}. Returns (sorted uids, per-device mean over seeds, NaN-aware)."""
    uids = sorted({u for r in runs.values() for u in r})
    mat = np.array([[runs[s].get(u, np.nan) for u in uids] for s in sorted(runs)], dtype=np.float64)
    with np.errstate(all="ignore"):
        mean = np.nanmean(mat, axis=0)
    return uids, mean


def paired_wilcoxon(a, b) -> float:
    """p-value of the paired two-sided Wilcoxon signed-rank test of a against b (scipy defaults)."""
    return float(sps.wilcoxon(np.asarray(a, float), np.asarray(b, float), alternative="two-sided").pvalue)


def holm(p) -> list:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return [float(x) for x in adj]


def bootstrap_median_diff_ci(a, b, resamples: int = 10000, level: float = 0.95, seed: int = 0) -> tuple:
    """Paired percentile bootstrap CI of median(a - b) over devices."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(resamples, d.size))
    meds = np.median(d[idx], axis=1)
    lo, hi = np.quantile(meds, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def verdict(p_holm: float, ci: tuple, alpha: float = 0.05) -> str:
    """"A" if A has the lower error (diff = A - B), "B" for the opposite, else "none"."""
    if not p_holm < alpha:
        return "none"
    lo, hi = ci
    if hi < 0.0:
        return "A"
    if lo > 0.0:
        return "B"
    return "none"


def compare_family(family: dict, resamples: int = 10000, level: float = 0.95, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    """`family`: name -> (errors_A, errors_B) per test device. Holm is applied across the family."""
    names = list(family)
    raw = [paired_wilcoxon(*family[n]) for n in names]
    adj = holm(raw)
    out = {}
    for n, p, ph in zip(names, raw, adj):
        a, b = (np.asarray(x, float) for x in family[n])
        ci = bootstrap_median_diff_ci(a, b, resamples, level, seed)
        out[n] = {"p": p, "p_holm": ph, "ci": list(ci), "median_diff": float(np.median(a - b)),
                  "median_a": float(np.median(a)), "median_b": float(np.median(b)),
                  "winner": verdict(ph, ci, alpha), "n_devices": int(a.size)}
    return out
