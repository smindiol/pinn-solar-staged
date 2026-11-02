"""Per-device measurements shared by the audit and remedy tests (read-only on the dPV files).

Three readings of Eq. (10) are evaluated on the stored reference solution of one device:
`expanded` (every signed term of a residual is its own summand T_iq, dPV interface currents),
`as_written` (each parenthesis of Eqs. 5, 7, 9 is one summand) and `continuous` (expanded, with
the currents of Eq. 6 formed on the nodes instead of read from dPV).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from Staged_PINN_Solar import data, loss
from Staged_PINN_Solar.residuals import residual, scale
from Staged_PINN_Solar.scaling import Scales
from sweep import configuration

TERMS = loss.TERMS
READINGS = {
    "expanded": dict(currents="dpv", grouping="expanded"),
    "as_written": dict(currents="dpv", grouping="as_written"),
    "continuous": dict(currents="continuous", grouping="expanded"),
}

#: sigma_i^2 of the proposal's own table in Section 4 (Eq. 10, measured by the author on E1A, seed 0).
PROPOSAL_SIGMA2 = {"poisson": 2.52e-3, "bcn": 0.204, "bcp": 0.334, "n": 1.70e-22, "p": 1.70e-22}


@lru_cache(maxsize=None)
def measure(index: int) -> dict:
    """{reading: {term: {"sigma2": mean (sum|T|)^2, "f": mean R^2, "count": n}}} for configuration `index`."""
    dev = data.load_device(configuration(index)["ref"])
    out = {}
    for name, kw in READINGS.items():
        terms = loss.reference_terms(dev, Scales(), **kw)
        out[name] = {
            t: {"sigma2": float(np.mean(scale(v).astype(np.float64) ** 2)),
                "f": float(np.mean(residual(v).astype(np.float64) ** 2)),
                "count": int(residual(v).size)}
            for t, v in terms.items()
        }
    return out


def pooled(indices, reading: str) -> dict:
    """Pooled sigma_i^2, f_i and rho_i over devices, weighting every residual sample equally."""
    acc = {t: [0.0, 0.0, 0] for t in TERMS}
    for i in indices:
        for t, m in measure(i)[reading].items():
            acc[t][0] += m["sigma2"] * m["count"]
            acc[t][1] += m["f"] * m["count"]
            acc[t][2] += m["count"]
    sigma2 = {t: a[0] / a[2] for t, a in acc.items()}
    f = {t: a[1] / a[2] for t, a in acc.items()}
    return {"sigma2": sigma2, "f": f, "rho": {t: f[t] / sigma2[t] for t in TERMS}}
