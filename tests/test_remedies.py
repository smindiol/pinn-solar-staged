"""Level 5 - remedies. Each proposed correction, validated at the same rigour.

A remedy is accepted only when the sweep shows BOTH poles: the correction
satisfies the criterion, and the declared formulation fails it. With one pole
only, nothing distinguishes a real improvement from a measurement that would
have passed whatever it was handed — `verify` reports such a test as a remedy
without control.

Every test starts by refusing to measure what was not ruled admissible:
soundness and expressibility are decided first, efficacy second.

The proposed replacements live here, never in src/.
"""

from __future__ import annotations

from admissibility import require_admissible
from sweep import SWEEP_SIZE, sweep  # noqa: F401


# def remedy_<name>(...):
#     """PROPOSED Eq. (<n>): <the replacement>."""


# def test_remedy_<finding_id>() -> None:
#     """Resolves it: ...   Preserves: ...   Control: the declared form fails it."""
#     require_admissible("<finding_id>")


import numpy as np

from Staged_PINN_Solar import data, loss
from Staged_PINN_Solar.residuals import residual, scale
from Staged_PINN_Solar.scaling import Scales
from measurement import PROPOSAL_SIGMA2, TERMS, measure, pooled
from sweep import configuration

ALL = range(SWEEP_SIZE)


def declared_formulation(index: int, **reading) -> dict:
    """The declared formulation of Eq. (10), evaluated by the package on one swept device:
    {term: (sigma2, f)}. `reading` selects how the unspecified parts of the text are read."""
    terms = loss.reference_terms(data.load_device(configuration(index)["ref"]), Scales(), **reading)
    return {t: (float(np.mean(scale(v).astype(np.float64) ** 2)),
                float(np.mean(residual(v).astype(np.float64) ** 2))) for t, v in terms.items()}


def remedy_eq10_explicit_terms(reading: str) -> dict:
    """PROPOSED Eq. (10): T_iq are the separately signed terms of R_i after expanding every parenthesis."""
    return pooled(ALL, reading)["sigma2"]


def _spread_orders(sigma2: dict) -> float:
    import math
    return math.log10(max(sigma2.values())) - math.log10(min(sigma2.values()))


# Both remedies were adopted by r03 and moved to tests/test_loss.py as invariants.
