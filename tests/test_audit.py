"""Level 4 - audit. Evidence that each declared finding is a real defect.

Every test sweeps SWEEP_SIZE independent configurations and reports the rate. A
finding declared `theorem` must hold in all of them; one declared `tendency`
must match its declared rate and is never asserted as a law.
"""

from __future__ import annotations

from findings import FINDINGS
from sweep import SWEEP_SIZE, sweep  # noqa: F401


def test_every_finding_declares_a_remedy() -> None:
    """A finding without a proposed correction is a complaint, not a finding."""
    for finding in FINDINGS:
        assert finding["remedy"].strip(), f"{finding['id']} has no remedy"
        if finding.get("remedy_in_prose") is True:
            # A prose remedy rewrites text that carries no tag, so it names no
            # equation; naming one as well would contradict the declaration.
            assert not finding.get("remedy_equations"), (
                f"{finding['id']} declares a prose remedy and remedy equations")
        else:
            assert finding["remedy_equations"], f"{finding['id']} does not say what it changes"
        assert finding["becomes_invariant"], f"{finding['id']} names no invariant to become"
        assert finding["adoption"]["absent"], f"{finding['id']} has no adoption marker"
        assert finding["status"] in {"theorem", "tendency"}
        assert finding["kind"] in {
            "ill-formed", "underspecified", "missing-complement", "overstated-claim",
            "ill-posed-objective", "loose-constant",
        }


# One test_finding_<id> per declared finding, measuring the defect over sweep().


import math

from measurement import measure


def _log10_gap(index: int, term: str) -> float:
    m = measure(index)
    return math.log10(m["expanded"][term]["sigma2"]) - math.log10(m["as_written"][term]["sigma2"])


def test_finding_eq10_summand_grouping() -> None:
    """theorem: for BCn and BCp the two readings of T_iq differ by >= 9 orders on every device."""
    gaps = {t: [_log10_gap(i, t) for i in range(SWEEP_SIZE)] for t in ("bcn", "bcp")}
    for term, g in gaps.items():
        assert len(g) == SWEEP_SIZE
        assert min(g) >= 9.0, f"{term}: smallest gap {min(g):.2f} orders"


def test_finding_floor_claim_depends_on_operator() -> None:
    """tendency (78/200): rho_n or rho_p >= 0.1 under node-formed currents, never under dPV's operator."""
    rho = lambda i, reading, t: measure(i)[reading][t]["f"] / measure(i)[reading][t]["sigma2"]  # noqa: E731
    node = sum(max(rho(i, "continuous", "n"), rho(i, "continuous", "p")) >= 0.1 for i in range(SWEEP_SIZE))
    dpv = sum(max(rho(i, "expanded", t) for t in ("n", "p")) >= 0.1 for i in range(SWEEP_SIZE))
    assert node == 78, f"declared rate 78/200, measured {node}/200"
    assert dpv == 0
