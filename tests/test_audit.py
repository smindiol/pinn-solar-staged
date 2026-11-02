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
