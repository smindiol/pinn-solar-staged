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
