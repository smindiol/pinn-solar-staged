"""The admissibility ruling, read before any remedy is measured.

Order is the point. Whether a correction is sound and expressible inside the
proposal is decided first; whether it resolves the finding is measured only
afterwards. Running the sweep on a remedy that cites a missing equation, or
leans on notation the revision never defines, produces numbers that read as
evidence and lend rigour to something that should not have reached the bench.

The ruling is produced by `implementation_cli.py admit --revision <rNN>`, which
reads the revision in the forge and writes only its verdict here. Without it,
every remedy test fails immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

RULING_PATH = Path(__file__).with_name("admissibility.json")

#: The revision this kit was scaffolded against -- the same token
#: `test_smoke.py` checks every module's own `__provenance__["revision"]`
#: against. `require_admissible` reads it too, so a ruling made against a
#: revision this suite was not written for is refused at measure time
#: instead of only surfacing later, at `verify`'s sha256 staleness check.
CURRENT_REVISION = "{{REVISION}}"

#: Still carrying its placeholder: this file was scaffolded before the bound
#: revision was known. The engine's `--stage objects` (SKILL.md step 9) fills
#: it in -- and re-seals the file -- once step 8 has declared `__implementation__`'s
#: `revision`; until then no ruling can be matched against it.
REVISION_UNANSWERED = CURRENT_REVISION.startswith("{" + "{")


def _ruling() -> dict:
    if not RULING_PATH.exists():
        raise AssertionError(
            "no admissibility ruling: run `implementation_cli.py admit --revision <rNN>` "
            "before measuring any remedy")
    return json.loads(RULING_PATH.read_text(encoding="utf-8"))


def require_admissible(finding_id: str) -> None:
    """Refuse to measure a remedy that was not ruled admissible, or whose
    ruling was made against a different revision than this suite is written
    for. Without this, a stale ruling on disk is silently trusted at measure
    time and the mismatch surfaces only later, at `verify`.
    """
    if REVISION_UNANSWERED:
        raise AssertionError(
            "tests/admissibility.py does not know the revision it is written "
            "against yet: it was scaffolded before step 8 declared one. Declare "
            "`__implementation__['revision']`, then run the engine's `--stage "
            "objects` (SKILL.md step 9), which fills it in, before measuring "
            "any remedy")
    revision = ruled_revision()
    if revision != CURRENT_REVISION:
        raise AssertionError(
            f"admissibility was ruled against revision {revision!r}, but "
            f"this suite is written against {CURRENT_REVISION!r}; re-run "
            f"`implementation_cli.py admit --revision {CURRENT_REVISION}` "
            "before measuring any remedy")
    verdicts = _ruling().get("findings", {})
    if finding_id not in verdicts:
        raise AssertionError(f"{finding_id} was never ruled on; efficacy cannot be measured")
    verdict = verdicts[finding_id]
    if not verdict.get("admissible"):
        raise AssertionError(
            f"{finding_id} is not admissible, so its efficacy is not measured: "
            + "; ".join(verdict.get("reasons", [])))


def ruled_revision() -> str:
    return _ruling().get("revision", "")
