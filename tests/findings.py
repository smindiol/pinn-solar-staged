"""Audit findings on the proposal, each with the remedy it proposes.

A finding is a defect in the mathematics itself: a term that is ill-formed, a
claim stated more strongly than the construction supports, a missing complement
the rest of the development assumes, or a constant that does not hold up. It is
not a bug in this code.

The contract, checked statically by `implementation_cli.py verify`:

- every `id` has a `test_finding_<id>` proving the defect is real, measured over
  the sweep, and a `test_remedy_<id>` proving the fix resolves it AND preserves
  what was already established;
- `status` is `theorem` (holds across the whole sweep) or `tendency` (declares
  its measured rate and is never asserted as a law);
- `uses` is notation the remedy leans on and must appear verbatim in the
  revision; `introduces` is notation it would add, which keeps the audit at
  `needs-deliberation` because adding notation is the deliberation's call;
- `adoption.absent` is text present in the revision today whose disappearance
  signals the remedy was taken in; `expect` are forms that confirm it;
- `becomes_invariant` names what the remedy turns into once published: at that
  point the remedy test retires and the claim moves to the invariant suite.
  The finding itself stays here, with its `test_finding_<id>`, as evidence;
  once adopted it owes no `test_remedy_<id>` and no longer holds the audit
  below `ok`;
- `remedy_block` is the corrected equation written out, exactly as it should
  read in the document, carrying the same `\\tag{n}` it replaces. A finding of
  local reach that omits it cannot be settled inline: the handoff defers it to
  a session of its own, because describing a correction in prose is not the
  same as writing it, and nothing here will paraphrase mathematics into a
  document. Findings that add notation are structural anyway and need none;
- `remedy_in_prose: True` declares a defect that lives in the prose (an
  overstated sentence, a term used before it is defined) rather than in a
  numbered equation. Such a finding leaves `remedy_equations` empty and
  carries no `remedy_block`: there is no `\\tag{n}` to substitute at, so
  `verify` never asks for one and `handoff` defers it to a deliberation
  session of its own (`prose-remedy`). It still owes `uses`, `adoption` and
  both tests like any other finding. Declaring it beside remedy equations is
  ruled inadmissible.

Remedies live in the tests, never in src/. Establishing that a correction is
sound is not the same as adopting it.
"""

FINDINGS = [
    # {
    #     "id": "<short_snake_case_id>",
    #     "kind": "loose-constant",   # or ill-formed | underspecified |
    #                                 # missing-complement | overstated-claim |
    #                                 # ill-posed-objective
    #     "equations": ["<n>"],            # where the defect lives
    #     "remedy_equations": ["<n>"],     # what the correction would rewrite
    #     "uses": [r"<symbol present in the revision>"],
    #     "introduces": [],                # notation the remedy would add
    #     "adoption": {"absent": r"<text present today>", "expect": [r"<what replaces it>"]},
    #     "becomes_invariant": "<invariant_id_once_adopted>",
    #     "status": "theorem",
    #     "rate": "<measured rate over the sweep>",
    #     "statement": "<the defect, stated precisely>",
    #     "remedy": "<the correction, and why it is expressible in the proposal>",
    #     "remedy_block": "$$\n<the corrected equation>\n\\tag{<n>}\n$$",
    # },
    # A defect in the prose instead: no remedy equation, no remedy_block.
    # {
    #     "id": "<short_snake_case_id>",
    #     "kind": "overstated-claim",
    #     "equations": [],
    #     "remedy_equations": [],
    #     "remedy_in_prose": True,
    #     ...the remaining fields exactly as above, minus remedy_block...
    # },
]
