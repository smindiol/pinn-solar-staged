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
    {
        "id": "eq10_summand_grouping",
        "document": "proposal",
        "kind": "underspecified",
        "equations": ["10"],
        "remedy_equations": ["10"],
        "uses": [r"\sigma_i^2", r"T_{iq}", r"\mathcal{R}_i"],
        "introduces": [],
        "adoption": {
            "absent": r"f_i = \bigl\langle \mathcal{R}_i^{2} \bigr\rangle_{\text{ref}}, \qquad i \in \mathcal{I}.",
            "expect": [r"\mathcal{R}_i = \sum_q T_{iq}"],
        },
        "becomes_invariant": "sigma_squared_elementary_terms_fixed",
        "status": "theorem",
        "rate": "200/200 devices: the contact terms differ by at least 9 orders of magnitude between the two readings",
        "statement": (
            "Eq. (10) sums |T_iq| but Eqs. (5), (7) and (9) do not fix what a summand T_iq is: "
            "reading each parenthesis as one summand and expanding every signed term give sigma_i^2 "
            "that differ by 9.8 to 16.1 orders of magnitude for the contact terms BCn and BCp on "
            "every one of the 200 swept devices. Only the expanded reading reproduces the proposal's "
            "own table of Section 4 and its 21-order spread of sigma^2; the parenthesis reading "
            "shrinks that spread to about 17 orders and changes every normalized weight lambda_i."
        ),
        "remedy": (
            "State that T_iq are the separately signed terms of R_i after expanding every "
            "parenthesis (n, p and Ndop in Poisson; G and R in the continuity equations; the flux and "
            "the two density terms in each contact residual). R_i = sum_q T_iq is the only extra symbol, "
            "already used in Eq. (10)."
        ),
        "remedy_block": (
            "$$\n"
            r"\sigma_i^2 = \Bigl\langle \Bigl(\sum_q \bigl|T_{iq}\bigr|\Bigr)^{2} \Bigr\rangle_{\text{ref}}, "
            r"\qquad f_i = \bigl\langle \mathcal{R}_i^{2} \bigr\rangle_{\text{ref}}, "
            r"\qquad \mathcal{R}_i = \sum_q T_{iq}, \qquad i \in \mathcal{I},"
            "\n"
            r"\text{with } T_{iq} \text{ the separately signed terms of } \mathcal{R}_i \text{ once every parenthesis is expanded.}"
            "\n"
            r"\tag{10}"
            "\n$$"
        ),
    },
    {
        "id": "floor_claim_depends_on_operator",
        "document": "proposal",
        "kind": "overstated-claim",
        "equations": ["10", "11"],
        "remedy_equations": [],
        "remedy_in_prose": True,
        "uses": [r"\rho_i", r"f_i", r"\partial\text{PV}"],
        "introduces": [],
        "adoption": {
            "absent": r"Todos los $\rho_i$ son mucho menores que uno: el piso de $\partial\text{PV}$ no limita el entrenamiento",
            "expect": ["operador discreto"],
        },
        "becomes_invariant": "reference_floor_below_unity_with_stated_operator",
        "status": "tendency",
        "rate": "78/200 devices have max(rho_n, rho_p) >= 0.1 when the currents are formed from Eq. (6) on the nodes; 0/200 with the dPV interface currents",
        "statement": (
            "Section 4 states that all rho_i are far below one, so the dPV floor does not limit training, "
            "but f_n and f_p depend on how the reference's currents are evaluated. With dPV's own "
            "Scharfetter-Gummel interface currents the largest rho over the 200 devices is 1e-6; forming "
            "the currents of Eq. (6) by differences on the nodes gives rho_n or rho_p of at least 0.1 on "
            "78 of 200 devices (up to 1.0). The conclusion holds for one discrete operator and the "
            "proposal does not say which."
        ),
        "remedy": (
            "Say in Section 4 that f_i is evaluated with the discrete operator of dPV (Scharfetter-Gummel "
            "interface currents), and that the conclusion rho_i << 1 is stated for that operator only."
        ),
    },
]
