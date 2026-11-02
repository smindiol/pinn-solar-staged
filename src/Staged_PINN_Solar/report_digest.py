"""The seal that binds a report to the code that produced it.

An executed notebook says its cells ran once, not that they ran against this
code. Without a seal an old report and a freshly generated one look identical,
and the old one keeps being believed while the code moves out from under it.

This module stamps it. The notebook prints it at the end, verification
recomputes it, and if the two disagree the report is a relic.

It lives beside the method's own package because Flow A must be able to
stamp before any comparison has ever been offered: it produces no proof of
its own and so declares no `__provenance__`, but every notebook that proves
something needs to import it, and Flow A's notebook runs before a benchmark
package exists at all.

**The two halves have to produce the same number.** The destination writes this
one and verification recomputes the other, so testing each against a fixture of
its own would verify both halves and never their union, which is the only thing
that matters here. The forge has a test that runs both over the same tree and
compares them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: What the notebook prints before the digest. Verification looks for it literally.
MARKER = "SOURCES-SHA256"


def source_digest(repository: Path, package: str) -> str:
    """One hash over everything a report's claims depend on.

    Modification times cannot serve here: a clone rewrites all of them with the
    checkout time and the ordering is gone. Content can.

    It covers all of `src/`, and nothing else. That boundary is the claim: a
    report depends on the code the run executes, and on nothing else.

    Naming the packages one by one failed in both directions. It left out prior
    work, which the benchmark imports — moving what an arm computes there left
    the notebooks reporting `executed` over stale numbers, and the separate
    session that goes and fixes prior work and comes back is precisely the case
    that triggers it. And it pulled in `tests/`, which no notebook imports:
    adding any test at all marked every report in the repository stale and asked
    for the campaign to be re-run to restamp a hash.

    The benchmark package stays inside, now by belonging to `src/` rather than
    by being named, and for the same reason as before: it is the module that
    renders the tables and writes the conclusions. Leaving it out allowed a
    conclusion to be corrected while the record went on asserting the old one
    with everything green.

    `package` no longer chooses what goes in. It stays in the signature because
    the two halves have to be called the same way and the destination passes it
    from `_here()`.
    """
    digest = hashlib.sha256()
    root = repository / "src"
    if root.is_dir():
        for file in sorted(root.rglob("*.py")):
            if "__pycache__" in file.parts:
                continue
            digest.update(str(file.relative_to(repository)).encode("utf-8"))
            digest.update(file.read_bytes())
    return digest.hexdigest()


def _here() -> tuple[Path, str]:
    """The repository and the package, deduced from where this file lives.

    It takes no arguments on purpose. The previous version asked for them, and
    the first notebook that did not use exactly the same names as the rest
    failed to stamp and was reported stale — the seal left out precisely the
    notebook that departed from the mould, which is the one that most needs
    watching.

    This file lives in `<repo>/src/<Package>/`, so both facts are in its own
    path and no notebook has to know them: `parents[1]` of its own directory
    is the repository root, and its own directory's name is the package,
    verbatim, with no suffix to strip.
    """
    package_dir = Path(__file__).resolve().parent
    return package_dir.parents[1], package_dir.name


def stamp(repository: Path | None = None, package: str | None = None) -> str:
    """The line the notebook prints to record what it ran against.

    It is called without arguments from any notebook; it accepts them only so it
    can be tested against a tree that is not this one.
    """
    if repository is None or package is None:
        found_repository, found_package = _here()
        repository = repository or found_repository
        package = package or found_package
    return f"{MARKER} {source_digest(repository, package)}"
