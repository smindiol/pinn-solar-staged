"""Deterministic fixtures. Fixed seed, no data, no network.

The seed is chosen per run so that levels 2 and 3 vary between runs while
staying reproducible within one.

Backend-neutral on purpose: this file imports no array library, so it never
decides which backend the implementation computes with (`src/<Package>/
__init__.py`'s `__backend__` does). Build the backend's own generator from
`SEED` where a test needs one, e.g. `numpy.random.default_rng(SEED)` or
`torch.Generator().manual_seed(SEED)`.
"""

from __future__ import annotations

import ast
import random
import re

import pytest

SEED = 42
TOL = 1e-10

#: A kit placeholder, spelled without writing one (this file is itself a
#: template, and a literal one here would be substituted away).
_PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")


class _UnansweredPlaceholders(pytest.Item):
    """Stands in for a test file that cannot be collected because kit
    placeholders still sit where Python needs identifiers. It fails, naming
    them, instead of the collection error a bare `SyntaxError` would be."""

    def __init__(self, *, placeholders, **kwargs):
        super().__init__(**kwargs)
        self.placeholders = placeholders

    def runtest(self):
        raise AssertionError(
            f"{self.path.name} still carries kit placeholders "
            f"{', '.join(self.placeholders)}: they are step 9's answers (the "
            "module, the function and the invariant or expectation it "
            "asserts). Author this file over the scaffolding, then declare it "
            f"authored with `implementation_cli.py` `--authored tests/{self.path.name}`.")

    def repr_failure(self, excinfo):
        return str(excinfo.value)

    def reportinfo(self):
        return self.path, None, f"{self.path.name}: unanswered kit placeholders"


class _UnansweredFile(pytest.File):
    def collect(self):
        text = self.path.read_text(encoding="utf-8")
        yield _UnansweredPlaceholders.from_parent(
            self, name="unanswered_kit_placeholders",
            placeholders=sorted(set(_PLACEHOLDER.findall(text))))


def pytest_pycollect_makemodule(module_path, parent):
    """A test file that does not parse AND still carries placeholders is
    collected as one failing item saying which; every other file is
    collected exactly as pytest would."""
    try:
        text = module_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not _PLACEHOLDER.search(text):
        return None
    try:
        ast.parse(text)
    except SyntaxError:
        return _UnansweredFile.from_parent(parent, path=module_path)
    return None


@pytest.fixture
def seed() -> int:
    return SEED


@pytest.fixture
def rng() -> random.Random:
    """A seeded standard-library generator; see the module docstring for a
    backend's own."""
    return random.Random(SEED)
