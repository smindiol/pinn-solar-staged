"""Level 1 - smoke. Does it run at all, on the smallest input?

Smoke never asserts mathematics; it asserts the code is reachable and every
public entry point returns something of the right shape. Mathematical claims
belong in test_invariants.py.

Until step 9 has run, this file is red on purpose: `MODULES` names a module
nobody has written yet, and `REVISION` the revision step 8 declares.
The engine's `--stage objects` (SKILL.md step 9) fills both in once they
are known.
"""

from __future__ import annotations

import importlib

MODULES = [
    "Staged_PINN_Solar.data",
    "Staged_PINN_Solar.scaling",
    "Staged_PINN_Solar.ansatz",
    "Staged_PINN_Solar.carriers",
    "Staged_PINN_Solar.residuals",
    "Staged_PINN_Solar.loss",
    "Staged_PINN_Solar.weights",
    "Staged_PINN_Solar.optimizers",
]
REVISION = "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r03.md"


def _unanswered(value: str) -> bool:
    return "{" + "{" in value


def test_package_imports() -> None:
    for module in MODULES:
        try:
            imported = importlib.import_module(module)
        except ModuleNotFoundError as error:
            if _unanswered(module):
                raise ModuleNotFoundError(
                    f"{error} -- step 9 has not named the module yet; "
                    "`implementation_cli.py` `--stage objects --module <name>` "
                    "writes it and fills its name in here") from error
            raise
        except SyntaxError as error:
            raise AssertionError(
                f"{module} does not parse ({error.msg}, line {error.lineno}); "
                "if it still carries kit placeholders, step 9 has not authored "
                "it yet") from error
        assert imported is not None


def test_every_module_declares_provenance() -> None:
    assert not _unanswered(REVISION), (
        "tests/test_smoke.py does not know the bound revision yet: it was "
        "scaffolded before step 8 declared one. The engine's `--stage objects` "
        "fills it in once `__implementation__['revision']` is declared")
    for module in MODULES:
        provenance = getattr(importlib.import_module(module), "__provenance__", None)
        assert provenance is not None, f"{module} has no __provenance__"
        assert provenance["revision"] == REVISION
