"""Independent units of work: one unit = one (experiment, arm, arch, seed) run.

Remote-execution campaign mode hands each healthy account a list of unit ids (`run_config["units"]`);
a step function runs exactly those. No unit reads another unit's output.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config

__provenance__ = {
    "revision": "experiments-protocolo-experimental-de-la-pinn-por-etapas-sobre-dpv-v03.md",
    "sections": ["E1", "E2", "E3", "E4", "E5"],
    "experiments": ["E1", "E2", "E3", "E4", "E5"],
    "invariants": [
        "unit_counts_are_30_40_10_20",
        "unit_ids_are_unique_parseable_and_filesystem_safe",
        "arms_follow_the_protocol",
        "shards_cover_without_overlap",
        "unknown_unit_is_refused",
    ],
}

TRAINING_STEPS = ("e1", "e2", "e5", "e4")
SINGLE_STEPS = ("pce_check", "reference", "analysis")
ARCHS = ("A", "B")
OPTIMIZERS = ("lbfgs", "soap")
LOSSES = ("unnormalized", "normalized", "relative")   # E1 arms: none | global sigma_i^2 (r03) | per-node relative (r04 Eq. 10)


@dataclass(frozen=True)
class UnitSpec:
    step: str
    arch: str
    optimizer: str
    loss: str            # "relative" (r04 Eq. 10) | "normalized" (r03 global sigma_i^2) | "unnormalized" (sigma_i^2 = 1)
    weighting: str       # "fixed" (lambda = 1) | "gradient" (Eqs. 12-13) | "bayes" (outer loop)
    reset_memory: bool   # L-BFGS memory reset when lambda is updated
    seed: int


def unit_id(s: UnitSpec) -> str:
    if s.step == "e1":
        return f"e1-{s.arch}-{s.loss}-s{s.seed}"
    if s.step == "e2":
        return f"e2-{s.arch}-{s.optimizer}-{s.weighting}-s{s.seed}"
    if s.step == "e5":
        return f"e5-{s.arch}-s{s.seed}"
    if s.step == "e4":
        return f"e4-{s.arch}-{s.optimizer}-s{s.seed}"
    raise ValueError(f"no training unit for step {s.step!r}")


def parse_unit(uid: str) -> UnitSpec:
    parts = uid.split("-")
    try:
        step, seed = parts[0], int(parts[-1][1:])
        if parts[-1][0] != "s":
            raise ValueError
        if step == "e1" and len(parts) == 4 and parts[1] in ARCHS and parts[2] in LOSSES:
            return UnitSpec("e1", parts[1], "lbfgs", parts[2], "fixed", True, seed)
        if step == "e2" and len(parts) == 5 and parts[1] in ARCHS and parts[2] in OPTIMIZERS \
                and parts[3] in ("fixed", "gradient"):
            return UnitSpec("e2", parts[1], parts[2], "relative", parts[3], True, seed)
        if step == "e5" and len(parts) == 3 and parts[1] in ARCHS:
            return UnitSpec("e5", parts[1], "lbfgs", "relative", "gradient", False, seed)
        if step == "e4" and len(parts) == 4 and parts[1] in ARCHS and parts[2] in OPTIMIZERS:
            return UnitSpec("e4", parts[1], parts[2], "relative", "bayes", True, seed)
    except (ValueError, IndexError):
        pass
    raise ValueError(f"not a unit id of this protocol: {uid!r}")


def enumerate_units(step: str, config: Config | None = None) -> list[str]:
    seeds = (config or Config()).seeds
    if step in SINGLE_STEPS:
        return [step]
    out = []
    if step == "e1":
        out = [f"e1-{a}-{l}-s{s}" for a in ARCHS for l in LOSSES for s in seeds]
    elif step == "e2":
        out = [f"e2-{a}-{o}-{w}-s{s}" for w in ("fixed", "gradient") for o in OPTIMIZERS
               for a in ARCHS for s in seeds]
    elif step == "e5":
        out = [f"e5-{a}-s{s}" for a in ARCHS for s in seeds]
    elif step == "e4":
        out = [f"e4-{a}-{o}-s{s}" for o in OPTIMIZERS for a in ARCHS for s in seeds]
    else:
        raise ValueError(f"unknown step {step!r}")
    return out


def shard(units: list[str], n: int) -> list[list[str]]:
    """Round-robin split of `units` over `n` workers: disjoint, exhaustive, sizes differ by at most one."""
    return [units[i::n] for i in range(n)]
