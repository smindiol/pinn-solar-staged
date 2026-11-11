"""Level 2 invariants of units.py (independent units of work that campaign mode shards)."""

from __future__ import annotations

import pytest

from Staged_PINN_Solar import units as U


def test_unit_counts_are_30_40_10_20() -> None:
    """Protocol v03 costs: E1 30 runs (3 losses), E2 40, E5 10 new, E4 20 (2 optimizers x 2 archs x 5 seeds)."""
    assert [len(U.enumerate_units(s)) for s in ("e1", "e2", "e5", "e4")] == [30, 40, 10, 20]
    assert len(U.enumerate_units("pce_check")) == len(U.enumerate_units("reference")) == 1
    assert len(U.enumerate_units("analysis")) == 1


def test_unit_ids_are_unique_parseable_and_filesystem_safe() -> None:
    seen = set()
    for step in U.TRAINING_STEPS:
        for uid in U.enumerate_units(step):
            assert uid not in seen and uid.replace("-", "").replace("_", "").isalnum()
            seen.add(uid)
            spec = U.parse_unit(uid)
            assert spec.step == step and U.unit_id(spec) == uid
    assert len(seen) == 100


def test_arms_follow_the_protocol() -> None:
    e1 = {U.parse_unit(u) for u in U.enumerate_units("e1")}
    assert {(s.arch, s.loss, s.optimizer, s.weighting) for s in e1} == {
        (a, l, "lbfgs", "fixed") for a in "AB" for l in ("unnormalized", "normalized", "relative")}
    e2 = {U.parse_unit(u) for u in U.enumerate_units("e2")}
    assert {(s.weighting, s.optimizer, s.arch) for s in e2} == {
        (w, o, a) for w in ("fixed", "gradient") for o in ("lbfgs", "soap") for a in "AB"}
    assert all(s.reset_memory and s.loss == "relative" for s in e2)
    e5 = {U.parse_unit(u) for u in U.enumerate_units("e5")}
    assert all(s.loss == "relative" for s in e5)
    e5 = {U.parse_unit(u) for u in U.enumerate_units("e5")}
    assert all(s.optimizer == "lbfgs" and s.weighting == "gradient" and not s.reset_memory for s in e5)
    e4 = {U.parse_unit(u) for u in U.enumerate_units("e4")}
    assert {(s.optimizer, s.arch) for s in e4} == {(o, a) for o in ("lbfgs", "soap") for a in "AB"}
    assert {s.seed for s in e4} == set(range(5))


def test_shards_cover_without_overlap() -> None:
    allu = U.enumerate_units("e2")
    shards = U.shard(allu, 4)
    assert sorted(sum(shards, [])) == sorted(allu) and len(shards) == 4
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_unknown_unit_is_refused() -> None:
    with pytest.raises(ValueError):
        U.parse_unit("e9-A-lbfgs-s0")


def test_shard_token_expands_to_its_round_robin_slice() -> None:
    """A requested id `shard-k-of-n` stands for `units.shard(every, n)[k]`, so one kernel carries a whole
    shard; the n shards are disjoint and exhaustive, and a malformed or out-of-range token is refused."""
    import pytest
    from Staged_PINN_Solar import steps, config as C

    every = U.enumerate_units("e1")
    got = [steps._requested("e1", [f"shard-{k}-of-3"], C.Config(), False) for k in range(3)]
    assert got == U.shard(every, 3)
    assert sorted(sum(got, [])) == sorted(every)
    mixed = steps._requested("e1", ["shard-0-of-30", every[5]], C.Config(), False)
    assert mixed == [every[0], every[5]]
    for bad in ("shard-3-of-3", "shard-0-of-0", "shard-x-of-2"):
        with pytest.raises(ValueError):
            steps._requested("e1", [bad], C.Config(), False)
