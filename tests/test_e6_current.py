"""r09 Eqs. 23-24: the current term of the E6 loss (experiments v09, E6)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import config as C
from Staged_PINN_Solar import e6, training

CFG = C.smoke().with_(max_stages=3)


@pytest.fixture(scope="module")
def ctx():
    prep = training.prepare(config=CFG)
    K = 3
    rng = np.random.default_rng(0)
    cur = {"train": rng.uniform(1e-3, 2e-2, size=(len(prep.train), K))}
    c = SimpleNamespace(prep=prep, cfg=CFG, K=K, currents=cur, dev=torch.device("cpu"),
                        sigma2=dict(prep.sigma2), fields=None)
    b = prep.bounds
    c.scale = torch.tensor([[b.D_phi, b.phi_n[1] - b.phi_n[0], b.phi_p[1] - b.phi_p[0]]], dtype=torch.float64)
    c.V = lambda j: j * CFG.voltage_step / prep.scales.V_th
    return c


def test_current_batch_is_fixed_per_seed_and_stage_on_the_full_mesh(ctx) -> None:
    a, b, c = e6.current_batch(ctx, 1, 0, 4), e6.current_batch(ctx, 1, 0, 4), e6.current_batch(ctx, 1, 1, 4)
    assert np.array_equal(a["devices"], b["devices"])
    assert a["x"].shape[0] == a["n"] * ctx.prep.train.x_hat.shape[1]          # every node of every device
    assert torch.equal(a["J_ref"], torch.tensor(ctx.currents["train"][a["devices"], 1]))
    assert not np.array_equal(a["devices"], c["devices"]) or len(ctx.prep.train) <= 4


def test_gr_current_torch_equals_the_evaluator_and_has_a_gradient(ctx) -> None:
    model = e6.new_model(ctx, 0)
    cb = e6.current_batch(ctx, 0, 0, 4)
    J = e6.gr_current_torch(model, cb, ctx.V(0), ctx)
    bank = ctx.prep.train
    sub = training.Bank(*[[getattr(bank, f)[i] for i in cb["devices"]] if isinstance(getattr(bank, f), list)
                          else getattr(bank, f)[cb["devices"]] for f in bank.__dataclass_fields__])
    ref = training.evaluate_currents(model, sub, ctx.V(0), ctx.prep.material, ctx.prep.scales, ctx.dev, 2)
    assert np.allclose(J.detach().numpy(), ref, rtol=1e-10, atol=1e-14)
    L = e6.current_loss(model, cb, ctx.V(0), ctx)
    g = torch.autograd.grad(L, list(model.parameters()))
    flat = torch.cat([x.reshape(-1) for x in g])
    assert torch.isfinite(flat).all() and flat.abs().max() > 0


def test_objective_has_the_current_term_only_when_given_its_batch(ctx) -> None:
    model = e6.new_model(ctx, 0)
    pb = e6.physics_batch(ctx, 0, 0) if ctx.fields is not None else None
    if pb is None:
        idx = np.tile(np.arange(4), (2, 1))
        pb = training._gather(ctx.prep.train, np.array([0, 1]), idx, ctx.dev)
        pb["target"] = torch.zeros(pb["x"].shape[0], 3, dtype=torch.float64)
    lam = {k: 1.0 for k in e6.TERMS}
    without = e6.PhasedObjective(ctx, model, pb, 0)
    with_j = e6.PhasedObjective(ctx, model, pb, 0, jbatch=e6.current_batch(ctx, 0, 0, 4))
    _, t0, _ = without.terms_and_grad(lam, list(model.parameters()))
    _, t1, _ = with_j.terms_and_grad({**lam, "J": 1.0}, list(model.parameters()))
    assert "J" not in t0 and "J" in t1 and t1["J"] > 0
    assert {k: t0[k] for k in t0} == pytest.approx({k: t1[k] for k in t0})
    assert "J" in with_j.term_grads(list(model.parameters()))


def test_smoke_verdict_applies_the_v09_rule() -> None:
    row = lambda j, c, f, d=False: {"stage": j, "after": {"current": c, "field": f}, "diverged": d}
    judged = e6.SMOKE9["judged"]
    ctl = {"rows": [row(j, 1.0, 1.0) for j in e6.SMOKE9["stages"]]}
    good = {"rows": [row(j, 0.7 if j in judged else 9.0, 1.1) for j in e6.SMOKE9["stages"]]}
    weak = {"rows": [row(j, 0.85, 1.0) for j in e6.SMOKE9["stages"]]}
    worse_field = {"rows": [row(j, 0.5, 1.3) for j in e6.SMOKE9["stages"]]}
    div = {"rows": [row(j, 0.5, 1.0, j == 20) for j in e6.SMOKE9["stages"]]}
    assert e6.smoke_verdict(good, ctl)["pass"]
    assert not e6.smoke_verdict(weak, ctl)["pass"]
    assert not e6.smoke_verdict(worse_field, ctl)["pass"]
    assert not e6.smoke_verdict(div, ctl)["pass"]


def test_final_plan_runs_every_seed_stage_once_with_equal_load_per_gpu() -> None:
    K = 24
    plan = e6.final_plan(seeds=5, accounts=4, gpus=2)
    assert len(plan) == 4 and all(len(packs) == 2 for packs in plan)
    seen, loads = [], []
    for packs in plan:
        for pack in packs:
            load = 0
            for uid in pack:
                seed, stages, base = e6.unit_spec(uid, K)
                assert base == f"{e6.UNIT}{seed}"
                seen += [(seed, j) for j in stages]
                load += len(stages)
            loads.append(load)
    assert sorted(seen) == [(s, j) for s in range(5) for j in range(K)]
    assert loads == [15] * 8
    assert e6.unit_spec(f"{e6.UNIT}3", K) == (3, list(range(K)), f"{e6.UNIT}3")
    with pytest.raises(ValueError):
        e6.unit_spec(f"{e6.UNIT}3.p2of2", K)


def test_merge_parts_joins_a_split_seed_into_the_unsplit_record(ctx, tmp_path) -> None:
    K, base = ctx.K, f"{e6.UNIT}7"
    n_test = len(ctx.prep.test)
    J = np.random.default_rng(1).uniform(-2e-2, 2e-2, size=(n_test, K))
    for p in range(2):
        uid = f"{base}.p{p}of2"
        stages = e6.unit_spec(uid, K)[1]
        jt = np.full((n_test, K), np.nan)
        jt[:, stages] = J[:, stages]
        np.save(tmp_path / f"{uid}.jtest.npy", jt)
        torch.save({"pretrained": None, "final": {j: {"w": torch.tensor(float(j))} for j in stages}},
                   tmp_path / f"{uid}.models.pt")
        rec = {"unit": uid, "complete": True, "seed": 7, "stages": stages, "stages_planned_total": K,
               "diverged": False, "stage_progress": [{"stage": j} for j in stages], "seconds": 10.0,
               "gpu_hours": 0.5, "per_device_error": None}
        (tmp_path / f"{uid}.json").write_text(__import__("json").dumps(rec))
        if p == 0:
            assert e6.merge_parts(tmp_path, ctx.prep, base) is None     # a part is missing
    rec = e6.merge_parts(tmp_path, ctx.prep, base)
    assert rec["stages_done"] == K and [r["stage"] for r in rec["stage_progress"]] == list(range(K))
    assert rec["gpu_hours"] == pytest.approx(1.0)
    assert rec["per_device_error"] == training.errors_from_curves(J, ctx.prep.test, K, ctx.prep.scales)
    final = torch.load(tmp_path / f"{base}.models.pt", weights_only=False)["final"]
    assert sorted(final) == list(range(K))


def test_final_unit_trains_only_its_stages_and_resumes_by_stage(ctx, tmp_path, monkeypatch) -> None:
    def pb(c, j, seed):
        idx = np.tile(np.arange(4), (2, 1))
        b = training._gather(c.prep.train, np.array([0, 1]), idx, c.dev)
        b["target"] = torch.zeros(b["x"].shape[0], 3, dtype=torch.float64)
        return b
    monkeypatch.setattr(e6, "physics_batch", pb)
    monkeypatch.setattr(e6, "field_error", lambda *a, **k: 0.0)
    ctx.sigma2 = {**ctx.sigma2, "data": 1.0}
    model = e6.new_model(ctx, 0)
    start = {j: e6._cpu_state(model) for j in range(ctx.K)}
    lam = {**{k: 1.0 for k in e6.TERMS}, "J": 1.0}
    uid = f"{e6.UNIT}0.p1of2"
    seed, stages, _ = e6.unit_spec(uid, ctx.K)
    rec = e6.final_unit(ctx, uid, seed, lam, tmp_path, time.time() + 600, adam_max=3, lbfgs_max=2,
                        start=start, stages=stages)
    assert [r["stage"] for r in rec["stage_progress"]] == stages == [1]
    assert rec["stages"] == [1] and rec["stages_planned_total"] == ctx.K
    assert sorted(torch.load(tmp_path / f"{uid}.models.pt", weights_only=False)["final"]) == [1]
    jt = np.load(tmp_path / f"{uid}.jtest.npy")
    assert np.isfinite(jt[:, 1]).all() and np.isnan(jt[:, [0, 2]]).all()


def test_scratch_centre_covers_every_term_and_join_picks_the_best_complete_trial(ctx, monkeypatch) -> None:
    from Staged_PINN_Solar import scratch

    def pb(c, j, seed):
        idx = np.tile(np.arange(4), (2, 1))
        b = training._gather(c.prep.train, np.array([0, 1]), idx, c.dev)
        b["target"] = torch.zeros(b["x"].shape[0], 3, dtype=torch.float64)
        return b
    monkeypatch.setattr(e6, "physics_batch", pb)
    monkeypatch.setattr(e6, "GRID", (0, 1))
    ctx.sigma2 = {**ctx.sigma2, "data": 1.0}
    model = e6.new_model(ctx, 0)
    pre = {j: e6._cpu_state(model) for j in range(ctx.K)}
    c = scratch.centre_all(ctx, pre, 0)
    assert set(c) == set(scratch.KEYS) and "J" in c and all(np.isfinite(v) and v > 0 for v in c.values())
    trial = lambda n, st, v, j: {"number": n, "state": st, "value": v, "params": {"log10_J": j, "log10_pde_phi": 0.0}}
    s0 = {"part": 0, "n_trials": 2, "complete": 1, "pruned": 1, "failed": 0, "centre": c,
          "trials": [trial(0, "COMPLETE", -1.0, 0.0), trial(1, "PRUNED", None, 1.0)]}
    s1 = {"part": 1, "n_trials": 1, "complete": 1, "pruned": 0, "failed": 0,
          "trials": [trial(0, "COMPLETE", -2.0, 1.0)]}
    j = scratch.join_studies([s0, s1])
    assert (j["part"], j["trial"], j["value"], j["n_trials"]) == (1, 0, -2.0, 3)
    assert j["lambda"]["J"] == pytest.approx(10.0) and j["lambda"]["data"] == e6.LAMBDA_DATA
    plan = e6.final_plan(unit=scratch.UNIT)
    assert all(u.startswith(scratch.UNIT) for packs in plan for p in packs for u in p)


def test_scratch_final_unit_pretrains_this_seed_and_runs_fixed_adam(ctx, tmp_path, monkeypatch) -> None:
    def pb(c, j, seed):
        idx = np.tile(np.arange(4), (2, 1))
        b = training._gather(c.prep.train, np.array([0, 1]), idx, c.dev)
        b["target"] = torch.zeros(b["x"].shape[0], 3, dtype=torch.float64)
        return b
    monkeypatch.setattr(e6, "physics_batch", pb)
    monkeypatch.setattr(e6, "stage_batch", lambda c, split, j, rng, nd, nn: pb(c, j, 0))
    monkeypatch.setattr(e6, "field_error", lambda *a, **k: 0.0)
    monkeypatch.setattr(e6, "PRE_STEPS", 3)
    monkeypatch.setattr(e6, "WINDOW", 2)
    ctx.sigma2 = {**ctx.sigma2, "data": 1.0}
    lam = {**{k: 1.0 for k in e6.TERMS}, "J": 1.0}
    uid = "e6s-A-s0.p0of3"
    seed, stages, _ = e6.unit_spec(uid, ctx.K)
    calls = []
    real = e6.pretrain
    monkeypatch.setattr(e6, "pretrain", lambda c, s, *a, **k: calls.append(s) or real(c, s, *a, **k))
    start = e6._scratch_start({"scratch": True}, seed)
    assert start is None                                  # None: final_unit pretrains this seed itself
    rec = e6.final_unit(ctx, uid, seed, lam, tmp_path, time.time() + 600, adam_max=8, lbfgs_max=2,
                        start=start, stages=stages, tau=None)
    assert calls == [seed]
    assert rec["stage_progress"][0]["adam_steps"] == 8 and rec["recipe"]["tau"] is None
    assert sorted(torch.load(tmp_path / f"{uid}.models.pt", weights_only=False)["final"]) == stages
