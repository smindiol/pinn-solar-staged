"""r09 Eqs. 23-24: the current term of the E6 loss (experiments v09, E6)."""

from __future__ import annotations

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
