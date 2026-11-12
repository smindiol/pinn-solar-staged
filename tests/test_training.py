"""Level 2 invariants of training.py: voltage-continuation training by blocks (Eqs. 3, 10-17, J-V of r10)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import config as C
from Staged_PINN_Solar import data, loss, residuals, training, units
from Staged_PINN_Solar.optimizers import LBFGS
from Staged_PINN_Solar.scaling import Scales

needs_dataset = pytest.mark.skipif(not (data.local_root() / "s1").exists(), reason="dPV set not available")

CFG = C.smoke().with_(max_stages=3, n_blocks=2, iters_per_block=3)


@pytest.fixture(scope="module")
def prep():
    return training.prepare(config=CFG)


def _spec(uid="e2-A-lbfgs-gradient-s0"):
    return units.parse_unit(uid)


def test_residual_terms_match_the_analytic_derivatives() -> None:
    """Autograd chain of Eqs. 5-7 and 9: for phi = 1 + 2x + 3x^2, phi_n = x/2, phi_p = -x/5 every derivative is known."""
    scales, mat = Scales(), training.const_material(Scales())

    class Fake(torch.nn.Module):
        def forward(self, x, xi, peq0, peq1, V):
            return 1 + 2 * x + 3 * x ** 2, 0.5 * x, -0.2 * x

    M, B = 4, 2
    x = torch.tensor([[0.1], [0.3], [0.5], [0.9]], dtype=torch.float64)
    one = torch.ones(M, 1, dtype=torch.float64)
    delta, ndop, g = 0.01 * one, 0.7 * one, 0.02 * one
    batch = dict(x=x, xi=0 * x, peq0=0 * x, peq1=0 * x, delta=delta, ndop=ndop, G=g,
                 xc=torch.tensor([[0.0], [1.0]] * B, dtype=torch.float64), xic=torch.zeros(2 * B, 1, dtype=torch.float64),
                 peq0c=torch.zeros(2 * B, 1, dtype=torch.float64), peq1c=torch.zeros(2 * B, 1, dtype=torch.float64),
                 deltac=0.01 * torch.ones(2 * B, 1, dtype=torch.float64), phi_eq=torch.zeros(B, 2, dtype=torch.float64))
    contacts = data.device_contacts(scales)
    res = training.residual_terms(Fake(), batch, 0.0, mat, contacts)
    from Staged_PINN_Solar import carriers
    phi, pn, pp = 1 + 2 * x + 3 * x ** 2, 0.5 * x, -0.2 * x
    m = replace(mat, Ndop=ndop).like(phi)
    n, p = carriers.electron_density(phi, pn, m), carriers.hole_density(phi, pp, m)
    want_poisson = 0.01 ** 2 * m.eps_r * 6.0 - n + p + ndop
    assert torch.allclose(res["poisson"], want_poisson, rtol=1e-12)
    dn = n * ((2 + 6 * x) + 0.5)                                     # d n / dx
    dJn = 0.01 * m.mu_n * (dn * 0.5)                                 # J_n = delta mu n (1/2)
    R = carriers.recombination(n, p, m)
    assert torch.allclose(res["n"], 0.01 * dJn + g - R, rtol=1e-10)
    assert res["bcn"].shape == (B, 2) and res["bcp"].shape == (B, 2)


@needs_dataset
def test_statistics_and_bounds_come_from_the_training_split_only(prep) -> None:
    """sigma_i^2 and the ansatz bounds are measured here, from the training devices of the run itself."""
    split = data.split_devices(data.inventory(), seed=CFG.split_seed)
    refs = split.train[:CFG.max_devices]
    want = loss.measure_reference_statistics(refs, Scales()).sigma2()
    assert prep.sigma2 == pytest.approx(want, rel=1e-12)
    assert prep.train.uids == [r.uid for r in refs]
    assert prep.test.uids == [r.uid for r in split.test[:CFG.max_devices]]
    assert prep.bounds.D_phi > 0


@needs_dataset
def test_batches_depend_on_seed_and_stage_but_not_on_the_arm(prep) -> None:
    b1 = training.make_batch(prep.train, 1, 0, CFG, torch.device("cpu"))
    b2 = training.make_batch(prep.train, 1, 0, CFG, torch.device("cpu"))
    b3 = training.make_batch(prep.train, 1, 1, CFG, torch.device("cpu"))
    assert torch.equal(b1["x"], b2["x"]) and not torch.equal(b1["x"], b3["x"])
    assert b1["x"].shape == (CFG.n_devices_per_batch * CFG.n_nodes_per_device, 1) or b1["x"].shape[0] <= \
        CFG.n_devices_per_batch * CFG.n_nodes_per_device
    assert float(b1["x"].min()) > 0.0 and float(b1["x"].max()) < 1.0       # interior nodes only


@needs_dataset
def test_arms_with_the_same_seed_share_the_initialization(prep) -> None:
    out = [training.initial_parameters(_spec(u), CFG, prep) for u in ("e1-A-normalized-s0", "e2-A-soap-gradient-s0")]
    assert all(torch.equal(p, q) for p, q in zip(*out))
    other = training.initial_parameters(_spec("e1-B-normalized-s0"), CFG, prep)
    assert len(other) != len(out[0])


@needs_dataset
def test_unnormalized_arm_uses_unit_scales(prep) -> None:
    r = training.train(_spec("e1-A-unnormalized-s0"), prep, CFG)
    assert set(r["sigma2_used"].values()) == {1.0}
    r2 = training.train(_spec("e1-A-normalized-s0"), prep, CFG)
    assert r2["sigma2_used"] == pytest.approx(prep.sigma2)


@needs_dataset
def test_training_step_decreases_loss_on_tiny_problem(prep) -> None:
    """L-BFGS blocks with Armijo backtracking never end a block above where it started."""
    r = training.train(_spec("e1-A-normalized-s0"), prep, CFG)
    for blk in r["blocks"]:
        assert blk["loss_end"] <= blk["loss_start"] * (1 + 1e-12)
    assert not r["diverged"] and r["stages_done"] == 3


@needs_dataset
def test_soap_runs_per_parameter_and_decreases_loss(prep) -> None:
    cfg = CFG.with_(iters_per_block=6)       # config lr and warmup: from the equilibrium start an unramped 1e-3 diverges
    r = training.train(_spec("e2-A-soap-fixed-s0"), prep, cfg)
    first = r["blocks"][0]
    assert first["loss_end"] < first["loss_start"]


@needs_dataset
def test_lambda_changes_only_between_blocks(prep) -> None:
    r = training.train(_spec("e2-A-lbfgs-gradient-s0"), prep, CFG)
    lams = [b["lambda"] for b in r["blocks"]]
    assert lams[0] == {k: 1.0 for k in lams[0]}                      # the first block runs with lambda = 1
    assert any(l != lams[0] for l in lams[1:])                       # the update acted between blocks
    for b in r["blocks"]:
        assert b["lambda_used_constant"] is True
    fixed = training.train(_spec("e2-A-lbfgs-fixed-s0"), prep, CFG)
    assert all(b["lambda"] == {k: 1.0 for k in b["lambda"]} for b in fixed["blocks"])


@needs_dataset
@pytest.mark.parametrize("uid, weighting_resets", [("e2-A-lbfgs-gradient-s0", True), ("e5-A-s0", False)])
def test_lbfgs_memory_is_reset_at_lambda_updates_only_in_the_reset_arm(prep, monkeypatch, uid, weighting_resets) -> None:
    calls = []
    orig = LBFGS.reset
    monkeypatch.setattr(LBFGS, "reset", lambda self: (calls.append(1), orig(self))[1])
    training.train(_spec(uid), prep, CFG)
    stages, blocks = 3, 3 * CFG.n_blocks
    assert len(calls) == stages + (blocks if weighting_resets else 0)


@needs_dataset
def test_divergence_is_flagged_and_stops_the_run(prep) -> None:
    r = training.train(_spec("e1-A-unnormalized-s0"), prep, CFG.with_(divergence_factor=1e-12))
    assert r["diverged"] is True and r["stages_done"] == 0 and len(r["stages"]) == 1
    assert all(np.isnan(v) for v in r["per_device_error"].values())


@needs_dataset
def test_conditioning_diagnostics_are_recorded_at_init_marks_and_block_ends(prep) -> None:
    r = training.train(_spec("e1-A-normalized-s0"), prep, CFG)
    d = r["diagnostics"]
    kinds = [x["at"] for x in d["records"]]
    assert kinds[0] == "init" and "mark" in kinds and kinds.count("block_end") == 3 * CFG.n_blocks
    init = d["records"][0]
    for k in ("max_share", "log10_norm_ratio", "lambda_max", "mean_trace", "curvature_ratio"):
        assert k in init and np.isfinite(init[k])
    assert 0.0 < d["early_max_gradient_share"] <= 1.0
    assert training.train(_spec("e2-A-lbfgs-fixed-s0"), prep, CFG)["diagnostics"] is None


@needs_dataset
def test_same_seed_reproduces_the_run(prep) -> None:
    a = training.train(_spec("e2-A-lbfgs-gradient-s1"), prep, CFG)
    b = training.train(_spec("e2-A-lbfgs-gradient-s1"), prep, CFG)
    assert a["per_device_error"] == b["per_device_error"]
    assert a["blocks"][-1]["loss_end"] == b["blocks"][-1]["loss_end"]


@needs_dataset
def test_error_is_relative_pce_percent_per_test_device(prep) -> None:
    r = training.train(_spec("e1-B-normalized-s0"), prep, CFG)
    assert set(r["per_device_error"]) == set(prep.test.uids)
    assert all(np.isfinite(v) and v >= 0 for v in r["per_device_error"].values())


@needs_dataset
def test_equilibrium_start_keeps_every_initial_normalized_term_bounded(prep) -> None:
    """From the equilibrium start every normalized term L_i at V = 0 is finite and O(1), for both
    architectures; the random start is kept as an explicit control and starts orders of magnitude higher."""
    from Staged_PINN_Solar import ansatz as ansatz_mod

    dev = torch.device("cpu")
    batch = training.make_batch(prep.train, 0, 0, CFG, dev)

    def initial(uid, cfg):
        model = ansatz_mod.Ansatz(training.build_network(_spec(uid), cfg, prep), prep.bounds)
        losses = training.Objective(model, batch, 0.0, prep.sigma2, prep.material, prep.contacts).losses()
        return {k: float(v.detach()) for k, v in losses.items()}

    for uid in ("e1-A-normalized-s0", "e1-B-normalized-s0"):
        eq = initial(uid, CFG)
        assert all(np.isfinite(v) and v < 10.0 for v in eq.values()), eq
    rnd = initial("e1-A-normalized-s0", CFG.with_(start="random"))
    assert max(rnd.values()) > 1e6 * max(eq.values())


@needs_dataset
def test_relative_arm_uses_the_per_node_residual(prep) -> None:
    """E1's r04 arm: the objective's L_i is Eq. (11) of r04 on the batch, with eps_i = sqrt(sigma_i^2)."""
    from Staged_PINN_Solar import ansatz as ansatz_mod
    dev = torch.device("cpu")
    batch = training.make_batch(prep.train, 0, 0, CFG, dev)
    model = ansatz_mod.Ansatz(training.build_network(_spec("e1-A-relative-s0"), CFG, prep), prep.bounds)
    obj = training.Objective(model, batch, 0.0, prep.sigma2, prep.material, prep.contacts, mode="relative")
    got = {k: float(v) for k, v in obj.losses().items()}
    t = training.term_summands(model, batch, 0.0, prep.material, prep.contacts)
    want = {k: float(v) for k, v in loss.relative_losses(t, loss.residual_floor(prep.sigma2)).items()}
    assert got == pytest.approx(want)
    assert all(0.0 <= v < 1.0 for v in got.values()), got
    r = training.train(_spec("e1-A-relative-s0"), prep, CFG)
    assert r["loss_mode"] == "relative" and r["sigma2_used"] == pytest.approx(prep.sigma2)


@needs_dataset
def test_full_batch_chunks_give_the_same_loss_and_gradient_as_one_batch(prep) -> None:
    """r04 Section 6: under `full_batch` every eligible device and interior node enters, and chunking
    (memory only) changes neither the loss nor its gradient."""
    from Staged_PINN_Solar import ansatz as ansatz_mod
    dev = torch.device("cpu")
    one = CFG.with_(full_batch=True, train_chunk_devices=10 ** 6)
    many = CFG.with_(full_batch=True, train_chunk_devices=3)
    b1, bm = training.make_batches(prep.train, 0, 0, one, dev), training.make_batches(prep.train, 0, 0, many, dev)
    assert len(b1) == 1 and len(bm) > 1
    n_eligible = int((prep.train.n_V > 0).sum())
    assert sum(len(c["devices"]) for c in bm) == n_eligible
    assert b1[0]["x"].shape[0] == n_eligible * (prep.train.x_hat.shape[1] - 2)
    model = ansatz_mod.Ansatz(training.build_network(_spec("e2-A-lbfgs-fixed-s0"), CFG, prep), prep.bounds)
    params = list(model.parameters())
    lam = {k: 1.0 for k in prep.sigma2}
    outs = []
    for b in (b1, bm):
        obj = training.Objective(model, b, 0.0, prep.sigma2, prep.material, prep.contacts, mode="relative")
        assert obj.n_devices == n_eligible
        total = obj.total(lam)
        g = torch.cat([x.reshape(-1) for x in torch.autograd.grad(total, params)])
        outs.append((float(total), g.detach().numpy()))
    assert outs[0][0] == pytest.approx(outs[1][0], rel=1e-12)
    np.testing.assert_allclose(outs[0][1], outs[1][1], rtol=1e-10, atol=1e-14)


@needs_dataset
def test_gradient_weights_never_exceed_lambda_max(prep) -> None:
    """Eq. (12) of r04 in the loop: the largest lambda ever used stays within [lambda_min, lambda_max]."""
    cfg = CFG.with_(lambda_max=50.0)
    r = training.train(_spec("e2-A-lbfgs-gradient-s0"), prep, cfg)
    assert r["lambda_max_seen"] <= 50.0
    assert all(cfg.lambda_min <= v <= 50.0 for b in r["blocks"] for v in b["lambda"].values())
    assert r["lambda_update_errors"] == []


@needs_dataset
def test_chunked_value_and_grad_equals_the_one_graph_gradient(prep) -> None:
    """Gradient accumulation frees each chunk's graph after its backward; the value, the total gradient and
    every per-term gradient must equal those of one graph over the whole full batch."""
    from Staged_PINN_Solar import ansatz as ansatz_mod
    dev = torch.device("cpu")
    chunks = training.make_batches(prep.train, 0, 0, CFG.with_(full_batch=True, train_chunk_devices=3), dev)
    assert len(chunks) > 1
    model = ansatz_mod.Ansatz(training.build_network(_spec("e2-A-lbfgs-gradient-s0"), CFG, prep), prep.bounds)
    params = list(model.parameters())
    lam = {"poisson": 2.0, "n": 0.5, "p": 3.0, "bcn": 1.0, "bcp": 7.0}
    obj = training.Objective(model, chunks, 0.0, prep.sigma2, prep.material, prep.contacts, mode="relative")
    total = obj.total(lam)
    want = torch.cat([g.reshape(-1) for g in torch.autograd.grad(total, params)]).detach().numpy()
    v, g = obj.value_and_grad(lam, params)
    assert float(v) == pytest.approx(float(total), rel=1e-12)
    np.testing.assert_allclose(torch.cat([x.reshape(-1) for x in g]).numpy(), want, rtol=1e-10, atol=1e-14)
    from Staged_PINN_Solar import weights as W
    per_term_want = W.term_gradients(obj.losses(), params)
    per_term = obj.term_grads(params)
    for k in per_term_want:
        np.testing.assert_allclose(per_term[k], per_term_want[k], rtol=1e-10, atol=1e-14)


def test_interrupted_run_resumes_from_its_stage_checkpoint_to_the_same_result(prep, tmp_path) -> None:
    spec = _spec("e2-A-lbfgs-gradient-s0")
    whole = training.train(spec, prep, CFG)
    ck = tmp_path / "u.ckpt.pt"
    calls = {"n": 0}

    def stop_after_two(_longest):
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(training.Interrupted) as stop:
        training.train(spec, prep, CFG, checkpoint=ck, should_stop=stop_after_two)
    assert stop.value.stages_done == 2 and ck.exists() and training.partial_path(ck).exists()
    resumed = training.train(spec, prep, CFG, checkpoint=ck)
    assert resumed["stages_done"] == whole["stages_done"] and len(resumed["stage_progress"]) == whole["stages_done"]
    a, b = whole["per_device_error"], resumed["per_device_error"]
    assert all(np.isclose(a[k], b[k], rtol=1e-6, equal_nan=True) for k in a)
    assert np.isclose(whole["blocks"][-1]["loss_end"], resumed["blocks"][-1]["loss_end"], rtol=1e-8)
