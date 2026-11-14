"""Network M (r11, S1-S8): per-device material threaded through banks, batches, residuals and currents; the
network size; stage eligibility (Eq. 27); final parts; the smoke verdict; and the A path left as it was."""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import ansatz as ansatz_mod
from Staged_PINN_Solar import config as C
from Staged_PINN_Solar import data, data_m, e6, network, scratch_m, training
from Staged_PINN_Solar.carriers import Material
from Staged_PINN_Solar.scaling import Scales

S = Scales()
needs_base_b = pytest.mark.skipif(not (data_m.local_root() / data_m.SPLIT_FILE).exists(),
                                  reason="base_b not available")
CPU = torch.device("cpu")


# --- network size ------------------------------------------------------------------------------------------------
def test_network_m_has_twelve_inputs_and_416003_parameters() -> None:
    """S5: 320 x 5, n_in = 1 + 11 = 12, three outputs, same activation and ansatz ranges as A."""
    cfg = C.network_m()
    assert (cfg.width, cfg.depth) == (320, 5) and len(data_m.XI_PARAMS) == 11
    net = network.build("A", len(data_m.XI_PARAMS), cfg.width, cfg.depth, seed=0)
    assert net.nets[0][0].in_features == 12 and net.nets[0][-1].out_features == 3
    assert sum(p.numel() for p in net.parameters()) == 416003
    out = net(torch.rand(5, 1, dtype=torch.float64), torch.rand(5, 11, dtype=torch.float64))
    assert out.shape == (5, 3) and out[:, 1:].min() >= 0 and out[:, 1:].max() <= 1


# --- two real devices of different materials ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def pair():
    """E1A (E1gen) and dpvB2 Si: banked together (BankM) with a small model on their bounds."""
    rows = data_m.read_split(data_m.local_root())
    s = data_m.split_m(data_m.local_root(), rows)
    refs = {r.uid: r for r in s.train + s.val + s.test}
    a = next(r for r in rows if r["origin"] == "E1A")
    devs = [data_m.load_device(refs[f"E1gen/{a['group']}"]), data_m.load_device(refs["Si/dpvB2__dev_000000"])]
    specs = data_m.xi_specs([r for r in rows if r["split"] == "train"])
    b = training._BankBuilder(S, specs)
    for d in devs:
        b.add(d)
    bank = b.finish()
    bounds = ansatz_mod.measure_bounds(devs, S)
    model = ansatz_mod.Ansatz(network.build("A", 11, 16, 2, seed=0), bounds)
    return SimpleNamespace(devs=devs, bank=bank, model=model, specs=specs)


def _alone(bank, d: int) -> tuple:
    """Device d's rows of the bank as an architecture-A Bank plus its constant scalar material."""
    sub = training.Bank(**{f.name: ([getattr(bank, f.name)[d]] if f.name == "uids" else getattr(bank, f.name)[[d]])
                           for f in fields(training.Bank)})
    mat = Material(**{f: np.float64(bank.mat[f][d]) for f in training.MAT_FIELDS}, Ndop=0.0)
    return sub, mat


@needs_base_b
def test_bank_m_carries_each_devices_material(pair) -> None:
    """S2: the bank of base-B devices is a BankM whose per-device constants equal data.device_material's."""
    assert isinstance(pair.bank, training.BankM)
    for d, dev in enumerate(pair.devs):
        m = data.device_material(dev, S)
        for f in training.MAT_FIELDS:
            assert pair.bank.mat[f][d] == getattr(m, f)[0]
    assert pair.bank.mat["Eg"][0] != pair.bank.mat["Eg"][1]


@needs_base_b
def test_per_device_material_residuals_equal_each_device_alone(pair) -> None:
    """Two devices of different materials in one batch give the residual summands of each device evaluated alone
    with its own constant material (the A code path with that material)."""
    idx = np.array([[3, 50, 120, 250, 400, 498]] * 2)
    V = 0.3 / S.V_th
    contacts = data.device_contacts(S)
    both = training._gather(pair.bank, np.array([0, 1]), idx, CPU)
    assert "mat" in both and both["mat"]["Eg"].shape == (12, 1) and both["matc"]["Eg"].shape == (4, 1)
    t_both = training.term_summands(pair.model, both, V, None, contacts)
    for d in range(2):
        sub, mat = _alone(pair.bank, d)
        one = training._gather(sub, np.array([0]), idx[:1], CPU)
        assert "mat" not in one
        t_one = training.term_summands(pair.model, one, V, mat, contacts)
        for k in ("poisson", "n", "p"):
            for a, b in zip(t_both[k], t_one[k]):
                assert torch.allclose(a[d * 6:(d + 1) * 6], b, rtol=1e-12, atol=0.0), k
        for k in ("bcn", "bcp"):
            for a, b in zip(t_both[k], t_one[k]):
                assert torch.allclose(a[d:d + 1], b, rtol=1e-12, atol=0.0), k


@needs_base_b
def test_per_device_material_current_equals_each_device_alone(pair) -> None:
    """Eq. 20 in evaluation (training.evaluate_currents) and with its graph (e6.gr_current_torch) uses each
    device's own material: the pair equals each device alone with its constant material."""
    V = 0.3 / S.V_th
    both = training.evaluate_currents(pair.model, pair.bank, V, None, S, CPU, 2)
    ctx = SimpleNamespace(prep=SimpleNamespace(train=pair.bank, material=None, scales=S), dev=CPU,
                          currents={"train": np.ones((2, 3))})
    cb = e6.current_batch(ctx, 0, 0, 2)
    J_graph = e6.gr_current_torch(pair.model, cb, V, ctx).detach().numpy()
    assert np.allclose(J_graph, both[cb["devices"]], rtol=1e-10)
    for d in range(2):
        sub, mat = _alone(pair.bank, d)
        assert training.evaluate_currents(pair.model, sub, V, mat, S, CPU, 1)[0] == pytest.approx(both[d], rel=1e-12)
    assert np.isfinite(both).all() and both[0] != both[1]


# --- stage eligibility (r11 Eq. 27) ------------------------------------------------------------------------------
def _stub_bank_m(n_V) -> training.BankM:
    D, N = len(n_V), 40
    x = np.tile(np.linspace(0, 1, N), (D, 1))
    nd = np.where(x <= 0.4, 0.01, -0.02)
    return training.BankM(uids=[f"d{i}" for i in range(D)], xi_hat=np.full((D, 11), 0.5), delta=np.full(D, 1e-4),
                          x_hat=x, G=np.full((D, N), 1e-12), Ndop=nd, phi_eq0=np.full(D, -150.0),
                          phi_eq1=np.full(D, -160.0), n_V=np.asarray(n_V), pce=np.full(D, 0.2),
                          p_in_std=np.full(D, 900.0), mat={f: np.arange(D, dtype=float) + 1 for f in training.MAT_FIELDS})


def test_stage_eligibility_excludes_devices_whose_curve_ends_before_the_stage() -> None:
    """S6 / r11 Eq. 27: stage j trains on devices with n_V > j. A Si-like device with n_V = 13 (voltages
    0..0.60 V) is in stage 12 and absent from stage 13 in every batch builder; K = max n_V over train."""
    n_V = [13, 25, 25, 25]
    bank = _stub_bank_m(n_V)
    cfg = C.Config(n_devices_per_batch=64, n_nodes_per_device=8)
    ctx = SimpleNamespace(prep=SimpleNamespace(train=bank), dev=CPU, cfg=cfg,
                          fields={"train": np.zeros((4, 25, 3, 40), dtype=np.float32)},
                          currents={"train": np.ones((4, 25))})
    for j, expect in ((12, {0, 1, 2, 3}), (13, {1, 2, 3}), (24, {1, 2, 3})):
        assert set(training.make_batch(bank, j, 0, cfg, CPU)["devices"]) == expect
        assert set(e6.stage_batch(ctx, "train", j, [0, j, 17], 64, 8)["devices"]) == expect
        assert set(e6.current_batch(ctx, j, 0, 16)["devices"]) == expect
    b = e6.stage_batch(ctx, "train", 13, [0, 13, 17], 64, 8)
    assert torch.equal(b["mat"]["Eg"][:, 0], torch.repeat_interleave(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64),
                                                                     b["x"].shape[0] // 3))
    assert max(n_V) == 25


def test_final_parts_split_the_k_stages_of_a_seed() -> None:
    """S8: a seed's units cover each of its K stages exactly once over the parts (stage j in part j mod n)."""
    for K, parts in ((25, 2), (30, 2), (7, 3)):
        units = scratch_m.final_units(1, parts)
        assert units[0] == "e7m-M-s1.p0of%d" % parts
        got = [e6.unit_spec(u, K) for u in units]
        assert all(seed == 1 and base == "e7m-M-s1" for seed, _, base in got)
        assert sorted(j for _, st, _ in got for j in st) == list(range(K))


def test_scratch_m_constants_follow_the_operator() -> None:
    """S6 / S8 and the operator's 2026-11-14 change: one study of 25 trials capped at 10.5 h, grid 0..24 by 4."""
    assert scratch_m.GRID == (0, 4, 8, 12, 16, 20, 24) and scratch_m.TRIALS == 25
    assert scratch_m.HOURS == {"smoke": 20 / 60, "study": 10.5, "final": 10.5} and scratch_m.SEEDS == (0, 1, 2)
    assert (scratch_m.STUDY, scratch_m.UNIT, scratch_m.BEST, scratch_m.JOURNAL) == \
        ("e7m", "e7m-M-s", "best_lambda_m.json", "optuna_journal_m.log")
    assert (scratch_m.ADAM, scratch_m.LBFGS, scratch_m.TAU) == (2000, 500, None)


def test_smoke_verdict_projects_and_judges() -> None:
    """The smoke's PASS needs every check: peak < 13 GB, finite, Adam lowers the loss, 25 trials fit 10.5 h on 2 GPUs."""
    w = dict(final_complete=True, trial_finite=True, all_finite=True, diverged=False, loss_start=2.0, loss_adam_end=1.0,
             trial_stage_seconds=300.0, s_per_adam=0.05, s_per_lbfgs_eval=0.1, stage_seconds=150.0, adam_steps=100,
             lbfgs_evaluations=50, study_peak_gb=4.0, final_peak_gb=5.0)
    v = scratch_m.smoke_verdict([w, w], s_pre_step=0.01, centre_stage_s=10.0, K=25, n_grid=7)
    # study: pretrain 25*500*0.01 s + 7*10 s + ceil(25/2) * 7 * 300 s
    assert v["projected_study_hours"] == pytest.approx((125.0 + 70.0 + 13 * 2100.0) / 3600.0)
    assert v["pass"] and v["peak_gb"] == 5.0
    assert not scratch_m.smoke_verdict([{**w, "final_peak_gb": 13.5}], s_pre_step=0.01, centre_stage_s=10.0,
                                       K=25, n_grid=7)["pass"]
    assert not scratch_m.smoke_verdict([{**w, "loss_adam_end": 3.0}], s_pre_step=0.01, centre_stage_s=10.0,
                                       K=25, n_grid=7)["pass"]
    assert not scratch_m.smoke_verdict([{**w, "trial_stage_seconds": 900.0}], s_pre_step=0.01, centre_stage_s=10.0,
                                       K=25, n_grid=7)["pass"]
    assert not scratch_m.smoke_verdict([{**w, "trial_finite": False}], s_pre_step=0.01, centre_stage_s=10.0,
                                       K=25, n_grid=7)["pass"]


# --- the whole M context on a few devices -----------------------------------------------------------------------
@needs_base_b
def test_prepare_m_builds_an_e6_context_whose_phased_loss_is_finite() -> None:
    """data_m.prepare on 3 devices per split: Prepared without a constant material, BankM banks, sigma2 from base-B
    train, K = max n_V over train, fields/currents shaped (D, K, 3, N) / (D, K); one E6 step (I'' with J) is finite."""
    cfg = C.network_m().with_(width=16, depth=2, n_devices_per_batch=3, n_nodes_per_device=8)
    prep, fields_, currents, K = data_m.prepare(data_m.local_root(), cfg, report=None, limit=3)
    assert prep.material is None and prep.n_xi == 11 and isinstance(prep.train, training.BankM)
    assert K == int(prep.train.n_V.max()) and set(prep.sigma2) == set(training.TERMS)
    assert fields_["val"].shape == (3, K, 3, 500) and currents["test"].shape == (3, K)
    ctx = e6.Ctx(prep, fields_, cfg, K, currents)
    model = e6.new_model(ctx, 0)
    obj = e6.PhasedObjective(ctx, model, e6.physics_batch(ctx, 0, 0), 0, jbatch=e6.current_batch(ctx, 0, 0, 2))
    lam = {k: 1.0 for k in training.TERMS + ("data", "J")}
    total, terms, g = obj.terms_and_grad(lam, list(model.parameters()))
    assert torch.isfinite(total) and set(terms) == set(lam) and all(torch.isfinite(x).all() for x in g)
    assert np.isfinite(e6.current_error(ctx, model, "val", 0))


# --- architecture A unchanged -------------------------------------------------------------------------------------
def _stub_device_a(i: int) -> data.Device:
    N, n_v = 30, 4
    x = np.linspace(0.0, 2e-4 / S.L_D, N)
    pots = np.random.default_rng(i).normal(size=(n_v, N))
    ref = data.DeviceRef("E1B", 0, i, data.local_root())
    return data.Device(ref=ref, xi={"Nd_n": 1e17 * (i + 1), "Na_p": 3e17, "L_n": 0.8e-4, "L_p": 1.2e-4},
                       u=np.zeros(4), V=0.05 * np.arange(n_v), x_debye=x, G=np.full(N, 1e-12), phi_eq=np.zeros(N),
                       phi=pots, phi_n=pots, phi_p=pots, J_n=np.zeros((n_v, N - 1)), J_p=np.zeros((n_v, N - 1)),
                       pce=0.2)


def test_a_path_bank_batch_and_material_are_unchanged() -> None:
    """Regression of the A path: a stub E1 device (no own material) banks into a plain Bank, its batch has exactly
    the keys and shapes it had, and the material is E1_MATERIAL (constant) with the device's doping step."""
    specs = (("Nd_n", "log", 1e16, 1e18), ("Na_p", "log", 1e16, 1e18), ("L_n", "lin", 5e-5, 2e-4),
             ("L_p", "lin", 5e-5, 2e-4))
    b = training._BankBuilder(S, specs)
    for i in range(3):
        b.add(_stub_device_a(i))
    bank = b.finish()
    assert type(bank) is training.Bank and not hasattr(bank, "mat")
    assert bank.xi_hat.shape == (3, 4) and bank.x_hat.shape == (3, 30) and bank.Ndop.shape == (3, 30)
    m = data.device_material(_stub_device_a(1), S)
    e1 = data.E1_MATERIAL
    assert np.all(m.Eg == e1["Eg"] / S.V_th) and np.all(m.Nc == e1["Nc"] / S.C0) and np.all(m.mu_n == e1["mn"])
    assert np.all(m.tau_n == e1["tn"] / S.t0) and np.all(m.Et == 0.0) and np.all(m.Br == 0.0)
    assert np.array_equal(bank.Ndop[1], m.Ndop)
    assert set(np.round(bank.Ndop[1] * S.C0)) == {2e17, -3e17}
    batch = training._gather(bank, np.array([0, 2]), np.array([[1, 5, 9], [2, 6, 10]]), CPU)
    assert set(batch) == {"x", "G", "ndop", "xi", "delta", "peq0", "peq1", "xc", "xic", "deltac", "peq0c", "peq1c",
                          "phi_eq", "devices"}
    assert batch["x"].shape == (6, 1) and batch["xi"].shape == (6, 4) and batch["xc"].shape == (4, 1)
    cm = training.const_material(S)
    assert cm.Eg == e1["Eg"] / S.V_th and cm.eps_r == e1["eps"] and cm.mu_p == e1["mp"]
    # the residuals of the A path equal the explicit constant-material computation they always did
    bounds = ansatz_mod.AnsatzBounds(D_phi=5.0, phi_n=(-1.0, 1.0), phi_p=(-1.0, 1.0))
    model = ansatz_mod.Ansatz(network.build("A", 4, 8, 2, seed=0), bounds)
    t = training.term_summands(model, batch, 0.0, cm, data.device_contacts(S))
    x = batch["x"].clone().requires_grad_(True)
    phi, phi_n, _ = model(x, batch["xi"], batch["peq0"], batch["peq1"], 0.0)
    mm = replace(cm, Ndop=batch["ndop"]).like(phi)
    from Staged_PINN_Solar import carriers
    n = carriers.electron_density(phi, phi_n, mm)
    assert torch.equal(t["poisson"][1], -n) and torch.equal(t["poisson"][3], batch["ndop"] + 0.0 * n)
    J = training.evaluate_currents(model, bank, 0.0, cm, S, CPU, 2)
    assert J.shape == (3,) and np.isfinite(J).all()


@needs_base_b
def test_run_smoke_plumbing_on_a_tiny_cpu_context(tmp_path, monkeypatch) -> None:
    """run_smoke end to end on CPU at toy scale (3 devices per split, 16 x 2 net, a few steps): the reduced
    theta_pre, the centre, one Optuna trial and the cut final recipe run, and e7m-smoke.json carries the
    projections, the checks and a verdict."""
    monkeypatch.setenv("STAGED_PINN_RESULTS", str(tmp_path))
    monkeypatch.setenv("STAGED_PINN_DEVICE", "cpu")
    cfg = C.network_m().with_(width=16, depth=2, n_devices_per_batch=3, n_nodes_per_device=8, max_stages=2)
    real = scratch_m.build_ctx
    monkeypatch.setattr(scratch_m, "build_ctx", lambda: real(cfg, limit=3, root=data_m.local_root()))
    monkeypatch.setattr(scratch_m, "SMOKE", dict(pre_steps=4, adam=30, lbfgs=5))
    monkeypatch.setattr(e6, "TRIAL_STEPS", 3)
    rec = scratch_m.run_smoke()
    assert rec["stage"] == 0 and rec["n_gpu"] == 0 and len(rec["workers"]) == 1
    w = rec["workers"][0]
    assert w["final_complete"] and w["adam_steps"] == 30 and w["lbfgs_evaluations"] >= 1
    assert np.isfinite(rec["projected_study_hours"]) and np.isfinite(rec["projected_final_hours"])
    assert set(rec["checks"]) == {"peak_below_limit", "finite", "adam_decreases_loss", "study_fits"}
    assert (tmp_path / "e7m" / "e7m-smoke.json").exists()


def test_e7m_notebooks_bind_the_repository_and_call_the_entry_points() -> None:
    """S8: smoke, ONE study and three final seeds; the first cell is the repository binding, byte for byte."""
    import json
    from pathlib import Path

    nb = Path(__file__).resolve().parents[1] / "Staged-PINN-Solar" / "Notebooks"
    cells = lambda n: json.loads((nb / f"{n}.ipynb").read_text())["cells"]
    calls = {"e7ms": "scratch_m.run_smoke()", "e7mp": "scratch_m.run_study()",
             **{f"e7mf{s}": f"scratch_m.run_final_seed({s})" for s in scratch_m.SEEDS}}
    assert sorted(p.stem for p in nb.glob("e7m*.ipynb")) == sorted(calls)
    for name, call in calls.items():
        c = cells(name)
        assert c[0]["source"] == cells("verification")[0]["source"]
        assert call in "".join(c[-1]["source"])
