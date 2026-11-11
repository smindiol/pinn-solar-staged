"""Level 2 invariants of steps.py: units, resumability, record contents, analysis and rehearsal."""

from __future__ import annotations

import json

import numpy as np
import pytest

from Staged_PINN_Solar import config as C
from Staged_PINN_Solar import data, steps, units

needs_dataset = pytest.mark.skipif(not (data.local_root() / "s1").exists(), reason="dPV set not available")
CFG = C.smoke().with_(max_stages=3, n_blocks=1, iters_per_block=2)


@needs_dataset
def test_a_step_runs_exactly_the_requested_units_and_skips_complete_ones(tmp_path) -> None:
    ids = ["e1-A-normalized-s0", "e1-B-unnormalized-s0"]
    logs = []
    done = steps.run_e1(ids[:1], False, CFG, out=tmp_path)
    assert done == ids[:1]
    assert [p.name for p in sorted((tmp_path / "e1").glob("*.json"))] == [f"{ids[0]}.json"]
    mtime = (tmp_path / "e1" / f"{ids[0]}.json").stat().st_mtime_ns
    steps._run_training_step("e1", ids, False, CFG, out=tmp_path, progress=logs.append)
    assert (tmp_path / "e1" / f"{ids[0]}.json").stat().st_mtime_ns == mtime           # skipped, not recomputed
    assert any("complete, skipped" in l for l in logs)
    assert {p.name for p in (tmp_path / "e1").glob("*.json")} == {f"{u}.json" for u in ids}
    with pytest.raises(ValueError, match="not units of step"):
        steps.run_e1(["e2-A-lbfgs-fixed-s0"], False, CFG, out=tmp_path)


@needs_dataset
def test_unit_record_carries_config_sigma_and_seconds(tmp_path) -> None:
    steps.run_e1(["e1-A-normalized-s0"], False, CFG, out=tmp_path)
    rec = json.loads((tmp_path / "e1" / "e1-A-normalized-s0.json").read_text())
    assert rec["complete"] is True and rec["config"] == json.loads(json.dumps(CFG.to_dict()))
    assert set(rec["sigma2"]) == {"poisson", "bcn", "bcp", "n", "p"}
    assert rec["seconds"] > 0 and "gpu_hours" in rec and rec["spec"]["seed"] == 0
    assert rec["diagnostics"]["records"][0]["at"] == "init"


@needs_dataset
def test_shards_write_disjoint_files(tmp_path, monkeypatch) -> None:
    ids = units.enumerate_units("e5", CFG.with_(seeds=(0, 1)))
    a, b = units.shard(ids, 2)
    monkeypatch.setenv("FORGE_RUN_UNITS", json.dumps(a))
    steps.run_e5(config=CFG.with_(seeds=(0, 1)), smoke=False, out=tmp_path)           # units from the runner's env
    first = {p.name for p in (tmp_path / "e5").glob("*.json")}
    monkeypatch.setenv("FORGE_RUN_UNITS", json.dumps(b))
    steps.run_e5(config=CFG.with_(seeds=(0, 1)), smoke=False, out=tmp_path)
    both = {p.name for p in (tmp_path / "e5").glob("*.json")}
    assert first == {f"{u}.json" for u in a} and both == {f"{u}.json" for u in ids} and not set(a) & set(b)


@needs_dataset
def test_bayesian_trials_are_resumable_and_selected_on_validation(tmp_path) -> None:
    cfg = CFG.with_(bo_trials=3, bo_startup_trials=2)
    uid = "e4-A-lbfgs-s0"
    steps.run_e4([uid], False, cfg, out=tmp_path)
    rec = json.loads((tmp_path / "e4" / f"{uid}.json").read_text())
    trials = json.loads((tmp_path / "e4" / f"{uid}.trials.json").read_text())
    assert rec["bo"]["n_evaluations"] == 3 == len(trials) and rec["bo"]["objective_split"] == "validation"
    assert rec["bo"]["sampler"] == "optuna.samplers.GPSampler"
    best = rec["bo"]["best_index"]
    assert rec["bo"]["best_objective"] == min(t["objective"] for t in trials)
    assert rec["per_device_error"] == trials[best]["per_device_error"]                # test errors of the chosen trial
    lo, hi = cfg.bo_log10_lambda
    assert all(10 ** lo * 0.999 <= v <= 10 ** hi * 1.001 for t in trials for v in t["lambda"].values())
    # resume: drop the unit file and the last trial; the first two are replayed, not retrained
    (tmp_path / "e4" / f"{uid}.json").unlink()
    (tmp_path / "e4" / f"{uid}.trials.json").write_text(json.dumps(trials[:2]))
    marks = [t["seconds"] for t in trials[:2]]
    steps.run_e4([uid], False, cfg, out=tmp_path)
    again = json.loads((tmp_path / "e4" / f"{uid}.trials.json").read_text())
    assert [t["seconds"] for t in again[:2]] == marks and len(again) == 3


@needs_dataset
def test_reference_step_records_sigma_floor_and_rho(tmp_path) -> None:
    steps.run_reference(None, False, CFG, out=tmp_path)
    rec = json.loads((tmp_path / "reference" / "reference.json").read_text())
    assert set(rec["sigma2"]) == set(rec["floor"]) == set(rec["rho"]) == {"poisson", "bcn", "bcp", "n", "p"}
    assert rec["bounds"]["D_phi"] > 0 and rec["n_train_devices"] == CFG.max_devices


@needs_dataset
def test_pce_check_step_records_the_tolerance_and_a_verdict(tmp_path) -> None:
    steps.run_pce_check(None, False, CFG, out=tmp_path)
    rec = json.loads((tmp_path / "pce_check" / "pce_check.json").read_text())
    assert rec["tolerance"] == CFG.pce_tolerance and isinstance(rec["passed"], bool)
    assert rec["n_devices"] == CFG.max_devices and "profile_passed" in rec


def _synthetic_results(tmp_path, cfg, n_dev=60, seed=0):
    """Unit result files whose errors are known: the second arm of every pair is 0.5 worse."""
    rng = np.random.default_rng(seed)
    uids = [f"d{i:03d}" for i in range(n_dev)]
    base = rng.gamma(2.0, 1.0, n_dev)
    for step in ("e1", "e2", "e5", "e4"):
        for uid in units.enumerate_units(step, cfg):
            spec = units.parse_unit(uid)
            worse = {"e1": spec.loss != "relative",
                     "e2": spec.weighting == "fixed" or spec.optimizer == "soap",
                     "e5": True, "e4": False}[step]
            shift = {"e1": 0.5 * (spec.loss == "unnormalized") + 0.3 * (spec.loss == "normalized"), "e5": 0.5, "e4": 0.0,
                     "e2": 0.5 * (spec.weighting == "fixed") + 0.3 * (spec.optimizer == "soap")}[step]
            err = base + shift + 0.05 * rng.normal(size=n_dev)
            rec = {"unit": uid, "complete": True, "spec": spec.__dict__, "diverged": False,
                   "per_device_error": dict(zip(uids, err.tolist())), "wall_seconds": 1.0, "gpu_hours": 0.001,
                   "diagnostics": {"early_max_gradient_share": 0.9 if worse else 0.4,
                                   "log10_norm_ratio_init": 5.0 if worse else 1.0,
                                   "curvature_ratio_init": 100.0 if worse else 10.0} if step == "e1" else None}
            if step == "e4":
                rec["bo"] = {"n_evaluations": 10}
            (tmp_path / step).mkdir(exist_ok=True)
            (tmp_path / step / f"{uid}.json").write_text(json.dumps(rec))


def test_analysis_applies_the_decision_rule_to_unit_results(tmp_path) -> None:
    cfg = C.Config(seeds=(0, 1), bootstrap_resamples=300)
    _synthetic_results(tmp_path, cfg)
    res = steps.analyse(root=tmp_path, config=cfg)
    ex = res["experiments"]
    assert all(ex[k]["status"] == "complete" for k in ("E1", "E2", "E3", "E4", "E5"))
    assert set(ex["E1"]["comparisons"]) == {f"{a}|vs_{c}" for a in "AB" for c in ("normalized", "unnormalized")}
    for c in ex["E1"]["comparisons"].values():
        assert c["winner"] == "A" and c["p_holm"] < 0.05 and c["relative_better"] is True   # same divergence counts: not more
        assert c["causal_lower_diagnostics_at_init"] is True
    assert all(c["winner"] == "A" for c in ex["E2"]["comparisons"].values())          # gradient statistics better
    assert set(ex["E2"]["comparisons"]) == {"lbfgs|A", "lbfgs|B", "soap|A", "soap|B"}
    assert all(c["winner"] == "A" for c in ex["E3"]["comparisons"].values())          # L-BFGS lower error than SOAP
    assert ex["E4"]["selected_cell"]["optimizer"] == "lbfgs" and ex["E4"]["outer_loop_evaluations"] == [10, 10]
    assert ex["E5"]["comparisons"]["reset_vs_no_reset"]["winner"] == "A"
    md = steps.report_markdown(res)
    assert "## E1" in md and "| Normalized | A |" in md and "Bayesian optimization" in md


def test_analysis_reports_missing_units_instead_of_guessing(tmp_path) -> None:
    cfg = C.Config(seeds=(0, 1), bootstrap_resamples=100)
    _synthetic_results(tmp_path, cfg)
    victim = next((tmp_path / "e2").glob("*.json"))
    victim.unlink()
    res = steps.analyse(root=tmp_path, config=cfg)
    assert res["experiments"]["E2"]["status"] == "incomplete" and res["experiments"]["E2"]["missing_units"] == [victim.stem]
    assert res["experiments"]["E3"]["status"] == "incomplete" and res["experiments"]["E1"]["status"] == "complete"
    assert "incomplete" in steps.report_markdown(res)


@needs_dataset
def test_smoke_runs_every_training_step_at_rehearsal_scale(tmp_path) -> None:
    for fn in (steps.smoke_reference, steps.smoke_pce_check, steps.smoke_e1, steps.smoke_e2, steps.smoke_e5, steps.smoke_e4):
        fn(out=tmp_path)
    assert len(list((tmp_path / "e1" / "_smoke").glob("e1-*.json"))) == 6             # seed 0 only: 2 archs x 3 losses
    assert len(list((tmp_path / "e2" / "_smoke").glob("e2-*.json"))) == 8
    assert len(list((tmp_path / "e5" / "_smoke").glob("e5-*.json"))) == 2
    assert len(list((tmp_path / "e4" / "_smoke").glob("e4-*-s0.json"))) == 4
    assert not list((tmp_path / "e1").glob("*.json"))                                 # never in the real root
    steps.smoke_analysis(out=tmp_path)
    res = json.loads((tmp_path / "analysis" / "_smoke" / "analysis.json").read_text())
    assert all(res["experiments"][k]["status"] == "complete" for k in ("E1", "E2", "E3", "E4", "E5"))
    assert (tmp_path / "analysis" / "_smoke" / "report.md").read_text().startswith("# Report tables")


def test_results_root_under_the_runner_is_the_working_directory(tmp_path, monkeypatch) -> None:
    """Under a remote runner (FORGE_CLONE_ROOT set) results land in the process working directory, which is
    what the runner's fetch collects; the clone is the pinned input. Locally they stay under the product."""
    from Staged_PINN_Solar import steps

    monkeypatch.delenv(steps.RESULTS_ENV, raising=False)
    monkeypatch.delenv("FORGE_CLONE_ROOT", raising=False)
    assert steps.results_root().name == "Results" and steps.results_root().parent.name == steps.PRODUCT
    monkeypatch.setenv("FORGE_CLONE_ROOT", str(tmp_path / "clone"))
    monkeypatch.chdir(tmp_path)
    assert steps.results_root() == tmp_path / "Results"
    monkeypatch.setenv(steps.RESULTS_ENV, str(tmp_path / "elsewhere"))
    assert steps.results_root() == tmp_path / "elsewhere"


def test_wall_budget_stops_before_a_unit_that_would_not_fit(tmp_path, monkeypatch) -> None:
    """A kernel has a hard time limit: once elapsed + the longest unit seen would pass `wall_budget_seconds`,
    no new unit starts, nothing partial is written, and a later call resumes exactly the missing units."""
    from Staged_PINN_Solar import steps, training, config as C

    clock = {"t": 0.0}
    monkeypatch.setattr(steps.time, "time", lambda: clock["t"])
    monkeypatch.setattr(training, "prepare", lambda root=None, config=None, progress=None: None)
    monkeypatch.setattr(steps.data, "default_root", lambda: tmp_path)

    def fake_train(spec, prep, cfg, **kw):
        clock["t"] += 40.0
        return {"diverged": False, "per_device_error": {}}

    monkeypatch.setattr(training, "train", fake_train)
    uids = U_list = ["e1-A-normalized-s0", "e1-A-normalized-s1", "e1-A-normalized-s2"]
    cfg = C.Config().with_(wall_budget_seconds=100.0)
    done = steps._run_training_step("e1", uids, False, cfg, out=tmp_path, progress=lambda *a: None)
    assert done == uids[:2]                                     # 0 -> 40 -> 80; 80 + 40 > 100: stop
    assert not steps.unit_path("e1", uids[2], False, tmp_path).exists()
    clock["t"] = 0.0
    done = steps._run_training_step("e1", uids, False, cfg, out=tmp_path, progress=lambda *a: None)
    assert done == uids                                         # resumes only the missing one


def test_units_are_dealt_over_the_gpus_expensive_architecture_first():
    pending = ["e1-A-relative-s0", "e1-B-relative-s0", "e1-A-relative-s1", "e1-B-relative-s1", "e1-A-relative-s2"]
    a, b = steps.split_by_cost(pending, 2)
    assert sorted(a + b) == sorted(pending) and not set(a) & set(b)
    assert a[0].split("-")[1] == "B" and b[0].split("-")[1] == "B"
    assert abs(len(a) - len(b)) <= 1


def test_config_survives_the_json_round_trip_to_a_gpu_worker():
    import json
    from Staged_PINN_Solar.config import Config
    cfg = Config().with_(wall_budget_seconds=123.0)
    assert steps.config_from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
