"""The declared steps of the flow, each remote (Kaggle) and each decomposable into independent units.

A step function runs exactly the units it is given (a list of unit ids, or `FORGE_RUN_UNITS` from the
runner's environment, or every unit of the step when neither is given), writes ONE result file per unit
under `Results/<step>/` so shards never collide, and skips a unit whose result file is complete (resumable).
`smoke_<step>` is the cheap rehearsal: 8 devices per split, seed 0, a handful of iterations, written
under `Results/<step>/_smoke/` so it never resumes into, or overwrites, a real unit.

Nothing here reads another step's output except `analysis`, which runs after the others.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import data, loss, pce, stats, training, units, weights
from .config import Config, smoke as smoke_config

__provenance__ = {
    "revision": "experiments-protocolo-experimental-de-la-pinn-por-etapas-sobre-dpv-v02.md",
    "sections": ["E1", "E2", "E3", "E4", "E5"],
    "experiments": ["E1", "E2", "E3", "E4", "E5"],
    "invariants": [
        "a_step_runs_exactly_the_requested_units_and_skips_complete_ones",
        "unit_record_carries_config_sigma_and_seconds",
        "shards_write_disjoint_files",
        "bayesian_trials_are_resumable_and_selected_on_validation",
        "reference_step_records_sigma_floor_and_rho",
        "pce_check_step_records_the_tolerance_and_a_verdict",
        "analysis_applies_the_decision_rule_to_unit_results",
        "analysis_reports_missing_units_instead_of_guessing",
        "smoke_runs_every_training_step_at_rehearsal_scale",
    ],
}

RESULTS_ENV = "STAGED_PINN_RESULTS"
PRODUCT = "Staged-PINN-Solar"
STEPS = ("pce_check", "reference", "e1", "e2", "e5", "e4", "analysis")


def results_root() -> Path:
    env = os.environ.get(RESULTS_ENV)
    return Path(env) if env else Path(__file__).resolve().parents[2] / PRODUCT / "Results"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=_json_default, allow_nan=True))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} is not JSON serializable")


def unit_path(step: str, uid: str, smoke: bool = False, root: Path | None = None) -> Path:
    base = (root or results_root()) / step
    return base / "_smoke" / f"{uid}.json" if smoke else base / f"{uid}.json"


def is_complete(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text()).get("complete"))
    except (OSError, ValueError):
        return False


def _requested(step: str, requested, cfg: Config, smoke: bool) -> list:
    every = units.enumerate_units(step, cfg)
    if requested is None:
        env = os.environ.get("FORGE_RUN_UNITS")
        requested = json.loads(env) if env else None
    if requested is None:
        return every
    unknown = [u for u in requested if u not in every]
    if unknown:
        raise ValueError(f"not units of step {step!r}: {unknown}; its units are {every[:3]}... ({len(every)})")
    return list(requested)


def _resolve(step: str, units_arg, smoke, config):
    if smoke is None:
        smoke = os.environ.get("FORGE_RUN_MODE") == "smoke"
    cfg = smoke_config() if smoke else (config or Config())
    return smoke, cfg, _requested(step, units_arg, cfg, smoke)


def _record(step, uid, cfg, prep, smoke, body: dict) -> dict:
    return {"unit": uid, "step": step, "complete": True, "smoke": smoke, "created": _now(), "config": cfg.to_dict(),
            "dataset_root": prep.root if prep else str(data.default_root()),
            "n_devices": {k: len(getattr(prep, k)) for k in ("train", "val", "test")} if prep else None,
            "sigma2": prep.sigma2 if prep else None, **body}


# --- training steps ------------------------------------------------------------------------------------
def _run_bayes(spec, prep, cfg, uid, smoke, root) -> dict:
    """E4: an outer loop (optuna GPSampler) over log10 lambda_i, i != Poisson; each evaluation is a full inner
    training scored on the VALIDATION devices. Trials are cached beside the unit so a resume replays them."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    names = [k for k in loss.TERMS if k != "poisson"]
    lo, hi = cfg.bo_log10_lambda
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = optuna.samplers.GPSampler(seed=cfg.bo_seed + spec.seed, n_startup_trials=cfg.bo_startup_trials)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    cache_path = unit_path("e4", uid, smoke, root).with_suffix(".trials.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else []
    state = {"trial": None, "k": 0}

    def propose(history):
        if history:
            study.tell(state["trial"], history[-1][1])
        state["trial"] = study.ask()
        return {n: float(10.0 ** state["trial"].suggest_float(f"log10_{n}", lo, hi)) for n in names}

    def evaluate(lam):
        k = state["k"]
        state["k"] += 1
        if k < len(cache) and all(np.isclose(cache[k]["lambda"][n], lam[n], rtol=1e-6) for n in names):
            return cache[k]["objective"]
        r = training.train(spec, prep, cfg, eval_val=True, lam_fixed=lam)
        val = np.array(list(r["per_device_error_val"].values()))
        obj = cfg.bo_diverged_penalty if (r["diverged"] or not np.isfinite(val).any()) else float(np.nanmean(val))
        entry = {"index": k, "lambda": lam, "objective": obj, "diverged": r["diverged"], "seconds": r["seconds"],
                 "gpu_hours": r["gpu_hours"], "per_device_error": r["per_device_error"],
                 "per_device_error_val": r["per_device_error_val"], "blocks": r["blocks"],
                 "stages_done": r["stages_done"]}
        del cache[k:]
        cache.append(entry)
        _write(cache_path, cache)
        return obj

    best, history = weights.outer_loop_weights(propose, evaluate, cfg.bo_trials)
    study.tell(state["trial"], history[-1][1])
    best_i = int(np.argmin([h[1] for h in history]))
    chosen = cache[best_i]
    return {"per_device_error": chosen["per_device_error"], "diverged": chosen["diverged"],
            "stages_done": chosen["stages_done"], "lambda_final": best, "blocks": chosen["blocks"],
            "seconds": sum(c["seconds"] for c in cache), "gpu_hours": sum(c["gpu_hours"] for c in cache),
            "bo": {"n_evaluations": len(history), "objective_split": "validation",
                   "history": [{"lambda": h[0], "objective": h[1]} for h in history], "best_index": best_i,
                   "best_objective": history[best_i][1], "sampler": "optuna.samplers.GPSampler",
                   "log10_lambda_range": [lo, hi]}}


def _run_training_step(step: str, units_arg, smoke, config, root=None, out=None, progress=print) -> list:
    smoke, cfg, todo = _resolve(step, units_arg, smoke, config)
    done = []
    pending = [u for u in todo if not is_complete(unit_path(step, u, smoke, out))]
    prep = training.prepare(root, cfg) if pending else None
    for uid in todo:
        path = unit_path(step, uid, smoke, out)
        if uid not in pending:
            progress(f"[{step}] {uid}: complete, skipped")
            done.append(uid)
            continue
        spec = units.parse_unit(uid)
        t0 = time.time()
        if step == "e4":
            body = _run_bayes(spec, prep, cfg, uid, smoke, out)
        else:
            body = training.train(spec, prep, cfg)
        body["spec"] = asdict(spec)
        body["wall_seconds"] = time.time() - t0
        _write(path, _record(step, uid, cfg, prep, smoke, body))
        progress(f"[{step}] {uid}: written {path.name} diverged={body['diverged']} {body['wall_seconds']:.0f}s")
        done.append(uid)
    return done


def run_e1(units=None, smoke=None, config=None, root=None, out=None):
    """E1: 2 architectures x 2 losses (normalized vs unnormalized) x 5 seeds, L-BFGS, with conditioning diagnostics. 20 units."""
    return _run_training_step("e1", units, smoke, config, root, out)


def run_e2(units=None, smoke=None, config=None, root=None, out=None):
    """E2: 2 weightings x 2 optimizers x 2 architectures x 5 seeds (E3 reuses these runs). 40 units."""
    return _run_training_step("e2", units, smoke, config, root, out)


def run_e5(units=None, smoke=None, config=None, root=None, out=None):
    """E5: L-BFGS with gradient-statistics weights WITHOUT memory reset, 2 architectures x 5 seeds. 10 units."""
    return _run_training_step("e5", units, smoke, config, root, out)


def run_e4(units=None, smoke=None, config=None, root=None, out=None):
    """E4: Bayesian optimization of lambda for both optimizers x both architectures x 5 seeds. 20 units."""
    return _run_training_step("e4", units, smoke, config, root, out)


# --- reference and PCE check -----------------------------------------------------------------------------
def run_reference(units=None, smoke=None, config=None, root=None, out=None, progress=print):
    """sigma_i^2, floors f_i and rho_i over the training devices, plus the ansatz bounds (a record)."""
    smoke, cfg, todo = _resolve("reference", units, smoke, config)
    path = unit_path("reference", "reference", smoke, out)
    if is_complete(path):
        return todo
    prep = training.prepare(root, cfg)
    st = prep.stats
    body = {"sigma2": st.sigma2(), "floor": st.f(), "rho": st.rho(), "count": st.count(),
            "bounds": {"D_phi": prep.bounds.D_phi, "phi_n": list(prep.bounds.phi_n), "phi_p": list(prep.bounds.phi_p)},
            "split_seed": cfg.split_seed, "n_train_devices": len(prep.train), "operator": "dPV interface currents, expanded summands"}
    _write(path, {**_record("reference", "reference", cfg, prep, smoke, body), "sigma2": st.sigma2()})
    progress(f"[reference] written {path}")
    return todo


def run_pce_check(units=None, smoke=None, config=None, root=None, out=None, progress=print):
    """The prerequisite: the evaluator reproduces the PCE stored in dPV on the training devices (pass/fail + tolerance)."""
    smoke, cfg, todo = _resolve("pce_check", units, smoke, config)
    path = unit_path("pce_check", "pce_check", smoke, out)
    if is_complete(path):
        return todo
    root = Path(root) if root else data.default_root()
    split = data.split_devices(data.inventory(root), seed=cfg.split_seed)
    refs = split.train[:cfg.max_devices] if cfg.max_devices else split.train
    from .scaling import Scales
    t0 = time.time()
    rec = pce.check_devices(refs, Scales(), cfg.pce_tolerance, cfg.pce_profile_tolerance)
    rec["seconds"] = time.time() - t0
    _write(path, {"unit": "pce_check", "step": "pce_check", "complete": True, "smoke": smoke, "created": _now(),
                  "config": cfg.to_dict(), "dataset_root": str(root), **rec})
    progress(f"[pce_check] written {path} passed={rec['passed']} max_rel_error={rec['max_rel_error']:.3g}")
    return todo


# --- analysis --------------------------------------------------------------------------------------------
def _load_units(step: str, ids: list, smoke: bool, root) -> tuple:
    found, missing = {}, []
    for uid in ids:
        p = unit_path(step, uid, smoke, root)
        if is_complete(p):
            found[uid] = json.loads(p.read_text())
        else:
            missing.append(uid)
    return found, missing


def _arm(records: list) -> dict:
    """Per-device error averaged over the seeds of one arm, with the seed medians and the divergence count."""
    runs = {r["spec"]["seed"]: r["per_device_error"] for r in records}
    uids, mean = stats.seed_average(runs)
    seed_medians = [float(np.nanmedian(list(v.values()))) if not r["diverged"] else float("nan")
                    for v, r in zip((r["per_device_error"] for r in records), records)]
    return {"uids": uids, "mean": mean, "seed_medians": seed_medians, "n_runs": len(records),
            "n_diverged": int(sum(bool(r["diverged"]) for r in records)),
            "gpu_hours": float(sum(r.get("gpu_hours", 0.0) for r in records)),
            "wall_seconds": float(sum(r.get("wall_seconds", 0.0) for r in records))}


def _pair(a: dict, b: dict) -> tuple:
    ok = np.isfinite(a["mean"]) & np.isfinite(b["mean"])
    return a["mean"][ok], b["mean"][ok]


def _median_ci(x, cfg: Config, seed=0) -> list:
    x = np.asarray(x)[np.isfinite(x)]
    if x.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, x.size, (cfg.bootstrap_resamples, x.size))], axis=1)
    a = (1 - cfg.confidence_level) / 2
    return [float(q) for q in np.quantile(meds, [a, 1 - a])]


def _arm_row(arm: dict, cfg: Config) -> dict:
    m = arm["mean"][np.isfinite(arm["mean"])]
    sm = np.array(arm["seed_medians"])
    return {"median": float(np.median(m)) if m.size else float("nan"), "ci": _median_ci(arm["mean"], cfg),
            "seed_median_mean": float(np.nanmean(sm)) if np.isfinite(sm).any() else float("nan"),
            "seed_median_std": float(np.nanstd(sm)) if np.isfinite(sm).any() else float("nan"),
            "n_runs": arm["n_runs"], "n_diverged": arm["n_diverged"], "n_devices": int(m.size),
            "gpu_hours": arm["gpu_hours"], "wall_seconds": arm["wall_seconds"]}


def _group(records: dict, key) -> dict:
    out: dict = {}
    for uid, r in records.items():
        out.setdefault(key(r["spec"]), []).append(r)
    return out


def _diag_median(records: list, field: str) -> float:
    v = [r["diagnostics"][field] for r in records if r.get("diagnostics") and r["diagnostics"].get(field) is not None]
    return float(np.median(v)) if v else float("nan")


def analyse(smoke: bool = False, root=None, config=None) -> dict:
    """The common decision rule over the unit results of E1-E5 (reads Results/<step>/ only)."""
    cfg = smoke_config() if smoke else (config or Config())
    seeds = cfg.seeds
    out: dict = {"smoke": smoke, "created": _now(), "config": cfg.to_dict(), "experiments": {}}

    def need(step):
        ids = units.enumerate_units(step, cfg)
        found, missing = _load_units(step, ids, smoke, root)
        return found, missing

    def fam(comparisons: dict) -> dict:
        return stats.compare_family({k: _pair(*v) for k, v in comparisons.items()}, cfg.bootstrap_resamples,
                                    cfg.confidence_level, 0, cfg.holm_alpha)

    # E1 -------------------------------------------------------------------------------------------------
    e1, miss = need("e1")
    if miss:
        out["experiments"]["E1"] = {"status": "incomplete", "missing_units": miss}
    else:
        g = _group(e1, lambda s: (s["loss"], s["arch"]))
        arms = {k: _arm(v) for k, v in g.items()}
        cmp = fam({a: (arms[("normalized", a)], arms[("unnormalized", a)]) for a in "AB"})
        rows = {}
        for (l, a), arm in arms.items():
            recs = g[(l, a)]
            rows[f"{l}|{a}"] = {**_arm_row(arm, cfg), "divergent_runs": arm["n_diverged"],
                                "early_max_gradient_share": _diag_median(recs, "early_max_gradient_share"),
                                "log10_norm_ratio_init": _diag_median(recs, "log10_norm_ratio_init"),
                                "curvature_ratio_init": _diag_median(recs, "curvature_ratio_init")}
        for a in "AB":
            n, u = rows[f"normalized|{a}"], rows[f"unnormalized|{a}"]
            c = cmp[a]
            c["fewer_divergent_runs_for_normalized"] = bool(n["divergent_runs"] < u["divergent_runs"])
            c["normalized_better"] = bool(c["winner"] == "A" and c["fewer_divergent_runs_for_normalized"])
            c["causal_lower_diagnostics_at_init"] = bool(
                n["log10_norm_ratio_init"] < u["log10_norm_ratio_init"]
                and n["curvature_ratio_init"] < u["curvature_ratio_init"])
            c["causal_part_note"] = ("medians over seeds of per-run diagnostics; the device-level rule cannot be applied "
                                     "to 5 values (smallest exact two-sided Wilcoxon p = 0.0625)")
        out["experiments"]["E1"] = {"status": "complete", "arms": rows, "comparisons": cmp,
                                    "family": "normalized (A) vs unnormalized (B), Holm over 2 architectures"}

    # E2 / E3 / E4 / E5 -----------------------------------------------------------------------------------
    e2, miss2 = need("e2")
    gradient_arms = {}
    if miss2:
        for k in ("E2", "E3", "E4", "E5"):
            out["experiments"][k] = {"status": "incomplete", "missing_units": miss2, "needs": "e2"}
    else:
        g = _group(e2, lambda s: (s["weighting"], s["optimizer"], s["arch"]))
        arms = {k: _arm(v) for k, v in g.items()}
        cmp = fam({f"{o}|{a}": (arms[("gradient", o, a)], arms[("fixed", o, a)]) for o in ("lbfgs", "soap") for a in "AB"})
        out["experiments"]["E2"] = {"status": "complete", "arms": {"|".join(k): _arm_row(v, cfg) for k, v in arms.items()},
                                    "comparisons": cmp, "family": "gradient statistics (A) vs fixed (B), Holm over 4"}
        gradient_arms = {(o, a): arms[("gradient", o, a)] for o in ("lbfgs", "soap") for a in "AB"}
        cmp3 = fam({a: (gradient_arms[("lbfgs", a)], gradient_arms[("soap", a)]) for a in "AB"})
        out["experiments"]["E3"] = {"status": "complete", "comparisons": cmp3, "reuses": "e2 gradient arms",
                                    "family": "L-BFGS (A) vs SOAP (B), Holm over 2 architectures",
                                    "arms": {f"{o}|{a}": _arm_row(v, cfg) for (o, a), v in gradient_arms.items()}}
        e5, miss5 = need("e5")
        if miss5:
            out["experiments"]["E5"] = {"status": "incomplete", "missing_units": miss5}
        else:
            reset = [r for r in e2.values() if r["spec"]["weighting"] == "gradient" and r["spec"]["optimizer"] == "lbfgs"]
            pooled = {"reset": _arm(reset), "no_reset": _arm(list(e5.values()))}
            c = stats.compare_family({"reset_vs_no_reset": _pair(pooled["reset"], pooled["no_reset"])},
                                     cfg.bootstrap_resamples, cfg.confidence_level, 0, cfg.holm_alpha)
            per_arch = {}
            for a in "AB":
                ra = _arm([r for r in reset if r["spec"]["arch"] == a])
                na = _arm([r for r in e5.values() if r["spec"]["arch"] == a])
                per_arch[a] = stats.compare_family({"reset_vs_no_reset": _pair(ra, na)}, cfg.bootstrap_resamples,
                                                   cfg.confidence_level, 0, cfg.holm_alpha)["reset_vs_no_reset"]
            out["experiments"]["E5"] = {
                "status": "complete", "arms": {k: _arm_row(v, cfg) for k, v in pooled.items()},
                "comparisons": c, "family": "single comparison, pooled over the 2 architectures (reset = A)",
                "per_architecture_descriptive": per_arch}
        e4, miss4 = need("e4")
        if miss4:
            out["experiments"]["E4"] = {"status": "incomplete", "missing_units": miss4}
        else:
            medians = {k: float(np.nanmedian(v["mean"])) for k, v in gradient_arms.items()}
            opt, arch = min(medians, key=medians.get)
            bo = _arm([r for r in e4.values() if r["spec"]["optimizer"] == opt and r["spec"]["arch"] == arch])
            c = stats.compare_family({"bo_vs_gradient": _pair(bo, gradient_arms[(opt, arch)])}, cfg.bootstrap_resamples,
                                     cfg.confidence_level, 0, cfg.holm_alpha)
            evals = [r["bo"]["n_evaluations"] for r in e4.values() if r["spec"]["optimizer"] == opt and r["spec"]["arch"] == arch]
            out["experiments"]["E4"] = {
                "status": "complete", "selected_cell": {"optimizer": opt, "arch": arch},
                "selection_rule": "lowest median seed-averaged error among the E2 gradient-statistics arms (the cell E3 leaves)",
                "arms": {"gradient": _arm_row(gradient_arms[(opt, arch)], cfg), "bayes": _arm_row(bo, cfg)},
                "outer_loop_evaluations": evals, "comparisons": c, "family": "single comparison (bayes = A)"}
    out["seeds"] = list(seeds)
    return out


def _fmt(x, p=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}g}"


def report_markdown(res: dict) -> str:
    """The expected-report tables of E1-E5, filled from the analysis (data, never written into experiments/)."""
    L = ["# Report tables (generated by steps.analyse)", ""]
    ex = res["experiments"]
    ci = lambda r: f"[{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}]"
    e = ex.get("E1", {})
    L += ["## E1", ""]
    if e.get("status") != "complete":
        L += [f"incomplete: {e.get('missing_units')}", ""]
    else:
        L += ["| Arm | Arch | Median rel. PCE error (%) | 95% CI | Divergent runs | Max grad share | log10 norm ratio | lambda_max / mean trace |",
              "|---|---|---|---|---|---|---|---|"]
        for l, name in (("unnormalized", "Unnormalized"), ("normalized", "Normalized")):
            for a in "AB":
                r = e["arms"][f"{l}|{a}"]
                L.append(f"| {name} | {a} | {_fmt(r['median'])} | {ci(r)} | {r['divergent_runs']}/{r['n_runs']} | "
                         f"{_fmt(r['early_max_gradient_share'])} | {_fmt(r['log10_norm_ratio_init'])} | {_fmt(r['curvature_ratio_init'])} |")
        L += [""] + [f"- {a}: p_Holm={_fmt(c['p_holm'])}, CI diff={[_fmt(x) for x in c['ci']]}, winner={c['winner']}, "
                     f"normalized_better={c['normalized_better']}, causal_lower_diagnostics={c['causal_lower_diagnostics_at_init']}"
                     for a, c in e["comparisons"].items()] + [""]
    e = ex.get("E2", {})
    L += ["## E2", ""]
    if e.get("status") != "complete":
        L += [f"incomplete: {e.get('missing_units')}", ""]
    else:
        L += ["| Weighting | Optimizer | Arch | Median rel. PCE error (%) | 95% CI |", "|---|---|---|---|---|"]
        for w, name in (("fixed", "Fixed"), ("gradient", "Gradient statistics")):
            for o, on in (("lbfgs", "L-BFGS"), ("soap", "SOAP")):
                for a in "AB":
                    r = e["arms"][f"{w}|{o}|{a}"]
                    L.append(f"| {name} | {on} | {a} | {_fmt(r['median'])} | {ci(r)} |")
        L += [""] + [f"- {k}: p_Holm={_fmt(c['p_holm'])}, winner={c['winner']}" for k, c in e["comparisons"].items()] + [""]
    e = ex.get("E3", {})
    L += ["## E3", ""]
    if e.get("status") != "complete":
        L += [f"incomplete: {e.get('missing_units', e.get('needs'))}", ""]
    else:
        L += ["| Arch | Median (L-BFGS) | Median (SOAP) | Median difference | 95% CI | p Holm |", "|---|---|---|---|---|---|"]
        for a, c in e["comparisons"].items():
            L.append(f"| {a} | {_fmt(c['median_a'])} | {_fmt(c['median_b'])} | {_fmt(c['median_diff'])} | "
                     f"[{_fmt(c['ci'][0])}, {_fmt(c['ci'][1])}] | {_fmt(c['p_holm'])} |")
        L.append("")
    e = ex.get("E4", {})
    L += ["## E4", ""]
    if e.get("status") != "complete":
        L += [f"incomplete: {e.get('missing_units', e.get('needs'))}", ""]
    else:
        L += [f"Cell selected from E3: {e['selected_cell']}", "", "| lambda method | Median rel. PCE error (%) | 95% CI | outer-loop evaluations |",
              "|---|---|---|---|"]
        L.append(f"| Gradient statistics | {_fmt(e['arms']['gradient']['median'])} | {ci(e['arms']['gradient'])} | n/a |")
        L.append(f"| Bayesian optimization | {_fmt(e['arms']['bayes']['median'])} | {ci(e['arms']['bayes'])} | {e['outer_loop_evaluations']} |")
        c = e["comparisons"]["bo_vs_gradient"]
        L += ["", f"- p={_fmt(c['p'])}, winner={c['winner']}", ""]
    e = ex.get("E5", {})
    L += ["## E5", ""]
    if e.get("status") != "complete":
        L += [f"incomplete: {e.get('missing_units', e.get('needs'))}", ""]
    else:
        L += ["| Arm | Median rel. PCE error (%) | 95% CI |", "|---|---|---|"]
        for k, name in (("reset", "With memory reset"), ("no_reset", "Without memory reset")):
            r = e["arms"][k]
            L.append(f"| {name} | {_fmt(r['median'])} | {ci(r)} |")
        c = e["comparisons"]["reset_vs_no_reset"]
        L += ["", f"- p={_fmt(c['p'])}, winner={c['winner']} (A = reset)", ""]
    return "\n".join(L)


def run_analysis(units=None, smoke=None, config=None, root=None, out=None, progress=print):
    """Reads Results/<step>/ of every training step, applies the common decision rule, writes tables as data."""
    if smoke is None:
        smoke = os.environ.get("FORGE_RUN_MODE") == "smoke"
    base = (out or results_root()) / "analysis"
    res = analyse(smoke=smoke, root=out, config=config)
    folder = base / "_smoke" if smoke else base
    _write(folder / "analysis.json", {**res, "complete": True})
    (folder / "report.md").write_text(report_markdown(res))
    progress(f"[analysis] written {folder}")
    return ["analysis"]


# --- rehearsals ------------------------------------------------------------------------------------------
def _smoke(step):
    def run(out=None, root=None, config=None, progress=print):
        cfg = config or smoke_config()
        every = units.enumerate_units(step, cfg)
        return globals()[f"run_{step}"](every, True, cfg, root, out)
    run.__name__ = f"smoke_{step}"
    run.__doc__ = (f"Rehearsal of {step}: 8 devices per split, seed 0, a handful of iterations; "
                   f"written under Results/{step}/_smoke/.")
    return run


smoke_e1, smoke_e2, smoke_e5, smoke_e4 = (_smoke(s) for s in ("e1", "e2", "e5", "e4"))


def smoke_reference(out=None, root=None, config=None):
    return run_reference(None, True, config or smoke_config(), root, out)


def smoke_pce_check(out=None, root=None, config=None):
    return run_pce_check(None, True, config or smoke_config(), root, out)


def smoke_analysis(out=None, root=None, config=None):
    """Runs the smoke of every training step that has no smoke result yet, then analyses the smoke results."""
    cfg = config or smoke_config()
    for step in ("e1", "e2", "e5", "e4"):
        globals()[f"smoke_{step}"](out, root, cfg)
    return run_analysis(None, True, cfg, root, out)


def run_from_environment(step: str):
    """What a step notebook calls: the runner's FORGE_RUN_MODE / FORGE_RUN_UNITS decide smoke and units."""
    return globals()[f"run_{step}"]()
