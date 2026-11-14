"""Experiment E7 (protocol v10): network M on base B, with the scratch recipe of `scratch.py` unchanged.

Network M: the ansatz of Eq. 3 around a 256 x 5 perceptron (A's width, one layer more) whose input is x_hat plus the 11 parameters of
Eq. 25 (`data_m.XI_PARAMS`), per-device material in every residual, in the current of Eq. 20 and in evaluation
(S2), sigma_i^2 measured on the base-B training split (S7). Stage j trains on the devices with n_V > j and
K = max n_V over the training split (r11 Eq. 27). Everything else is `e6` / `scratch` as is: theta_pre on Eq. 18
alone from the random initialisation, the Eq. 12 centre at theta_pre over I'' with lambda_data = 1 (Eq. 26), one
Optuna study over I'' in a +-BOX_DECADES box, then per stage ADAM fixed Adam steps and up to LBFGS L-BFGS
evaluations from theta_pre of that stage.

  smoke  (`run_smoke`, <= 20 min) base built in memory, a reduced theta_pre, one single-stage trial per GPU and
         100 Adam + 50 L-BFGS on the stage with the most devices; peak GPU memory, time per step, projections and
         an explicit PASS / FAIL.
  study  (`run_study`) ONE account, both GPU workers on ONE journal, stopping at TRIALS trials or the cap
         (operator 2026-11-14).
  final  (`run_final_seed`) split in two kernels per seed (operator 2026-11-14): phase "adam" pretrains and runs
         ADAM Adam steps per stage, writing <UNIT_ADAM><seed>.models.pt; phase "lbfgs" starts every stage from
         that file and runs up to LBFGS L-BFGS evaluations with the same loss and lambda. Each kernel splits the
         K stages over its GPUs (stage j in part j mod n) and merges; seeds 0, 1, 2.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import data_m, e6, steps, training, weights
from .config import Config, network_m

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r10.md",
    "sections": ["5", "6"],
    "equations": ["12", "18", "20", "23", "26", "27"],
    "invariants": [
        "network_m_has_twelve_inputs_and_267267_parameters",
        "stage_eligibility_excludes_devices_whose_curve_ends_before_the_stage",
        "final_parts_split_the_k_stages_of_a_seed",
    ],
}

STUDY = "e7m"
UNIT = "e7m-M-s"
UNIT_ADAM = "e7m-Madam-s"             # Adam-only half of the split final
BEST = "best_lambda_m.json"
JOURNAL = "optuna_journal_m.log"
PRETRAINED = "pretrained_m.pt"
KEYS = training.TERMS + ("J",)
GRID = (0, 4, 8, 12, 16, 20, 24)        # S6: study trial grid
TRIALS = 15                             # operator 2026-11-14: one study, 15 trials, 10.5 h at most
ADAM, LBFGS, TAU = 2000, 500, None      # TAU None: no plateau switch, all ADAM steps run
HOURS = {"smoke": 20 / 60, "study": 10.5, "final": 10.5}
SEEDS = (0, 1, 2)
SMOKE = dict(pre_steps=100, adam=100, lbfgs=50)
SMOKE_LIMITS = dict(peak_gb=13.0, study_gpus=2)


def results_dir() -> Path:
    return steps.results_root() / "e7m"


def _say(msg: str) -> None:
    print(f"[e7m] {msg}", flush=True)


def build_ctx(cfg: Config | None = None, limit: int | None = None, root=None) -> e6.Ctx:
    """The E6 context on base B: banks with per-device material, sigma_i^2 and bounds from base-B train, stored
    fields and dPV currents; every device read once (`data_m.prepare`)."""
    cfg = cfg or network_m()
    prep, fields, currents, K = data_m.prepare(root, cfg, report=_say, limit=limit)
    return e6.Ctx(prep, fields, cfg, K, currents)


def _file(name: str, out: Path) -> Path | None:
    for p in (out / name, e6.handoff_dir() / name):
        if p.exists():
            return p
    return None


def theta_pre(ctx, out: Path, tel, deadline: float) -> tuple[dict, Path]:
    """theta_pre of seed 0 over the K stages (Eq. 18 alone, random initialisation): `pretrained_m.pt` when handed
    over (Results/e7m or handoff-e6/), pretrained here otherwise. Never E6's pretrained.pt (another network)."""
    p = _file(PRETRAINED, out)
    if p is not None:
        return torch.load(p, map_location="cpu", weights_only=False), p
    pre = e6.pretrain(ctx, 0, ctx.K, tel, deadline)
    p = out / PRETRAINED
    torch.save(pre, p)
    return pre, p


def centre_all(ctx, pre: dict, seed: int = 0, grid: tuple = GRID) -> dict:
    """r11 Eq. 26: Eq. 12 at theta_pre over I'', lambda_hat per grid stage rescaled so lambda_data = 1, geometric
    mean over the grid stages theta_pre holds."""
    model = e6.new_model(ctx, seed)
    params = list(model.parameters())
    logs = {k: [] for k in KEYS}
    for j in grid:
        if j not in pre:
            continue
        model.load_state_dict(pre[j])
        obj = e6.PhasedObjective(ctx, model, e6.physics_batch(ctx, j, seed), j,
                                 jbatch=e6.current_batch(ctx, j, seed))
        lh = weights.lambda_hat(obj.term_grads(params), "max", ctx.cfg.eps_g, ctx.cfg.lambda_min, ctx.cfg.lambda_max)
        for k in KEYS:
            logs[k].append(math.log10(lh[k] / lh["data"]))
    return {k: 10.0 ** float(np.mean(v)) for k, v in logs.items()}


def _grid(ctx) -> tuple:
    return tuple(j for j in GRID if j < ctx.K)


def study_progress_line(s: dict, elapsed: float, remaining: float) -> str:
    """One live progress line of the study: finished trials of TRIALS, best value, elapsed time and ETA."""
    done = int(s.get("complete", 0)) + int(s.get("pruned", 0)) + int(s.get("failed", 0))
    eta = elapsed / done * max(TRIALS - done, 0) if done else float("nan")
    eta = min(eta, remaining) if done else eta
    best = s.get("best_value")
    return (f"[e7m study] {done}/{TRIALS} finished (complete {s.get('complete', 0)} · pruned {s.get('pruned', 0)} · "
            f"failed {s.get('failed', 0)} · running {s.get('running_or_lost', 0)}) · best "
            f"{'-' if best is None else f'{best:.4g}'} · {elapsed / 3600:.2f} h elapsed · ETA "
            f"{'-' if eta != eta else f'{eta / 3600:.2f} h'}")


def _watch_study(journal, t0: float, deadline: float, stop, every: float = 120.0) -> None:
    """Print the study's progress every `every` seconds until `stop` is set (live Kaggle log)."""
    while not stop.wait(every):
        try:
            s = e6.study_summary(journal, STUDY)
            print(study_progress_line(s, time.time() - t0, deadline - time.time()), flush=True)
        except Exception as e:  # noqa: BLE001  (a progress line must never stop the study)
            print(f"[e7m study] progress unavailable: {e!r}", flush=True)


def run_study() -> dict:
    """The study on ONE account: theta_pre, the Eq. 26 centre, then both GPU workers on ONE journal until TRIALS
    trials or HOURS["study"]. Writes study-m.json and, when a trial completed, best_lambda_m.json."""
    start = time.time()
    deadline = start + HOURS["study"] * 3600.0
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    ctx = build_ctx()
    tel = e6.Telemetry(out, "e7m-study")
    pre, pre_path = theta_pre(ctx, out, tel, deadline)
    c = centre_all(ctx, pre, 0, _grid(ctx))
    tel.line(f"Eq. 26 centre over I'' at theta_pre (lambda_data = 1): "
             f"{json.dumps({k: float(f'{v:.3e}') for k, v in c.items()})}")
    journal = out / JOURNAL
    e6.open_study(journal, 0, create_centre=c, study_name=STUDY)
    if ctx.dev.type == "cuda":
        torch.cuda.empty_cache()
    n = min(max(e6._n_gpu(), 1), 2)
    stop = threading.Event()
    watch = threading.Thread(target=_watch_study, args=(journal, time.time(), deadline, stop), daemon=True)
    watch.start()
    e6._run_payloads([{"role": "tune", "worker": 10 + w, "start": str(pre_path), "journal": str(journal),
                       "out": str(out), "deadline": deadline, "study": STUDY, "keys": list(KEYS), "target": TRIALS,
                       "grid": list(_grid(ctx))} for w in range(n)], ctx, deadline)
    stop.set()
    watch.join(timeout=5)
    s = e6.study_summary(journal, STUDY)
    s.update(part=0, centre=c, created=steps._now(), pretrained=str(pre_path), hours=(time.time() - start) / 3600.0)
    steps._write(out / "study-m.json", s)
    if s["complete"]:
        steps._write(out / BEST, join_studies([s]))
    _say(f"study: trials {s['n_trials']} · complete {s['complete']} · pruned {s['pruned']} · "
         f"best {s.get('best_value')} · {s['hours']:.2f} h")
    return s


def join_studies(summaries: list[dict]) -> dict:
    """The best complete trial over the study summaries, as the content of best_lambda_m.json."""
    best, total = None, {"n_trials": 0, "complete": 0, "pruned": 0, "failed": 0}
    for s in summaries:
        for k in total:
            total[k] += int(s.get(k, 0))
        for t in s["trials"]:
            if t["state"] == "COMPLETE" and t["value"] is not None and (best is None or t["value"] < best[1]["value"]):
                best = (s.get("part", 0), t)
    if best is None:
        raise ValueError("no complete trial in the study")
    part, t = best
    lam = e6.full_lambda({k[6:]: 10.0 ** v for k, v in t["params"].items() if k.startswith("log10_")})
    return {"lambda": lam, "part": part, "trial": t["number"], "value": t["value"], **total,
            "centre": summaries[0].get("centre"), "rule": "E7 network M: Optuna over I'' from theta_pre (r11 Eq. 26)"}


def final_units(seed: int, parts: int, unit: str = UNIT) -> list:
    """The units of one seed split into `parts` (stage j in part j mod parts, `e6.unit_spec`)."""
    return [f"{unit}{seed}.p{p}of{parts}" for p in range(parts)]


def final_payloads(seed: int, parts: int, phase: str, lam: dict, out: Path, deadline: float,
                   adam_file: Path | None = None) -> list:
    """Payloads of one half of the split final. "adam": theta_pre then ADAM Adam steps per stage, no L-BFGS.
    "lbfgs": no Adam, up to LBFGS evaluations per stage from the per-stage Adam-end models of `adam_file`."""
    base = dict(role="final", **{"lambda": lam}, out=str(out), deadline=deadline, scratch=True, tau=TAU)
    if phase == "adam":
        return [{**base, "units": [u], "adam_max": ADAM, "lbfgs_max": 0} for u in final_units(seed, parts, UNIT_ADAM)]
    if phase == "lbfgs":
        if adam_file is None:
            raise FileNotFoundError(f"{UNIT_ADAM}{seed}.models.pt is needed for the L-BFGS half")
        return [{**base, "units": [u], "adam_max": 0, "lbfgs_max": LBFGS, "start_file": str(adam_file),
                 "start_key": "final"} for u in final_units(seed, parts)]
    raise ValueError(f"phase {phase!r}: 'adam' or 'lbfgs'")


def run_final_seed(seed: int = 0, parts: int | None = None, phase: str = "adam") -> list:
    """One half of one seed of the final on one kernel, its stages split over the kernel's GPUs, then merged."""
    start = time.time()
    deadline = start + HOURS["final"] * 3600.0
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    p = _file(BEST, out)
    if p is None:
        raise FileNotFoundError(f"{BEST} is neither in Results/e7m nor in handoff-e6/")
    lam = json.loads(p.read_text())["lambda"]
    adam_file = _file(f"{UNIT_ADAM}{seed}.models.pt", out) if phase == "lbfgs" else None
    ctx = build_ctx()
    n = parts or max(e6._n_gpu(), 1)
    pays = final_payloads(seed, n, phase, lam, out, deadline, adam_file)
    e6._run_payloads(pays, ctx, deadline)
    unit = UNIT_ADAM if phase == "adam" else UNIT
    e6.merge_parts(out, ctx.prep, f"{unit}{seed}")
    return [u for q in pays for u in q["units"] if steps.is_complete(out / f"{u}.json")]


# --- smoke ------------------------------------------------------------------------------------------------------
def _history(out: Path, name: str) -> list:
    p = out / f"{name}.history.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def _per_step(recs: list, phase: str) -> float:
    """Seconds per step between the first and last logged step of `phase` (logged every e6.EVERY steps)."""
    r = [x for x in recs if x.get("phase") == phase and "step" in x]
    if len(r) < 2 or r[-1]["step"] == r[0]["step"]:
        return float("nan")
    return (r[-1]["wall"] - r[0]["wall"]) / (r[-1]["step"] - r[0]["step"])


def _seconds_per_eval(ctx, pre: dict, j: int, lam: dict, n: int = 5) -> float:
    """Seconds of one loss-and-gradient evaluation on the final batch of stage j (one L-BFGS closure call),
    timed directly: the fallback when too few L-BFGS evaluations were logged to read it from the history."""
    model = e6.new_model(ctx, 0)
    model.load_state_dict(pre[j])
    params = list(model.parameters())
    obj = e6.PhasedObjective(ctx, model, e6.physics_batch(ctx, j, 0), j, jbatch=e6.current_batch(ctx, j, 0))
    obj.terms_and_grad(lam, params)
    sync = torch.cuda.synchronize if ctx.dev.type == "cuda" else (lambda: None)
    sync()
    t = time.time()
    for _ in range(n):
        obj.terms_and_grad(lam, params)
    sync()
    return (time.time() - t) / n


def _peak_gb(dev) -> float | None:
    return torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None


def _reset_peak(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def _smoke_worker(ctx, a: dict) -> dict:
    """One GPU of the smoke: one single-stage Optuna trial (the study batch), then SMOKE adam + lbfgs of the final
    recipe on the same stage (the final batch). Peak memory is measured separately for each. Writes smoke-w<w>.json."""
    import optuna
    from optuna.trial import TrialState

    out, w, j, dev = Path(a["out"]), int(a["worker"]), int(a["stage"]), ctx.dev
    pre = torch.load(a["start_file"], map_location="cpu", weights_only=False)
    tel = e6.Telemetry(out, f"smoke-w{w}")
    rec = {"worker": w, "stage": j, "device": str(dev)}
    _reset_peak(dev)
    study = e6.open_study(Path(a["journal"]), w, study_name=a["study"])
    c = study.user_attrs["centre"]
    trial = study.ask()
    lam = e6.full_lambda({k: 10.0 ** trial.suggest_float(f"log10_{k}", math.log10(c[k]) - e6.BOX_DECADES,
                                                         math.log10(c[k]) + e6.BOX_DECADES) for k in KEYS})
    t = time.time()
    try:
        value = e6.run_trial(ctx, pre, lam, 0, a["deadline"], trial, tel, with_j=True, grid=(j,))
        study.tell(trial, value)
    except (optuna.TrialPruned, TimeoutError) as e:
        study.tell(trial, state=TrialState.FAIL)
        value = float("nan")
        tel.line(f"trial stopped: {e!r}")
    rec.update(trial_value=value, trial_finite=bool(math.isfinite(value) and value < e6.PENALTY),
               trial_stage_seconds=time.time() - t, study_peak_gb=_peak_gb(dev))
    tel.line(f"trial: value {value:.3e} · {rec['trial_stage_seconds']:.0f}s · peak {rec['study_peak_gb']} GB")
    _reset_peak(dev)
    uid = f"e7m-smoke-w{w}-s0"
    t = time.time()
    lam_c = e6.full_lambda({k: c[k] for k in KEYS})
    r = e6.final_unit(ctx, uid, 0, lam_c, out, a["deadline"], start=pre,
                      stages=[j], adam_max=SMOKE["adam"], lbfgs_max=SMOKE["lbfgs"], tau=None)
    rec["final_peak_gb"] = _peak_gb(dev)
    rec["final_seconds"] = time.time() - t
    if r is None or not r["stage_progress"]:
        rec.update(final_complete=False)
    else:
        row = r["stage_progress"][0]
        hist = _history(out, uid)
        losses = [x["total"] for x in hist if x.get("phase") in ("adam", "lbfgs")]
        rec.update(final_complete=True, diverged=bool(r["diverged"]), loss_start=row.get("loss_start"),
                   loss_adam_end=row.get("loss_adam_end"), loss_end=row.get("loss_end"),
                   adam_steps=row.get("adam_steps"), lbfgs_evaluations=row.get("lbfgs_evaluations"),
                   stage_seconds=row.get("seconds"), s_per_adam=_per_step(hist, "adam"),
                   s_per_lbfgs_eval=_per_step(hist, "lbfgs"), s_per_eval_direct=_seconds_per_eval(ctx, pre, j, lam_c),
                   all_finite=bool(all(math.isfinite(v) for v in losses + [row.get("loss_end", float("nan"))])))
    if rec.get("final_complete"):
        for k in ("s_per_adam", "s_per_lbfgs_eval"):          # too few logged steps: the direct timing
            if not math.isfinite(rec[k]):
                rec[k] = rec["s_per_eval_direct"]
    steps._write(out / f"smoke-w{w}.json", rec)
    tel.line(f"done: {json.dumps({k: rec.get(k) for k in ('study_peak_gb', 'final_peak_gb', 's_per_adam', 's_per_lbfgs_eval')})}")
    return rec


def smoke_verdict(workers: list, *, s_pre_step: float, centre_stage_s: float, K: int, n_grid: int,
                  peak_limit_gb: float = SMOKE_LIMITS["peak_gb"], study_gpus: int = SMOKE_LIMITS["study_gpus"]) -> dict:
    """Projections and the PASS / FAIL of the smoke (operator 2026-11-14): peak GPU memory < 13 GB, no NaN / inf,
    loss decreasing over the Adam steps, and TRIALS complete trials projected within HOURS["study"] on 2 GPUs."""
    ok = [w for w in workers if w.get("final_complete")]
    peaks = [p for w in workers for p in (w.get("study_peak_gb"), w.get("final_peak_gb")) if p is not None]
    peak = max(peaks) if peaks else None
    trial_stage_s = max([w["trial_stage_seconds"] for w in workers], default=float("nan"))
    s_adam = max([w["s_per_adam"] for w in ok], default=float("nan"))
    s_lbfgs = max([w["s_per_lbfgs_eval"] for w in ok], default=float("nan"))
    overhead = max([w["stage_seconds"] - w["adam_steps"] * s_adam - w["lbfgs_evaluations"] * s_lbfgs for w in ok],
                   default=float("nan"))
    pretrain_h = K * e6.PRE_STEPS * s_pre_step / 3600.0
    trial_h = n_grid * trial_stage_s / 3600.0                       # upper bound: no pruning
    study_h = pretrain_h + n_grid * centre_stage_s / 3600.0 + math.ceil(TRIALS / study_gpus) * trial_h
    final_h = pretrain_h + math.ceil(K / study_gpus) * (ADAM * s_adam + LBFGS * s_lbfgs + max(overhead, 0.0)) / 3600.0
    checks = {
        "peak_below_limit": peak is None or peak < peak_limit_gb,
        "finite": bool(workers) and all(w.get("trial_finite") for w in workers)
                  and len(ok) == len(workers) and all(w["all_finite"] and not w["diverged"] for w in ok),
        "adam_decreases_loss": bool(ok) and all(w["loss_adam_end"] < w["loss_start"] for w in ok),
        "study_fits": bool(math.isfinite(study_h) and study_h <= HOURS["study"]),
    }
    return {"peak_gb": peak, "trial_stage_seconds": trial_stage_s, "s_per_adam": s_adam, "s_per_lbfgs_eval": s_lbfgs,
            "stage_overhead_seconds": overhead, "s_per_pretrain_step": s_pre_step,
            "projected_pretrain_hours": pretrain_h, "projected_trial_hours": trial_h,
            "projected_study_hours": study_h, "projected_final_hours": final_h,
            "final_fits": bool(math.isfinite(final_h) and final_h <= HOURS["final"]),
            "checks": checks, "pass": all(checks.values())}


def run_smoke() -> dict:
    """<= 20 min before the long kernels (S8). Builds base B in memory, a reduced theta_pre up to the stage with the
    most devices, the Eq. 26 centre on that stage, then one smoke worker per GPU (`_smoke_worker`)."""
    start = time.time()
    deadline = start + HOURS["smoke"] * 3600.0
    out = results_dir() / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    _say(f"smoke: building base M (deadline {HOURS['smoke'] * 60:.0f} min)")
    ctx = build_ctx()
    t_ctx = time.time() - start
    counts = [int((ctx.prep.train.n_V > j).sum()) for j in range(ctx.K)]
    stage = int(np.argmax(counts))
    tel = e6.Telemetry(out, "smoke-pretrain")
    t = time.time()
    pre = e6.pretrain(ctx, 0, stage + 1, tel, deadline, n_steps=SMOKE["pre_steps"])
    s_pre_step = (time.time() - t) / ((stage + 1) * SMOKE["pre_steps"])
    pre_path = out / "pretrained_m_smoke.pt"
    torch.save(pre, pre_path)
    _say(f"smoke: reduced theta_pre ({stage + 1} stage(s) x {SMOKE['pre_steps']} steps) {s_pre_step * 1e3:.1f} ms/step")
    t = time.time()
    c = centre_all(ctx, pre, 0, (stage,))
    centre_stage_s = time.time() - t
    journal = out / "optuna_journal_smoke.log"
    e6.open_study(journal, 0, create_centre=c, study_name=f"{STUDY}-smoke")
    if ctx.dev.type == "cuda":
        torch.cuda.empty_cache()
    n = max(e6._n_gpu(), 1)
    _say(f"smoke: centre {json.dumps({k: float(f'{v:.3e}') for k, v in c.items()})}; {n} worker(s) on stage {stage} "
         f"({counts[stage]} train devices)")
    e6._run_payloads([{"role": "call", "fn": "Staged_PINN_Solar.scratch_m:_smoke_worker", "worker": w,
                       "stage": stage, "start_file": str(pre_path), "journal": str(journal), "out": str(out),
                       "deadline": deadline, "study": f"{STUDY}-smoke"} for w in range(n)], ctx, deadline)
    workers = [json.loads((out / f"smoke-w{w}.json").read_text()) for w in range(n) if (out / f"smoke-w{w}.json").exists()]
    v = smoke_verdict(workers, s_pre_step=s_pre_step, centre_stage_s=centre_stage_s, K=ctx.K, n_grid=len(_grid(ctx)))
    if len(workers) < n:
        v["checks"]["all_workers_reported"] = False
        v["pass"] = False
    rec = {"unit": "e7m-smoke", "complete": True, "created": steps._now(), "n_gpu": e6._n_gpu(), "K": ctx.K,
           "stage": stage, "stage_devices": counts[stage], "ctx_seconds": t_ctx, "centre": c, "workers": workers,
           **v, "seconds": time.time() - start}
    steps._write(results_dir() / "e7m-smoke.json", rec)
    _say(f"smoke {'PASS' if rec['pass'] else 'FAIL'} · {json.dumps({k: rec[k] for k in ('n_gpu', 'peak_gb', 's_per_adam', 's_per_lbfgs_eval', 'projected_study_hours', 'projected_final_hours', 'checks')})} · "
         f"{rec['seconds'] / 60:.1f} min")
    return rec
