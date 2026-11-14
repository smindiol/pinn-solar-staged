"""Scratch control of r09 (operator 2026-11-14).

Question: is the r09 result owed to the chain of earlier trainings (r05 -> r06 -> r08 -> r09, each stage warm-started
from the previous run) or does the r09 recipe reach it from nothing? Nothing of the recipe changes: the same network,
the same normalisation (sigma2 measured on the training split, per-node relative residual, J / J_sc), the same loss
over I'' = physics terms + data + J (r09 Eq. 23), and the Optuna study run exactly like the last one (r08 P/Q): the
Eq. 12 rule at theta_pre gives the centre of every lambda, each searched in a box of +-BOX_DECADES around it,
lambda_data = 1 as the reference. What changes is the start and the budget:

  study  every lambda of I'' re-tuned, trials from theta_pre (pretraining on Eq. 18 alone from a random
         initialisation, seed 0, as before), centre by Eq. 12 at theta_pre over I'' (the r08 study's rule, now
         including J). One account, both GPU workers on ONE journal (every trial sees the others), TRIALS trials.
  final  every seed pretrained here (as r05's final did), then per stage ADAM fixed Adam steps (no plateau switch)
         and up to LBFGS L-BFGS evaluations, nothing carried over from earlier runs. Seeds split over every GPU with
         e6.final_plan, as in r09 F.
  smoke  one short kernel (~10 min): one single-stage trial per GPU worker and one cut near-Voc stage per GPU
         from theta_pre, to measure time per step and peak GPU memory and project the study and final wall times.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import e6, steps, training, weights

STUDY = "e6s"
UNIT = "e6s-A-s"
BEST = "best_lambda_s.json"
JOURNAL = "optuna_journal_s.log"
KEYS = training.TERMS + ("J",)
TRIALS = 40                             # operator 2026-11-14: 40 trials, 3.5 h at most
ADAM, LBFGS, TAU = 2000, 500, None      # TAU None: no plateau switch, all ADAM steps run
HOURS = {"study": 3.5, "final": 10.5, "smoke": 0.4}
PRETRAIN_S = 240.0                      # measured: theta_pre of 24 stages took 240 s on one T4 (r08 P, 2026-11-13)
SMOKE = dict(units=(f"{UNIT}0.p21of24", f"{UNIT}1.p20of24"), adam=100, lbfgs=100, grid=(20,))


def theta_pre(ctx, out: Path, tel, deadline: float) -> tuple[dict, Path]:
    """theta_pre of seed 0 (Eq. 18 alone, from a random initialisation): the copy handed over in handoff-e6/
    (the network the last Optuna study was centred on), pretrained here only if it is missing."""
    p = e6._handoff_file("pretrained.pt")
    if p is not None:
        return torch.load(p, map_location="cpu", weights_only=False), p
    pre = e6.pretrain(ctx, 0, ctx.K, tel, deadline)
    p = out / "pretrained.pt"
    torch.save(pre, p)
    return pre, p


def centre_all(ctx, pre: dict, seed: int = 0) -> dict:
    """Eq. 12 at theta_pre over I'' (the r08 study's centre, extended to the current term like r09 Eq. 24):
    lambda_hat per grid stage, rescaled so lambda_data = 1, geometric mean over GRID."""
    model = e6.new_model(ctx, seed)
    params = list(model.parameters())
    logs = {k: [] for k in KEYS}
    for j in e6.GRID:
        if j not in pre:
            continue
        model.load_state_dict(pre[j])
        obj = e6.PhasedObjective(ctx, model, e6.physics_batch(ctx, j, seed), j,
                                 jbatch=e6.current_batch(ctx, j, seed))
        lh = weights.lambda_hat(obj.term_grads(params), "max", ctx.cfg.eps_g, ctx.cfg.lambda_min, ctx.cfg.lambda_max)
        for k in KEYS:
            logs[k].append(math.log10(lh[k] / lh["data"]))
    return {k: 10.0 ** float(np.mean(v)) for k, v in logs.items()}


def _study(ctx, out: Path, deadline: float, target: int, journal: Path, grid: tuple | None = None) -> dict:
    tel = e6.Telemetry(out, "scratch-study")
    pre, pre_path = theta_pre(ctx, out, tel, deadline)
    c = centre_all(ctx, pre, 0)
    tel.line(f"Eq. 12 centre over I'' at theta_pre (lambda_data = 1): "
             f"{json.dumps({k: float(f'{v:.3e}') for k, v in c.items()})}")
    e6.open_study(journal, 0, create_centre=c, study_name=STUDY)
    if ctx.dev.type == "cuda":
        torch.cuda.empty_cache()
    n = min(max(e6._n_gpu(), 1), 2)
    e6._run_payloads([{"role": "tune", "worker": 10 + w, "start": str(pre_path), "journal": str(journal),
                       "out": str(out), "deadline": deadline, "study": STUDY, "keys": list(KEYS), "target": target,
                       "grid": list(grid) if grid else None} for w in range(n)], ctx, deadline)
    s = e6.study_summary(journal, STUDY)
    s.update(part=0, centre=c, created=steps._now(), pretrained=str(pre_path))
    return s


def run_study() -> dict:
    """The scratch study on one account: both GPU workers share one journal, stopping at TRIALS trials."""
    start = time.time()
    out = e6.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    ctx = e6.build_ctx()
    s = _study(ctx, out, start + HOURS["study"] * 3600.0, TRIALS, out / JOURNAL)
    s["hours"] = (time.time() - start) / 3600.0
    steps._write(out / "study-s.json", s)
    print(f"[scratch study] trials {s['n_trials']} · complete {s['complete']} · pruned {s['pruned']} · "
          f"best {s.get('best_value')} · {s['hours']:.2f} h", flush=True)
    return s


def join_studies(summaries: list[dict]) -> dict:
    """The best complete trial over the study summaries (one, or several parts with the same objective and box)."""
    best, total = None, {"n_trials": 0, "complete": 0, "pruned": 0, "failed": 0}
    for s in summaries:
        for k in total:
            total[k] += int(s.get(k, 0))
        for t in s["trials"]:
            if t["state"] == "COMPLETE" and t["value"] is not None and (best is None or t["value"] < best[1]["value"]):
                best = (s["part"], t)
    if best is None:
        raise ValueError("no complete trial in any part")
    part, t = best
    lam = e6.full_lambda({k[6:]: 10.0 ** v for k, v in t["params"].items() if k.startswith("log10_")})
    return {"lambda": lam, "part": part, "trial": t["number"], "value": t["value"], **total,
            "centre": summaries[0].get("centre"), "rule": "scratch study: Optuna over I'' from theta_pre"}


def run_final(account: int) -> list:
    """The scratch final runs of `account` (e6.final_plan with UNIT): every seed pretrained here, then ADAM fixed
    Adam steps and up to LBFGS L-BFGS evaluations per stage, lambda from best_lambda_s.json."""
    start = time.time()
    deadline = start + HOURS["final"] * 3600.0
    out = e6.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    lam = e6._lambda_file(BEST)
    packs = e6.final_plan(unit=UNIT)[account]
    ctx = e6.build_ctx()
    e6._run_payloads([{"role": "final", "units": p, "lambda": lam, "out": str(out), "deadline": deadline,
                       "scratch": True, "adam_max": ADAM, "lbfgs_max": LBFGS, "tau": TAU} for p in packs], ctx, deadline)
    for base in sorted({e6.unit_spec(u, ctx.K)[2] for p in packs for u in p}):
        e6.merge_parts(out, ctx.prep, base)
    return [u for p in packs for u in p if steps.is_complete(out / f"{u}.json")]


def _history(out: Path, name: str) -> list:
    p = out / f"{name}.history.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def run_smoke() -> dict:
    """~10 min check before the long kernels: a 2-trial study on one grid stage (one trial per GPU worker) and one
    near-Voc stage per GPU from theta_pre with a cut Adam/L-BFGS budget. Projects the study and final wall times."""
    start = time.time()
    deadline = start + HOURS["smoke"] * 3600.0
    out = e6.results_dir() / "smoke-s"
    out.mkdir(parents=True, exist_ok=True)
    ctx = e6.build_ctx()
    t0 = time.time()
    study = _study(ctx, out, deadline, 2, out / "optuna_journal_smoke.log", grid=SMOKE["grid"])
    t_study = time.time() - t0
    trials = [r for w in (10, 11) for r in _history(out, f"tune-w{w}") if r.get("kind") == "trial"]
    lam = e6.full_lambda(dict(study["centre"]))                       # the centre itself: values not judged
    t1 = time.time()
    e6._run_payloads([{"role": "final", "units": [u], "lambda": lam, "out": str(out), "deadline": deadline,
                       "scratch": True, "start_file": study["pretrained"], "adam_max": SMOKE["adam"],
                       "lbfgs_max": SMOKE["lbfgs"], "tau": TAU} for u in SMOKE["units"]], ctx, deadline)
    t_final = time.time() - t1
    units = {}
    for u in SMOKE["units"]:
        p = out / f"{u}.json"
        if not steps.is_complete(p):
            units[u] = None
            continue
        r = json.loads(p.read_text())
        row = r["stage_progress"][0]
        evals = row["adam_steps"] + row["lbfgs_evaluations"]
        units[u] = {"stage": row["stage"], "seconds_stage": row["seconds"], "adam_steps": row["adam_steps"],
                    "lbfgs_evaluations": row["lbfgs_evaluations"], "s_per_eval": row["seconds"] / max(evals, 1),
                    "diverged": r["diverged"], "peak_mem_gb": r.get("peak_mem_gb")}
    ok = [v for v in units.values() if v]
    s_eval = max(v["s_per_eval"] for v in ok) if ok else float("nan")
    trial_stage_s = max([t["seconds"] for t in trials], default=float("nan")) / len(SMOKE["grid"])
    stages_per_gpu = max(sum(len(e6.unit_spec(u, ctx.K)[1]) for u in pack)
                         for packs in e6.final_plan(unit=UNIT) for pack in packs)
    eta_final_h = (stages_per_gpu * (ADAM + LBFGS) * s_eval + 2 * PRETRAIN_S) / 3600.0
    eta_study_h = math.ceil(TRIALS / 2) * len(e6.GRID) * trial_stage_s / 3600.0     # upper bound: no pruning
    rec = {"unit": "e6s-smoke", "complete": True, "created": steps._now(), "n_gpu": e6._n_gpu(),
           "study": {"trials": len(trials), "states": [t.get("state") for t in trials],
                     "trial_stage_seconds": trial_stage_s, "seconds": t_study, "centre": study["centre"]},
           "final": units, "final_seconds": t_final, "s_per_eval": s_eval, "stages_per_gpu": stages_per_gpu,
           "eta_study_hours": eta_study_h, "eta_final_hours": eta_final_h, "seconds": time.time() - start}
    rec["pass"] = bool(len(trials) >= 2 and len(ok) == len(SMOKE["units"]) and not any(v["diverged"] for v in ok)
                       and eta_final_h <= HOURS["final"] - 0.5)      # the study stops itself at HOURS["study"]
    steps._write(e6.results_dir() / "e6s-smoke.json", rec)
    print(f"[scratch smoke] {json.dumps({k: rec[k] for k in ('n_gpu', 's_per_eval', 'eta_study_hours', 'eta_final_hours', 'pass')})}",
          flush=True)
    return rec
