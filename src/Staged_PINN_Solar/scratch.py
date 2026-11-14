"""Scratch control of r09 (operator 2026-11-14).

Question: is the r09 result owed to the chain of earlier trainings (r05 -> r06 -> r08 -> r09, each stage warm-started
from the previous run) or does the r09 recipe reach it from nothing? Nothing of the recipe changes: the same network,
the same loss over I'' = physics terms + data + J (r09 Eq. 23), the same Eq. 12 centre and Optuna box, the same
Adam -> L-BFGS phases. What changes is the start and the budget:

  study  every lambda of I'' except lambda_data = 1 (the reference) re-tuned by Optuna, trials from theta_pre
         (data-only pretraining from a random initialisation, seed 0), centre by Eq. 12 at theta_pre over I''.
         Two independent parts (one account each, two GPU workers), TRIALS_PER_PART trials each; the parts are
         joined locally by `join_studies` (same objective, same box), best complete trial wins.
  final  every seed pretrained here from a random initialisation, then per stage ADAM fixed Adam steps (no plateau
         switch) and up to LBFGS L-BFGS evaluations: about the steps the r09 networks accumulated over
         r05, r06, r08 and r09 (each stopped Adam at ~150-270 steps and used ~430 L-BFGS evaluations).
         Seeds split over every GPU with e6.final_plan, as in r09 F.
  smoke  one short kernel: one trial per GPU worker and one heavy stage per GPU with a cut budget, to measure
         time per step and peak GPU memory and to project the study and final wall times.
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
PRE = "pretrained_s0.pt"
KEYS = training.TERMS + ("J",)
TRIALS_PER_PART = 25            # two parts -> 50 trials
ADAM, LBFGS, TAU = 1000, 1700, None      # TAU None: no plateau switch, all ADAM steps run
HOURS = {"study": 4.5, "final": 10.5, "smoke": 0.75}
SMOKE = dict(units=(f"{UNIT}0.p21of24", f"{UNIT}1.p20of24"), adam=150, lbfgs=150)


def journal_name(part: int) -> str:
    return f"optuna_journal_s{part}.log"


def theta_pre(ctx, out: Path, tel, deadline: float) -> tuple[dict, Path]:
    """theta_pre of seed 0 (Eq. 18 alone, from a random initialisation): the handed-over copy when present,
    pretrained here otherwise."""
    p = e6._handoff_file(PRE) or e6._handoff_file("pretrained.pt")
    if p is not None:
        return torch.load(p, map_location="cpu", weights_only=False), p
    pre = e6.pretrain(ctx, 0, ctx.K, tel, deadline)
    p = out / PRE
    torch.save(pre, p)
    return pre, p


def centre_all(ctx, pre: dict, seed: int = 0) -> dict:
    """Eq. 12 at theta_pre over I'' (the r09 Eq. 24 rule extended to every term): lambda_hat per grid stage,
    rescaled so lambda_data = 1, geometric mean over GRID."""
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


def _study(ctx, part: int, out: Path, deadline: float, target: int, journal: Path | None = None) -> dict:
    tel = e6.Telemetry(out, f"scratch-study-p{part}")
    pre, pre_path = theta_pre(ctx, out, tel, deadline)
    journal = journal or out / journal_name(part)
    c = centre_all(ctx, pre, 0)
    tel.line(f"Eq. 12 centre over I'' at theta_pre (lambda_data = 1): "
             f"{json.dumps({k: float(f'{v:.3e}') for k, v in c.items()})}")
    e6.open_study(journal, 0, create_centre=c, study_name=STUDY)
    if ctx.dev.type == "cuda":
        torch.cuda.empty_cache()
    n = min(max(e6._n_gpu(), 1), 2)
    e6._run_payloads([{"role": "tune", "worker": 10 * (part + 1) + w, "start": str(pre_path), "journal": str(journal),
                       "out": str(out), "deadline": deadline, "study": STUDY, "keys": list(KEYS), "target": target}
                      for w in range(n)], ctx, deadline)
    s = e6.study_summary(journal, STUDY)
    s.update(part=part, centre=c, created=steps._now())
    return s


def run_study(part: int) -> dict:
    """One part of the scratch study (one account, both GPUs), stopping at TRIALS_PER_PART trials."""
    start = time.time()
    out = e6.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    ctx = e6.build_ctx()
    s = _study(ctx, part, out, start + HOURS["study"] * 3600.0, TRIALS_PER_PART)
    s["hours"] = (time.time() - start) / 3600.0
    steps._write(out / f"study-s{part}.json", s)
    print(f"[scratch study p{part}] trials {s['n_trials']} · complete {s['complete']} · pruned {s['pruned']} · "
          f"best {s.get('best_value')} · {s['hours']:.2f} h", flush=True)
    return s


def join_studies(summaries: list[dict]) -> dict:
    """Join the parts (same objective, same centre and box): the best complete trial over all of them."""
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
    """The scratch final runs of `account` (e6.final_plan with UNIT): every seed pretrained here, fixed ADAM Adam
    steps then up to LBFGS L-BFGS evaluations per stage, lambda from best_lambda_s.json."""
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
    """Short check before the long kernels: a 2-trial study (one per GPU worker) and one heavy near-Voc stage per
    GPU, each from scratch (pretraining included) with a cut Adam/L-BFGS budget. Projects the wall times."""
    start = time.time()
    deadline = start + HOURS["smoke"] * 3600.0
    out = e6.results_dir() / "smoke-s"
    out.mkdir(parents=True, exist_ok=True)
    ctx = e6.build_ctx()
    t0 = time.time()
    study = _study(ctx, 9, out, deadline, 2, journal=out / "optuna_journal_smoke.log")
    t_study = time.time() - t0
    trials = [r for w in (100, 101) for r in _history(out, f"tune-w{w}") if r.get("kind") == "trial"]
    lam = e6.full_lambda({k: v for k, v in study["centre"].items()})      # the centre itself: values not judged
    t1 = time.time()
    e6._run_payloads([{"role": "final", "units": [u], "lambda": lam, "out": str(out), "deadline": deadline,
                       "scratch": True, "adam_max": SMOKE["adam"], "lbfgs_max": SMOKE["lbfgs"], "tau": TAU}
                      for u in SMOKE["units"]], ctx, deadline)
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
        units[u] = {"stage": row["stage"], "seconds_stage": row["seconds"], "evals": evals,
                    "s_per_eval": row["seconds"] / max(evals, 1), "seconds_unit": r["seconds"],
                    "diverged": r["diverged"], "peak_mem_gb": r.get("peak_mem_gb")}
    ok = [v for v in units.values() if v]
    s_eval = max(v["s_per_eval"] for v in ok) if ok else float("nan")
    pre_s = max(v["seconds_unit"] - v["seconds_stage"] for v in ok) if ok else float("nan")
    trial_s = max([t["seconds"] for t in trials], default=float("nan"))
    stages_per_gpu = max(sum(len(e6.unit_spec(u, ctx.K)[1]) for u in pack)
                         for packs in e6.final_plan(unit=UNIT) for pack in packs)
    eta_final_h = (stages_per_gpu * (ADAM + LBFGS) * s_eval + 2 * pre_s) / 3600.0
    eta_study_h = math.ceil(TRIALS_PER_PART / 2) * trial_s / 3600.0
    rec = {"unit": "e6s-smoke", "complete": True, "created": steps._now(), "n_gpu": e6._n_gpu(),
           "study": {"trials": len(trials), "states": [t.get("state") for t in trials], "trial_seconds": trial_s,
                     "seconds": t_study, "centre": study["centre"]},
           "final": units, "final_seconds": t_final, "pretrain_seconds": pre_s, "s_per_eval": s_eval,
           "stages_per_gpu": stages_per_gpu, "eta_study_hours": eta_study_h, "eta_final_hours": eta_final_h,
           "seconds": time.time() - start}
    rec["pass"] = bool(len(trials) >= 2 and len(ok) == len(SMOKE["units"]) and not any(v["diverged"] for v in ok)
                       and eta_study_h <= HOURS["study"] and eta_final_h <= HOURS["final"] - 0.5)
    steps._write(e6.results_dir() / "e6s-smoke.json", rec)
    print(f"[scratch smoke] {json.dumps({k: rec[k] for k in ('n_gpu', 's_per_eval', 'pretrain_seconds', 'eta_study_hours', 'eta_final_hours', 'pass')})}",
          flush=True)
    return rec
