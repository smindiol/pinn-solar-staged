"""Diagnostic of network M on the dpvB2-only base (operator 2026-11-15): do the continuity terms stop being ignored?

The F-Adam run of protocol v12 left the continuity residuals n, p at 0.6-0.7 while Optuna's lambda_n, lambda_p were
tiny. Two short arms, same structure as the final (theta_pre per stage, same batch, current term, clip, cosine lr),
STEPS Adam steps on every stage of GRID, starting from the Eq. 12 centre at theta_pre (no Optuna):

  "adaptive"  lambda recomputed by Eq. 12 every UPDATE_EVERY steps (rescaled so lambda_data = 1) and smoothed by
              the Eq. 13 EMA with ALPHA.
  "pcgrad"    lambda fixed at the centre; the weighted per-term gradients lambda_i grad L_i are combined by PCGrad
              (Yu et al. 2020): each one loses its component along every other term's gradient it conflicts with,
              in a random order, and the projected gradients are summed.

Per stage the record holds every term's loss and gradient share at the start and the end, the validation current
error (r06 Eq. 13) and field error (Eq. 18), the lambda trajectory (adaptive) or the conflict counts (pcgrad).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import diagnostics, e6, scratch_m, steps, weights

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r12.md",
    "sections": ["5", "6"],
    "equations": ["12", "13", "18", "23", "26"],
    "invariants": [
        "pcgrad_removes_every_pairwise_conflict_of_two_terms",
        "pcgrad_leaves_agreeing_gradients_unchanged",
        "adaptive_lambda_keeps_the_data_weight_at_one",
    ],
}

VARIANTS = ("adaptive", "pcgrad")
STEPS = 500                 # Adam steps per stage (operator 2026-11-15)
UPDATE_EVERY = 100          # adaptive: Eq. 12 recomputed every UPDATE_EVERY steps
ALPHA = 0.1                 # adaptive: Eq. 13 EMA weight
EVERY = 25                  # progress line / history record every EVERY steps
SEED = 0
GRID = scratch_m.GRID
HOURS = 9.0
PRETRAINED = "pretrained_m_dpvb2.pt"


def results_dir() -> Path:
    return steps.results_root() / "e7md"


def _flat(g, params) -> torch.Tensor:
    return torch.cat([torch.zeros_like(p).reshape(-1) if x is None else x.reshape(-1) for p, x in zip(params, g)])


def _unflat(vec: torch.Tensor, params) -> list:
    out, o = [], 0
    for p in params:
        out.append(vec[o:o + p.numel()].view_as(p))
        o += p.numel()
    return out


def term_grads(obj, params) -> tuple[dict, dict]:
    """Per-term losses and flat gradients grad L_i on the stage batch (one forward, one backward per term)."""
    L, _ = obj._chunk_losses(obj.batch)
    names = list(L)
    grads = {}
    for i, k in enumerate(names):
        g = torch.autograd.grad(L[k], params, retain_graph=i < len(names) - 1, allow_unused=True)
        grads[k] = _flat(g, params).detach()
    return {k: float(v.detach()) for k, v in L.items()}, grads


def pcgrad(weighted: dict, rng: np.random.Generator) -> tuple[torch.Tensor, int]:
    """PCGrad (Yu et al. 2020) over flat weighted gradients: for each term, project out its component along every
    other term's gradient it has a negative inner product with (others visited in a random order), then sum.
    Returns the combined gradient and the number of conflicting pairs met."""
    names = list(weighted)
    out = torch.zeros_like(weighted[names[0]])
    conflicts = 0
    for k in names:
        g = weighted[k].clone()
        for j in rng.permutation([n for n in names if n != k]):
            h = weighted[j]
            dot = torch.dot(g, h)
            hh = torch.dot(h, h)
            if dot < 0 and hh > 0:
                g = g - dot / hh * h
                conflicts += 1
        out = out + g
    return out, conflicts


def adaptive_update(lam: dict, grads: dict, cfg, alpha: float = ALPHA) -> dict:
    """Eq. 12 on the current per-term gradients, rescaled so lambda_data = 1, then the Eq. 13 EMA."""
    lh = weights.lambda_hat({k: v.cpu().numpy() for k, v in grads.items()}, "max", cfg.eps_g, cfg.lambda_min,
                            cfg.lambda_max)
    lh = {k: v / lh["data"] for k, v in lh.items()}
    return weights.ema_update(lam, lh, alpha)


def _snapshot(ctx, model, obj, params, lam, j) -> dict:
    terms, g = term_grads(obj, params)
    shares = diagnostics.gradient_shares({k: v.cpu().numpy() for k, v in g.items()}, lam)
    return {"terms": terms, "gradient_shares": shares, "total": sum(lam[k] * terms[k] for k in terms),
            "current_error_val": e6.current_error(ctx, model, "val", j),
            "field_error_val": e6.field_error(ctx, model, "val", j)}


def run_stage(ctx, variant: str, pre_j: dict, lam0: dict, j: int, tel, n_steps: int = STEPS,
              stage_no: int = 1, n_stages: int = 1) -> dict:
    """STEPS Adam steps on stage j from theta_pre of that stage, with `variant`'s gradient."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {VARIANTS}")
    model = e6.new_model(ctx, SEED)
    params = list(model.parameters())
    model.load_state_dict(pre_j)
    obj = e6.PhasedObjective(ctx, model, e6.physics_batch(ctx, j, SEED), j, jbatch=e6.current_batch(ctx, j, SEED))
    lam = dict(lam0)
    t0 = time.time()
    start = _snapshot(ctx, model, obj, params, lam, j)
    opt, sched = e6._adam(params, n_steps)
    rng = np.random.default_rng([SEED, j, 31])
    lam_traj, conflicts, diverged = [{"step": 0, **lam}], [], False
    for s in range(n_steps):
        if variant == "adaptive":
            if s > 0 and s % UPDATE_EVERY == 0:
                _, g = term_grads(obj, params)
                lam = adaptive_update(lam, g, ctx.cfg)
                lam_traj.append({"step": s, **lam})
            total, terms, grads = obj.terms_and_grad(lam, params)
            n_conf = None
        else:
            terms, g = term_grads(obj, params)
            total = torch.tensor(sum(lam[k] * terms[k] for k in terms))
            vec, n_conf = pcgrad({k: lam[k] * g[k] for k in g}, rng)
            grads = _unflat(vec, params)
            conflicts.append(n_conf)
        if not torch.isfinite(total):
            diverged = True
            tel.line(f"stage {j}: non-finite loss at step {s}, stopped")
            break
        e6._apply(params, grads)
        torch.nn.utils.clip_grad_norm_(params, e6.CLIP)
        opt.step()
        sched.step()
        if s % EVERY == 0 or s == n_steps - 1:
            rate = (time.time() - t0) / (s + 1)
            tel.log(kind="step", variant=variant, stage=j, step=s, total=float(total), terms=terms,
                    conflicts=n_conf, **{"lambda": lam})
            tel.line(f"{variant} · stage {j} ({stage_no}/{n_stages}) · Adam step {s + 1}/{n_steps} · "
                     f"total {float(total):.3e} | " + " ".join(f"{k} {v:.3e}" for k, v in terms.items())
                     + (f" | conflicts {n_conf}" if n_conf is not None else
                        " | lambda " + " ".join(f"{k} {v:.3g}" for k, v in lam.items()))
                     + f" | {rate:.2f} s/step · stage ETA {(n_steps - s - 1) * rate / 60:.1f} min")
    end = _snapshot(ctx, model, obj, params, lam, j)
    return {"variant": variant, "stage": j, "steps": n_steps, "seconds": time.time() - t0, "diverged": diverged,
            "start": start, "end": end, "lambda_start": lam0, "lambda_end": lam, "lambda_trajectory": lam_traj,
            "mean_conflicts": float(np.mean(conflicts)) if conflicts else None,
            "model": e6._cpu_state(model)}


def _stage_file(out: Path, variant: str, j: int) -> Path:
    return out / f"e7md-{variant}-j{j:02d}.json"


def diag_worker(ctx, a: dict) -> None:
    """Role "call": the stages of one GPU, each written as soon as it ends (a rerun skips the written ones)."""
    out = Path(a["out"])
    pre = torch.load(a["start_file"], map_location="cpu", weights_only=False)
    tel = e6.Telemetry(out, f"e7md-{a['variant']}-w{a['worker']}")
    for i, j in enumerate(a["stages"]):
        f = _stage_file(out, a["variant"], j)
        if f.exists():
            continue
        if time.time() > a["deadline"] - 600:
            tel.line(f"wall budget: stage {j} not started")
            return
        rec = run_stage(ctx, a["variant"], pre[j], a["lambda"], j, tel, a.get("steps", STEPS), i + 1, len(a["stages"]))
        torch.save(rec.pop("model"), out / f"e7md-{a['variant']}-j{j:02d}.pt")
        steps._write(f, rec)
        tel.line(f"stage {j} done ({i + 1}/{len(a['stages'])}): current error val "
                 f"{rec['start']['current_error_val']:.3e} -> {rec['end']['current_error_val']:.3e} | "
                 + " ".join(f"{k} {rec['start']['terms'][k]:.2e}->{rec['end']['terms'][k]:.2e}"
                            for k in rec['end']['terms']))


def summarize(out: Path, variant: str) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(out.glob(f"e7md-{variant}-j*.json"))]
    ratio = lambda r, k: r["end"]["terms"][k] / r["start"]["terms"][k] if r["start"]["terms"][k] > 0 else math.nan
    keys = list(rows[0]["start"]["terms"]) if rows else []
    s = {"variant": variant, "stages": [r["stage"] for r in rows],
         "current_error_val_end": {r["stage"]: r["end"]["current_error_val"] for r in rows},
         "field_error_val_end": {r["stage"]: r["end"]["field_error_val"] for r in rows},
         "median_current_error_val_end": float(np.median([r["end"]["current_error_val"] for r in rows])) if rows else None,
         "median_term_ratio_end_over_start": {k: float(np.nanmedian([ratio(r, k) for r in rows])) for k in keys},
         "diverged": [r["stage"] for r in rows if r["diverged"]]}
    steps._write(out / f"e7md-{variant}.summary.json", s)
    return s


def run_diag(variant: str = "adaptive", n_steps: int = STEPS, parts: int | None = None) -> dict:
    """One arm on one kernel: theta_pre on the dpvB2-only base, the Eq. 12 centre, then STEPS Adam steps on every
    GRID stage, stages split over the kernel's GPUs."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {VARIANTS}")
    start = time.time()
    deadline = start + HOURS * 3600.0
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    ctx = scratch_m.build_ctx()                                  # data_m's default: dpvB2 devices only
    tel = e6.Telemetry(out, f"e7md-{variant}")
    pre_path = out / PRETRAINED
    if pre_path.exists():
        pre = torch.load(pre_path, map_location="cpu", weights_only=False)
    else:
        pre = e6.pretrain(ctx, SEED, ctx.K, tel, deadline)       # never the mixed-base pretrained_m.pt
        torch.save(pre, pre_path)
    grid = scratch_m._grid(ctx)
    lam0 = e6.full_lambda({k: v for k, v in scratch_m.centre_all(ctx, pre, SEED, grid).items() if k != "data"})
    steps._write(out / "e7md-centre.json", {"lambda": lam0, "grid": list(grid)})
    tel.line(f"centre lambda (Eq. 12, data = 1): " + ", ".join(f"{k} {v:.3g}" for k, v in lam0.items()))
    n = parts or max(e6._n_gpu(), 1)
    pays = [{"role": "call", "fn": "Staged_PINN_Solar.diag_m:diag_worker", "variant": variant, "worker": w,
             "stages": [j for i, j in enumerate(grid) if i % n == w], "lambda": lam0, "steps": n_steps,
             "start_file": str(pre_path), "out": str(out), "deadline": deadline} for w in range(n)]
    e6._run_payloads(pays, ctx, deadline)
    s = summarize(out, variant)
    tel.line("SUMMARY " + json.dumps(s, default=float))
    return s
