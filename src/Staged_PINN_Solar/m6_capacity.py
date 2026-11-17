"""T4 capacity/timing rehearsal, not an accuracy experiment."""
from __future__ import annotations

import hashlib
import json
import time
import numpy as np
import torch

from . import m6c, m6d, m6_data, m6_sampling as sampling, steps


def fingerprint(model):
    h = hashlib.sha256()
    for p in model.state_dict().values():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def worker(gpu, root, limit=None):
    arm = ("Control", "PKC")[gpu]
    abc = m6c.ARMS[arm]
    out = steps.results_root() / "e9c_capacity" / arm
    log = m6c.Log(out, arm)
    ctx = m6c.Ctx.load(root, torch.device(f"cuda:{gpu}"), limit)
    model = m6c.new_model(ctx, 0)
    plan = sampling.SpatialPlan(ctx, "train", np.random.default_rng(1234), 254)
    result = {"purpose": "capacity_and_time_only", "arm": arm, "gpu": gpu,
              "initial_weights": fingerprint(model), "devices": len(ctx.ids["train"]),
              "nodes": 150, "adam_devices": len(plan.ids), "adam_pairs": len(plan.pd),
              "adam_rows": plan.rows, "stages": []}
    path = out / "capacity.json"
    steps._write(path, result)
    try:
        capacity = sampling.calibrate(ctx, model, plan, log, abc=abc, probe_to_oom=True)
        result["capacity"] = capacity
        rows = capacity["rows"]
        steps._write(path, result)
        # Identical single data warm-up in both processes, independent of capacity.
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        sampling.objective(ctx, model, plan, {"data": 1.}, (0, 0, 0), 1024, True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), m6c.CLIP)
        opt.step()
        result["after_common_warmup"] = fingerprint(model)
        unit = {"data": 1., **{k: 1. for k in m6c.PHYS_TERMS}}
        values, _ = sampling.objective(ctx, model, plan, unit, (1, 1, 1), 1024)
        lam = m6c.lambdas(values)
        result["lambda"] = lam
        opt = torch.optim.Adam(model.parameters(), lr=m6c.ADAM_LR)
        for update in range(2):
            p = sampling.SpatialPlan(ctx, "train", np.random.default_rng(3000 + update), 254)
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(ctx.dev)
            started = time.monotonic()
            values, loss = sampling.objective(ctx, model, p, lam, abc, rows, True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), m6c.CLIP)
            opt.step()
            torch.cuda.synchronize(ctx.dev)
            stage = {"phase": "adam", "update": update + 1, "seconds": time.monotonic() - started,
                     "loss": float(loss), "terms": {k: float(v) for k, v in values.items()}}
            result["stages"].append(stage)
            log.line(json.dumps(stage))
            steps._write(path, result)
        # Closure cost grows with subset size, but GPU graph size stays bounded.
        sizes = sorted(set(min(n, len(ctx.ids["train"])) for n in (254, 508, 1016, len(ctx.ids["train"]))))
        for n in sizes:
            p = sampling.SpatialPlan(ctx, "train", np.random.default_rng(4000), n)
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(ctx.dev)
            torch.cuda.synchronize(ctx.dev)
            started = time.monotonic()
            values, loss = sampling.objective(ctx, model, p, lam, abc, rows, True)
            torch.cuda.synchronize(ctx.dev)
            stage = {"phase": "lbfgs_closure_probe", "devices": n, "pairs": len(p.pd),
                     "rows": p.rows, "seconds": time.monotonic() - started,
                     "peak_reserved": torch.cuda.max_memory_reserved(ctx.dev), "loss": float(loss)}
            result["stages"].append(stage)
            log.line(json.dumps(stage))
            steps._write(path, result)
        # Two real iterations on the largest fixed set; Wolfe may call closure more often.
        opt = torch.optim.LBFGS(model.parameters(), max_iter=2, max_eval=5,
                                history_size=50, line_search_fn="strong_wolfe")
        evaluations = 0
        started = time.monotonic()

        def closure():
            nonlocal evaluations
            opt.zero_grad(set_to_none=True)
            values, loss = sampling.objective(ctx, model, p, lam, abc, rows, True)
            evaluations += 1
            if evaluations == 1 or evaluations % 50 == 0:
                log.line(f"LBFGS_SMOKE {arm} evaluation={evaluations} loss={float(loss):.5e} {m6c._fmt(values)}")
            return loss

        opt.step(closure)
        torch.cuda.synchronize(ctx.dev)
        iterations = next(iter(opt.state.values())).get("n_iter", 0)
        result["lbfgs"] = {"iterations": iterations, "evaluations": evaluations,
                            "devices": len(p.ids), "seconds": time.monotonic() - started}
        adam_seconds = np.median([x["seconds"] for x in result["stages"] if x["phase"] == "adam"])
        closure_seconds = result["stages"][-1]["seconds"]
        result["projection_excludes_pretraining"] = {
            "adam_2000_hours": float(2000 * adam_seconds / 3600),
            "lbfgs_500_hours_at_1_2_5_evals_per_iteration": [float(500 * e * closure_seconds / 3600) for e in (1, 2, 5)],
            "warning": "Cost scenarios, not a convergence or runtime guarantee"}
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        steps._write(path, result)
        raise
    steps._write(path, result)
    log.line("CAPACITY_DONE " + json.dumps(result))


def run_smoke(root=None, limit=None):
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Capacity rehearsal requires two GPUs")
    root = str(m6_data.ensure(root))
    torch.multiprocessing.spawn(worker, args=(root, limit), nprocs=2, join=True)
    out = steps.results_root() / "e9c_capacity"
    records = {a: json.loads((out / a / "capacity.json").read_text()) for a in ("Control", "PKC")}
    for key in ("initial_weights", "after_common_warmup"):
        if records["Control"][key] != records["PKC"][key]:
            raise RuntimeError(f"Arm weights differ at {key}")
    summary = {"status": "ok", "same_initial_weights": True, "same_warmup_weights": True, "arms": records}
    steps._write(out / "summary.json", summary)
    return summary
