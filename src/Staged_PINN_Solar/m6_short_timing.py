"""Three Adam updates + at most three strong-Wolfe LBFGS iterations per batch-size case.

No pretraining or accuracy claim. Full voltage curves, 150 stratified nodes,
production 256x5 float64 network. Timing is distinct from convergence.
"""
from __future__ import annotations

import json
import time
import numpy as np
import torch

from . import m6c, m6_data, m6_sampling as sampling, steps
from .m6_capacity import fingerprint

CASES = ((64, 16), (64, 32), (64, 64), (128, 16), (128, 32), (128, 64))


def synchronize(ctx):
    if ctx.dev.type == "cuda":
        torch.cuda.synchronize(ctx.dev)


def run_case(ctx, model, arm, lam, rows, adam_devices, lbfgs_devices, log):
    abc = m6c.ARMS[arm]
    result = {"adam_devices": adam_devices, "lbfgs_devices": lbfgs_devices,
              "initial_weights": fingerprint(model), "chunk_rows": rows,
              "adam": [], "lbfgs": [], "line_search": "strong_wolfe"}
    opt = torch.optim.Adam(model.parameters(), lr=m6c.ADAM_LR)
    for i in range(3):
        b = sampling.SpatialPlan(ctx, "train", np.random.default_rng(3000 + i), adam_devices)
        opt.zero_grad(set_to_none=True)
        synchronize(ctx)
        start = time.monotonic()
        terms, loss = sampling.objective(ctx, model, b, lam, abc, rows, True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), m6c.CLIP)
        opt.step()
        synchronize(ctx)
        r = {"step": i + 1, "seconds": time.monotonic() - start,
             "devices": len(b.ids), "pairs": len(b.pd), "rows": b.rows,
             "loss": float(loss), "terms": {k: float(v) for k, v in terms.items()}}
        result["adam"].append(r)
        log.line(f"SHORT {arm} {adam_devices}/{lbfgs_devices} ADAM " + json.dumps(r))
    b = sampling.SpatialPlan(ctx, "train", np.random.default_rng(4000), lbfgs_devices)
    # Same optimizer as the training run; cap iterations, count every closure.
    # Wolfe may evaluate several times within an iteration.
    opt = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=3, max_eval=20,
                            history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        synchronize(ctx)
        start = time.monotonic()
        terms, loss = sampling.objective(ctx, model, b, lam, abc, rows, True)
        synchronize(ctx)
        r = {"evaluation": len(result["lbfgs"]) + 1, "seconds": time.monotonic() - start,
             "devices": len(b.ids), "pairs": len(b.pd), "rows": b.rows,
             "loss": float(loss), "terms": {k: float(v) for k, v in terms.items()}}
        result["lbfgs"].append(r)
        log.line(f"SHORT {arm} {adam_devices}/{lbfgs_devices} LBFGS " + json.dumps(r))
        return loss

    synchronize(ctx)
    start = time.monotonic()
    opt.step(closure)
    synchronize(ctx)
    result["lbfgs_total_seconds"] = time.monotonic() - start
    state = next(iter(opt.state.values()))
    result["lbfgs_iterations"] = state.get("n_iter", 0)
    result["lbfgs_history"] = {
        "configured_capacity": opt.param_groups[0]["history_size"],
        "stored_pairs": len(state.get("old_dirs", [])),
        "parameter_difference_vectors": len(state.get("old_stps", [])),
        "function_evaluations": state.get("func_evals", 0),
        "recorded_closure_calls": len(result["lbfgs"]),
    }
    log.line(f"SHORT {arm} {adam_devices}/{lbfgs_devices} HISTORY " + json.dumps(result["lbfgs_history"]))
    if result["lbfgs_iterations"] > 3:
        raise AssertionError("Short smoke exceeded its three-iteration budget")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite parameters after LBFGS")
    a = float(np.median([r["seconds"] for r in result["adam"]]))
    l = float(np.median([r["seconds"] for r in result["lbfgs"]]))
    result["median_adam_seconds"] = a
    result["median_lbfgs_evaluation_seconds"] = l
    result["projected_2000_adam_500_lbfgs_hours_at_1_2_3_evals"] = [
        (2000 * a + 500 * e * l) / 3600 for e in (1, 2, 3)]
    result["projection_excludes"] = ["pretraining", "validation", "sampling", "line_search_overhead"]
    result["status"] = "ok"
    return result


def worker(gpu, root):
    torch.cuda.set_device(gpu)
    arm = ("Control", "PKC")[gpu]
    ctx = m6c.Ctx.load(root, torch.device(f"cuda:{gpu}"))
    model = m6c.new_model(ctx, 0)
    initial = m6c._state(model)
    # Resident chunks already succeeded in the preceding capacity rehearsal.
    rows = 262144 if arm == "Control" else 32768
    out = steps.results_root() / "e9c_short_timing" / arm
    log = m6c.Log(out, arm)
    p = sampling.SpatialPlan(ctx, "train", np.random.default_rng(2000), 16)
    unit = {"data": 1., **{k: 1. for k in m6c.PHYS_TERMS}}
    values, _ = sampling.objective(ctx, model, p, unit, (1, 1, 1), 1024)
    lam = m6c.lambdas(values)
    summary = {"arm": arm, "gpu": gpu, "initial_weights": fingerprint(model),
               "lambda": lam, "cases": [], "status": "running"}
    path = out / "timing.json"
    steps._write(path, summary)
    for adam_devices, lbfgs_devices in CASES:
        model.load_state_dict(initial)
        try:
            case = run_case(ctx, model, arm, lam, rows, adam_devices, lbfgs_devices, log)
        except Exception as exc:
            case = {"adam_devices": adam_devices, "lbfgs_devices": lbfgs_devices,
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        summary["cases"].append(case)
        steps._write(path, summary)
    summary["status"] = "ok" if all(c["status"] == "ok" for c in summary["cases"]) else "failed"
    steps._write(path, summary)


def run_smoke(root=None):
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two T4 GPUs required")
    root = str(m6_data.ensure(root))
    torch.multiprocessing.spawn(worker, args=(root,), nprocs=2, join=True)
    out = steps.results_root() / "e9c_short_timing"
    arms = {a: json.loads((out / a / "timing.json").read_text()) for a in ("Control", "PKC")}
    if arms["Control"]["initial_weights"] != arms["PKC"]["initial_weights"]:
        raise AssertionError("Initial weights differ between arms")
    summary = {"status": "ok" if all(a["status"] == "ok" for a in arms.values()) else "failed",
               "arms": arms}
    steps._write(out / "summary.json", summary)
    return summary
