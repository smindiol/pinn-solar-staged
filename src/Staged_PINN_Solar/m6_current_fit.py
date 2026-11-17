"""Terminal-current supervision and separately balanced data groups, all PKC.

64 devices/update, all voltages, 75 spatial samples; fixed 64-device LBFGS.
No local training required. Optional terminal-label noise is off by default.
"""
from __future__ import annotations

import gc
import json
import time
import numpy as np
import torch

from . import m6c as base, m6_resolution as spatial, m6_data, steps
from .m6_capacity import fingerprint

GROUPS = {"potential": ("phi", "phi_n", "phi_p"), "current": ("J_n", "J_p"),
          "terminal": ("JV",)}


def terminal(model, ctx, plan, noise_sigma=0., noise_seed=0):
    """One supervised prediction per device/voltage, equal device weights.

    ctx.J_np is in A/cm2; ctx.devc.Jgen is dimensionless. The prediction
    raw[:,3] at x=0 is already normalized by device generation current.
    """
    d = torch.as_tensor(plan.pd, device=ctx.dev)
    v = torch.as_tensor(plan.pv, device=ctx.dev)
    s = torch.zeros((len(d), 1), dtype=torch.float64, device=ctx.dev)
    raw = model.raw(s, ctx.xi[d], (ctx.V[d, v] / ctx.V_max)[:, None])
    reference = torch.as_tensor(ctx.J_np[plan.pd, plan.pv], device=ctx.dev)
    reference = reference / (ctx.attrs["J0"] * ctx.devc["Jgen"][d])
    if noise_sigma:
        # Fixed deterministic labels for each plan; no randomness inside closure.
        rng = np.random.default_rng(noise_seed)
        reference = reference + torch.as_tensor(rng.normal(0., noise_sigma, len(d)), device=ctx.dev)
    weights = torch.as_tensor(1. / (len(plan.ids) * ctx.n_V[plan.pd]), device=ctx.dev)
    return (weights * (raw[:, 3] - reference).square()).sum()


def component_terms(model, b, physics):
    return spatial.terms(model, b, physics)


def objective(ctx, model, plan, lam, physics=True, backward=False,
              noise_sigma=0., noise_seed=0, rows=32768):
    values = {}
    total = torch.zeros((), dtype=torch.float64, device=ctx.dev)
    for start in range(0, plan.rows, rows):
        b = plan.batch(ctx, start, start + rows)
        current = component_terms(model, b, physics)
        weight = b["n_rows"] / plan.rows
        loss = weight * sum(lam[k] * v for k, v in current.items())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite spatial loss")
        if backward:
            loss.backward()
        total += loss.detach()
        for key, val in current.items():
            values[key] = values.get(key, 0.) + weight * val.detach()
    jv = terminal(model, ctx, plan, noise_sigma, noise_seed)
    loss = lam.get("JV", 0.) * jv
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite terminal loss")
    if backward and lam.get("JV", 0.):
        loss.backward()
    total += loss.detach()
    values["JV"] = jv.detach()
    return values, total


def gradient_weights(ctx, model, plan):
    """Balance DATA group gradient norms, once on clean training data.

    Potential group is the reference (weight 1); constrain ratios to [.01,100].
    Physical-term weights remain Eq39. Persist all norms and weights.
    """
    keys = base.DATA_TERMS + base.PHYS_TERMS + ("JV",)
    norms = {}
    for name, group in GROUPS.items():
        lam = {k: float(k in group) for k in keys}
        model.zero_grad(set_to_none=True)
        objective(ctx, model, plan, lam, physics=False, backward=True)
        norms[name] = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()
                                         if p.grad is not None)))
    model.zero_grad(set_to_none=True)
    if not all(np.isfinite(v) and v > 0 for v in norms.values()):
        raise FloatingPointError(f"Invalid calibration gradient norms: {norms}")
    reference = max(norms["potential"], 1e-12)
    weights = {k: float(np.clip(reference / max(v, 1e-12), .01, 100.)) for k, v in norms.items()}
    return weights, norms


def make_weights(ctx, model, plan, variant):
    unit = {k: 1. for k in base.DATA_TERMS + base.PHYS_TERMS + ("JV",)}
    values, _ = objective(ctx, model, plan, unit)
    old = base.lambdas(values)
    lam = {k: old["data"] for k in base.DATA_TERMS}
    lam.update({k: old[k] for k in base.PHYS_TERMS})
    lam["JV"] = 0.
    record = {"variant": variant, "calibration_values": {k: float(v) for k, v in values.items()}}
    if variant == "terminal":
        # One common baseline scale; new term starts at order one.
        lam["JV"] = 10. ** -np.floor(np.log10(max(float(values["JV"]), 1e-12)))
    elif variant == "balanced":
        weights, norms = gradient_weights(ctx, model, plan)
        for group, keys in GROUPS.items():
            for key in keys:
                lam[key] = old["data"] * weights[group]
        record.update(data_gradient_norms=norms, data_group_multipliers=weights)
    elif variant != "base":
        raise ValueError("variant must be base, terminal or balanced")
    record["lambda"] = lam
    return lam, record


def evaluate(ctx, model, lam):
    plan = spatial.Plan(ctx, "val", 0, 0, len(ctx.ids["val"]), 250)
    values, loss = objective(ctx, model, plan, lam)
    return {"jv": base.jv_metrics(ctx, model), "clean_dense_terms": {k: float(v) for k, v in values.items()},
            "weighted_loss": float(loss), "nodes": 250, "devices": len(plan.ids)}


def run_case(root, variant="balanced", noise_sigma=0., seed=0, gpu=0,
             pre_steps=300, adam_steps=100, lbfgs_iters=0, limit=None):
    """New pilot defaults to 100 Adam steps; production can request 2000/500.

    All variants start from identical RANDOM weights, but have their own
    pretraining: compares the full supervision strategy, not common fine-tuning.
    """
    if min(pre_steps, adam_steps) < 1 or lbfgs_iters < 0 or noise_sigma < 0:
        raise ValueError("Invalid training lengths or noise")
    if noise_sigma and variant != "balanced":
        raise ValueError("Noise ablation is defined relative to balanced variant")
    ctx = base.Ctx.load(root, torch.device(f"cuda:{gpu}"), limit)
    model = base.new_model(ctx, seed)
    suffix = f"{variant}_noise{noise_sigma:g}_seed{seed}"
    out = steps.results_root() / "e9_current" / suffix
    log = base.Log(out, suffix)
    calibration = spatial.Plan(ctx, "train", seed, 20000, 64, 75)
    lam, calibration_pre = make_weights(ctx, model, calibration, variant)
    result = {"variant": variant, "noise_sigma_in_Jgen_units": noise_sigma,
              "status": "running", "initial_weights": fingerprint(model),
              "pre_calibration": calibration_pre, "devices": 64, "nodes": 75,
              "pre_steps_requested": pre_steps, "adam_steps_requested": adam_steps,
              "lbfgs_iters_requested": lbfgs_iters, "checkpoints": []}
    steps._write(out / "summary.json", result)
    for phase, n, lr, use_physics in (("pre", pre_steps, base.PRE_LR, False),
                                       ("adam", adam_steps, base.ADAM_LR, True)):
        if phase == "adam":
            lam, record = make_weights(ctx, model, calibration, variant)
            result["post_pre_calibration"] = record
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = base._cosine(opt, n, lr, base.LR_MIN)
        started = time.monotonic()
        for i in range(n):
            step_seed = 10000+i if phase == "pre" else 30000+i
            plan = spatial.Plan(ctx, "train", seed, step_seed, 64, 75)
            opt.zero_grad(set_to_none=True)
            values, loss = objective(ctx, model, plan, lam, use_physics, True,
                                     noise_sigma, seed*1000000+step_seed)
            torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
            opt.step()
            sch.step()
            if i == 0 or (i+1) % 50 == 0 or i+1 == n:
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError(f"Nonfinite parameters in {phase}")
                torch.cuda.synchronize(ctx.dev)
                record = {"phase": phase, "step": i+1, "loss": float(loss),
                          "terms": {k: float(v) for k, v in values.items()},
                          "seconds": time.monotonic()-started, "validation": base.jv_metrics(ctx, model)}
                result["checkpoints"].append(record)
                log.rec(**record)
                log.line(f"{phase} {variant} step={i+1}/{n} total={float(loss):.4e} {base._fmt(values)}")
                steps._write(out / "summary.json", result)
                torch.save({"state": base._state(model), "lambda": lam,
                            "phase": phase, "step": i+1}, out / "latest.pt")
        result[f"{phase}_validation"] = evaluate(ctx, model, lam)
        torch.save({"state": base._state(model), "lambda": lam,
                    "phase": phase, "step": n}, out / f"{phase}.pt")
        steps._write(out / "summary.json", result)
    if lbfgs_iters:
        # Fixed clean labels: identical deterministic objective at every closure.
        plan = spatial.Plan(ctx, "train", seed, 40000, 64, 75)
        opt = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=lbfgs_iters,
                                history_size=50, line_search_fn="strong_wolfe")
        count = 0

        def closure():
            nonlocal count
            opt.zero_grad(set_to_none=True)
            values, loss = objective(ctx, model, plan, lam, True, True)
            count += 1
            if count == 1 or count % 50 == 0:
                log.line(f"LBFGS {variant} evaluation={count} total={float(loss):.4e} {base._fmt(values)}")
            return loss

        opt.step(closure)
        result["lbfgs_evaluations"] = count
        result["lbfgs_iterations"] = next(iter(opt.state.values())).get("n_iter", 0)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite final parameters")
    result.update(status="ok", final=evaluate(ctx, model, lam))
    torch.save({"state": base._state(model), "lambda": lam}, out / "final.pt")
    steps._write(out / "summary.json", result)
    return result


def _worker(gpu, root, include_noise, kwargs):
    torch.cuda.set_device(gpu)
    if gpu == 0:
        run_case(root, "base", gpu=gpu, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        run_case(root, "terminal", gpu=gpu, **kwargs)
    else:
        run_case(root, "balanced", gpu=gpu, **kwargs)
        if include_noise:
            gc.collect()
            torch.cuda.empty_cache()
            run_case(root, "balanced", noise_sigma=.005, gpu=gpu, **kwargs)


def run(root=None, include_noise=False, **kwargs):
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two GPUs required")
    root = str(m6_data.ensure(root))
    torch.multiprocessing.spawn(_worker, args=(root, include_noise, kwargs), nprocs=2, join=True)
    return {"out": str(steps.results_root() / "e9_current"), "include_noise": include_noise}


def _comparison_worker(gpu, root, kwargs):
    torch.cuda.set_device(gpu)
    variant = ("base", "balanced")[gpu]
    try:
        run_case(root, variant, gpu=gpu, **kwargs)
    except Exception as exc:
        out = steps.results_root() / "e9_current" / f"{variant}_noise0_seed{kwargs.get('seed', 0)}"
        steps._write(out / "failure.json", {"status": "failed", "error": repr(exc)})
        raise


def run_comparison(root=None, pre_steps=300, adam_steps=100, lbfgs_iters=0, limit=None):
    """Base-PKC versus corrected-PKC, one seed, one arm per T4; no noise."""
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two GPUs required")
    root = str(m6_data.ensure(root))
    kwargs = dict(pre_steps=pre_steps, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters,
                  limit=limit, seed=0)
    torch.multiprocessing.spawn(_comparison_worker, args=(root, kwargs), nprocs=2, join=True)
    out = steps.results_root() / "e9_current"
    arms = {v: json.loads((out / f"{v}_noise0_seed0" / "summary.json").read_text())
            for v in ("base", "balanced")}
    if len({r["initial_weights"] for r in arms.values()}) != 1:
        raise AssertionError("Arms did not start from identical random weights")
    if not all(r["status"] == "ok" for r in arms.values()):
        raise RuntimeError("A comparison arm did not complete")
    result = {"status": "ok", "seed": 0, "noise": False,
              "configuration": kwargs, "arms": arms}
    steps._write(out / "comparison.json", result)
    return {"status": "ok", "out": str(out), "configuration": kwargs}


def run_smoke(root=None):
    return run_comparison(root=root, pre_steps=2, adam_steps=3, lbfgs_iters=2, limit=4)
