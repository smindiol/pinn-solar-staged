"""PKC-only 2x2 pilot: device diversity versus spatial samples per update.

Every case starts from ONE common data-pretrained checkpoint. The expected
spatial objective is identical across resolutions, including odd edge quotas.
This tests refinement after pretraining, not from-scratch sample efficiency.
"""
from __future__ import annotations

import gc
import json
import time
import numpy as np
import torch

from . import m6c as base, m6_data, m6_sampling, steps
from .m6_capacity import fingerprint

CASES = ((64, 75), (64, 150), (128, 75), (128, 150))
ASSIGNMENTS = (((64, 75), (128, 150)), ((64, 150), (128, 75)))
QUOTAS = {75: (7, 45, 8, 15), 150: (15, 90, 15, 30), 250: (25, 150, 25, 50)}
ROWS = 32768


def output_dir():
    return steps.results_root() / "e9_resolution"


class Plan(m6_sampling.SpatialPlan):
    def __init__(self, ctx, split, seed, step, n_dev, nodes):
        self.ctx, self.k = ctx, nodes
        if nodes not in QUOTAS:
            raise ValueError("Resolution must be 75, 150 or 250")
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 0]))
        # Prefixes pair the 64-device and 128-device batches at each step.
        self.ids = np.sort(rng.permutation(ctx.ids[split])[:n_dev])
        if not hasattr(ctx, "resolution_regions"):
            s = ctx.nodes["s"].detach().cpu().numpy()
            nd = ctx.nodes["Ndop"].detach().cpu().numpy()
            ctx.resolution_regions = [m6_sampling.regions(x, d) for x, d in zip(s, nd)]
        self.nodes = np.zeros((len(ctx.uid), nodes), dtype=np.int64)
        self.weights = np.zeros((len(ctx.uid), nodes), dtype=np.float64)
        for d in self.ids:
            # Independent of batch size/resolution: common devices get nested nodes.
            rng = np.random.default_rng(np.random.SeedSequence([seed, step, int(d), 1]))
            indices, weights = [], []
            for group, q in zip(ctx.resolution_regions[d], QUOTAS[nodes]):
                indices.extend(rng.permutation(group)[:q])
                # Inverse inclusion weights preserve the 250-node mean exactly
                # in expectation: 10% left / 60% junction / 10% right / 20% bulk.
                weights.extend([nodes * len(group) / (250 * q)] * q)
            order = np.argsort(indices)
            self.nodes[d] = np.asarray(indices)[order]
            self.weights[d] = np.asarray(weights)[order]
        self.pd = np.repeat(self.ids, ctx.n_V[self.ids])
        self.pv = np.concatenate([np.arange(ctx.n_V[d]) for d in self.ids])
        self.rows = len(self.pd) * nodes

    def batch(self, ctx, start, stop):
        b = super().batch(ctx, start, stop)
        flat = np.arange(start, min(stop, self.rows))
        b["weight"] = torch.as_tensor(self.weights[self.pd[flat // self.k], flat % self.k],
                                      device=ctx.dev)[:, None]
        return b


def terms(model, b, physics):
    q, w = b["q"], b["weight"]
    r0, r1 = base._ends(model, b)
    phi, pn, pp, _, _ = model(q["s"], q, r0, r1)
    _, _, _, jn, jp = model(q["s_mid"], q, r0, r1)
    errors = torch.cat((phi, pn, pp), 1) - q["target"]
    currents = (torch.cat((jn, jp), 1) - q["jtarget"]) / q["Jgen"]
    values = {name: (w * errors[:, k:k+1] ** 2).mean()
              for k, name in enumerate(base.DATA_TERMS[:3])}
    values.update({name: (w * currents[:, k:k+1] ** 2).mean()
                   for k, name in enumerate(base.DATA_TERMS[3:])})
    if physics:
        values.update({name: (w * value ** 2).mean()
                       for name, value in base.residuals(model, b).items()})
    return values


def objective(ctx, model, plan, lam, physics=True, backward=False, rows=ROWS):
    values, total = {}, torch.zeros((), device=ctx.dev, dtype=torch.float64)
    for start in range(0, plan.rows, rows):
        b = plan.batch(ctx, start, start + rows)
        current = terms(model, b, physics)
        weight = b["n_rows"] / plan.rows
        loss = weight * base.total(current, lam, (1, 1, 1) if physics else (0, 0, 0))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite resolution-pilot objective")
        if backward:
            loss.backward()
        total += loss.detach()
        for key, val in current.items():
            values[key] = values.get(key, 0.) + weight * val.detach()
    return values, total


def evaluate(ctx, model, lam):
    # All validation devices, every stored voltage, all 250 spatial nodes.
    plan = Plan(ctx, "val", 0, 0, len(ctx.ids["val"]), 250)
    values, _ = objective(ctx, model, plan, lam)
    predicted = base.predicted_currents(ctx, model)
    per_device = []
    for d, current in predicted.items():
        reference = ctx.J_np[d, :ctx.n_V[d]]
        per_device.append({"uid": ctx.uid[d], "material": str(ctx.material[d]),
                           "curve_rmse_mA_cm2": float(1000 * np.sqrt(np.mean((current-reference)**2)))})
    return {"jv": base.jv_metrics(ctx, model), "dense_terms": {k: float(v) for k, v in values.items()},
            "per_device": per_device, "validation_devices": len(plan.ids), "nodes": 250}


def prepare(root, pre_steps, limit=None):
    ctx = base.Ctx.load(root, torch.device("cuda:0"), limit)
    model = base.new_model(ctx, 0)
    out = output_dir()
    log = base.Log(out, "pretrain")
    opt = torch.optim.Adam(model.parameters(), lr=base.PRE_LR)
    sch = base._cosine(opt, pre_steps, base.PRE_LR, base.LR_MIN)
    started = time.monotonic()
    for step in range(pre_steps):
        plan = Plan(ctx, "train", 0, 10000 + step, 254, 150)
        opt.zero_grad(set_to_none=True)
        values, loss = objective(ctx, model, plan, {"data": 1.}, False, True, rows=65536)
        torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
        opt.step()
        sch.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == pre_steps:
            log.line(f"PRE step={step+1}/{pre_steps} loss={float(loss):.5e} {base._fmt(values)}")
    unit = {"data": 1., **{k: 1. for k in base.PHYS_TERMS}}
    values, _ = objective(ctx, model, Plan(ctx, "train", 0, 20000, 128, 250), unit)
    lam = base.lambdas(values)
    state = {"state": base._state(model), "lambda": lam, "steps": pre_steps,
             "fingerprint": fingerprint(model), "seconds": time.monotonic() - started,
             "validation": evaluate(ctx, model, lam)}
    torch.save(state, out / "common_pre.pt")
    steps._write(out / "pretrain.json", {k: v for k, v in state.items() if k != "state"})


def train_case(ctx, model, common, devices, nodes, updates, out, log):
    model.load_state_dict(common["state"])
    if fingerprint(model) != common["fingerprint"]:
        raise AssertionError("Initial model differs from common pretrained state")
    lam = common["lambda"]
    opt = torch.optim.Adam(model.parameters(), lr=base.ADAM_LR)
    sch = base._cosine(opt, updates, base.ADAM_LR, base.LR_MIN)
    visits = np.zeros(len(ctx.uid), dtype=int)
    seen_nodes = np.zeros((len(ctx.uid), 250), dtype=bool)
    elapsed, total_rows = 0., 0
    result = {"devices_per_update": devices, "nodes_per_device": nodes, "physics": "PKC",
              "initial_weights": common["fingerprint"], "lambda": lam,
              "checkpoints": [{"step": 0, "training_seconds": 0., "jv": common["validation"]["jv"]}]}
    for step in range(updates):
        started = time.monotonic()
        plan = Plan(ctx, "train", 0, 30000 + step, devices, nodes)
        visits[plan.ids] += 1
        seen_nodes[plan.ids[:, None], plan.nodes[plan.ids]] = True
        opt.zero_grad(set_to_none=True)
        values, loss = objective(ctx, model, plan, lam, True, True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
        opt.step()
        sch.step()
        if ctx.dev.type == "cuda":
            torch.cuda.synchronize(ctx.dev)
        elapsed += time.monotonic() - started
        total_rows += plan.rows
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == updates:
            metric = base.jv_metrics(ctx, model)
            record = {"step": step+1, "training_seconds": elapsed, "sampled_rows": total_rows,
                      "jv": metric, "loss": float(loss), "terms": {k: float(v) for k, v in values.items()}}
            result["checkpoints"].append(record)
            log.line(f"PKC devices={devices} nodes={nodes} step={step+1}/{updates} "
                     f"loss={float(loss):.5e} val_curve={metric['all']['curve']:.5f} seconds={elapsed:.1f}")
            log.rec(**record)
            torch.save({"state": base._state(model), "step": step+1}, out / "latest.pt")
            steps._write(out / "result.json", result)
    result.update({"status": "ok", "updates": updates, "training_seconds": elapsed,
                   "sampled_rows": total_rows, "final": evaluate(ctx, model, lam),
                   "device_visits": visits[ctx.ids["train"]].tolist(),
                   "unique_nodes_per_device": seen_nodes[ctx.ids["train"]].sum(1).tolist(),
                   "unique_train_devices_seen": int(np.count_nonzero(visits))})
    steps._write(out / "result.json", result)
    return result


def worker(gpu, root, updates, limit):
    torch.cuda.set_device(gpu)
    ctx = base.Ctx.load(root, torch.device(f"cuda:{gpu}"), limit)
    model = base.new_model(ctx, 0)
    common = torch.load(output_dir() / "common_pre.pt", map_location="cpu", weights_only=False)
    for devices, nodes in ASSIGNMENTS[gpu]:
        out = output_dir() / f"d{devices}_n{nodes}"
        log = base.Log(out, "train")
        train_case(ctx, model, common, devices, nodes, updates, out, log)


def run(root=None, pre_steps=300, updates=200, limit=None):
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This pilot requires two T4 GPUs")
    if pre_steps < 1 or updates < 1:
        raise ValueError("Positive training lengths required")
    root = str(m6_data.ensure(root))
    prepare(root, pre_steps, limit)
    gc.collect()
    torch.cuda.empty_cache()
    torch.multiprocessing.spawn(worker, args=(root, updates, limit), nprocs=2, join=True)
    cases = {f"d{d}_n{n}": json.loads((output_dir() / f"d{d}_n{n}" / "result.json").read_text())
             for d, n in CASES}
    summary = {"status": "ok", "seed": 0, "pre_steps": pre_steps, "updates": updates,
               "scope": "PKC refinement after common data pretraining; exploratory, one seed",
               "cases": cases}
    steps._write(output_dir() / "summary.json", summary)
    return summary
