"""Streaming accumulation and stratified spatial sampling for the repaired M6.

Separate from the already submitted m6c pilot. Each stochastic update selects devices and includes all their voltages.
GPU chunks are an implementation detail.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from . import m6c


def regions(s, nd):
    """Reproduce e6.junction_nodes windows using the original grid indices."""
    idx = np.rint(s * 499).astype(int)
    if len(s) != 250 or not np.allclose(idx / 499, s, atol=1e-7):
        raise ValueError("Expected the stored 250-node subset of the 500-node grid")
    changes = np.flatnonzero(np.diff(np.sign(nd)) != 0)
    if len(changes) != 1:
        raise ValueError("Expected one doping sign change")
    lo = int(np.clip(idx[changes[0]] - 75, 26, 324))
    left, right = idx <= 25, idx >= 474
    center = (idx >= lo) & (idx < lo + 150)
    masks = (left, center, right, ~(left | center | right))
    if tuple(int(m.sum()) for m in masks) != (25, 150, 25, 50):
        raise ValueError("Unexpected stored window counts")
    return [np.flatnonzero(m) for m in masks]


def sample_nodes(s, nd, rng):
    """150 nodes: 15 left + 90 junction + 15 right + 30 bulk.

    Ordered strata and mandatory region extremes prevent accidental clustering.
    The same selection is reused for every voltage of that device in an update.
    """
    groups = regions(s, nd)
    selected = []
    for group, count in zip(groups, (15, 90, 15, 30)):
        parts = np.array_split(group[1:-1], count - 2)
        selected.append(np.sort(np.r_[group[0], [rng.choice(x) for x in parts], group[-1]]))
    return np.sort(np.concatenate(selected)), selected


class SpatialPlan:
    """CPU index plan, streamed onto the GPU; no full-batch graph allocation."""
    def __init__(self, ctx, split, rng, n_dev):
        self.ctx = ctx
        self.ids = np.sort(rng.choice(ctx.ids[split], min(n_dev, len(ctx.ids[split])), replace=False))
        if not hasattr(ctx, "sampling_arrays"):
            ctx.sampling_arrays = (ctx.nodes["s"].detach().cpu().numpy(),
                                   ctx.nodes["Ndop"].detach().cpu().numpy())
        s, nd = ctx.sampling_arrays
        self.k = 150
        self.nodes = np.zeros((len(s), self.k), dtype=np.int64)
        for d in self.ids:
            self.nodes[d], _ = sample_nodes(s[d], nd[d], rng)
        self.pd = np.repeat(self.ids, ctx.n_V[self.ids])
        self.pv = np.concatenate([np.arange(ctx.n_V[d]) for d in self.ids])
        self.rows = len(self.pd) * self.k

    def batch(self, ctx, start, stop):
        flat = np.arange(start, min(stop, self.rows))
        pair_idx = flat // self.k
        lo, hi = pair_idx[0], pair_idx[-1] + 1
        rd, rv = self.pd[pair_idx], self.pv[pair_idx]
        return m6c.gather(ctx, self.pd[lo:hi], self.pv[lo:hi], rd, rv,
                         pair_idx - lo, self.nodes[rd, flat % self.k])


def objective(ctx, model, plan, lam, abc, rows, backward=False):
    """Exact row-mean objective/gradient, independent of GPU chunk size.

    Reset gradients once before a backward call, never once per chunk.
    No graph from an earlier chunk is retained.
    """
    loss_sum = torch.zeros((), dtype=torch.float64, device=ctx.dev)
    acc = {}
    for start in range(0, plan.rows, rows):
        b = plan.batch(ctx, start, start + rows)
        terms = m6c.terms(model, b, abc)
        weight = b["n_rows"] / plan.rows
        loss = weight * m6c.total(terms, lam, abc)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite full-train objective")
        if backward:
            loss.backward()
        loss_sum += loss.detach()
        for key, val in terms.items():
            acc[key] = acc.get(key, 0.) + weight * val.detach()
    return acc, loss_sum


def calibrate(ctx, model, plan, log=None, ceiling=262144, memory_fraction=.85, abc=(1, 1, 1), probe_to_oom=False):
    """Find the largest tested chunk under 85% VRAM, using ALL physics.

    The same safe chunk is used for both arms. This does not change their
    effective batch. CPU runs use a small chunk; no GPU capacity is inferred.
    """
    if ctx.dev.type != "cuda":
        return {"rows": min(256, plan.rows), "device": "cpu", "gpu_verified": False}
    ceiling = min(ceiling, plan.rows)
    budget = memory_fraction * torch.cuda.get_device_properties(ctx.dev).total_memory
    lam = {"data": 1., **{k: 1. for k in m6c.PHYS_TERMS}}
    safe = 0
    good, bad, candidate = 0, ceiling + 1, min(1024, ceiling)
    trials = []
    while candidate > good and candidate < bad:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(ctx.dev)
        started = time.monotonic()
        ok = False
        oom = False
        try:
            b = plan.batch(ctx, 0, candidate)
            values = m6c.terms(model, b, abc)
            loss = m6c.total(values, lam, abc)
            loss.backward()
            torch.cuda.synchronize(ctx.dev)
            peak = torch.cuda.max_memory_reserved(ctx.dev)
            if peak <= budget:
                safe = max(safe, candidate)
            ok = probe_to_oom or peak <= budget
        except torch.cuda.OutOfMemoryError:
            peak = None
            oom = True
        finally:
            b = values = loss = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        trials.append({"rows": candidate, "accepted": ok, "oom": oom, "peak_reserved": peak,
                       "seconds": time.monotonic() - started})
        if log:
            log.line(f"MEMORY_PROBE {trials[-1]}")
        if ok:
            good = candidate
            if good == ceiling:
                break
            candidate = min(2 * good, ceiling) if bad == ceiling + 1 else (good + bad) // 2
        else:
            bad = candidate
            candidate = (good + bad) // 2
        if good and bad - good <= 256:
            break
    if not good:
        raise RuntimeError("No safe GPU chunk found")
    return {"rows": safe or max(1, good // 2), "largest_successful_rows": good,
            "oom_observed": any(x["oom"] for x in trials), "device": str(ctx.dev), "gpu_verified": True,
            "memory_fraction": memory_fraction, "trials": trials}

