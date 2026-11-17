"""E9d queue for a single local GPU (RTX A5000 workstation).

Runs the arms one after another, skips arms that already finished (status ok),
and re-executes the report notebook after each arm so results can be checked
on the workstation at any time.

Usage:
    python tools/gpu/e9d_queue.py            # full queue
    python tools/gpu/e9d_queue.py --smoke    # short memory/timing check
    python tools/gpu/e9d_queue.py --tag e9e --arms control_anchor:0
                                             # E9e: control with Voc/MPP anchors
    python tools/gpu/e9d_queue.py --tag e9f_smoke --arms control:0,balanced:0 \
        --steps 300,500,50 --weight-cap none --recalib 250 --adam-lr 3e-4 --lbfgs-val 25
                                             # E9f: exact group balance, periodic recalibration

Environment:
    DPV_BASE_M6          folder holding base_m6.h5
    STAGED_PINN_RESULTS  results root (outputs land in <root>/e9d or e9d_smoke)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from Staged_PINN_Solar import m6_current_fit as fit, m6_data, steps  # noqa: E402

# Order matters: control first, then the matching balanced arm, then a new seed.
# An arm is (variant, seed, anchors); "<variant>_anchor:<seed>" on the CLI sets anchors.
ARMS = [("control", 0, False), ("balanced", 0, False), ("balanced", 1, False)]
FULL = dict(pre_steps=300, adam_steps=4000, lbfgs_iters=500, devices=128, lb_devices=128)
SMOKE = dict(pre_steps=2, adam_steps=3, lbfgs_iters=2, devices=128, lb_devices=128)
NOTEBOOK = Path(__file__).with_name("e9d_report.ipynb")


def parse_arms(text):
    arms = []
    for item in text.split(","):
        name, seed = item.split(":")
        anchors = name.endswith("_anchor")
        arms.append((name.removesuffix("_anchor"), int(seed), anchors))
    return arms


def status(tag, variant, seed, anchors=False):
    name = f"{variant}{'_anchor' if anchors else ''}_noise0_seed{seed}"
    path = steps.results_root() / tag / name / "summary.json"
    return json.loads(path.read_text()).get("status") if path.exists() else None


def report(tag):
    out = steps.results_root() / tag / "report.ipynb"
    env = dict(os.environ, E9D_TAG=tag)
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    str(NOTEBOOK), "--output", str(out)], env=env, check=False)
    print(f"[queue] report -> {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", default="e9d")
    parser.add_argument("--arms", default=None, help="e.g. control_anchor:0,balanced_anchor:0")
    parser.add_argument("--steps", default=None, help="pre,adam,lbfgs overriding the config, e.g. 300,500,50")
    parser.add_argument("--weight-cap", default="legacy", choices=("legacy", "none"),
                        help="legacy = clip group multipliers to [.01,100]; none = exact balance")
    parser.add_argument("--recalib", type=int, default=0, help="recompute multipliers every N Adam steps")
    parser.add_argument("--adam-lr", type=float, default=None, help="Adam peak learning rate")
    parser.add_argument("--lbfgs-val", type=int, default=0, help="validate every N L-BFGS evaluations")
    args = parser.parse_args()
    arms = parse_arms(args.arms) if args.arms else ARMS
    tag, config = (f"{args.tag}_smoke", dict(SMOKE)) if args.smoke else (args.tag, dict(FULL))
    if args.steps:
        pre, adam, lbfgs = (int(x) for x in args.steps.split(","))
        config.update(pre_steps=pre, adam_steps=adam, lbfgs_iters=lbfgs)
    config.update(weight_cap=None if args.weight_cap == "none" else fit.LEGACY_CAP,
                  recalib_every=args.recalib, adam_lr=args.adam_lr, lbfgs_val_every=args.lbfgs_val)
    root = str(m6_data.ensure(None))
    print(f"[queue] tag={tag} config={config} data={root}", flush=True)
    for variant, seed, anchors in arms:
        if not args.smoke and status(tag, variant, seed, anchors) == "ok":
            print(f"[queue] skip {variant} seed={seed} anchors={anchors} (already ok)", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        print(f"[queue] start {variant} seed={seed} anchors={anchors}", flush=True)
        result = fit.run_case(root, variant, seed=seed, gpu=0, tag=tag, anchors=anchors, **config)
        print(f"[queue] done {variant} seed={seed} anchors={anchors} {time.time()-started:.0f}s "
              f"peak={result.get('peak_memory_gb', 0):.2f}GB "
              f"final_curve={result['final']['jv']['all']['curve']:.3f}", flush=True)
        del result
        gc.collect()
        torch.cuda.empty_cache()
        report(tag)


if __name__ == "__main__":
    main()
