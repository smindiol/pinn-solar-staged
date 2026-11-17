"""Validation J-V metrics (incl. fine-grid Voc) of saved checkpoints, on CPU.

Usage:
    python tools/gpu/eval_ckpt.py <checkpoint.pt> [<checkpoint.pt> ...]
Prints one JSON line per checkpoint with the "all" metrics and per-material invalid counts.
Environment: DPV_BASE_M6 as for training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from Staged_PINN_Solar import m6c as base, m6_data  # noqa: E402


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("cpu")
    ctx = base.Ctx.load(str(m6_data.ensure(None)), dev, None)
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location=dev)
        model = base.new_model(ctx, 0)
        model.load_state_dict(ck["state"])
        m = base.jv_metrics(ctx, model)
        print(json.dumps({"checkpoint": path, "phase": ck.get("phase"), "step": ck.get("step"),
                          "all": m["all"],
                          "per_material": {k: {"Voc_bias_mV": v["Voc_bias_mV"], "invalid_Voc": v["invalid_Voc"],
                                               "curve_to_voc": v["curve_to_voc"]}
                                           for k, v in m.items() if k != "all"}}), flush=True)


if __name__ == "__main__":
    main()
