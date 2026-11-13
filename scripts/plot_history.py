#!/usr/bin/env python3
"""Replot an E6 history locally: `python scripts/plot_history.py Results/e6/e6-A-s0.history.jsonl [out.png]`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Staged_PINN_Solar.e6 import plot_history  # noqa: E402

h = Path(sys.argv[1])
plot_history(h, Path(sys.argv[2]) if len(sys.argv) > 2 else h.with_suffix(".png"))
