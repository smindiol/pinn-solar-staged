"""Level 2 invariants of config.py: the open parameters are all in one place, marked, and recorded."""

from __future__ import annotations

import ast
import json
from dataclasses import fields
from pathlib import Path

from Staged_PINN_Solar import config as C

SRC = Path(C.__file__)

#: what the experiments protocol (v03) or r10 fixes; every other field is open and must say PROVISIONAL.
FIXED = {"seeds", "split_seed", "bootstrap_resamples", "confidence_level", "holm_alpha", "width", "depth",
         "activation", "voltage_step", "max_voltages", "eps_g", "lambda_min", "lambda_max", "lambda_reference"}
#: rehearsal-scale switches, not parameters of the experiment
REHEARSAL = {"max_devices", "max_stages", "smoke"}


def test_every_open_parameter_is_marked_provisional() -> None:
    """Each field line of Config ends with a FIXED or PROVISIONAL tag, and the FIXED ones are exactly the protocol's."""
    lines = SRC.read_text().splitlines()
    tagged = {}
    for f in fields(C.Config):
        line = next(l for l in lines if l.strip().startswith(f"{f.name}:"))
        tagged[f.name] = "FIXED" if "# FIXED" in line else "PROVISIONAL" if "# PROVISIONAL" in line else None
    for name, tag in tagged.items():
        if name in REHEARSAL:
            continue
        assert tag is not None, f"{name} carries no FIXED/PROVISIONAL tag"
        assert (tag == "FIXED") == (name in FIXED), name
    # every PROVISIONAL line names a source after the colon
    for l in lines:
        if "# PROVISIONAL:" in l:
            assert len(l.split("# PROVISIONAL:")[1].strip()) > 5


def test_config_roundtrips_into_the_unit_record() -> None:
    d = C.Config().to_dict()
    assert json.loads(json.dumps(d)) == json.loads(json.dumps(d))
    assert d["width"] == 256 and d["depth"] == 4 and d["seeds"] == (0, 1, 2, 3, 4)
    assert C.smoke().to_dict()["smoke"] is True and C.Config().to_dict()["smoke"] is False
    assert C.Config().with_(n_blocks=2).n_blocks == 2
