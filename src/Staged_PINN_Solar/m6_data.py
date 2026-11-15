"""Reduced, pre-split dataset of experiments v15 E9 (r14 Section 8): about 250 fixed nodes per device.

For every dpvB2 device of `split_m.csv` the builder keeps, ONCE and for every voltage of its curve:

* every node within 0.15 of the p-n junction (a window of width 0.30 centred on that device's junction),
* every node within 0.05 of each contact,
* 50 more nodes drawn uniformly among the remaining interior nodes with a fixed seed,

so M = 250 nodes per device (dPV's grid is uniform with N = 500, so the windows hold the same count everywhere).
The three potentials are stored at those nodes and the two carrier currents at the interface to the right of each
node (midpoint x_{i+1/2}), where dPV stores them by Scharfetter-Gummel. Every field is deltapv's dimensionless
value (global scales C0 = 1e19, mu0 = 1); V is in volts and the terminal current J in A/cm^2, as stored.

The file `base_m6.h5` holds everything a run needs, so a Kaggle kernel never reads the 500-node base.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from . import data, data_m, e6
from .scaling import Scales

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r14.md",
    "sections": ["8"],
    "equations": ["34", "37"],
    "invariants": [
        "m6_dataset_has_250_fixed_nodes_per_device",
        "m6_ohmic_contact_potentials_match_dpv",
        "m6_dataset_keeps_the_device_split",
    ],
}

FILE = "base_m6.h5"
DATASET_NAME = "dpv-pinn-base-m6"
HALF_WINDOW, EDGE, N_RANDOM, NODE_SEED = 0.15, 0.05, 50, 0
SPLITS = ("train", "val", "test")
MAT_KEYS = ("Nc", "Nv", "chi", "Eg", "eps_r", "mu_n", "mu_p", "tau_n", "tau_p", "Et", "Br")


def ohmic_potentials(Nc, Nv, chi, Eg, Nd, Na) -> tuple:
    """r14 Eq. 34 (dPV bcond.py): phi_eq at the n contact (x = 0) and at the p contact (x = L), normalized."""
    return -chi + np.log(Nd / Nc), -chi - Eg - np.log(Na / Nv)


def build(out_dir, root=None, limit: int | None = None, report=print) -> dict:
    """Write `<out_dir>/base_m6.h5`; `limit` caps each split (tests)."""
    root = Path(root) if root else data_m.ensure_dataset()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = Scales()
    contacts = data.device_contacts(scales)
    rows = data_m.read_split(root)
    split = data_m.split_m(root, rows)
    specs = data_m.xi_specs([r for r in rows if r["split"] == "train"])
    refs = []
    for s, name in enumerate(SPLITS):
        sel = getattr(split, name)
        refs += [(s, r) for r in (sel[:limit] if limit else sel)]
    D = len(refs)
    K = max(r.n_V for _, r in refs)
    rng = np.random.default_rng(NODE_SEED)
    t0 = time.time()
    cols = {k: np.zeros(D) for k in MAT_KEYS + ("delta", "Nd", "Na", "phi_eq0", "phi_eq1", "phi_eq0_dpv",
                                                 "phi_eq1_dpv", "pce", "L_cm")}
    xi_hat = np.zeros((D, len(specs)))
    n_V = np.zeros(D, dtype=np.int32)
    split_id = np.array([s for s, _ in refs], dtype=np.int8)
    V = np.full((D, K), np.nan)
    J = np.full((D, K), np.nan)
    M = None
    for d, (_, ref) in enumerate(refs):
        dev = data_m.load_device(ref, specs)
        mat = data.device_material(dev, scales)
        if M is None:
            idx_probe = e6.junction_nodes(dev.x_hat[None], mat.Ndop[None], N_RANDOM, np.random.default_rng(0),
                                          half_width=HALF_WINDOW, edge_width=EDGE)
            M = idx_probe.shape[1]
            nodes = {k: np.zeros((D, M)) for k in ("s", "s_mid", "G", "Ndop")}
            idx_all = np.zeros((D, M), dtype=np.int16)
            fields = np.full((D, K, 3, M), np.nan, dtype=np.float32)
            cur = np.full((D, K, 2, M), np.nan, dtype=np.float32)
        idx = e6.junction_nodes(dev.x_hat[None], mat.Ndop[None], N_RANDOM, rng, half_width=HALF_WINDOW,
                                edge_width=EDGE)[0]
        N = dev.x_hat.size
        im = np.minimum(idx, N - 2)
        idx_all[d] = idx
        nodes["s"][d] = dev.x_hat[idx]
        nodes["s_mid"][d] = 0.5 * (dev.x_hat[im] + dev.x_hat[im + 1])
        nodes["G"][d] = dev.G[idx]
        nodes["Ndop"][d] = mat.Ndop[idx]
        n = len(dev.V)
        n_V[d] = n
        V[d, :n], J[d, :n] = dev.V, dev.J
        fields[d, :n, 0], fields[d, :n, 1], fields[d, :n, 2] = dev.phi[:, idx], dev.phi_n[:, idx], dev.phi_p[:, idx]
        cur[d, :n, 0], cur[d, :n, 1] = np.asarray(dev.J_n)[:, im], np.asarray(dev.J_p)[:, im]
        for k in MAT_KEYS:
            cols[k][d] = float(np.asarray(getattr(mat, k)).ravel()[0])
        cols["Nd"][d], cols["Na"][d] = dev.xi["Nd_n"] / scales.C0, dev.xi["Na_p"] / scales.C0
        cols["phi_eq0"][d], cols["phi_eq1"][d] = ohmic_potentials(cols["Nc"][d], cols["Nv"][d], cols["chi"][d],
                                                                  cols["Eg"][d], cols["Nd"][d], cols["Na"][d])
        cols["phi_eq0_dpv"][d], cols["phi_eq1_dpv"][d] = float(dev.phi_eq[0]), float(dev.phi_eq[-1])
        cols["delta"][d] = dev.delta(scales)
        cols["pce"][d] = dev.pce
        cols["L_cm"][d] = dev.xi["L_n"] + dev.xi["L_p"]
        xi_hat[d] = dev.u
        if report and (d + 1) % 250 == 0:
            report(f"[m6_data] {d + 1}/{D} devices, {time.time() - t0:.0f}s")
    path = out_dir / FILE
    with h5py.File(path, "w") as f:
        z = dict(compression="gzip", compression_opts=4, shuffle=True)
        f.attrs["specs"] = json.dumps(specs)
        f.attrs["splits"] = json.dumps(SPLITS)
        f.attrs["S_n0"], f.attrs["S_p1"] = contacts.S_n0, contacts.S_p1
        f.attrs["S_n1"], f.attrs["S_p0"] = contacts.S_n1, contacts.S_p0
        f.attrs["V_th"], f.attrs["J0"], f.attrs["P_in"] = scales.V_th, scales.J0, data_m.P_IN_STD
        f.attrs["nodes"] = json.dumps({"half_window": HALF_WINDOW, "edge": EDGE, "random": N_RANDOM,
                                       "seed": NODE_SEED, "M": int(M)})
        f["uid"] = np.array([r.uid for _, r in refs], dtype="S64")
        f["material"] = np.array([r.material for _, r in refs], dtype="S16")
        f["split"], f["n_V"], f["xi_hat"], f["V"], f["J"] = split_id, n_V, xi_hat, V, J
        f["idx"] = idx_all
        for k, v in cols.items():
            f[f"dev/{k}"] = v
        for k, v in nodes.items():
            f.create_dataset(f"nodes/{k}", data=v, **z)
        f.create_dataset("fields", data=fields, **z)
        f.create_dataset("currents", data=cur, **z)
    out = {"devices": D, "K": int(K), "M": int(M), "bytes": path.stat().st_size, "seconds": time.time() - t0,
           "per_split": {n: int((split_id == s).sum()) for s, n in enumerate(SPLITS)},
           "max_phi_eq_gap": float(max(np.max(np.abs(cols["phi_eq0"] - cols["phi_eq0_dpv"])),
                                       np.max(np.abs(cols["phi_eq1"] - cols["phi_eq1_dpv"]))))}
    if report:
        report(f"[m6_data] {json.dumps(out)}")
    return out


def default_root() -> Path:
    """Env DPV_BASE_M6, then any /kaggle/input/**/base_m6.h5, then the workspace copy."""
    import os
    env = os.environ.get("DPV_BASE_M6")
    if env:
        return Path(env)
    hits = sorted(Path("/kaggle/input").glob(f"**/{FILE}")) if Path("/kaggle/input").is_dir() else []
    if hits:
        return hits[0].parent
    return data_m.local_root().parent / "base_m6"


def load(root=None) -> dict:
    """Every array of base_m6.h5 in memory (float64 except the float32 targets)."""
    root = Path(root) if root else default_root()
    out = {}
    with h5py.File(root / FILE, "r") as f:
        out["attrs"] = {k: (json.loads(v) if k in ("specs", "splits", "nodes") else float(v))
                        for k, v in f.attrs.items()}
        out["uid"] = [u.decode() for u in f["uid"][()]]
        out["material"] = [u.decode() for u in f["material"][()]]
        for k in ("split", "n_V", "xi_hat", "V", "J", "idx", "fields", "currents"):
            out[k] = f[k][()]
        out["dev"] = {k: f[f"dev/{k}"][()] for k in f["dev"]}
        out["nodes"] = {k: f[f"nodes/{k}"][()] for k in f["nodes"]}
    return out
