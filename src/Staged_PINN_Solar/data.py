"""Reader of the dPV reference set (E1A + E1B, 4096 devices) with the device-level split.

Sources live under `dpv-dataset/salidas/` and are opened read-only, one file
at a time and one device at a time: the 9.7 GB set is never loaded at once.
Two things the dataset's own documentation does not say, both measured here
(see `tests/test_invariants.py`):

* the stored coordinate `x_cm` is NOT in centimetres: it is the Debye
  coordinate x/L_D with C0 = 1e19 cm^-3 (x[-1] * L_D equals L_n + L_p);
* the stored generation `G` is dPV's dimensionless generation for
  mu0 = 1 cm^2/(V s) (continuity closes to 1e-8 with that choice).

E1B block 1 exists twice (session s1 stopped at 814/1075 simulations,
session s4 completed 1024/1024); only the complete one is used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .carriers import Material
from .scaling import Scales

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r03.md",
    "sections": ["1", "2"],
    "equations": ["2", "3"],
    "invariants": [
        "inventory_is_4096_ok_devices_from_e1a_and_e1b",
        "split_is_device_level_80_10_10_and_seeded",
        "loaded_device_respects_dpv_contract",
        "device_material_doping_step_and_contacts",
    ],
}

SEED = 42
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)

# (family-stage, block) -> file relative to the salidas root. E1B block 1 is taken from s4.
SOURCES = {
    ("E1A", 0): "s1/dpv_E1A_b000.h5",
    ("E1A", 1): "s1/dpv_E1A_b001.h5",
    ("E1B", 0): "s1/dpv_E1B_b000.h5",
    ("E1B", 1): "s4/dpv_E1B_b001.h5",
}

# dpvgen/simulacion.py `_material_e1` and the contacts of `construir("E1", ...)`.
E1_MATERIAL = dict(Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100.0, mp=100.0,
                   Et=0.0, tn=1e-8, tp=1e-8, Br=0.0)
E1_CONTACT_VELOCITIES_CM_S = dict(Snl=1e7, Snr=0.0, Spl=0.0, Spr=1e7)


def default_root() -> Path:
    env = os.environ.get("DPV_SALIDAS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4] / "dpv-dataset" / "salidas"


@dataclass(frozen=True)
class DeviceRef:
    """A pointer to one device; nothing is read until `load_device`."""

    family_stage: str
    block: int
    idx: int
    source: Path

    @property
    def uid(self) -> str:
        return f"{self.family_stage}-b{self.block:03d}-{self.idx:06d}"


@dataclass
class Device:
    ref: DeviceRef
    xi: dict
    u: np.ndarray            # xi normalized to [0, 1] (g_j then min-max, Eq. 3)
    V: np.ndarray            # applied voltages [V]
    x_debye: np.ndarray      # nodes in units of L_D (stored as "x_cm")
    G: np.ndarray            # dimensionless generation
    phi_eq: np.ndarray
    phi: np.ndarray          # (n_V, N)
    phi_n: np.ndarray
    phi_p: np.ndarray
    J_n: np.ndarray          # (n_V, N-1) Scharfetter-Gummel interface currents (dimensionless)
    J_p: np.ndarray
    pce: float               # dPV efficiency (fraction)
    xi_specs: tuple = ()     # per active parameter: (name, scale 'log'|'lin', min, max)

    def delta(self, scales: Scales) -> float:
        """Eq. (2): delta = L_D / L_total = 1 / x_D(L_total)."""
        return 1.0 / float(self.x_debye[-1])

    @property
    def x_hat(self) -> np.ndarray:
        return self.x_debye / self.x_debye[-1]


def inventory(root: Path | None = None) -> list[DeviceRef]:
    """All Sobol devices of E1A+E1B whose state is 'ok', read from the per-file index only."""
    root = Path(root) if root else default_root()
    refs: list[DeviceRef] = []
    for (fs, block), rel in SOURCES.items():
        path = root / rel
        with h5py.File(path, "r") as f:
            idx = f["indice/idx"][()]
            estado = f["indice/estado"][()]
        refs += [DeviceRef(fs, block, int(i), path) for i, e in zip(idx, estado) if e == b"ok"]
    return refs


@dataclass
class Split:
    train: list[DeviceRef]
    val: list[DeviceRef]
    test: list[DeviceRef]


def split_devices(refs: list[DeviceRef], seed: int = SEED, fractions=SPLIT_FRACTIONS) -> Split:
    """Device-level split: a seeded permutation cut at round(0.8 N) and round(0.9 N)."""
    order = np.random.default_rng(seed).permutation(len(refs))
    n = len(refs)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    pick = lambda ids: [refs[i] for i in ids]
    return Split(pick(order[:n_train]), pick(order[n_train:n_train + n_val]),
                 pick(order[n_train + n_val:]))


def load_device(ref: DeviceRef) -> Device:
    with h5py.File(ref.source, "r") as f:
        g = f["sims"][f"{ref.idx:06d}"]
        specs = tuple((p["nombre"], p["escala"], float(p["min"]), float(p["max"]))
                      for p in json.loads(f.attrs["parametros_xi"]))
        n_v = int(g.attrs["n_voltajes"])
        pots = g["pots"][:n_v]
        return Device(
            ref=ref, xi=json.loads(g.attrs["xi"]), u=g["u"][()], V=g["V"][:n_v],
            x_debye=g["x_cm"][()], G=g["G"][()], phi_eq=g["phi_eq"][()],
            phi=pots[:, 0], phi_n=pots[:, 1], phi_p=pots[:, 2],
            J_n=g["Jn"][:n_v], J_p=g["Jp"][:n_v], pce=float(g.attrs["eff"]), xi_specs=specs)


def device_material(dev: Device, scales: Scales) -> Material:
    """Dimensionless E1 material with the device's doping step (layer 1 holds x <= L_n)."""
    m, n = E1_MATERIAL, len(dev.x_debye)
    x_cm = dev.x_debye * scales.L_D
    L_n = dev.xi["L_n"]
    donor = x_cm <= L_n * (1.0 + 1e-9)
    Ndop = np.where(donor, dev.xi["Nd_n"], -dev.xi["Na_p"]) / scales.C0
    full = lambda v: np.full(n, float(v))
    return Material(
        Nc=full(m["Nc"] / scales.C0), Nv=full(m["Nv"] / scales.C0),
        chi=full(m["Chi"] / scales.V_th), Eg=full(m["Eg"] / scales.V_th), eps_r=full(m["eps"]),
        mu_n=full(m["mn"] / scales.mu0), mu_p=full(m["mp"] / scales.mu0),
        tau_n=full(m["tn"] / scales.t0), tau_p=full(m["tp"] / scales.t0),
        Et=full(m["Et"] / scales.V_th), Br=full(m["Br"] * scales.C0 * scales.t0), Ndop=Ndop)


@dataclass(frozen=True)
class Contacts:
    """Dimensionless surface velocities S / v0 of Eq. (9)."""

    S_n0: float
    S_n1: float
    S_p0: float
    S_p1: float


def device_contacts(scales: Scales) -> Contacts:
    c = E1_CONTACT_VELOCITIES_CM_S
    return Contacts(c["Snl"] / scales.v0, c["Snr"] / scales.v0, c["Spl"] / scales.v0,
                    c["Spr"] / scales.v0)
