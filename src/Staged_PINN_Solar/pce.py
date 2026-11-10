"""Terminal current, J-V sweep and conversion efficiency (r10 Eqs. 13-14, handoff/historial-TDG, Section 5) and the
`pce_check` prerequisite. (r03 numbers its own Eqs. 13-14 differently: they are the lambda update and L-BFGS.)

dPV's `spline.calcPmax` is reimplemented as in the previous TDG implementation (a piecewise quadratic
through (V_j, V_j J_j) with continuous slope and a = 0 on the first piece); the standard-spectrum power
of Eq. 14's denominator is the per-device stored `P_in_std`. `check_devices` reproduces the stored PCE
(Device.pce) from the stored interface currents and, separately, from the stored potentials.
"""

from __future__ import annotations

import numpy as np

from . import carriers, residuals
from .scaling import Scales

__provenance__ = {
    "revision": "experiments-protocolo-experimental-de-la-pinn-por-etapas-sobre-dpv-v02.md",
    "sections": ["E1", "E2", "E3", "E4", "E5"],
    "experiments": ["E1", "E2", "E3", "E4", "E5"],
    "invariants": [
        "calcpmax_exact_on_quadratic_power",
        "sweep_is_cut_where_dpv_stops",
        "jv_summary_of_a_known_diode_curve",
        "relative_error_percent",
        "stored_pce_reproduced_on_dataset_devices",
        "pce_check_records_tolerance_and_verdict",
    ],
}

DEFAULT_P_IN_STD = 899.9167906040965   # W/m^2, dPV's standard spectrum (stored per device as P_in_std)
MAX_STEPS = 100


def qspline(x, y):
    """Piecewise quadratic through (x, y), continuous slope, a = 0 on the first piece (dPV spline)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = x.size
    M = np.zeros((3 * (n - 1), 3 * (n - 1)))
    z = np.zeros(3 * (n - 1))
    M[0, 0] = 1
    z[1] = y[0]
    for i in range(n - 1):
        M[3 * i + 1, 3 * i:3 * i + 3] = [x[i] ** 2, x[i], 1]
        z[3 * i + 1] = y[i]
        M[3 * i + 2, 3 * i:3 * i + 3] = [x[i + 1] ** 2, x[i + 1], 1]
        z[3 * i + 2] = y[i + 1]
    for i in range(n - 2):
        M[3 * i + 3, 3 * i:3 * i + 6] = [2 * x[i + 1], 1, 0, -2 * x[i + 1], -1, 0]
    coef = np.linalg.solve(M, z)
    return coef[::3], coef[1::3], coef[2::3]


def _quad(x, coef):
    a, b, c = coef
    return a * x ** 2 + b * x + c


def calc_pmax(v, j):
    """(P_max, V_max) of dPV's calcPmax for J in A/m^2: largest spline value among piece ends and vertices."""
    v = np.asarray(v, dtype=np.float64)
    p = v * np.asarray(j, dtype=np.float64)
    coef = qspline(v, p)
    a, b, _ = coef
    xl, xu = v[:-1], v[1:]
    xm = np.clip(-b / (2 * np.where(a != 0, a, 1)), xl, xu)
    xall = np.concatenate([xl, xu, xm])
    yall = np.concatenate([_quad(xl, coef), _quad(xu, coef), _quad(xm, coef)])
    k = int(np.argmax(yall))
    return float(yall[k]), float(xall[k])


def sweep_length(j) -> int:
    """Number of voltages dPV keeps: it stops at the first k >= 2 with J_{k-1} J_k <= 0 or J_k < 0."""
    j = np.asarray(j, dtype=np.float64)
    for k in range(2, min(j.size, MAX_STEPS)):
        if j[k - 1] * j[k] <= 0 or j[k] < 0:
            return k + 1
    return min(j.size, MAX_STEPS)


def jv_summary(v, j_acm2, p_in_std: float = DEFAULT_P_IN_STD) -> dict:
    """PCE, Pmax, Vmax, Jsc, Voc and FF of a J-V curve (V, A/cm^2), cut where dPV would stop it (Eq. 14)."""
    j_all = np.asarray(j_acm2, dtype=np.float64)
    if not np.all(np.isfinite(j_all)) or j_all.size < 3:
        return {"pce": float("nan"), "pmax": float("nan"), "vmax": float("nan"), "jsc": float("nan"),
                "voc": float("nan"), "ff": float("nan"), "n_voltages": int(j_all.size)}
    k = sweep_length(j_all)
    v = np.asarray(v, dtype=np.float64)[:k]
    j = j_all[:k]
    pmax, vmax = calc_pmax(v, j * 1e4)
    jsc = j[0] * 1e4
    voc = float("nan")
    if j.size >= 2 and j[-1] * j[-2] <= 0 and j[-2] != j[-1]:
        voc = float(v[-2] + j[-2] * (v[-1] - v[-2]) / (j[-2] - j[-1]))
    ff = pmax / (voc * jsc) if np.isfinite(voc) and jsc > 0 else float("nan")
    return {"pce": pmax / p_in_std, "pmax": pmax, "vmax": vmax, "jsc": jsc, "voc": voc, "ff": ff,
            "n_voltages": int(k)}


def relative_error_percent(pce_pred: float, pce_ref: float) -> float:
    """The metric of the protocol: |eta - eta_ref| / eta_ref in percent (NaN stays NaN)."""
    return 100.0 * abs(pce_pred - pce_ref) / abs(pce_ref)


def terminal_current_stored(dev, scales: Scales) -> np.ndarray:
    """Eq. 13 on the dPV interface currents: J_0 times the spatial mean of J_n + J_p, per voltage [A/cm^2]."""
    return scales.J0 * np.mean(np.asarray(dev.J_n) + np.asarray(dev.J_p), axis=-1)


def terminal_current_profile(dev, scales: Scales) -> np.ndarray:
    """Eq. 13 as written: J_0 delta mean(mu_n n dphi_n/dx + mu_p p dphi_p/dx), potentials differentiated on dPV's nodes."""
    from .data import device_material

    m = device_material(dev, scales)
    n = carriers.electron_density(dev.phi, dev.phi_n, m)
    p = carriers.hole_density(dev.phi, dev.phi_p, m)
    xh = dev.x_hat
    jn = residuals.current_density(m.mu_n, n, np.gradient(dev.phi_n, xh, axis=-1), dev.delta(scales))
    jp = residuals.current_density(m.mu_p, p, np.gradient(dev.phi_p, xh, axis=-1), dev.delta(scales))
    return scales.J0 * np.mean(jn + jp, axis=-1)


def check_device_objects(devs, scales: Scales, tolerance: float, profile_tolerance: float | None = None) -> dict:
    """The record of `pce_check`: per-device relative error of the reproduced PCE, the tolerance and a verdict."""
    out = {}
    worst_stored, worst_profile = 0.0, None
    for dev in devs:
        v = np.asarray(dev.V)
        s = jv_summary(v, terminal_current_stored(dev, scales), dev.p_in_std)
        rec = {"pce": s["pce"], "pce_ref": float(dev.pce),
               "rel_error": abs(s["pce"] - dev.pce) / abs(dev.pce) if np.isfinite(s["pce"]) else float("inf")}
        worst_stored = max(worst_stored, rec["rel_error"])
        if hasattr(dev, "phi"):
            sp = jv_summary(v, terminal_current_profile(dev, scales), dev.p_in_std)
            rec["pce_profile"] = sp["pce"]
            rec["profile_rel_error"] = (abs(sp["pce"] - dev.pce) / abs(dev.pce)
                                        if np.isfinite(sp["pce"]) else float("inf"))
            worst_profile = max(worst_profile or 0.0, rec["profile_rel_error"])
        out[dev.ref.uid] = rec
    rec = {"n_devices": len(out), "tolerance": tolerance, "max_rel_error": worst_stored,
           "passed": bool(len(out) and worst_stored <= tolerance), "devices": out,
           "evaluator": "stored interface currents (dPV operator), Eq. 13 mean over nodes + calcPmax, Eq. 14"}
    if worst_profile is not None:
        ptol = profile_tolerance if profile_tolerance is not None else tolerance
        rec.update(profile_tolerance=ptol, profile_max_rel_error=worst_profile,
                   profile_passed=bool(worst_profile <= ptol))
    return rec


def check_devices(refs, scales: Scales, tolerance: float, profile_tolerance: float | None = None,
                  progress=None) -> dict:
    """`check_device_objects` over dataset references, streaming one device at a time."""
    from .data import load_device

    def gen():
        for k, r in enumerate(refs):
            yield load_device(r)
            if progress:
                progress(k + 1, len(refs))

    return check_device_objects(gen(), scales, tolerance, profile_tolerance)
