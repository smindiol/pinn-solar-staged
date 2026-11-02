"""Per-term normalized loss (Eqs. 10-11) and its measurement on the dPV reference.

`ReferenceStatistics` accumulates, over the training devices, all voltages and
all nodes, the two quantities of Eq. (10): sigma_i^2 = <(sum_q |T_iq|)^2> and
the floor f_i = <R_i^2>. `reference_terms` evaluates the summands T_iq on a
stored dPV solution with the staggered scheme of `residuals.py`; the contact
currents are the dPV interface currents stored with the solution.
"""

from __future__ import annotations

import numpy as np

from . import carriers, residuals
from .data import Device, device_contacts, device_material
from .residuals import residual, scale
from .scaling import Scales

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r02.md",
    "sections": ["4"],
    "equations": ["10", "11"],
    "invariants": [
        "sigma_squared_dominates_residual_squared",
        "loss_at_reference_equals_rho",
        "normalization_removes_the_residual_scale",
        "total_loss_is_the_weighted_sum",
        "streaming_statistics_equal_batch_statistics",
        "reference_solution_closes_the_equations",
        "reference_continuity_closes_with_the_dpv_operator",
    ],
}

#: The residual index set I of Section 4, in the order of the proposal's table.
TERMS = ("poisson", "bcn", "bcp", "n", "p")


def reference_terms(dev: Device, scales: Scales, currents: str = "continuous") -> dict:
    """Summands T_iq of the five residuals on one stored dPV solution.

    Domain residuals live at the N-2 interior nodes, contact residuals at the two contacts;
    every array has the voltage axis first.

    `currents` selects how J_n, J_p enter the continuity residuals: "continuous" forms Eq. (6)
    at the half nodes (logarithmic mean of the density, difference of the quasi-Fermi
    potential), the discretization-independent reading of the equations; "dpv" uses the
    Scharfetter-Gummel interface currents stored with the solution, i.e. the reference's own
    discrete operator.
    """
    if currents not in ("continuous", "dpv"):
        raise ValueError(f"currents must be 'continuous' or 'dpv', got {currents!r}")
    m = device_material(dev, scales)
    c = device_contacts(scales)
    xh, d = dev.x_hat, dev.delta(scales)
    n = carriers.electron_density(dev.phi, dev.phi_n, m)
    p = carriers.hole_density(dev.phi, dev.phi_p, m)
    R = carriers.recombination(n, p, m)
    h = np.diff(xh)
    half = 0.5 * (h[1:] + h[:-1])

    flux_dx = residuals.staggered_flux_derivative(xh, dev.phi, m.eps_r)
    interior = slice(1, -1)
    out = {"poisson": residuals.poisson_terms(flux_dx, n[:, interior], p[:, interior],
                                              m.Ndop[interior], d)}

    def mid(a):
        return 0.5 * (a[..., 1:] + a[..., :-1])

    def log_mean(a):
        """Logarithmic mean of the neighbours: exact for a density that is exponential
        between two nodes, which is what Eq. (4) makes it."""
        lo, hi = a[..., :-1], a[..., 1:]
        same = np.isclose(lo, hi, rtol=1e-12, atol=0.0)
        safe = np.where(same, 1.0, hi / lo)
        return np.where(same, lo, (hi - lo) / np.log(safe))

    Jn = residuals.current_density(mid(m.mu_n), log_mean(n), np.diff(dev.phi_n, axis=-1) / h, d)
    Jp = residuals.current_density(mid(m.mu_p), log_mean(p), np.diff(dev.phi_p, axis=-1) / h, d)
    if currents == "dpv":
        Jn, Jp = dev.J_n, dev.J_p
    dJn = (Jn[:, 1:] - Jn[:, :-1]) / half
    dJp = (Jp[:, 1:] - Jp[:, :-1]) / half
    out["n"] = residuals.continuity_n_terms(dJn, dev.G[interior], R[:, interior], d)
    out["p"] = residuals.continuity_p_terms(dJp, dev.G[interior], R[:, interior], d)

    n_eq, p_eq = carriers.equilibrium_densities(dev.phi_eq, m)
    ends = [0, -1]
    out["bcn"] = residuals.contact_n_terms(dev.J_n[:, ends], n[:, ends], n_eq[ends],
                                           np.array([c.S_n0, c.S_n1]))
    out["bcp"] = residuals.contact_p_terms(dev.J_p[:, ends], p[:, ends], p_eq[ends],
                                           np.array([c.S_p0, c.S_p1]))
    return out


class ReferenceStatistics:
    """Pooled sigma_i^2 and f_i of Eq. (10), accumulated chunk by chunk."""

    def __init__(self) -> None:
        self._sigma: dict[str, float] = {}
        self._floor: dict[str, float] = {}
        self._count: dict[str, int] = {}

    def add(self, terms: dict) -> None:
        for name, t in terms.items():
            s, r = scale(t), residual(t)
            self._sigma[name] = self._sigma.get(name, 0.0) + float(np.sum(s.astype(np.float64) ** 2))
            self._floor[name] = self._floor.get(name, 0.0) + float(np.sum(r.astype(np.float64) ** 2))
            self._count[name] = self._count.get(name, 0) + int(r.size)

    def count(self) -> dict:
        return dict(self._count)

    def sigma2(self) -> dict:
        return {k: self._sigma[k] / self._count[k] for k in self._sigma}

    def f(self) -> dict:
        return {k: self._floor[k] / self._count[k] for k in self._floor}

    def rho(self) -> dict:
        s, f = self.sigma2(), self.f()
        return {k: f[k] / s[k] for k in s}


def measure_reference_statistics(refs, scales: Scales | None = None, progress=None) -> ReferenceStatistics:
    """Stream the devices one at a time; nothing but the running sums stays in memory."""
    from .data import load_device
    scales = scales or Scales()
    stats = ReferenceStatistics()
    for k, ref in enumerate(refs):
        stats.add(reference_terms(load_device(ref), scales))
        if progress is not None:
            progress(k + 1, len(refs))
    return stats


def normalized_losses(res: dict, sigma2: dict) -> dict:
    """Eq. (11): L_i = <R_i^2> / sigma_i^2 (numpy arrays or torch tensors)."""
    out = {}
    for name, r in res.items():
        out[name] = (r ** 2).mean() / sigma2[name]
        if not hasattr(out[name], "backward"):
            out[name] = float(out[name])
    return out


def total_loss(L: dict, lam: dict):
    """Eq. (11): L(theta; lambda) = sum_i lambda_i L_i(theta)."""
    total = 0.0
    for name, value in L.items():
        total = total + lam[name] * value
    return total
