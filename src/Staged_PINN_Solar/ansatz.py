"""Hard-constrained ansatz for (phi, phi_n, phi_p) (Eq. 3).

The network itself is an *interface*: any callable ``arch(x_hat, xi_hat) -> (B, 3)``
returning the raw outputs (u_phi, u_n, u_p) with u_n, u_p in [0, 1]. The
proposal refers to "the two architectures of the previous proposal" without
defining them (r03, section 7), so none is shipped here: this is an open gap.
D_phi and the carrier bounds Y_k^min/max are defined by the r10 construction that
r03 cites unchanged: D_phi = max |phi - phi_lin| and Y_k = the extremes of phi_n,
phi_p over every training device, voltage and node. `measure_bounds` computes them
from the training split; `AnsatzBounds` stays a plain container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r04.md",
    "sections": ["2"],
    "equations": ["3"],
    "invariants": [
        "contacts_are_exact_for_any_network",
        "interior_correction_is_bounded_by_d_phi",
        "carrier_outputs_stay_inside_their_bounds",
        "normalized_parameters_match_the_dataset_u",
        "bounds_cover_the_training_profiles",
        "bounds_contain_the_equilibrium_value",
    ],
}

Architecture = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AnsatzBounds:
    """Eq. (3) constants defined by r10: D_phi and (Y_min, Y_max) for phi_n, phi_p."""

    D_phi: float
    phi_n: tuple
    phi_p: tuple


def measure_bounds(devices, scales=None, margin: float = 0.05) -> AnsatzBounds:
    """D_phi = max |phi - phi_lin| and the extremes of phi_n, phi_p over training devices (r10, Section 2).

    Each quasi-Fermi box is then widened to contain 0, the equilibrium value at V = 0, and by `margin`
    of its range on each side. dPV's stored phi_p peaks at about -1e-8, so the raw extremes put 0 on the
    edge of the box, where the sigmoid of Eq. (3) can never reach it and the equilibrium start is
    impossible. With margin = 0 only the inclusion of 0 is applied.

    `devices` yields `data.Device` objects (stream them; nothing is kept). phi_lin is the linear
    interpolation of the contact potentials with the applied voltage V / V_th at the right contact.
    """
    from .scaling import Scales

    scales = scales or Scales()
    d_phi = 0.0
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for dev in devices:
        v_tilde = (np.asarray(dev.V) / scales.V_th)[:, None]
        line = dev.phi_eq[0] + (dev.phi_eq[-1] + v_tilde - dev.phi_eq[0]) * dev.x_hat[None, :]
        d_phi = max(d_phi, float(np.max(np.abs(dev.phi - line))))
        lo = np.minimum(lo, [dev.phi_n.min(), dev.phi_p.min()])
        hi = np.maximum(hi, [dev.phi_n.max(), dev.phi_p.max()])
    lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)
    pad = margin * (hi - lo)
    lo, hi = lo - pad, hi + pad
    return AnsatzBounds(D_phi=d_phi, phi_n=(float(lo[0]), float(hi[0])), phi_p=(float(lo[1]), float(hi[1])))


def normalize_xi(xi: dict, specs) -> np.ndarray:
    """Eq. (3): xi_hat_j = (g_j(xi_j) - g_j^min) / (g_j^max - g_j^min), g_j = log10 or identity."""
    out = []
    for name, kind, lo, hi in specs:
        g = (lambda v: np.log10(v)) if kind == "log" else (lambda v: v)
        out.append((g(xi[name]) - g(lo)) / (g(hi) - g(lo)))
    return np.asarray(out)


class Ansatz(torch.nn.Module):
    """Eq. (3): phi exact at both contacts, phi_n and phi_p boxed into [Y_min, Y_max]."""

    def __init__(self, arch: Architecture, bounds: AnsatzBounds) -> None:
        super().__init__()
        self.arch = arch
        self.bounds = bounds

    def forward(self, x_hat, xi_hat, phi_eq0, phi_eq1, V_tilde):
        u = self.arch(x_hat, xi_hat)
        u_phi, u_n, u_p = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        line = phi_eq0 + (phi_eq1 + V_tilde - phi_eq0) * x_hat
        phi = line + 4.0 * x_hat * (1.0 - x_hat) * self.bounds.D_phi * u_phi
        lo_n, hi_n = self.bounds.phi_n
        lo_p, hi_p = self.bounds.phi_p
        return phi, lo_n + (hi_n - lo_n) * u_n, lo_p + (hi_p - lo_p) * u_p
