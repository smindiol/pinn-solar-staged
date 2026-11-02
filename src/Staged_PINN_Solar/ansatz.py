"""Hard-constrained ansatz for (phi, phi_n, phi_p) (Eq. 3).

The network itself is an *interface*: any callable ``arch(x_hat, xi_hat) -> (B, 3)``
returning the raw outputs (u_phi, u_n, u_p) with u_n, u_p in [0, 1]. The
proposal refers to "the two architectures of the previous proposal" without
defining them (r02, section 7), so none is shipped here: this is an open gap.
D_phi and the carrier bounds Y_k^min/max are likewise not given by r02 (they
belong to the r10 construction it cites) and are therefore explicit inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r03.md",
    "sections": ["2"],
    "equations": ["3"],
    "invariants": [
        "contacts_are_exact_for_any_network",
        "interior_correction_is_bounded_by_d_phi",
        "carrier_outputs_stay_inside_their_bounds",
        "normalized_parameters_match_the_dataset_u",
    ],
}

Architecture = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AnsatzBounds:
    """Eq. (3) constants that r02 does not define: D_phi and (Y_min, Y_max) for phi_n, phi_p."""

    D_phi: float
    phi_n: tuple
    phi_p: tuple


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
