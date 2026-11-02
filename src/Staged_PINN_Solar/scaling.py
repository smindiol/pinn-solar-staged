"""Debye scaling and normalized coordinates of the thin-film solar cell (Eqs. 1-2).

`Scales` holds the characteristic scales of Eq. (1) for a chosen reference
concentration `C0` and mobility `mu0`; `Coordinates` implements the
per-device normalized coordinate of Eq. (2). The proposal leaves `C0` and
`mu0` as free choices of the nondimensionalization; the defaults below
(`C0 = 1e19 cm^-3`, `mu0 = 1 cm^2/(V s)`) are the values that reproduce the
dPV dataset's stored coordinate and generation (measured, see
`data.py`), not a number taken from the proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r02.md",
    "sections": ["2"],
    "equations": ["1", "2"],
    "invariants": [
        "debye_scale_identities",
        "second_derivative_chain_rule",
        "normalized_coordinate_range",
    ],
}

# CODATA 2018 exact values, cgs-style units used by dPV (lengths in cm).
Q_ELEMENTARY = 1.602176634e-19      # C
K_BOLTZMANN = 1.380649e-23          # J / K
EPS0_F_PER_CM = 8.8541878128e-14    # F / cm
TEMPERATURE = 300.0                 # K (Section 2)

DEFAULT_C0 = 1.0e19                 # cm^-3
DEFAULT_MU0 = 1.0                   # cm^2 / (V s)


@dataclass(frozen=True)
class Scales:
    """Eq. (1): V_th, L_D, t_0, v_0, G_0 and J_0 for one (C0, mu0)."""

    C0: float = DEFAULT_C0
    mu0: float = DEFAULT_MU0
    T: float = TEMPERATURE

    @property
    def V_th(self) -> float:
        return K_BOLTZMANN * self.T / Q_ELEMENTARY

    @property
    def L_D(self) -> float:
        return sqrt(EPS0_F_PER_CM * K_BOLTZMANN * self.T / (Q_ELEMENTARY ** 2 * self.C0))

    @property
    def t0(self) -> float:
        return EPS0_F_PER_CM / (Q_ELEMENTARY * self.C0 * self.mu0)

    @property
    def v0(self) -> float:
        return self.L_D / self.t0

    @property
    def G0(self) -> float:
        return self.C0 / self.t0

    @property
    def J0(self) -> float:
        return Q_ELEMENTARY * self.C0 * self.L_D / self.t0


def delta(scales: Scales, L_total_cm: float) -> float:
    """Eq. (2): delta = L_D / L_total."""
    return scales.L_D / L_total_cm


class Coordinates:
    """Eq. (2): x_hat = x / L_total = delta * x_D in [0, 1] for one device."""

    def __init__(self, scales: Scales, L_total_cm: float) -> None:
        self.scales = scales
        self.L_total = float(L_total_cm)
        self.delta = delta(scales, self.L_total)

    def to_hat(self, x_debye: np.ndarray) -> np.ndarray:
        return self.delta * np.asarray(x_debye)

    def to_debye(self, x_hat: np.ndarray) -> np.ndarray:
        return np.asarray(x_hat) / self.delta

    def d_dxD(self, d_dxhat: np.ndarray) -> np.ndarray:
        """d/dx_D = delta d/dx_hat."""
        return self.delta * np.asarray(d_dxhat)

    def d2_dxD2(self, d2_dxhat2: np.ndarray) -> np.ndarray:
        """d^2/dx_D^2 = delta^2 d^2/dx_hat^2."""
        return self.delta ** 2 * np.asarray(d2_dxhat2)
