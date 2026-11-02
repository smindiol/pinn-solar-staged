"""Boltzmann carrier densities and total recombination (Eqs. 4 and 8).

Every function accepts numpy arrays (the dPV reference) or torch tensors
(the PINN, differentiable); the material fields are converted to the
backend of the potentials they meet. All quantities are dimensionless in the
sense of `scaling.py` (densities over C0, energies over V_th, times over t0).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r03.md",
    "sections": ["3"],
    "equations": ["4", "8"],
    "invariants": [
        "equilibrium_np_product_is_ni_squared",
        "quasi_fermi_shifts_scale_densities",
        "recombination_zero_at_equilibrium_and_signed",
        "recombination_srh_closed_form_and_radiative_term",
        "torch_and_numpy_agree",
    ],
}


@dataclass(frozen=True)
class Material:
    """Dimensionless per-node material fields of Eqs. (4)-(8).

    Nc, Nv: effective densities; chi: electron affinity; Eg: gap; eps_r:
    relative permittivity; mu_n, mu_p: mobilities; tau_n, tau_p: SRH
    lifetimes; Et: trap level; Br: radiative coefficient; Ndop: net doping
    (donors positive).
    """

    Nc: np.ndarray
    Nv: np.ndarray
    chi: np.ndarray
    Eg: np.ndarray
    eps_r: np.ndarray
    mu_n: np.ndarray
    mu_p: np.ndarray
    tau_n: np.ndarray
    tau_p: np.ndarray
    Et: np.ndarray
    Br: np.ndarray
    Ndop: np.ndarray

    def like(self, ref):
        """The fields as tensors/arrays of the backend (and dtype) of `ref`."""
        if isinstance(ref, torch.Tensor):
            return Material(**{f.name: torch.as_tensor(getattr(self, f.name), dtype=ref.dtype,
                                                       device=ref.device)
                               for f in fields(self)})
        return self


def _exp(x):
    return torch.exp(x) if isinstance(x, torch.Tensor) else np.exp(x)


def _sqrt(x):
    return torch.sqrt(x) if isinstance(x, torch.Tensor) else np.sqrt(x)


def electron_density(phi, phi_n, m: Material):
    """Eq. (4): n = Nc exp(phi + phi_n + chi)."""
    m = m.like(phi)
    return m.Nc * _exp(phi + phi_n + m.chi)


def hole_density(phi, phi_p, m: Material):
    """Eq. (4): p = Nv exp(-phi - phi_p - chi - Eg)."""
    m = m.like(phi)
    return m.Nv * _exp(-phi - phi_p - m.chi - m.Eg)


def equilibrium_densities(phi, m: Material):
    """Eq. (4) with phi_n = phi_p = 0 (n_eq, p_eq of Eq. 9)."""
    zero = phi * 0.0
    return electron_density(phi, zero, m), hole_density(phi, zero, m)


def intrinsic_density_squared(m: Material):
    """n_i^2 = Nc Nv exp(-Eg) (Section 3)."""
    return m.Nc * m.Nv * (np.exp(-m.Eg) if not isinstance(m.Eg, torch.Tensor) else torch.exp(-m.Eg))


def recombination(n, p, m: Material):
    """Eq. (8): R = (n p - n_i^2) [Br + 1/(tau_p (n + n_i e^Et) + tau_n (p + n_i e^-Et))]."""
    m = m.like(n)
    ni2 = intrinsic_density_squared(m)
    ni = _sqrt(ni2)
    srh = 1.0 / (m.tau_p * (n + ni * _exp(m.Et)) + m.tau_n * (p + ni * _exp(-m.Et)))
    return (n * p - ni2) * (m.Br + srh)
