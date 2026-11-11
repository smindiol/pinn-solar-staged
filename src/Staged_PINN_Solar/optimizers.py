"""Second-order optimizers: L-BFGS (Eqs. 14-15) and SOAP (Eqs. 16-17).

`LBFGS` keeps the last m pairs (s_k, y_k), applies the inverse-Hessian
approximation by the two-loop recursion and must be `reset()` whenever lambda
changes (section 5: the objective is fixed only inside a block).

`SOAP` follows Eqs. (16)-(17) for one weight matrix: exponential moving averages of
G G^T and G^T G, their eigenvector matrices Q_L, Q_R refreshed every
`precondition_frequency` steps, and bias-corrected Adam moments of the rotated
gradient with epsilon inside the root, exactly as Eq. (17) is written. The
first moment is accumulated in the original basis and rotated when used, and the
second moment is kept in the rotated basis as it stood when it was accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r04.md",
    "sections": ["5", "6"],
    "equations": ["14", "15", "16", "17"],
    "invariants": [
        "lbfgs_inverse_hessian_satisfies_the_secant_condition",
        "soap_warmup_scales_the_first_steps_linearly",
        "lbfgs_memory_is_bounded_and_reset_clears_it",
        "lbfgs_step_decreases_the_objective_monotonically",
        "lbfgs_reset_on_objective_change_keeps_descending",
        "soap_covariances_are_ema_of_gram_matrices",
        "soap_rotation_matrices_are_orthogonal",
        "soap_update_matches_equation_17_in_the_identity_basis",
        "soap_rotated_update_is_equivariant",
        "lbfgs_solves_ill_conditioned_quadratic_and_soap_decreases_it",
    ],
}


class LBFGS:
    """Limited-memory BFGS with an Armijo backtracking line search (Eq. 14, a_k)."""

    def __init__(self, memory: int = 10, c1: float = 1e-4, shrink: float = 0.5,
                 max_backtracks: int = 40) -> None:
        self.memory = memory
        self.c1, self.shrink, self.max_backtracks = c1, shrink, max_backtracks
        self._s: list[torch.Tensor] = []
        self._y: list[torch.Tensor] = []

    @property
    def history_size(self) -> int:
        return len(self._s)

    def reset(self) -> None:
        """Forget the curvature pairs (to be called whenever lambda changes)."""
        self._s.clear()
        self._y.clear()

    def update(self, s: torch.Tensor, y: torch.Tensor) -> bool:
        """Store (s_k, y_k) if it satisfies the curvature condition s^T y > 0."""
        sy = float(s @ y)
        if sy <= 1e-12 * float(s.norm() * y.norm()):
            return False
        self._s.append(s.clone())
        self._y.append(y.clone())
        if len(self._s) > self.memory:
            self._s.pop(0)
            self._y.pop(0)
        return True

    def apply_inverse_hessian(self, g: torch.Tensor) -> torch.Tensor:
        """Two-loop recursion; H_0 = (s^T y / y^T y) I from the newest pair, identity if empty."""
        if not self._s:
            return g.clone()
        q = g.clone()
        alphas = []
        for s, y in zip(reversed(self._s), reversed(self._y)):
            a = (s @ q) / (s @ y)
            alphas.append(a)
            q = q - a * y
        s, y = self._s[-1], self._y[-1]
        r = q * ((s @ y) / (y @ y))
        for (s, y), a in zip(zip(self._s, self._y), reversed(alphas)):
            b = (y @ r) / (s @ y)
            r = r + s * (a - b)
        return r

    def minimize(self, x0: torch.Tensor, fun: Callable[[torch.Tensor], tuple], max_iter: int = 100,
                 gtol: float = 1e-8):
        """Run one block of iterations from x0 on a fixed objective `fun(x) -> (f, grad)`."""
        x = x0.clone()
        f, g = fun(x)
        history = [float(f)]
        it = 0
        while it < max_iter and float(g.norm()) > gtol:
            d = -self.apply_inverse_hessian(g)
            slope = float(g @ d)
            if slope >= 0.0:           # not a descent direction: fall back to steepest descent
                self.reset()
                d, slope = -g, -float(g @ g)
            step = 1.0 if self.history_size else min(1.0, 1.0 / float(g.norm()))
            for _ in range(self.max_backtracks):
                f_new, g_new = fun(x + step * d)
                if float(f_new) <= float(f) + self.c1 * step * slope:
                    break
                # minimizer of the quadratic through f, slope and f_new, safeguarded
                denom = 2.0 * (float(f_new) - float(f) - slope * step)
                trial = -slope * step ** 2 / denom if denom > 0.0 else self.shrink * step
                step = min(max(trial, 0.1 * step), self.shrink * step)
            else:
                break                   # no acceptable step: stop the block
            s_k = step * d
            self.update(s_k, g_new - g)
            x, f, g = x + s_k, f_new, g_new
            history.append(float(f))
            it += 1
        return x, {"iterations": it, "grad_norm": float(g.norm()), "loss": float(f), "history": history}


@dataclass
class SOAPState:
    step: int = 0
    P_L: torch.Tensor | None = None
    P_R: torch.Tensor | None = None
    Q_L: torch.Tensor | None = None
    Q_R: torch.Tensor | None = None
    m: torch.Tensor | None = None
    v: torch.Tensor | None = None


class SOAP:
    """Eqs. (16)-(17) for a single matrix parameter."""

    def __init__(self, lr: float = 3e-3, betas=(0.95, 0.95), shampoo_beta: float = 0.95,
                 eps: float = 1e-8, precondition_frequency: int = 10, warmup: int = 0) -> None:
        """`warmup` > 0 ramps the step size linearly over the first `warmup` steps (Vyas et al. warm up SOAP's
        learning rate). Adam-type steps are about lr * sign(g) per weight whatever |g| is, and Eq. (3) amplifies
        the outputs by D_phi and the box widths (tens of thermal voltages) before an exponential, so from the
        equilibrium start an unramped step overshoots: measured, Adam at lr 1e-5 spikes the loss from 0.43 to 175."""
        self.lr, self.betas, self.shampoo_beta = lr, betas, shampoo_beta
        self.warmup = int(warmup)
        self.eps, self.freq = eps, precondition_frequency
        self.state = SOAPState()

    def step(self, W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        st, (b1, b2), sb = self.state, self.betas, self.shampoo_beta
        m_dim, n_dim = G.shape
        if st.P_L is None:
            st.P_L = torch.zeros(m_dim, m_dim, dtype=G.dtype, device=G.device)
            st.P_R = torch.zeros(n_dim, n_dim, dtype=G.dtype, device=G.device)
            st.m = torch.zeros_like(G)
            st.v = torch.zeros_like(G)
        if st.Q_L is None:
            st.Q_L = torch.eye(m_dim, dtype=G.dtype, device=G.device)
            st.Q_R = torch.eye(n_dim, dtype=G.dtype, device=G.device)
        st.step += 1
        st.P_L = sb * st.P_L + (1.0 - sb) * G @ G.T                 # Eq. (16)
        st.P_R = sb * st.P_R + (1.0 - sb) * G.T @ G
        if st.step % self.freq == 0:
            st.Q_L = torch.linalg.eigh(st.P_L)[1]
            st.Q_R = torch.linalg.eigh(st.P_R)[1]
        Gp = st.Q_L.T @ G @ st.Q_R                                   # Eq. (17)
        st.m = b1 * st.m + (1.0 - b1) * G
        st.v = b2 * st.v + (1.0 - b2) * Gp ** 2
        m_hat = (st.Q_L.T @ st.m @ st.Q_R) / (1.0 - b1 ** st.step)
        v_hat = st.v / (1.0 - b2 ** st.step)
        lr = self.lr * (min(1.0, st.step / self.warmup) if self.warmup > 0 else 1.0)
        return W - lr * st.Q_L @ (m_hat / torch.sqrt(v_hat + self.eps)) @ st.Q_R.T
