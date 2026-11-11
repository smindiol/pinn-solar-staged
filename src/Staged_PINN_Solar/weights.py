"""Loss-weight optimization (Eqs. 12-13 of r04) and the outer-loop alternative hook (section 5).

The weights follow the learning-rate annealing of wang2020 in its original form: the
reference is the term of largest gradient at the iteration, every weight chases
max_j max|grad L_j| / (mean|grad L_i| + eps_g), clipped to [lambda_min, lambda_max],
through an exponential moving average. The r03 form (Poisson as a fixed reference, no
floor, no cap) is kept under `reference="poisson"` as the measured control. A quasi-Newton method needs a fixed
objective while it accumulates memory, so `BlockWeights` lets lambda change
only between blocks of iterations and tells registered optimizers to reset.

The Bayesian-optimization alternative of section 5 is left as a hook
(`outer_loop_weights`): the surrogate/acquisition is a caller-supplied
proposer; no BO library is bundled and none was specified by the proposal.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r04.md",
    "sections": ["5"],
    "equations": ["12", "13"],
    "invariants": [
        "lambda_hat_balances_mean_gradient_magnitudes",
        "moving_average_update_is_convex_and_has_the_fixed_point",
        "poisson_weight_is_the_fixed_reference",
        "satisfied_term_keeps_its_weight",
        "weights_change_only_between_blocks",
        "update_notifies_the_optimizer_to_reset_its_memory",
        "vanishing_term_gradient_is_refused",
        "bayesian_outer_loop_hook_returns_the_best_proposal",
        "term_gradients_match_autograd",
        "max_reference_weights_are_clipped_and_floored",
        "satisfied_term_gets_at_most_lambda_max",
    ],
}


def term_gradients(losses: dict, params: Iterable[torch.Tensor]) -> dict:
    """Flat grad_theta L_i for every term (retains the graph so every term can be differentiated)."""
    params = list(params)
    out = {}
    for name, value in losses.items():
        grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        flat = [torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                for p, g in zip(params, grads)]
        out[name] = torch.cat(flat).detach().cpu().numpy()
    return out


def lambda_hat(grads: dict, reference: str = "max", eps_g: float = 0.0,
               lam_min: float | None = None, lam_max: float | None = None) -> dict:
    """Eq. (12). `reference="max"` (r04): clip(max_j max|grad L_j| / (mean|grad L_i| + eps_g), lam_min, lam_max)
    for EVERY i. `reference=<term>` (r03 control): max|grad L_ref| / mean|grad L_i| for every i != ref."""
    if reference == "max":
        top = max(float(np.max(np.abs(g))) for g in grads.values())
        names = list(grads)
    else:
        top = float(np.max(np.abs(grads[reference])))
        names = [k for k in grads if k != reference]
    out = {}
    for name in names:
        mean = float(np.mean(np.abs(grads[name]))) + eps_g
        if mean == 0.0:
            raise ValueError(f"Eq. (12) is undefined for {name!r}: its mean gradient magnitude is zero")
        v = top / mean
        if lam_min is not None:
            v = max(v, lam_min)
        if lam_max is not None:
            v = min(v, lam_max)
        out[name] = v
    return out


def ema_update(lam: dict, lam_hat: dict, alpha: float) -> dict:
    """Eq. (13): lambda_i <- (1 - alpha) lambda_i + alpha lambda_hat_i."""
    return {k: (1.0 - alpha) * lam[k] + alpha * lam_hat[k] for k in lam}


class BlockWeights:
    """lambda that changes only at block boundaries. With `reference="max"` (r04) every term is
    weighted, floored and capped; with `reference="poisson"` (r03 control) Poisson stays at 1."""

    def __init__(self, names=("poisson", "bcn", "bcp", "n", "p"), alpha: float = 0.1,
                 reference: str = "max", on_update: Iterable[Callable[[], None]] = (),
                 eps_g: float = 1e-8, lam_min: float = 1e-2, lam_max: float = 1e4) -> None:
        self.reference = reference
        self.alpha = alpha
        self.eps_g, self.lam_min, self.lam_max = eps_g, lam_min, lam_max
        self.values = {k: 1.0 for k in names}
        self.max_seen = 1.0
        self.generation = 0
        self._in_block = False
        self._callbacks = list(on_update)

    def begin_block(self) -> None:
        self._in_block = True

    def update_inside_block(self, grads: dict) -> None:
        """Refused by design: lambda must stay fixed while a block accumulates curvature."""
        raise RuntimeError("lambda cannot be updated inside a block of iterations")

    def end_block(self, grads: dict) -> dict:
        self._in_block = False
        if self.reference == "max":
            lh = lambda_hat(grads, "max", self.eps_g, self.lam_min, self.lam_max)
            self.values = ema_update(self.values, {**self.values, **lh}, self.alpha)
            self.max_seen = max(self.max_seen, max(self.values.values()))
            self.generation += 1
            for cb in self._callbacks:
                cb()
            return dict(self.values)
        # A term with zero gradient is exactly satisfied (the contacts at the equilibrium start): Eq. (12)
        # is undefined for it, so its lambda is kept; every other term updates by Eqs. (12)-(13).
        live = {k: g for k, g in grads.items()
                if k == self.reference or float(np.mean(np.abs(g))) > 0.0}
        lh = lambda_hat(live, self.reference)
        new = {k: v for k, v in self.values.items() if k != self.reference}
        new.update(ema_update({k: new[k] for k in lh}, lh, self.alpha))
        self.values = {self.reference: 1.0, **new}
        self.generation += 1
        for cb in self._callbacks:
            cb()
        return dict(self.values)


def outer_loop_weights(propose: Callable[[list], dict], evaluate: Callable[[dict], float],
                       n_trials: int):
    """Section 5 alternative: an outer loop proposes weights, an inner training scores them.

    `propose(history)` returns a weight dict given the list of (weights, objective) so far;
    a Bayesian-optimization proposer plugs in here. Returns the best weights and the history.
    """
    history: list = []
    for _ in range(n_trials):
        lam = propose(history)
        history.append((lam, float(evaluate(lam))))
    best = min(history, key=lambda h: h[1])[0]
    return best, history
