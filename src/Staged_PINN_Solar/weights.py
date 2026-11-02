"""Loss-weight optimization (Eqs. 12-13) and the outer-loop alternative hook (section 5).

The weights follow the learning-rate annealing of wang2020: Poisson is the
reference term, every other weight chases max|grad L_Poisson| / mean|grad L_i|
through an exponential moving average. A quasi-Newton method needs a fixed
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
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r03.md",
    "sections": ["5"],
    "equations": ["12", "13"],
    "invariants": [
        "lambda_hat_balances_mean_gradient_magnitudes",
        "moving_average_update_is_convex_and_has_the_fixed_point",
        "poisson_weight_is_the_fixed_reference",
        "weights_change_only_between_blocks",
        "update_notifies_the_optimizer_to_reset_its_memory",
        "vanishing_term_gradient_is_refused",
        "bayesian_outer_loop_hook_returns_the_best_proposal",
        "term_gradients_match_autograd",
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


def lambda_hat(grads: dict, reference: str = "poisson") -> dict:
    """Eq. (12): max|grad L_ref| / mean|grad L_i| for every i != reference."""
    top = float(np.max(np.abs(grads[reference])))
    out = {}
    for name, g in grads.items():
        if name == reference:
            continue
        mean = float(np.mean(np.abs(g)))
        if mean == 0.0:
            raise ValueError(f"Eq. (12) is undefined for {name!r}: its mean gradient magnitude is zero")
        out[name] = top / mean
    return out


def ema_update(lam: dict, lam_hat: dict, alpha: float) -> dict:
    """Eq. (13): lambda_i <- (1 - alpha) lambda_i + alpha lambda_hat_i."""
    return {k: (1.0 - alpha) * lam[k] + alpha * lam_hat[k] for k in lam}


class BlockWeights:
    """lambda that changes only at block boundaries; Poisson stays at 1."""

    def __init__(self, names=("poisson", "bcn", "bcp", "n", "p"), alpha: float = 0.1,
                 reference: str = "poisson", on_update: Iterable[Callable[[], None]] = ()) -> None:
        self.reference = reference
        self.alpha = alpha
        self.values = {k: 1.0 for k in names}
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
        lh = lambda_hat(grads, self.reference)
        new = ema_update({k: v for k, v in self.values.items() if k != self.reference}, lh, self.alpha)
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
