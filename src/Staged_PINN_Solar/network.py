"""The two architectures of the protocol (experiments v03, "Arquitecturas del protocolo").

* A: a Kaiming-initialized perceptron of width 256 and depth 4 with the activation of zhang2025,
  tanh(z) exp(-z^2/2), no Fourier features, ONE network for the three outputs (phi, phi_n, phi_p).
* B: the same perceptron, one separate network per field.

Inputs are (x_hat, xi_hat); the raw outputs are mapped to the ranges Eq. (3)'s ansatz assumes:
u_phi = tanh(.) in [-1, 1] (the interior correction is then bounded by D_phi) and
u_n, u_p = sigmoid(.) in [0, 1]. Everything is float64.

`equilibrium_start` sets the output layer so that the run starts from the known V = 0 solution of the
quasi-Fermi potentials (phi_n = phi_p = 0) and from phi = phi_lin, instead of from a random point of the
boxes. From a random start the densities exp(+-(phi - phi_n,p)) are off by tens of thermal voltages and
the normalized loss starts near 1e34-1e39; from this start every normalized term is O(0.1) (measured on
dPV, training split, seed 0). The previous TDG used the same start (`qf_start="equilibrium"`).
"""

from __future__ import annotations

import torch

__provenance__ = {
    "revision": "experiments-protocolo-experimental-de-la-pinn-por-etapas-sobre-dpv-v03.md",
    "sections": ["E1", "E2", "E3", "E4", "E5"],
    "experiments": ["E1", "E2", "E3", "E4", "E5"],
    "invariants": [
        "architecture_a_is_one_net_for_three_outputs",
        "architecture_b_is_one_net_per_field",
        "protocol_size_is_256_by_4",
        "outputs_are_in_the_ranges_the_ansatz_assumes",
        "activation_is_tanh_times_gaussian",
        "same_seed_gives_same_initialization_and_seeds_differ",
        "float64",
        "equilibrium_start_gives_zero_quasi_fermi_and_linear_potential",
    ],
}


def tanh_gauss(z: torch.Tensor) -> torch.Tensor:
    """zhang2025's activation: tanh(z) exp(-z^2 / 2)."""
    return torch.tanh(z) * torch.exp(-0.5 * z * z)


class _Act(torch.nn.Module):
    def forward(self, z):
        return tanh_gauss(z)


def _mlp(n_in: int, n_out: int, width: int, depth: int) -> torch.nn.Sequential:
    layers, d = [], n_in
    for _ in range(depth):
        lin = torch.nn.Linear(d, width, dtype=torch.float64)
        torch.nn.init.kaiming_normal_(lin.weight)      # Kaiming (He) normal, fan-in
        torch.nn.init.zeros_(lin.bias)
        layers += [lin, _Act()]
        d = width
    out = torch.nn.Linear(d, n_out, dtype=torch.float64)
    torch.nn.init.kaiming_normal_(out.weight)
    torch.nn.init.zeros_(out.bias)
    return torch.nn.Sequential(*layers, out)


class Architecture(torch.nn.Module):
    """`arch(x_hat, xi_hat) -> (B, 3)` as `ansatz.Architecture` expects."""

    def __init__(self, kind: str, n_xi: int, width: int, depth: int) -> None:
        super().__init__()
        if kind not in ("A", "B"):
            raise ValueError(f"architecture must be 'A' or 'B', got {kind!r}")
        self.kind = kind
        n_in = 1 + n_xi
        self.nets = torch.nn.ModuleList(
            [_mlp(n_in, 3, width, depth)] if kind == "A" else [_mlp(n_in, 1, width, depth) for _ in range(3)])

    def forward(self, x_hat, xi_hat):
        z = torch.cat([x_hat, xi_hat], dim=1)
        raw = self.nets[0](z) if self.kind == "A" else torch.cat([net(z) for net in self.nets], dim=1)
        return torch.cat([torch.tanh(raw[:, 0:1]), torch.sigmoid(raw[:, 1:2]), torch.sigmoid(raw[:, 2:3])], dim=1)


def build(kind: str, n_xi: int, width: int = 256, depth: int = 4, seed: int = 0) -> Architecture:
    """A network whose initialization is a function of (kind, seed) only, so arms share it."""
    state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        return Architecture(kind, n_xi, width, depth)
    finally:
        torch.random.set_rng_state(state)


def equilibrium_start(arch: Architecture, bounds) -> Architecture:
    """Zero the output rows and set the biases so that u_phi = 0 and phi_n = phi_p = 0 for every input.

    Needs Y_min < 0 < Y_max for both quasi-Fermi boxes (`ansatz.measure_bounds` guarantees it). Only the
    last layer changes: the hidden layers keep their seeded Kaiming initialization, and the zeroed rows
    still receive gradient through their weights.
    """
    targets = []
    for lo, hi in (bounds.phi_n, bounds.phi_p):
        if not lo < 0.0 < hi:
            raise ValueError(f"equilibrium start needs Y_min < 0 < Y_max, got ({lo}, {hi})")
        u = -lo / (hi - lo)
        targets.append(float(torch.logit(torch.tensor(u, dtype=torch.float64))))
    biases = [0.0] + targets                       # u_phi = tanh(0) = 0; u_k = sigmoid(logit(u_k))
    with torch.no_grad():
        if arch.kind == "A":
            last = arch.nets[0][-1]
            last.weight.zero_()
            last.bias.copy_(torch.tensor(biases, dtype=last.bias.dtype))
        else:
            for net, b in zip(arch.nets, biases):
                net[-1].weight.zero_()
                net[-1].bias.fill_(b)
    return arch
