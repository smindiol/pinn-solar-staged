"""The ONE configuration of the experiments: every number the protocol does not fix lives here.

Each field is tagged in its comment:

* ``FIXED`` -- stated by the experiments protocol (v02) or by r03/r10; changing it changes the experiment.
* ``PROVISIONAL`` -- the protocol and r03 leave it open; the value is a provisional default taken from the
  source named in the comment (previous TDG code, a library default, or r10) and is awaiting the
  operator's decision. The record of every unit carries the whole configuration (`Config.to_dict`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

__provenance__ = {
    "revision": "experiments-protocolo-experimental-de-la-pinn-por-etapas-sobre-dpv-v02.md",
    "sections": ["E1", "E2", "E3", "E4", "E5"],
    "experiments": ["E1", "E2", "E3", "E4", "E5"],
    "invariants": [
        "every_open_parameter_is_marked_provisional",
        "config_roundtrips_into_the_unit_record",
    ],
}


@dataclass(frozen=True)
class Config:
    # --- fixed by the protocol (parameters table of experiments v02) -----------------------------
    seeds: tuple = (0, 1, 2, 3, 4)                 # FIXED: training seeds
    split_seed: int = 42                           # FIXED: device split seed (80/10/10)
    bootstrap_resamples: int = 10000               # FIXED
    confidence_level: float = 0.95                 # FIXED
    holm_alpha: float = 0.05                       # FIXED: common decision rule
    width: int = 256                               # FIXED: network width
    depth: int = 4                                 # FIXED: hidden layers
    activation: str = "tanh_gauss"                 # FIXED: tanh(z) exp(-z^2/2) of zhang2025
    voltage_step: float = 0.05                     # FIXED by r10 Eq. 14 (dPV vincr, 20 values per volt)
    max_voltages: int = 100                        # FIXED by dPV's sweep (contacts.MAX_STEPS of the previous code)

    # --- NOT fixed by the protocol: PROVISIONAL, operator to decide ---------------------------
    n_blocks: int = 5                              # PROVISIONAL: blocks per voltage; previous TDG lbfgs_iters 500 / gummel_block 100
    iters_per_block: int = 100                     # PROVISIONAL: iterations per block; previous TDG TrainConfig.gummel_block = 100
    n_devices_per_batch: int = 64                  # PROVISIONAL: previous TDG TrainConfig.data_batch = 64
    n_nodes_per_device: int = 32                   # PROVISIONAL: 64 devices x 32 nodes ~ previous TDG n_q = 2000 interior points
    alpha_lambda: float = 0.1                      # PROVISIONAL: weights.BlockWeights default (wang2020 anneals with 0.9)
    lbfgs_memory: int = 10                         # PROVISIONAL: optimizers.LBFGS default
    soap_lr: float = 1e-4                          # PROVISIONAL: measured from the equilibrium start (Adam proxy, 256x4, 200 its): 1e-4 with warmup reaches 0.35 from 0.43; 3e-3 down to 3e-5 without warmup diverge
    soap_warmup: int = 100                         # PROVISIONAL: linear lr warmup steps (Vyas et al. warm up SOAP); no warmup spikes the loss 400x at lr 1e-5
    soap_precondition_frequency: int = 10          # PROVISIONAL: optimizers.SOAP default (Vyas et al. default)
    pce_tolerance: float = 1e-6                    # PROVISIONAL: relative PCE error of pce_check; previous TDG test asserts 1e-12 absolute
    pce_profile_tolerance: float = 5e-2            # PROVISIONAL: same, for the evaluator that differentiates the stored potentials
    bo_trials: int = 10                            # PROVISIONAL: Bayesian-optimization budget (outer-loop evaluations per unit); no prior source
    bo_startup_trials: int = 5                     # PROVISIONAL: optuna GPSampler random start (its own default is 10)
    bo_log10_lambda: tuple = (-2.0, 4.0)           # PROVISIONAL: search range of log10(lambda_i), i != Poisson; no prior source
    bo_diverged_penalty: float = 1.0e3             # PROVISIONAL: objective assigned to a diverged trial (percent error)
    bo_seed: int = 0                               # PROVISIONAL: sampler seed (the training seed varies with the unit)
    divergence_factor: float = 1.0e3               # PROVISIONAL: a run diverges if non-finite or loss > factor x loss at the stage start
    early_marks: tuple = (0, 10, 50)               # PROVISIONAL: iterations of the first block at which the early gradient share is read
    hvp_power_iters: int = 20                      # PROVISIONAL: power-iteration steps for lambda_max
    hvp_probes: int = 10                           # PROVISIONAL: Hutchinson probes for the trace
    start: str = "equilibrium"                     # PROVISIONAL: phi_n = phi_p = 0, phi = phi_lin at the first stage (previous TDG qf_start="equilibrium"); "random" = plain seeded init
    qf_box_margin: float = 0.05                    # PROVISIONAL: quasi-Fermi boxes widened to contain 0 and by 5% of their range; 2% and 5% gave the same initial loss
    wall_budget_seconds: float = 39600.0           # PROVISIONAL: Kaggle stops a session at 12 h; 11 h leaves room for setup, the dataset download and the output upload
    eval_chunk: int = 32                           # PROVISIONAL: devices per evaluation chunk (memory only, no effect on results)

    # --- rehearsal scale (smoke), never a decision of the experiment --------------------------
    max_devices: int | None = None                 # cap per split; None = the whole split
    max_stages: int | None = None                  # cap on voltage stages; None = every stage the data holds
    smoke: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)


#: Rehearsal scale: 8 devices per split, tiny network, a handful of iterations. Exercises every code path.
SMOKE = dict(seeds=(0,), bootstrap_resamples=200, max_devices=8, max_stages=4, n_blocks=2, iters_per_block=3, n_devices_per_batch=8,
             n_nodes_per_device=8, width=16, depth=2, hvp_power_iters=3, hvp_probes=2,
             early_marks=(0, 1), bo_trials=3, bo_startup_trials=2, smoke=True)


def default() -> Config:
    return Config()


def smoke() -> Config:
    return Config(**SMOKE)
