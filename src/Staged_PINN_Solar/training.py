"""Training by voltage continuation, in blocks, with the evaluator of the protocol.

One run (`train`) = one (architecture, optimizer, loss, weighting, seed) of the experiments:

* the ansatz of Eq. (3) around `network.Architecture`; hard contacts, bounded carriers;
* the loss of Eq. (11) assembled from the residual summands of `residuals.py` evaluated at dPV's own
  nodes (so G_LDS, stored per node, needs no optics); sigma_i^2 (`loss.measure_reference_statistics`)
  and the ansatz bounds (`ansatz.measure_bounds`) are measured here from the TRAINING split, so no
  unit reads another step's output; the unnormalized arm uses sigma_i^2 = 1;
* the voltage enters only through Eq. (3)/(11): stage j trains at V_j = j * 0.05 V on the training
  devices whose stored curve reaches j, warm-started from stage j-1 (r10, Section 5);
* each stage is `n_blocks` blocks of `iters_per_block` iterations on ONE fixed collocation batch;
  lambda (Eqs. 12-13, `weights.BlockWeights`) changes only between blocks and L-BFGS memory is reset
  at each update only when the arm asks for it (E5's variable); L-BFGS is also reset when the voltage
  changes, since the objective then changes;
* after each stage the terminal current of Eq. (13) is evaluated on the test devices; the PCE of
  Eq. (14) comes from that curve (`pce.jv_summary`) and the metric is the relative PCE error in percent.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import ansatz as ansatz_mod
from . import carriers, data, diagnostics, loss, network, pce, residuals, weights
from .carriers import Material
from .config import Config
from .optimizers import LBFGS, SOAP
from .scaling import Scales
from .units import UnitSpec

__provenance__ = {
    "revision": "experiments-protocolo-experimental-de-la-pinn-por-etapas-sobre-dpv-v02.md",
    "sections": ["E1", "E2", "E3", "E4", "E5"],
    "experiments": ["E1", "E2", "E3", "E4", "E5"],
    "invariants": [
        "residual_terms_match_the_analytic_derivatives",
        "statistics_and_bounds_come_from_the_training_split_only",
        "batches_depend_on_seed_and_stage_but_not_on_the_arm",
        "arms_with_the_same_seed_share_the_initialization",
        "unnormalized_arm_uses_unit_scales",
        "training_step_decreases_loss_on_tiny_problem",
        "soap_runs_per_parameter_and_decreases_loss",
        "lambda_changes_only_between_blocks",
        "lbfgs_memory_is_reset_at_lambda_updates_only_in_the_reset_arm",
        "divergence_is_flagged_and_stops_the_run",
        "conditioning_diagnostics_are_recorded_at_init_marks_and_block_ends",
        "same_seed_reproduces_the_run",
        "error_is_relative_pce_percent_per_test_device",
    ],
}

TERMS = loss.TERMS


def device() -> torch.device:
    """cuda when present, unless STAGED_PINN_DEVICE says otherwise (e.g. 'cpu')."""
    want = os.environ.get("STAGED_PINN_DEVICE")
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- data banks ------------------------------------------------------------------------------------
def const_material(scales: Scales) -> Material:
    """The E1 material constants as scalars (only the doping step varies per node), via `data.device_material`."""
    stub = SimpleNamespace(x_debye=np.array([0.0, 1.0]), xi={"L_n": 1.0, "Nd_n": 1e17, "Na_p": 1e17})
    m = data.device_material(stub, scales)
    from dataclasses import fields
    return Material(**{f.name: np.float64(getattr(m, f.name)[0]) for f in fields(m)})


@dataclass
class Bank:
    """The devices of one split in the arrays the training needs (nothing is read from disk after this)."""

    uids: list
    xi_hat: np.ndarray        # (D, n_xi)
    delta: np.ndarray         # (D,)
    x_hat: np.ndarray         # (D, N)
    G: np.ndarray             # (D, N)
    Ndop: np.ndarray          # (D, N)
    phi_eq0: np.ndarray       # (D,)
    phi_eq1: np.ndarray
    n_V: np.ndarray           # (D,)
    pce: np.ndarray           # (D,) dPV efficiency
    p_in_std: np.ndarray      # (D,)

    def __len__(self) -> int:
        return len(self.uids)


def _union_specs(root: Path) -> tuple:
    """The four parameters of E1B (E1A fixes L_n, L_p): the network input is the same for every device."""
    import json

    import h5py

    with h5py.File(data.source_path(root, data.SOURCES[("E1B", 0)]), "r") as f:
        return tuple((p["nombre"], p["escala"], float(p["min"]), float(p["max"]))
                     for p in json.loads(f.attrs["parametros_xi"]))


class _BankBuilder:
    def __init__(self, scales: Scales, specs: tuple) -> None:
        self.scales, self.specs = scales, specs
        self.rows = []

    def add(self, dev) -> None:
        n_v = len(dev.V)
        if not np.allclose(dev.V, 0.05 * np.arange(n_v), atol=1e-9):
            raise ValueError(f"{dev.ref.uid}: the voltage grid is not V_j = 0.05 j")
        mat = data.device_material(dev, self.scales)
        self.rows.append((dev.ref.uid, ansatz_mod.normalize_xi(dev.xi, self.specs), dev.delta(self.scales),
                          dev.x_hat, dev.G, mat.Ndop, float(dev.phi_eq[0]), float(dev.phi_eq[-1]), n_v,
                          float(dev.pce), float(dev.p_in_std)))

    def finish(self) -> Bank:
        r = self.rows
        n_nodes = {len(x[3]) for x in r}
        if len(n_nodes) != 1:
            raise ValueError(f"devices have different node counts {sorted(n_nodes)}")
        col = lambda k, dt=np.float64: np.asarray([x[k] for x in r], dtype=dt)
        return Bank(uids=[x[0] for x in r], xi_hat=np.stack([x[1] for x in r]), delta=col(2),
                    x_hat=np.stack([x[3] for x in r]), G=np.stack([x[4] for x in r]),
                    Ndop=np.stack([x[5] for x in r]), phi_eq0=col(6), phi_eq1=col(7),
                    n_V=col(8, np.int64), pce=col(9), p_in_std=col(10))


@dataclass
class Prepared:
    scales: Scales
    sigma2: dict
    bounds: ansatz_mod.AnsatzBounds
    train: Bank
    val: Bank
    test: Bank
    n_xi: int
    material: Material
    contacts: object
    root: str
    stats: object = None      # the ReferenceStatistics the sigma_i^2 come from (f_i, rho_i, counts)
    refs: dict = None         # split name -> list of DeviceRef


_PREPARED: dict = {}


def prepare(root=None, config: Config | None = None, progress=None) -> Prepared:
    """Measure sigma_i^2 and the ansatz bounds on the training split and bank the three splits (cached per process)."""
    cfg = config or Config()
    root = Path(root) if root else data.default_root()
    key = (str(root), cfg.split_seed, cfg.max_devices, cfg.qf_box_margin)
    if key in _PREPARED:
        return _PREPARED[key]
    scales = Scales()
    split = data.split_devices(data.inventory(root), seed=cfg.split_seed)
    cap = (lambda refs: refs[:cfg.max_devices]) if cfg.max_devices else (lambda refs: refs)
    train_refs, val_refs, test_refs = cap(split.train), cap(split.val), cap(split.test)
    specs = _union_specs(root)
    stats = loss.measure_reference_statistics(train_refs, scales, progress=progress)
    sigma2 = stats.sigma2()
    builder = _BankBuilder(scales, specs)

    def tap(refs, b):
        for ref in refs:
            dev = data.load_device(ref)
            b.add(dev)
            yield dev

    bounds = ansatz_mod.measure_bounds(tap(train_refs, builder), scales, margin=cfg.qf_box_margin)
    banks = [builder.finish()]
    for refs in (val_refs, test_refs):
        b = _BankBuilder(scales, specs)
        for _ in tap(refs, b):
            pass
        banks.append(b.finish())
    prep = Prepared(scales, sigma2, bounds, banks[0], banks[1], banks[2], len(specs),
                    const_material(scales), data.device_contacts(scales), str(root), stats,
                    {"train": train_refs, "val": val_refs, "test": test_refs})
    _PREPARED[key] = prep
    return prep


# --- collocation batches ------------------------------------------------------------------------
def _t(a, dev):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)


def make_batch(bank: Bank, stage: int, seed: int, cfg: Config, dev) -> dict:
    """Interior nodes (and both contacts) of the devices whose stored curve reaches `stage`: a function of (seed, stage)."""
    eligible = np.nonzero(bank.n_V > stage)[0]
    rng = np.random.default_rng([int(seed), int(stage), 17])
    pick = np.sort(rng.choice(eligible, size=min(cfg.n_devices_per_batch, eligible.size), replace=False))
    n_nodes = bank.x_hat.shape[1]
    k = min(cfg.n_nodes_per_device, n_nodes - 2)
    idx = np.argsort(rng.random((pick.size, n_nodes - 2)), axis=1)[:, :k] + 1
    rows = pick[:, None]
    rep = lambda a: np.repeat(a[pick], k, axis=0).reshape(-1, a.shape[1] if a.ndim > 1 else 1)
    col = lambda a: np.repeat(a[pick], k)[:, None]
    rep2 = lambda a: np.repeat(a[pick], 2)[:, None]
    return {
        "x": _t(bank.x_hat[rows, idx].reshape(-1, 1), dev),
        "G": _t(bank.G[rows, idx].reshape(-1, 1), dev),
        "ndop": _t(bank.Ndop[rows, idx].reshape(-1, 1), dev),
        "xi": _t(rep(bank.xi_hat), dev), "delta": _t(col(bank.delta), dev),
        "peq0": _t(col(bank.phi_eq0), dev), "peq1": _t(col(bank.phi_eq1), dev),
        "xc": _t(np.tile([[0.0], [1.0]], (pick.size, 1)), dev),
        "xic": _t(np.repeat(bank.xi_hat[pick], 2, axis=0), dev), "deltac": _t(rep2(bank.delta), dev),
        "peq0c": _t(rep2(bank.phi_eq0), dev), "peq1c": _t(rep2(bank.phi_eq1), dev),
        "phi_eq": _t(np.stack([bank.phi_eq0[pick], bank.phi_eq1[pick]], axis=1), dev),
        "devices": pick,
    }


# --- residuals and loss ----------------------------------------------------------------------------
def residual_terms(model, b: dict, V_tilde: float, mat: Material, contacts) -> dict:
    """R_i of Eqs. (5), (7), (9) on a batch, by autograd on the ansatz `model(x, xi, peq0, peq1, V_tilde)`."""
    x = b["x"].clone().requires_grad_(True)
    phi, phi_n, phi_p = model(x, b["xi"], b["peq0"], b["peq1"], V_tilde)
    m = replace(mat, Ndop=b["ndop"]).like(phi)
    n, p = carriers.electron_density(phi, phi_n, m), carriers.hole_density(phi, phi_p, m)
    R = carriers.recombination(n, p, m)
    d = lambda y, v: residuals.autograd_derivative(y, v)
    delta = b["delta"]
    flux_dx = d(m.eps_r * d(phi, x), x)
    Jn = residuals.current_density(m.mu_n, n, d(phi_n, x), delta)
    Jp = residuals.current_density(m.mu_p, p, d(phi_p, x), delta)
    terms = {"poisson": residuals.poisson_terms(flux_dx, n, p, m.Ndop, delta),
             "n": residuals.continuity_n_terms(d(Jn, x), b["G"], R, delta),
             "p": residuals.continuity_p_terms(d(Jp, x), b["G"], R, delta)}
    xc = b["xc"].clone().requires_grad_(True)
    phic, phin_c, phip_c = model(xc, b["xic"], b["peq0c"], b["peq1c"], V_tilde)
    mc = replace(mat, Ndop=0.0 * phic).like(phic)
    nc, pc = carriers.electron_density(phic, phin_c, mc), carriers.hole_density(phic, phip_c, mc)
    Jnc = residuals.current_density(mc.mu_n, nc, d(phin_c, xc), b["deltac"]).reshape(-1, 2)
    Jpc = residuals.current_density(mc.mu_p, pc, d(phip_c, xc), b["deltac"]).reshape(-1, 2)
    n_eq, p_eq = carriers.equilibrium_densities(b["phi_eq"], mc)
    nc, pc = nc.reshape(-1, 2), pc.reshape(-1, 2)
    terms["bcn"] = residuals.contact_n_terms(Jnc, nc, n_eq, np.array([contacts.S_n0, contacts.S_n1]))
    terms["bcp"] = residuals.contact_p_terms(Jpc, pc, p_eq, np.array([contacts.S_p0, contacts.S_p1]))
    return {k: residuals.residual(v) for k, v in terms.items()}


class Objective:
    """Eq. (11) on one fixed batch at one voltage."""

    def __init__(self, model, batch, V_tilde, sigma2, mat, contacts) -> None:
        self.model, self.batch, self.V, self.sigma2 = model, batch, V_tilde, sigma2
        self.mat, self.contacts = mat, contacts

    def losses(self) -> dict:
        return loss.normalized_losses(residual_terms(self.model, self.batch, self.V, self.mat, self.contacts),
                                      self.sigma2)

    def total(self, lam: dict):
        return loss.total_loss(self.losses(), lam)


def _flat(tensors) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _set(params, vec: torch.Tensor) -> None:
    with torch.no_grad():
        o = 0
        for p in params:
            p.copy_(vec[o:o + p.numel()].view_as(p))
            o += p.numel()


# --- evaluation (Eqs. 13-14) ------------------------------------------------------------------------
def evaluate_currents(model, bank: Bank, V_tilde: float, mat: Material, scales: Scales, dev, chunk: int) -> np.ndarray:
    """J(V) = J_0 mean_nodes (J_n + J_p) of the network at dPV's nodes for every device of `bank` [A/cm^2]."""
    out = np.empty(len(bank))
    N = bank.x_hat.shape[1]
    for s in range(0, len(bank), chunk):
        sl = slice(s, min(s + chunk, len(bank)))
        c = bank.x_hat[sl].shape[0]
        x = _t(bank.x_hat[sl].reshape(-1, 1), dev).requires_grad_(True)
        rep = lambda a: _t(np.repeat(a[sl], N)[:, None], dev)
        with torch.enable_grad():
            phi, phi_n, phi_p = model(x, _t(np.repeat(bank.xi_hat[sl], N, axis=0), dev),
                                      rep(bank.phi_eq0), rep(bank.phi_eq1), V_tilde)
            m = mat.like(phi)
            n, p = carriers.electron_density(phi, phi_n, m), carriers.hole_density(phi, phi_p, m)
            dn = torch.autograd.grad(phi_n.sum(), x, retain_graph=True)[0]
            dp = torch.autograd.grad(phi_p.sum(), x)[0]
        delta = rep(bank.delta)
        j = residuals.current_density(m.mu_n, n, dn, delta) + residuals.current_density(m.mu_p, p, dp, delta)
        out[sl] = (scales.J0 * j.detach().reshape(c, N).mean(dim=1)).cpu().numpy()
    return out


def errors_from_curves(J: np.ndarray, bank: Bank, stages_done: int, scales: Scales) -> dict:
    """uid -> relative PCE error (%) from each device's predicted J-V curve (NaN where it cannot be formed)."""
    out = {}
    for d, uid in enumerate(bank.uids):
        n = min(int(bank.n_V[d]), stages_done)
        v = 0.05 * np.arange(n)
        s = pce.jv_summary(v, J[d, :n], float(bank.p_in_std[d]))
        out[uid] = pce.relative_error_percent(s["pce"], float(bank.pce[d])) if np.isfinite(s["pce"]) else float("nan")
    return out


# --- the run -----------------------------------------------------------------------------------------
def build_network(spec: UnitSpec, cfg: Config, prep: Prepared):
    """The seeded architecture of the unit, set to the equilibrium start unless `cfg.start == "random"`."""
    if cfg.start not in ("equilibrium", "random"):
        raise ValueError(f"unknown start {cfg.start!r}")
    arch = network.build(spec.arch, prep.n_xi, cfg.width, cfg.depth, spec.seed)
    return network.equilibrium_start(arch, prep.bounds) if cfg.start == "equilibrium" else arch


def initial_parameters(spec: UnitSpec, cfg: Config, prep: Prepared) -> list:
    """The initial parameter tensors of (architecture, seed): identical for every arm of that pair."""
    return [p.detach().clone() for p in build_network(spec, cfg, prep).parameters()]


def _soap_block(params, soaps, objective, lam, iters, history) -> None:
    for _ in range(iters):
        total = objective.total(lam)
        history.append(float(total.detach()))
        grads = torch.autograd.grad(total, params)
        with torch.no_grad():
            for p, s, g in zip(params, soaps, grads):
                w = p.detach().reshape(p.shape[0], -1)
                p.copy_(s.step(w, g.reshape(w.shape)).reshape(p.shape))


def train(spec: UnitSpec, prep: Prepared, cfg: Config, eval_val: bool = False, lam_fixed: dict | None = None,
          progress=None) -> dict:
    """One run of the experiments. `lam_fixed` (the outer loop of E4) fixes lambda for the whole run."""
    t0, dev = time.time(), device()
    model = ansatz_mod.Ansatz(build_network(spec, cfg, prep).to(dev), prep.bounds)
    params = list(model.parameters())
    sigma2 = dict(prep.sigma2) if spec.loss == "normalized" else {k: 1.0 for k in prep.sigma2}
    lbfgs = LBFGS(memory=cfg.lbfgs_memory)
    on_update = [lbfgs.reset] if (spec.weighting == "gradient" and spec.reset_memory) else []
    bw = weights.BlockWeights(alpha=cfg.alpha_lambda, on_update=on_update)
    if lam_fixed is not None:
        bw.values = {"poisson": 1.0, **{k: float(v) for k, v in lam_fixed.items()}}
    soaps = [SOAP(lr=cfg.soap_lr, precondition_frequency=cfg.soap_precondition_frequency, warmup=cfg.soap_warmup) for _ in params] \
        if spec.optimizer == "soap" else None
    diag_on = spec.step == "e1"
    K = min(int(prep.train.n_V.max()), cfg.max_voltages, cfg.max_stages or 10 ** 9)
    J_test = np.full((len(prep.test), K), np.nan)
    J_val = np.full((len(prep.val), K), np.nan) if eval_val else None
    blocks, diag_records, stages, diverged, stages_done = [], [], [], False, 0
    early = {}
    lambda_errors = []

    for j in range(K):
        V_tilde = j * cfg.voltage_step / prep.scales.V_th
        obj = Objective(model, make_batch(prep.train, j, spec.seed, cfg, dev), V_tilde, sigma2, prep.material,
                        prep.contacts)
        lbfgs.reset()                                   # the objective changed: the voltage moved

        def fun(x):
            _set(params, x)
            total = obj.total(bw.values)
            g = torch.autograd.grad(total, params)
            return total.detach(), _flat(g).detach()

        def diagnose(at: str, it: int, hessian: bool) -> None:
            losses = obj.losses()
            grads = weights.term_gradients(losses, params)
            rec = {"stage": j, "at": at, "iteration": it, "lambda": dict(bw.values)}
            sh = diagnostics.gradient_shares(grads, bw.values)
            rec.update(shares=sh, max_share=max(sh.values()),
                       log10_norm_ratio=diagnostics.log10_norm_ratio(grads, bw.values))
            if hessian:
                total = loss.total_loss(losses, bw.values)
                g = _flat(torch.autograd.grad(total, params, create_graph=True))

                def hvp(v):
                    hv = torch.autograd.grad(g @ v.to(dev), params, retain_graph=True)
                    return _flat(hv).detach().cpu()

                lam_max = diagnostics.power_iteration(hvp, g.numel(), cfg.hvp_power_iters, spec.seed)
                tr = diagnostics.hutchinson_trace(hvp, g.numel(), cfg.hvp_probes, spec.seed) / g.numel()
                rec.update(lambda_max=lam_max, mean_trace=tr, curvature_ratio=lam_max / tr if tr else float("nan"))
            diag_records.append(rec)

        x = _flat([p.detach() for p in params])
        loss_start_stage = float(fun(x)[0])
        if diag_on and j == 0:
            diagnose("init", 0, hessian=True)
            early[0] = diag_records[-1]["max_share"]
        for b in range(cfg.n_blocks):
            bw.begin_block()
            lam_block = dict(bw.values)
            x = _flat([p.detach() for p in params])
            loss_start = float(fun(x)[0])
            history = []
            if spec.optimizer == "lbfgs":
                marks = sorted({m for m in cfg.early_marks if 0 < m < cfg.iters_per_block}) \
                    if (diag_on and j == 0 and b == 0) else []
                done, info = 0, {"iterations": 0, "grad_norm": float("nan")}
                for stop in marks + [cfg.iters_per_block]:
                    x, info_seg = lbfgs.minimize(x, fun, max_iter=stop - done)
                    history += info_seg["history"]
                    info = {"iterations": info["iterations"] + info_seg["iterations"],
                            "grad_norm": info_seg["grad_norm"]}
                    done = stop
                    _set(params, x)
                    if stop in marks:
                        diagnose("mark", stop, hessian=False)
                        early[stop] = diag_records[-1]["max_share"]
                iters_done, grad_norm = info["iterations"], info["grad_norm"]
            else:
                _soap_block(params, soaps, obj, bw.values, cfg.iters_per_block, history)
                iters_done = cfg.iters_per_block
                grad_norm = float("nan")
            x = _flat([p.detach() for p in params])
            f_end = float(fun(x)[0])
            finite = bool(np.isfinite(f_end)) and bool(torch.isfinite(x).all())
            blk = {"stage": j, "block": b, "iterations": iters_done, "loss_start": loss_start, "loss_end": f_end,
                   "grad_norm": grad_norm, "lambda": lam_block, "lambda_used_constant": True,
                   "lbfgs_memory": lbfgs.history_size}
            if diag_on and finite:
                diagnose("block_end", cfg.iters_per_block, hessian=True)
            if not finite or f_end > cfg.divergence_factor * max(loss_start_stage, 1e-300):
                blk["diverged"] = True
                blocks.append(blk)
                diverged = True
                break
            if spec.weighting == "gradient":
                grads = weights.term_gradients(obj.losses(), params)
                try:
                    bw.end_block(grads)
                except ValueError as e:                 # Eq. (12) undefined: keep lambda, record it
                    lambda_errors.append({"stage": j, "block": b, "error": str(e)})
            blocks.append(blk)
        stages.append({"stage": j, "V": j * cfg.voltage_step, "n_devices": int(len(obj.batch["devices"])),
                       "loss_start": loss_start_stage})
        if diverged:
            stages_done = j                       # the stage that diverged is not a finished stage
            break
        mat = prep.material
        J_test[:, j] = evaluate_currents(model, prep.test, V_tilde, mat, prep.scales, dev, cfg.eval_chunk)
        if eval_val:
            J_val[:, j] = evaluate_currents(model, prep.val, V_tilde, mat, prep.scales, dev, cfg.eval_chunk)
        stages_done = j + 1
        if progress:
            progress(spec, j + 1, K)

    def errs(bank, J):
        if diverged:
            return {uid: float("nan") for uid in bank.uids}
        return errors_from_curves(J, bank, stages_done, prep.scales)

    seconds = time.time() - t0
    result = {"sigma2_used": sigma2, "blocks": blocks, "stages": stages, "stages_done": stages_done,
              "stages_planned": K, "diverged": diverged, "lambda_final": dict(bw.values),
              "lambda_update_errors": lambda_errors, "per_device_error": errs(prep.test, J_test),
              "seconds": seconds, "device": str(dev),
              "gpu_hours": seconds / 3600.0 if dev.type == "cuda" else 0.0,
              "bounds": {"D_phi": prep.bounds.D_phi, "phi_n": list(prep.bounds.phi_n), "phi_p": list(prep.bounds.phi_p)}}
    if eval_val:
        result["per_device_error_val"] = errs(prep.val, J_val)
    if diag_on:
        init = diag_records[0] if diag_records else {}
        result["diagnostics"] = {
            "records": diag_records, "early_max_gradient_share": max(early.values()) if early else float("nan"),
            "early_share_by_iteration": early,
            "log10_norm_ratio_init": init.get("log10_norm_ratio"), "curvature_ratio_init": init.get("curvature_ratio"),
            "lambda_max_init": init.get("lambda_max"), "mean_trace_init": init.get("mean_trace")}
    else:
        result["diagnostics"] = None
    return result
