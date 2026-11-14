"""E6 of experiments v04 (r05, Sections 4-6): phased training with fixed weights chosen by Optuna.

Four remote stages, one per account, each a notebook calling `run_from_environment(<stage>)`:

* ``p``: data pretraining by voltage continuation (Eq. 18 only, Adam), the Eq. 12 centre evaluated once at the
  pretrained network, and the first part of the Optuna study (TPE, median pruner, journal file);
* ``q``: the same study continued from the journal and the pretrained network handed over in ``handoff-e6/``;
* ``h``: the single smoke run of the final recipe with the chosen weights (seed 0, stages 0-2);
* ``f``: the final runs, one per seed: pretraining, then per stage Adam -> L-BFGS from the pretrained network of
  that stage on the SAME loss (Eq. 11 + Eq. 18) with the SAME fixed weights lambda*.

Everything reused from E1 is reused as is: `training.prepare` (sigma_i^2, bounds, splits), the residuals and the
per-node relative loss of `training.Objective`, the collocation batch of `training.make_batch` (same seed and
stage give the same nodes), `training.evaluate_currents` and `training.errors_from_curves` (relative PCE error).
Telemetry: `<tag>.history.jsonl` every 25 steps, `<tag>.progress.json`, one flushed progress line per 25 steps
with an ETA, PNG plots, per-stage checkpoints plus one every 15 minutes inside Adam, a clean stop before the
wall budget.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import ansatz as ansatz_mod
from . import carriers, data, diagnostics, pce, steps, training, weights
from .config import Config
from .units import UnitSpec

__provenance__ = {
    "revision": "experiments-protocolo-experimental-de-la-pinn-por-etapas-sobre-dpv-v04.md",
    "sections": ["E6"],
    "experiments": ["E6"],
    "equations": [12, 13, 18, 19, 20, 21, 22, 23, 24],
    "invariants": [],
}
# r06 (proposal): Eq. 13 scores a trial by the terminal-current error (normalized by J_sc) instead of the field
# error of Eq. 18, and both the study and the final training start from theta_prev, the final networks of the r05
# run (one per seed and stage, handed over as handoff-e6/prev-s<seed>.pt).

# --- the parameters table of E6 (experiments v04) ---------------------------------------------------------
PRE_STEPS = 500             # pretraining Adam steps per stage
LR0, LR_END = 1e-4, 5e-6    # Adam, cosine from LR0 to LR_END (r08: low rate, every start is an already trained network)
CLIP = 1.0                  # gradient-norm clip
LAMBDA_DATA = 1.0           # reference weight
BOX_DECADES = 2.0           # half-width b of the search box, in log10
TPE_SEED, TPE_STARTUP = 0, 5
GRID = (0, 4, 8, 12, 16, 20)  # trial stage grid
TRIAL_STEPS = 300           # T, Adam steps per trial stage
PENALTY = 1e3               # objective of a diverged trial, in log10 units (finite log10 values stay below 309)
ADAM_MAX = 1000             # final Adam, steps per stage
TAU, WINDOW = 1e-3, 100     # switch when the relative improvement over WINDOW steps falls below TAU
LBFGS_HISTORY, LBFGS_MAX = 50, 500
SEEDS = (0, 1, 2, 3, 4)
SMOKE = dict(stages=3, adam_max=100, lbfgs_max=50)
EVERY = 25                  # history / progress line every EVERY steps
GRAD_EVERY = 100            # per-term gradient norms every GRAD_EVERY Adam steps (6 backward passes each)
CKPT_SECONDS = 15 * 60
BUDGET_HOURS = {"p": 2.2, "q": 2.2, "h": 0.4, "f": 10.5}    # h: paired smoke capped at 25 min (operator, 2026-11-14)

PRE_DEVICES, PRE_NODES = 64, 32   # data batch of one pretraining step (same size as the physics batch of E1)
LOSS_MODE = "relative"            # per-node relative residual (r04 Eq. 10), the control arm of E1
TERMS = training.TERMS + ("data",)
JUNCTION_HALF_WIDTH, EDGE_WIDTH = 0.05, 0.02   # node sampling: every node within these x_hat windows, the rest uniform
STUDY = "e6r9"                    # r09: no study; r08 weights + lambda_J by Eq. 24 (best_lambda_r09.json)
JOURNAL = "optuna_journal_r08.log"
BEST = "best_lambda_r09.json"
BEST_R08 = "best_lambda_r08.json"
UNIT = "e6r9-A-s"                 # final units of r09; r08 units were e6r8-A-s<seed>, r05 units e6-A-s<seed>
J_DEVICES = 16                    # D_J: training devices per stage on their full mesh for the current term
SMOKE9 = dict(stages=(0, 1, 10, 19, 20, 21), adam=200, judged=(0, 1, 19, 20, 21))   # v09 paired smoke


def results_dir() -> Path:
    return steps.results_root() / "e6"


def handoff_dir() -> Path:
    """`handoff-e6/` at the repository root: what one account hands the next through a commit."""
    return Path(__file__).resolve().parents[2] / "handoff-e6"


# --- context: E1's prepared splits plus the stored potentials -------------------------------------------
def _currents(refs, K: int, scales) -> np.ndarray:
    """(D, K) dPV terminal current J^dPV(V_j) of every device (r06 Eq. 13), NaN past the device's curve."""
    out = np.full((len(refs), K), np.nan)
    for d, ref in enumerate(refs):
        j = pce.terminal_current_stored(data.load_device(ref), scales)
        n = min(len(j), K)
        out[d, :n] = j[:n]
    return out


def _fields(refs, K: int) -> np.ndarray:
    """(D, K, 3, N) float32 stored phi, phi_n, phi_p of every device and stage (NaN past the device's curve)."""
    out = None
    for d, ref in enumerate(refs):
        dev = data.load_device(ref)
        if out is None:
            out = np.full((len(refs), K, 3, len(dev.x_hat)), np.nan, dtype=np.float32)
        n = min(len(dev.V), K)
        out[d, :n, 0], out[d, :n, 1], out[d, :n, 2] = dev.phi[:n], dev.phi_n[:n], dev.phi_p[:n]
    return out


class Ctx:
    def __init__(self, prep, fields: dict, cfg: Config, K: int, currents: dict) -> None:
        self.prep, self.fields, self.cfg, self.K = prep, fields, cfg, K
        self.currents = currents          # split -> (D, K) dPV terminal current [A/cm^2], NaN past the curve
        self.dev = training.device()
        b = prep.bounds
        self.scale = torch.tensor([[b.D_phi, b.phi_n[1] - b.phi_n[0], b.phi_p[1] - b.phi_p[0]]],
                                  dtype=torch.float64, device=self.dev)
        self.sigma2 = dict(prep.sigma2)

    def V(self, j: int) -> float:
        return j * self.cfg.voltage_step / self.prep.scales.V_th


def build_ctx(cfg: Config | None = None, report=print) -> Ctx:
    cfg = cfg or Config()
    t0 = time.time()
    prep = training.prepare(None, cfg)
    K = min(int(prep.train.n_V.max()), cfg.max_voltages, cfg.max_stages or 10 ** 9)
    fields = {s: _fields(prep.refs[s], K) for s in ("train", "val", "test")}
    currents = {s: _currents(prep.refs[s], K, prep.scales) for s in ("train", "val", "test")}
    report(f"[e6] data ready: {len(prep.train)}/{len(prep.val)}/{len(prep.test)} devices, {K} stages, "
           f"{time.time() - t0:.0f}s")
    return Ctx(prep, fields, cfg, K, currents)


def _save_ctx(ctx: Ctx) -> str:
    path = Path(tempfile.gettempdir()) / "e6-ctx.pkl"
    with open(path, "wb") as f:
        pickle.dump({"prep": ctx.prep, "fields": ctx.fields, "cfg": ctx.cfg, "K": ctx.K,
                     "currents": ctx.currents}, f, protocol=5)
    return str(path)


def _load_ctx(path: str) -> Ctx:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return Ctx(d["prep"], d["fields"], d["cfg"], d["K"], d["currents"])


# --- batches and the loss ------------------------------------------------------------------------------
def junction_position(x_hat: np.ndarray, Ndop: np.ndarray) -> np.ndarray:
    """x_hat of the p-n junction per device: the node where the doping changes sign. (P, N) -> (P,)."""
    j = np.argmax(np.diff(np.sign(Ndop), axis=1) != 0, axis=1)
    return x_hat[np.arange(x_hat.shape[0]), j]


def junction_nodes(x_hat: np.ndarray, Ndop: np.ndarray, k: int, rng, half_width: float = None,
                   edge_width: float = None) -> np.ndarray:
    """Interior nodes per device: EVERY node within `half_width` of the p-n junction x_j and EVERY node within
    `edge_width` of either contact, plus `k` nodes drawn uniformly among the remaining interior nodes. On dPV's
    uniform grid the windows hold the same number of nodes in every device, so the batch is rectangular.
    (P, n_window + k) sorted indices into the full grid (contacts excluded)."""
    half_width = JUNCTION_HALF_WIDTH if half_width is None else half_width
    edge_width = EDGE_WIDTH if edge_width is None else edge_width
    P, N = x_hat.shape
    h = 1.0 / (N - 1)
    nj, ne = int(round(2 * half_width / h)), int(round(edge_width / h))
    inner = np.arange(1, N - 1)
    jidx = np.argmin(np.abs(x_hat - junction_position(x_hat, Ndop)[:, None]), axis=1)
    out = []
    for d in range(P):
        lo = int(np.clip(jidx[d] - nj // 2, 1 + ne, N - 1 - ne - nj))
        fixed = np.unique(np.concatenate([inner[:ne], np.arange(lo, lo + nj), inner[-ne:] if ne else []]).astype(int))
        rest = np.setdiff1d(inner, fixed)
        out.append(np.sort(np.concatenate([fixed, rng.choice(rest, size=min(k, rest.size), replace=False)])))
    return np.stack(out)


def stage_batch(ctx: Ctx, split: str, stage: int, key, n_dev: int, n_nodes: int) -> dict:
    """The devices `training.make_batch` picks for the key, with the nodes of `junction_nodes` (every node of the
    junction and contact windows, where the r05 networks keep more than half of their field error on a tenth
    of the nodes, plus a uniform draw of the rest) and the stored outputs at those nodes."""
    bank = getattr(ctx.prep, split)
    eligible = np.nonzero(bank.n_V > stage)[0]
    rng = np.random.default_rng(key)
    pick = np.sort(rng.choice(eligible, size=min(n_dev, eligible.size), replace=False))
    N = bank.x_hat.shape[1]
    k = min(n_nodes, N - 2)
    idx = junction_nodes(bank.x_hat[pick], bank.Ndop[pick], k, rng)
    b = training._gather(bank, pick, idx, ctx.dev)
    tgt = ctx.fields[split][pick[:, None], stage, :, idx]            # (P, k, 3)
    b["target"] = training._t(tgt.reshape(-1, 3).astype(np.float64), ctx.dev)
    return b


def physics_batch(ctx: Ctx, stage: int, seed: int) -> dict:
    """The fixed batch of a stage: a function of (seed, stage), junction-concentrated nodes."""
    return stage_batch(ctx, "train", stage, [int(seed), int(stage), 17], ctx.cfg.n_devices_per_batch,
                       ctx.cfg.n_nodes_per_device)


def data_loss(model, b: dict, V_tilde: float, scale: torch.Tensor) -> torch.Tensor:
    """Eq. 18 of r05 on a batch: < ((phi-phi*)/D_phi)^2 + sum_k ((Y_k-Y_k*)/(Y_k^max-Y_k^min))^2 >."""
    phi, pn, pp = model(b["x"], b["xi"], b["peq0"], b["peq1"], V_tilde)
    pred = torch.cat([phi, pn, pp], dim=1)
    return (((pred - b["target"]) / scale) ** 2).sum(dim=1).mean()


def current_batch(ctx: Ctx, stage: int, seed: int, n_dev: int = None) -> dict:
    """D_J of r09 Eq. 23: `n_dev` training devices whose curve reaches `stage`, on their FULL dPV mesh (Eq. 20 is
    a quadrature over every node), with the dPV current at the stage voltage and the dPV short-circuit current.
    A function of (seed, stage): fixed during the stage, so L-BFGS sees one objective."""
    bank = ctx.prep.train
    n_dev = n_dev or J_DEVICES
    eligible = np.nonzero(bank.n_V > stage)[0]
    rng = np.random.default_rng([int(seed), int(stage), 23])
    pick = np.sort(rng.choice(eligible, size=min(n_dev, eligible.size), replace=False))
    N = bank.x_hat.shape[1]
    t = lambda a: training._t(np.asarray(a, dtype=np.float64), ctx.dev)
    ref = ctx.currents["train"][pick]
    return {"x": t(bank.x_hat[pick].reshape(-1, 1)), "xi": t(np.repeat(bank.xi_hat[pick], N, axis=0)),
            "peq0": t(np.repeat(bank.phi_eq0[pick], N)[:, None]), "peq1": t(np.repeat(bank.phi_eq1[pick], N)[:, None]),
            "G": t(bank.G[pick]), "x_hat": t(bank.x_hat[pick]), "delta": t(bank.delta[pick]),
            "J_ref": t(ref[:, stage]), "J_sc": t(ref[:, 0]), "n": int(pick.size), "N": int(N), "devices": pick}


def gr_current_torch(model, cb: dict, V_tilde: float, ctx: Ctx) -> torch.Tensor:
    """Eq. 20 of r07 with its graph (r09 Eq. 23 differentiates it): J_0 [ (1/delta) int (G - R) dx_hat
    - S_n1 (n(1) - n_eq(1)) - S_p0 (p(0) - p_eq(0)) ] per device [A/cm^2]. Same quadrature as `training.gr_current`."""
    m, contacts, scales = ctx.prep.material, data.device_contacts(ctx.prep.scales), ctx.prep.scales
    phi, phi_n, phi_p = model(cb["x"], cb["xi"], cb["peq0"], cb["peq1"], V_tilde)
    shape = lambda a: a.reshape(cb["n"], cb["N"])
    phi, phi_n, phi_p = shape(phi), shape(phi_n), shape(phi_p)
    mm = m.like(phi)
    n, p = carriers.electron_density(phi, phi_n, mm), carriers.hole_density(phi, phi_p, mm)
    n_eq, p_eq = carriers.equilibrium_densities(phi, mm)
    f = cb["G"] - carriers.recombination(n, p, mm)
    integral = (0.5 * (f[:, 1:] + f[:, :-1]) * (cb["x_hat"][:, 1:] - cb["x_hat"][:, :-1])).sum(dim=1)
    j = integral / cb["delta"] - contacts.S_n1 * (n[:, -1] - n_eq[:, -1]) - contacts.S_p0 * (p[:, 0] - p_eq[:, 0])
    return scales.J0 * j


def current_loss(model, cb: dict, V_tilde: float, ctx: Ctx) -> torch.Tensor:
    """r09 Eq. 23: < ((J_GR(V_j; theta) - J_dPV(V_j)) / J_sc_dPV)^2 > over D_J."""
    return (((gr_current_torch(model, cb, V_tilde, ctx) - cb["J_ref"]) / cb["J_sc"]) ** 2).mean()


class PhasedObjective(training.Objective):
    """Eq. 11 of r05 over I' = I u {data}: E1's relative residual terms plus the data term on the same batch.
    With `jbatch` (r09), also the current term of Eq. 23 over I'' = I' u {J}, on its own full-mesh devices."""

    def __init__(self, ctx: Ctx, model, batch: dict, stage: int, jbatch: dict | None = None) -> None:
        super().__init__(model, batch, ctx.V(stage), ctx.sigma2, ctx.prep.material, ctx.prep.contacts,
                         mode=LOSS_MODE)
        self.scale, self.ctx, self.jbatch = ctx.scale, ctx, jbatch

    def _chunk_losses(self, b: dict) -> tuple:
        L, n = super()._chunk_losses(b)
        L["data"] = data_loss(self.model, b, self.V, self.scale)
        n["data"] = int(b["x"].shape[0])
        if self.jbatch is not None:
            L["J"] = current_loss(self.model, self.jbatch, self.V, self.ctx)
            n["J"] = 1
        return L, n

    def _counts(self) -> list:
        out = super()._counts()
        for c, b in zip(out, self.chunks):
            c["data"] = int(b["x"].shape[0])
            if self.jbatch is not None:
                c["J"] = 1
        return out

    def terms_and_grad(self, lam: dict, params: list) -> tuple:
        L, _ = self._chunk_losses(self.batch)
        total = sum(lam[k] * L[k] for k in L)
        g = torch.autograd.grad(total, params)
        return total.detach(), {k: float(v.detach()) for k, v in L.items()}, g


@torch.no_grad()
def field_error(ctx: Ctx, model, split: str, stage: int, chunk: int = 32) -> float:
    """Eq. 18 over every node of every device of `split` whose stored curve reaches `stage`."""
    bank, F = getattr(ctx.prep, split), ctx.fields[split]
    eligible = np.nonzero(bank.n_V > stage)[0]
    N = bank.x_hat.shape[1]
    total, count = 0.0, 0
    for s in range(0, eligible.size, chunk):
        pick = eligible[s:s + chunk]
        idx = np.tile(np.arange(N), (pick.size, 1))
        b = training._gather(bank, pick, idx, ctx.dev)
        b["target"] = training._t(F[pick[:, None], stage, :, idx].reshape(-1, 3).astype(np.float64), ctx.dev)
        n = int(b["x"].shape[0])
        total += float(data_loss(model, b, ctx.V(stage), ctx.scale)) * n
        count += n
    return total / count if count else float("nan")


def current_error(ctx: Ctx, model, split: str, stage: int) -> float:
    """r06 Eq. 13: mean over the devices of `split` whose curve reaches `stage` of
    ((J(V_j; theta) - J^dPV(V_j)) / J_sc^dPV)^2, with J_sc^dPV = J^dPV(0)."""
    bank, ref = getattr(ctx.prep, split), ctx.currents[split]
    eligible = np.nonzero(bank.n_V > stage)[0]
    if eligible.size == 0:
        return float("nan")
    J = training.evaluate_currents(model, bank, ctx.V(stage), ctx.prep.material, ctx.prep.scales, ctx.dev,
                                   ctx.cfg.eval_chunk)
    r = (J[eligible] - ref[eligible, stage]) / ref[eligible, 0]
    return float(np.mean(r ** 2))


# --- telemetry ------------------------------------------------------------------------------------------
def _hms(s: float) -> str:
    if not math.isfinite(s) or s < 0:
        return "?"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Telemetry:
    """history.jsonl (append), progress.json (overwrite), one flushed line per call to `line`."""

    def __init__(self, out: Path, tag: str) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.out, self.tag, self.t0 = out, tag, time.time()
        self.history = out / f"{tag}.history.jsonl"
        self.progress_path = out / f"{tag}.progress.json"

    def log(self, **rec) -> None:
        rec["wall"] = time.time() - self.t0
        with open(self.history, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def line(self, msg: str) -> None:
        print(f"[{self.tag}] {msg}", flush=True)

    def progress(self, fraction: float, **kw) -> str:
        el = time.time() - self.t0
        eta = el / fraction * (1.0 - fraction) if fraction > 0 else float("nan")
        steps._write(self.progress_path, {"tag": self.tag, "fraction": fraction, "elapsed_s": el, "eta_s": eta,
                                          "updated": steps._now(), **kw})
        return _hms(eta)

    def plot(self) -> None:
        try:
            plot_history(self.history, self.out / f"{self.tag}.png")
        except Exception as e:                       # a plot never stops a run
            self.line(f"plot skipped: {e!r}")


def plot_history(history: Path, png: Path) -> None:
    """Loss per term, gradient norm per term, lambda / lr, and validation error, against the global step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in Path(history).read_text().splitlines() if l.strip()]
    if not recs:
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    steps_ = [r for r in recs if "terms" in r]
    x = np.arange(len(steps_))
    if steps_:
        ax[0, 0].semilogy(x, [r["total"] for r in steps_], "k", lw=1.2, label="total")
        for k in TERMS + ("J",):
            ax[0, 0].semilogy(x, [r["terms"].get(k, np.nan) for r in steps_], lw=0.8, label=k)
        for i, r in enumerate(steps_):
            if i and r.get("phase") != steps_[i - 1].get("phase"):
                ax[0, 0].axvline(i, color="grey", lw=0.5, ls=":")
        ax[0, 0].set_title("loss per term (every 25 steps; dotted = phase change)")
        ax[0, 0].legend(fontsize=7)
        ax[1, 0].semilogy(x, [r.get("lr", np.nan) or np.nan for r in steps_])
        ax[1, 0].set_title("learning rate (Adam)")
    g = [r for r in recs if "grad_norms" in r]
    if g:
        for k in TERMS + ("J",):
            ax[0, 1].semilogy([r["grad_norms"].get(k, np.nan) for r in g], ".-", lw=0.8, label=k)
        ax[0, 1].set_title("weighted gradient norm per term")
        ax[0, 1].legend(fontsize=7)
    v = [r for r in recs if r.get("kind") == "stage"]
    if v:
        ax[1, 1].semilogy([r["stage"] for r in v], [r.get("field_error_test", np.nan) for r in v], "o-",
                          label="field error test (Eq. 18)")
        ax[1, 1].set_title("per stage")
        ax[1, 1].legend(fontsize=7)
    tr = [r for r in recs if r.get("kind") == "trial"]
    if tr:
        ax[1, 1].semilogy([r["trial"] for r in tr], [r["value"] for r in tr], "o", label="trial objective")
        ax[1, 1].set_title("Optuna trials (validation Eq. 18)")
        ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


# --- networks -------------------------------------------------------------------------------------------
def new_model(ctx: Ctx, seed: int):
    spec = UnitSpec("e6", "A", "adam", LOSS_MODE, "fixed", False, int(seed))
    return ansatz_mod.Ansatz(training.build_network(spec, ctx.cfg, ctx.prep).to(ctx.dev), ctx.prep.bounds)


def _cpu_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _adam(params, n_steps: int):
    opt = torch.optim.Adam(params, lr=LR0)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps, 1), eta_min=LR_END)


def _apply(params, grads) -> None:
    for p, g in zip(params, grads):
        p.grad = g.detach().clone()


def pretrain(ctx: Ctx, seed: int, n_stages: int, tel: Telemetry, deadline: float | None = None) -> dict:
    """theta_pre: Adam on Eq. 18 alone, PRE_STEPS per stage, stages 0..n_stages-1 in continuation from the
    equilibrium start. A fresh data batch per step (devices whose curve reaches the stage)."""
    torch.manual_seed(seed)
    model = new_model(ctx, seed)
    params = list(model.parameters())
    states = {}
    for j in range(n_stages):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"wall budget reached during pretraining at stage {j}")
        opt, sched = _adam(params, PRE_STEPS)
        for s in range(PRE_STEPS):
            b = stage_batch(ctx, "train", j, [int(seed), j, s, 29], PRE_DEVICES, PRE_NODES)
            L = data_loss(model, b, ctx.V(j), ctx.scale)
            if not torch.isfinite(L):
                raise FloatingPointError(f"pretraining diverged at stage {j} step {s}")
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
            if s % EVERY == 0 or s == PRE_STEPS - 1:
                tel.log(phase="pretrain", stage=j, step=s, total=float(L), terms={"data": float(L)},
                        lr=sched.get_last_lr()[0])
        states[j] = _cpu_state(model)
        err = field_error(ctx, model, "val", j)
        tel.log(kind="pretrain_stage", stage=j, field_error_val=err)
        eta = tel.progress((j + 1) / n_stages, phase="pretrain", stage=j)
        tel.line(f"pretrain stage {j + 1}/{n_stages} · data loss {float(L):.3e} · val field err {err:.3e} · ETA {eta}")
    return states


# --- Eq. 12 centre ----------------------------------------------------------------------------------------
def centre(ctx: Ctx, pre: dict, seed: int) -> dict:
    """Eq. 12 at theta_pre, once: lambda_hat per grid stage, rescaled so lambda_data = 1 (Adam is invariant to a
    global rescaling), then the geometric mean over the grid. Returns lambda^0 for the five physics terms."""
    model = new_model(ctx, seed)
    params = list(model.parameters())
    logs = {k: [] for k in training.TERMS}
    for j in GRID:
        if j not in pre:
            continue
        model.load_state_dict(pre[j])
        g = PhasedObjective(ctx, model, physics_batch(ctx, j, seed), j).term_grads(params)
        lh = weights.lambda_hat(g, "max", ctx.cfg.eps_g, ctx.cfg.lambda_min, ctx.cfg.lambda_max)
        for k in logs:
            logs[k].append(math.log10(lh[k] / lh["data"]))
    return {k: 10.0 ** float(np.mean(v)) for k, v in logs.items()}


def centre_current(ctx: Ctx, prev: dict, seed: int = 0) -> float:
    """r09 Eq. 24: the Eq. 12 rule for the current term at theta_prev^r08 over I'' (reference: the largest
    gradient of any term), rescaled so lambda_data = 1 like `centre`, geometric mean over GRID."""
    model = new_model(ctx, seed)
    params = list(model.parameters())
    logs = []
    for j in GRID:
        if j not in prev:
            continue
        model.load_state_dict(prev[j])
        obj = PhasedObjective(ctx, model, physics_batch(ctx, j, seed), j, jbatch=current_batch(ctx, j, seed))
        lh = weights.lambda_hat(obj.term_grads(params), "max", ctx.cfg.eps_g, ctx.cfg.lambda_min, ctx.cfg.lambda_max)
        logs.append(math.log10(lh["J"] / lh["data"]))
    return 10.0 ** float(np.mean(logs))


# --- the Optuna study -------------------------------------------------------------------------------------
def _storage(journal: Path):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:                                   # optuna >= 4.0
        from optuna.storages.journal import JournalFileBackend, JournalStorage
        return JournalStorage(JournalFileBackend(str(journal)))
    except ImportError:                    # optuna 3.x
        from optuna.storages import JournalFileStorage, JournalStorage
        return JournalStorage(JournalFileStorage(str(journal)))


def open_study(journal: Path, worker: int = 0, create_centre: dict | None = None, study_name: str | None = None):
    import optuna
    sampler = optuna.samplers.TPESampler(seed=TPE_SEED + worker, n_startup_trials=TPE_STARTUP)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=TPE_STARTUP, n_warmup_steps=1)
    study = optuna.create_study(study_name=study_name or STUDY, storage=_storage(journal), direction="minimize",
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    if create_centre is not None and "centre" not in study.user_attrs:
        study.set_user_attr("centre", create_centre)
    return study


def full_lambda(phys: dict) -> dict:
    return {**{k: float(v) for k, v in phys.items()}, "data": LAMBDA_DATA}


def run_trial(ctx: Ctx, start: dict, lam: dict, seed: int, deadline: float, trial=None, tel=None,
              with_j: bool = False) -> float:
    """theta_T(lambda): per grid stage, TRIAL_STEPS Adam steps from theta_prev of that stage (r06); the objective
    is the mean validation current error of r06 Eq. 13 over the stages done, reported after each stage to the
    pruner. The study minimizes log10 of that mean: a monotone transform, so the argmin of Eq. 13, TPE's
    quantile split and the median pruner's decisions are unchanged, while values that span 1e-4 to 1e24 stay
    comparable to the fixed penalty of a diverged trial."""
    import optuna
    model = new_model(ctx, seed)
    params = list(model.parameters())
    vals = []
    for i, j in enumerate(GRID):
        if time.time() > deadline:
            raise TimeoutError("deadline inside a trial")
        model.load_state_dict(start[j])
        obj = PhasedObjective(ctx, model, physics_batch(ctx, j, seed), j,
                              jbatch=current_batch(ctx, j, seed) if with_j else None)
        opt, sched = _adam(params, TRIAL_STEPS)
        for s in range(TRIAL_STEPS):
            total, _, g = obj.terms_and_grad(lam, params)
            if not torch.isfinite(total):
                return PENALTY
            _apply(params, g)
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
        e = current_error(ctx, model, "val", j)
        if not math.isfinite(e):
            return PENALTY
        vals.append(e)
        if tel is not None:
            tel.log(kind="trial_stage", trial=trial.number if trial else None, stage=j, current_error_val=e,
                    field_error_val=field_error(ctx, model, "val", j))
        if trial is not None:
            trial.report(float(np.log10(np.mean(vals))), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.log10(np.mean(vals)))


def tune_loop(ctx: Ctx, start: dict, journal: Path, worker: int, deadline: float, tel: Telemetry,
              study_name: str | None = None, keys: tuple | None = None, target: int | None = None) -> None:
    """`keys` (default the physics terms) are searched in the box around the study's centre; "J" in `keys` adds
    the current term to every trial's loss. `target` stops the worker once the study holds that many trials."""
    import optuna
    from optuna.trial import TrialState
    study = open_study(journal, worker, study_name=study_name)
    c = study.user_attrs["centre"]
    keys = tuple(keys or training.TERMS)
    longest = 0.0
    while True:
        if time.time() + 1.05 * longest > deadline:
            tel.line(f"wall budget: no room for another trial (longest {longest:.0f}s); stopping")
            return
        if target is not None and len(study.get_trials(deepcopy=False)) >= target:
            tel.line(f"study holds {target} trials; stopping")
            return
        trial = study.ask()
        lam = full_lambda({k: 10.0 ** trial.suggest_float(f"log10_{k}", math.log10(c[k]) - BOX_DECADES,
                                                          math.log10(c[k]) + BOX_DECADES) for k in keys})
        t = time.time()
        try:
            value = run_trial(ctx, start, lam, 0, deadline, trial, tel, with_j="J" in keys)
            study.tell(trial, value)
            state = "complete" if value < PENALTY else "diverged"
        except optuna.TrialPruned:
            study.tell(trial, state=TrialState.PRUNED)
            value, state = float("nan"), "pruned"
        except TimeoutError:
            study.tell(trial, state=TrialState.FAIL)
            tel.line(f"trial {trial.number} cut by the wall budget; stopping")
            return
        dt = time.time() - t
        longest = max(longest, dt)
        try:
            best = study.best_value
        except ValueError:
            best = float("nan")
        tel.log(kind="trial", trial=trial.number, value=value, state=state, seconds=dt, lam=lam)
        eta = _hms(deadline - time.time())
        tel.line(f"trial {trial.number} {state} · value {value:.3e} · {dt:.0f}s · best {best:.3e} · budget left {eta}")
        tel.plot()


def study_summary(journal: Path, study_name: str | None = None) -> dict:
    from optuna.trial import TrialState
    study = open_study(journal, study_name=study_name)
    trials = study.get_trials(deepcopy=False)
    count = lambda s: sum(t.state == s for t in trials)
    out = {"n_trials": len(trials), "complete": count(TrialState.COMPLETE), "pruned": count(TrialState.PRUNED),
           "failed": count(TrialState.FAIL), "running_or_lost": count(TrialState.RUNNING),
           "centre": study.user_attrs.get("centre"), "box_decades": BOX_DECADES,
           "trials": [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params}
                      for t in trials]}
    done = [t for t in trials if t.state == TrialState.COMPLETE]
    if done:
        b = min(done, key=lambda t: t.value)
        out.update(best_trial=b.number, best_value=b.value,
                   best_lambda=full_lambda({k[6:]: 10.0 ** v for k, v in b.params.items() if k.startswith("log10_")}))
    return out


# --- the final recipe -----------------------------------------------------------------------------------
_PART = re.compile(r"^(?P<base>.+-s(?P<seed>\d+))(?:\.p(?P<p>\d+)of(?P<n>\d+))?$")


def unit_spec(uid: str, K: int) -> tuple:
    """`e6r9-A-s4` -> (4, all stages, "e6r9-A-s4"); `e6r9-A-s4.p3of8` -> (4, [3, 11, 19], "e6r9-A-s4").
    r09 starts every stage from theta_prev^r08 of that stage, so the stages of one seed are independent and a
    seed can be split across GPUs; interleaving (j mod n) spreads the slow near-Voc stages over the parts."""
    m = _PART.match(uid)
    if m is None:
        raise ValueError(f"unit id {uid!r} names no seed")
    p, n = (int(m["p"]), int(m["n"])) if m["p"] is not None else (0, 1)
    if not 0 <= p < n:
        raise ValueError(f"unit id {uid!r}: part {p} of {n}")
    return int(m["seed"]), [j for j in range(K) if j % n == p], m["base"]


def final_plan(seeds: int = 5, accounts: int = 4, gpus: int = 2, unit: str | None = None) -> list:
    """v09 F split (operator 2026-11-14: no idle GPU). Account a runs seed a split over its GPUs, and the seeds
    left over (a >= accounts) are split over every GPU of every account, so all GPUs carry the same load
    (5 seeds, 4 accounts x 2 T4: 12 + 3 = 15 stages per GPU). Returns, per account, one pack per GPU."""
    total = accounts * gpus
    extra = list(range(accounts, seeds))
    plan = []
    for a in range(accounts):
        packs = []
        for g in range(gpus):
            pack = [f"{unit or UNIT}{a}.p{g}of{gpus}"] if a < seeds else []
            pack += [f"{unit or UNIT}{s}.p{a * gpus + g}of{total}" for s in extra]
            packs.append(pack)
        plan.append(packs)
    return plan


def merge_parts(out: Path, prep, base: str) -> dict | None:
    """Join the complete parts `<base>.pXofN` of one seed into `<base>.json` and `<base>.models.pt`, the same
    files an unsplit unit writes, with the PCE error computed on the joined curve. None while a part is missing."""
    recs = [json.loads(p.read_text()) for p in sorted(out.glob(f"{base}.p*of*.json"))]
    recs = [r for r in recs if r.get("complete")]
    if not recs:
        return None
    n = int(_PART.match(recs[0]["unit"])["n"])
    if sorted(int(_PART.match(r["unit"])["p"]) for r in recs) != list(range(n)):
        return None
    K = recs[0]["stages_planned_total"]
    J = np.full((len(prep.test), K), np.nan)
    rows, final, pre = [], {}, {}
    for r in recs:
        part = torch.load(out / f"{r['unit']}.models.pt", map_location="cpu", weights_only=False)
        final.update(part["final"])
        jt = np.load(out / f"{r['unit']}.jtest.npy")
        for j in r["stages"]:
            J[:, j] = jt[:, j]
        rows += r["stage_progress"]
    rows.sort(key=lambda r: r["stage"])
    diverged = any(r["diverged"] for r in recs)
    done = 0
    while done < K and done in final:
        done += 1
    errs = training.errors_from_curves(J, prep.test, done, prep.scales)
    torch.save({"pretrained": None, "final": final}, out / f"{base}.models.pt")
    np.save(out / f"{base}.jtest.npy", J)
    rec = {**{k: v for k, v in recs[0].items() if k not in ("unit", "stages", "stage_progress", "per_device_error")},
           "unit": base, "parts": [r["unit"] for r in recs], "stages_done": done, "stages_planned": K,
           "diverged": diverged, "stage_progress": rows, "per_device_error": errs,
           "seconds": sum(r["seconds"] for r in recs), "gpu_hours": sum(r["gpu_hours"] for r in recs),
           "created": steps._now()}
    steps._write(out / f"{base}.json", rec)
    return rec


def final_unit(ctx: Ctx, uid: str, seed: int, lam: dict, out: Path, deadline: float, n_stages: int | None = None,
               adam_max: int = ADAM_MAX, lbfgs_max: int = LBFGS_MAX, start: dict | None = None,
               stages: list | None = None, tau: float | None = TAU) -> dict | None:
    # tau None: Adam runs all adam_max steps (no plateau switch; scratch control of r09)
    """One seed: pretraining, then per stage Adam -> L-BFGS from theta_pre of that stage with the same loss and the
    same lambda. Resumable from `<uid>.ckpt.pt` (per stage, and every 15 min inside Adam). Returns the record, or
    None when the wall budget stopped it (checkpoint and partial record on disk)."""
    prep, dev = ctx.prep, ctx.dev
    K = min(n_stages or ctx.K, ctx.K)
    tel = Telemetry(out, uid)
    ckpt = out / f"{uid}.ckpt.pt"
    st = {"pre": None, "stage_models": {}, "rows": [], "J_test": np.full((len(prep.test), K), np.nan),
          "mid": None, "seconds": 0.0, "diverged": False}
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        tel.line(f"resumed: {len(st['rows'])}/{K} stages done")
    t0 = time.time()

    def save():
        tmp = ckpt.with_suffix(".tmp")
        torch.save({**st, "seconds": st["seconds"] + time.time() - t0}, tmp)
        os.replace(tmp, ckpt)

    if st["pre"] is None:
        # r06: start from theta_prev of this seed when handed over; otherwise pretrain as in r05
        st["pre"] = start if start is not None else pretrain(ctx, seed, K, tel, deadline)
        save()
    model = new_model(ctx, seed)
    params = list(model.parameters())
    longest = max([r["seconds"] for r in st["rows"]], default=0.0)
    stages = [j for j in (stages if stages is not None else range(K)) if j < K]
    whole = stages == list(range(K))
    Kp = len(stages)
    for j in [j for j in stages if j not in {r["stage"] for r in st["rows"]}]:
        pos = stages.index(j)
        if st["diverged"]:
            break
        if time.time() + 1.05 * longest > deadline:
            tel.line(f"wall budget: stage {j} would not fit (longest {longest:.0f}s); checkpoint kept")
            save()
            return None
        ts = time.time()
        model.load_state_dict(st["pre"][j])
        # r09: the current term of Eq. 23 enters exactly when its weight is in lambda (best_lambda_r09.json)
        obj = PhasedObjective(ctx, model, physics_batch(ctx, j, seed), j,
                              jbatch=current_batch(ctx, j, seed) if "J" in lam else None)
        g0 = obj.term_grads(params)
        shares = diagnostics.gradient_shares(g0, lam)
        loss0 = float(obj.terms_and_grad(lam, params)[0])
        opt, sched = _adam(params, adam_max)
        losses, s0, switched = [], 0, None
        mid = st.get("mid")
        if mid and mid["stage"] == j:
            model.load_state_dict(mid["model"])
            opt.load_state_dict(mid["opt"])
            sched.load_state_dict(mid["sched"])
            losses, s0 = mid["losses"], mid["step"]
        last_ck = time.time()
        # --- phase 2: Adam ---
        for s in range(s0, adam_max):
            total, terms, g = obj.terms_and_grad(lam, params)
            f = float(total)
            if not math.isfinite(f) or f > ctx.cfg.divergence_factor * max(loss0, 1e-300):
                st["diverged"] = True
                tel.line(f"stage {j}: Adam diverged at step {s} (loss {f:.3e})")
                break
            losses.append(f)
            _apply(params, g)
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
            if s % EVERY == 0:
                rec = dict(phase="adam", stage=j, step=s, total=f, terms=terms, lr=sched.get_last_lr()[0])
                if s % GRAD_EVERY == 0:
                    gt = obj.term_grads(params)
                    rec["grad_norms"] = {k: float(lam[k] * np.linalg.norm(v)) for k, v in gt.items()}
                tel.log(**rec)
                frac = (pos + (s + 1) / (adam_max + lbfgs_max)) / Kp
                eta = tel.progress(frac, phase="adam", stage=j, step=s)
                tel.line(f"stage {j + 1}/{K} · adam {s}/{adam_max} · loss {f:.3e} · ETA {eta}")
            if tau is not None and len(losses) > WINDOW and (losses[-WINDOW - 1] - losses[-1]) / abs(losses[-WINDOW - 1]) < tau:
                switched = s
                break
            if time.time() - last_ck > CKPT_SECONDS:
                st["mid"] = {"stage": j, "model": _cpu_state(model), "opt": opt.state_dict(),
                             "sched": sched.state_dict(), "losses": losses, "step": s + 1}
                save()
                last_ck = time.time()
        st["mid"] = None
        if st["diverged"]:
            st["rows"].append({"stage": j, "diverged": True, "seconds": time.time() - ts})
            save()
            break
        adam_state = _cpu_state(model)
        f_adam = float(obj.terms_and_grad(lam, params)[0])
        # --- phase 3: L-BFGS on the same objective and the same lambda ---
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_max, history_size=LBFGS_HISTORY,
                               line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
        n_eval = {"k": 0}

        def closure():
            total, terms, g = obj.terms_and_grad(lam, params)
            _apply(params, g)
            k = n_eval["k"]
            n_eval["k"] += 1
            if k % EVERY == 0:
                tel.log(phase="lbfgs", stage=j, step=k, total=float(total), terms=terms, lr=None)
                frac = (pos + (adam_max + min(k, lbfgs_max)) / (adam_max + lbfgs_max)) / Kp
                eta = tel.progress(frac, phase="lbfgs", stage=j, step=k)
                tel.line(f"stage {j + 1}/{K} · lbfgs eval {k} · loss {float(total):.3e} · ETA {eta}")
            return total

        try:
            lb.step(closure)
        except RuntimeError as e:
            tel.line(f"stage {j}: L-BFGS raised {e!r}")
        f_lb = float(obj.terms_and_grad(lam, params)[0])
        lb_iters = int(lb.state[lb._params[0]].get("n_iter", 0)) if lb._params and lb._params[0] in lb.state else 0
        reverted = not math.isfinite(f_lb) or f_lb > f_adam
        if reverted:
            model.load_state_dict(adam_state)
            f_lb = f_adam
        Vt = ctx.V(j)
        st["J_test"][:, j] = training.evaluate_currents(model, prep.test, Vt, prep.material, prep.scales, dev,
                                                        ctx.cfg.eval_chunk)
        st["stage_models"][j] = _cpu_state(model)
        partial = (np.array(list(training.errors_from_curves(st["J_test"], prep.test, j + 1, prep.scales).values()))
                   if whole else np.array([np.nan]))      # a split seed's PCE error is formed by merge_parts
        row = {"stage": j, "V": j * ctx.cfg.voltage_step, "loss_start": loss0, "loss_adam_end": f_adam,
               "loss_end": f_lb, "adam_steps": len(losses), "switched_at": switched, "lbfgs_iterations": lb_iters,
               "lbfgs_evaluations": n_eval["k"], "lbfgs_reverted": reverted, "gradient_shares_start": shares,
               "max_gradient_share_start": max(shares.values()),
               "field_error_test": field_error(ctx, model, "test", j),
               "partial_pce_error_median": float(np.nanmedian(partial)) if np.isfinite(partial).any() else float("nan"),
               "seconds": time.time() - ts}
        st["rows"].append(row)
        longest = max(longest, row["seconds"])
        tel.log(kind="stage", **row)
        save()
        steps._write(out / f"{uid}.partial.json", {"unit": uid, "complete": False, "rows": st["rows"]})
        tel.line(f"stage {j + 1}/{K} done · loss {loss0:.3e} -> adam {f_adam:.3e} -> lbfgs {f_lb:.3e}"
                 f"{' (reverted)' if reverted else ''} · field err {row['field_error_test']:.3e} · "
                 f"partial PCE err median {row['partial_pce_error_median']:.2f}% · {row['seconds']:.0f}s")
        tel.plot()
    done = len([r for r in st["rows"] if not r.get("diverged")])
    errs = ({u: float("nan") for u in prep.test.uids} if st["diverged"] or not whole
            else training.errors_from_curves(st["J_test"], prep.test, done, prep.scales))
    seconds = st["seconds"] + time.time() - t0
    torch.save({"pretrained": st["pre"] if whole else None, "final": st["stage_models"]}, out / f"{uid}.models.pt")
    np.save(out / f"{uid}.jtest.npy", st["J_test"])
    rec = {"unit": uid, "step": "e6", "complete": True, "created": steps._now(), "seed": seed,
           "config": ctx.cfg.to_dict(), "lambda": lam, "loss_mode": LOSS_MODE, "sigma2": ctx.sigma2,
           "recipe": {"pre_steps": PRE_STEPS, "adam_max": adam_max, "tau": tau, "window": WINDOW,
                      "lbfgs_max": lbfgs_max, "lbfgs_history": LBFGS_HISTORY, "lr": [LR0, LR_END], "clip": CLIP},
           "stages": stages, "stages_planned_total": K,
           "stages_done": done, "stages_planned": Kp, "diverged": st["diverged"], "stage_progress": st["rows"],
           "per_device_error": errs, "seconds": seconds,
           "gpu_hours": seconds / 3600.0 if dev.type == "cuda" else 0.0, "device": str(dev),
           "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None}
    steps._write(out / f"{uid}.json", rec)
    (out / f"{uid}.partial.json").unlink(missing_ok=True)
    ckpt.unlink(missing_ok=True)
    e = np.array(list(errs.values()))
    tel.line(f"DONE · {done}/{K} stages · diverged={st['diverged']} · median PCE err "
             f"{np.nanmedian(e) if np.isfinite(e).any() else float('nan'):.2f}% · {seconds / 3600:.2f} h")
    tel.plot()
    return rec


# --- workers (one process per GPU) -------------------------------------------------------------------------
def _worker(payload: str) -> None:
    a = json.loads(payload)
    ctx = _load_ctx(a["ctx"])
    out = Path(a["out"])
    if a["role"] == "tune":
        start = torch.load(a["start"], map_location="cpu", weights_only=False)
        tune_loop(ctx, start, Path(a["journal"]), a["worker"], a["deadline"], Telemetry(out, f"tune-w{a['worker']}"),
                  study_name=a.get("study"), keys=a.get("keys"), target=a.get("target"))
    else:
        lam = a["lambda"]
        for uid in a["units"]:
            if steps.is_complete(out / f"{uid}.json"):
                print(f"[{uid}] complete, skipped", flush=True)
                continue
            seed, stages, _ = unit_spec(uid, ctx.K)
            stages = stages[:a["stage_limit"]] if a.get("stage_limit") else stages
            start = None if a.get("scratch") else prev_models(seed)       # scratch: pretrain this seed here
            if final_unit(ctx, uid, seed, lam, out, a["deadline"], start=start, stages=stages,
                          adam_max=a.get("adam_max", ADAM_MAX), lbfgs_max=a.get("lbfgs_max", LBFGS_MAX),
                          tau=a.get("tau", TAU)) is None:
                break


def _spawn(payloads: list, deadline: float, report=print) -> None:
    """One worker process per payload (payload i on GPU i), restarted if it dies while the budget allows."""
    src = str(Path(__file__).resolve().parents[1])

    def supervise(gpu: int, payload: dict) -> None:
        for attempt in range(3):
            if time.time() > deadline - 120:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), STAGED_PINN_WORKER="1",
                       PYTHONPATH=os.pathsep.join([src] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
            proc = subprocess.Popen([sys.executable, "-u", "-c",
                                     "import sys; from Staged_PINN_Solar import e6; e6._worker(sys.argv[1])",
                                     json.dumps(payload)], env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                report(f"[gpu{gpu}] {line.rstrip()}")
            if proc.wait() == 0:
                return
            report(f"[gpu{gpu}] worker exited with code {proc.returncode} (attempt {attempt + 1})")

    threads = [threading.Thread(target=supervise, args=(g, p)) for g, p in enumerate(payloads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _n_gpu() -> int:
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def _run_payloads(payloads: list, ctx: Ctx, deadline: float) -> None:
    if _n_gpu() >= 2 and len(payloads) >= 2:
        path = _save_ctx(ctx)
        for p in payloads:
            p["ctx"] = path
        _spawn(payloads, deadline)
        return
    for p in payloads:                     # one device: the payloads run here, one after the other
        if p["role"] == "tune":
            start = torch.load(p["start"], map_location="cpu", weights_only=False)
            tune_loop(ctx, start, Path(p["journal"]), p["worker"], deadline, Telemetry(Path(p["out"]), f"tune-w{p['worker']}"),
                      study_name=p.get("study"), keys=p.get("keys"), target=p.get("target"))
        else:
            for uid in p["units"]:
                if not steps.is_complete(Path(p["out"]) / f"{uid}.json"):
                    seed, stages, _ = unit_spec(uid, ctx.K)
                    stages = stages[:p["stage_limit"]] if p.get("stage_limit") else stages
                    if final_unit(ctx, uid, seed, p["lambda"], Path(p["out"]), deadline,
                                  start=None if p.get("scratch") else prev_models(seed), stages=stages,
                                  adam_max=p.get("adam_max", ADAM_MAX), lbfgs_max=p.get("lbfgs_max", LBFGS_MAX),
                                  tau=p.get("tau", TAU)) is None:
                        return


# --- the four stages ---------------------------------------------------------------------------------------
def _tokens() -> list:
    env = os.environ.get("FORGE_RUN_UNITS")
    return json.loads(env) if env else []


def _deadline(stage: str, start: float) -> float:
    b = steps.wall_budget_of(_tokens())
    return start + (b if b is not None else BUDGET_HOURS[stage] * 3600.0)


def _handoff_file(name: str) -> Path | None:
    for p in (results_dir() / name, handoff_dir() / name):
        if p.exists():
            return p
    return None


def prev_path(seed: int) -> Path | None:
    """theta_prev of `seed`. r09: the final networks of the r08 run, one state per stage
    (handoff-e6/prev-r8-s<seed>.pt); r06-r08 used those of r05 (prev-s<seed>.pt)."""
    return _handoff_file(f"prev-r8-s{seed}.pt")


def prev_models(seed: int) -> dict | None:
    p = prev_path(seed)
    return torch.load(p, map_location="cpu", weights_only=False) if p is not None else None


def run_p() -> dict:
    """Eq. 12 centre at theta_pre (pretrained.pt, pretrained here if absent) + the first part of the r06 study,
    whose trials start from theta_prev of seed 0, on every GPU."""
    start = time.time()
    deadline = _deadline("p", start)
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    ctx = build_ctx()
    tel = Telemetry(out, "pretrain-s0")
    pre_path = _handoff_file("pretrained.pt")
    if pre_path is not None:
        pre = torch.load(pre_path, map_location="cpu", weights_only=False)
    else:
        pre_path = out / "pretrained.pt"
        pre = pretrain(ctx, 0, ctx.K, tel, deadline)
        torch.save(pre, pre_path)
        tel.plot()
    start_path = prev_path(0)
    if start_path is None:
        raise FileNotFoundError("prev-s0.pt (theta_prev, r06) is neither in Results/e6 nor in handoff-e6/")
    journal = out / JOURNAL
    c = centre(ctx, pre, 0)
    tel.line(f"Eq. 12 centre (lambda_data = 1): {json.dumps({k: float(f'{v:.3e}') for k, v in c.items()})}")
    open_study(journal, 0, create_centre=c)
    if ctx.dev.type == "cuda":
        torch.cuda.empty_cache()
    n = max(_n_gpu(), 1)
    _run_payloads([{"role": "tune", "worker": w, "start": str(start_path), "journal": str(journal), "out": str(out),
                    "deadline": deadline} for w in range(min(n, 2))], ctx, deadline)
    return _close_study(out, journal, "p", start)


def run_q() -> dict:
    """The same study, continued from the handed-over journal and pretrained network."""
    start = time.time()
    deadline = _deadline("q", start)
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    journal = out / JOURNAL
    if not journal.exists():
        shutil.copy(handoff_dir() / JOURNAL, journal)
    start_path = prev_path(0)
    if start_path is None:
        raise FileNotFoundError("prev-s0.pt (theta_prev, r06) is neither in Results/e6 nor in handoff-e6/")
    ctx = build_ctx()
    n = max(_n_gpu(), 1)
    _run_payloads([{"role": "tune", "worker": 2 + w, "start": str(start_path), "journal": str(journal),
                    "out": str(out), "deadline": deadline} for w in range(min(n, 2))], ctx, deadline)
    return _close_study(out, journal, "q", start)


def _close_study(out: Path, journal: Path, stage: str, start: float) -> dict:
    s = study_summary(journal)
    s.update(stage=stage, hours=(time.time() - start) / 3600.0, created=steps._now())
    steps._write(out / f"study-{stage}.json", s)
    if "best_lambda" in s:
        steps._write(out / BEST, {"lambda": s["best_lambda"], "trial": s["best_trial"],
                                                "value": s["best_value"], "after_stage": stage})
    print(f"[e6{stage}] trials {s['n_trials']} · complete {s['complete']} · pruned {s['pruned']} · "
          f"failed {s['failed']} · best {s.get('best_value')}", flush=True)
    try:
        plot_history(out / "tune-w0.history.jsonl", out / f"study-{stage}.png")
    except Exception:
        pass
    return s


def _lambda_file(name: str) -> dict:
    p = _handoff_file(name)
    if p is None:
        raise FileNotFoundError(f"{name} is neither in Results/e6 nor in handoff-e6/")
    return json.loads(p.read_text())["lambda"]


def _lambda_star() -> dict:
    return _lambda_file(BEST)


def smoke_arm(ctx: Ctx, arm: str, lam: dict, out: Path, deadline: float, seed: int = 0) -> dict:
    """v09 paired smoke, one arm: from theta_prev^r08 of `seed`, SMOKE9["adam"] Adam steps on each stage of
    SMOKE9["stages"] (no L-BFGS), with the r09 loss when "J" is in `lam` (arm "r09") or the r08 loss ("ctl").
    Before and after each stage: validation current error (r07 Eq. 13 form, Eq. 20 current) and field error."""
    prev = prev_models(seed)
    tel = Telemetry(out, f"e6r9-smoke-{arm}")
    model = new_model(ctx, seed)
    params = list(model.parameters())
    rows = []
    for j in SMOKE9["stages"]:
        if time.time() > deadline - 60:
            tel.line(f"wall budget: stopping before stage {j}")
            break
        ts = time.time()
        model.load_state_dict(prev[j])
        before = {"current": current_error(ctx, model, "val", j), "field": field_error(ctx, model, "val", j)}
        obj = PhasedObjective(ctx, model, physics_batch(ctx, j, seed), j,
                              jbatch=current_batch(ctx, j, seed) if "J" in lam else None)
        opt, sched = _adam(params, SMOKE9["adam"])
        loss0, diverged = None, False
        for s in range(SMOKE9["adam"]):
            total, terms, g = obj.terms_and_grad(lam, params)
            f = float(total)
            loss0 = f if loss0 is None else loss0
            if not math.isfinite(f) or f > ctx.cfg.divergence_factor * max(loss0, 1e-300):
                diverged = True
                break
            _apply(params, g)
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
            if s % EVERY == 0:
                tel.log(phase="adam", stage=j, step=s, total=f, terms=terms, lr=sched.get_last_lr()[0])
        after = {"current": current_error(ctx, model, "val", j), "field": field_error(ctx, model, "val", j)}
        rows.append({"stage": j, "V": j * ctx.cfg.voltage_step, "before": before, "after": after,
                     "diverged": diverged, "seconds": time.time() - ts})
        tel.line(f"[{arm}] stage {j}: current err {before['current']:.3e} -> {after['current']:.3e} · "
                 f"field err {before['field']:.3e} -> {after['field']:.3e}{' DIVERGED' if diverged else ''} · "
                 f"{rows[-1]['seconds']:.0f}s")
    rec = {"unit": f"e6r9-smoke-{arm}", "step": "e6", "complete": True, "created": steps._now(), "arm": arm,
           "seed": seed, "lambda": lam, "smoke": SMOKE9, "rows": rows}
    steps._write(out / f"e6r9-smoke-{arm}.json", rec)
    return rec


def smoke_verdict(r09: dict, ctl: dict) -> dict:
    """The v09 pass rule: over SMOKE9["judged"], mean validation current error after the steps at least 20 %
    lower in r09 than in the control, mean field error of r09 at most 20 % above the control's, no divergence."""
    pick = lambda rec, key: [r["after"][key] for r in rec["rows"] if r["stage"] in SMOKE9["judged"]]
    cur9, curc = float(np.mean(pick(r09, "current"))), float(np.mean(pick(ctl, "current")))
    fld9, fldc = float(np.mean(pick(r09, "field"))), float(np.mean(pick(ctl, "field")))
    div = any(r["diverged"] for r in r09["rows"] + ctl["rows"])
    ok = (len(pick(r09, "current")) == len(SMOKE9["judged"]) == len(pick(ctl, "current"))
          and cur9 <= 0.8 * curc and fld9 <= 1.2 * fldc and not div)
    return {"current_r09": cur9, "current_ctl": curc, "field_r09": fld9, "field_ctl": fldc, "diverged": div,
            "pass": bool(ok)}


def run_h(arm: str | None = None) -> dict:
    """The v09 smoke, one arm per kernel: `arm` "r09" or "ctl" (or FORGE_RUN_UNITS ["arm:..."]; default r09)."""
    start = time.time()
    deadline = _deadline("h", start)
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    arm = arm or next((str(t)[4:] for t in _tokens() if str(t).startswith("arm:")), "r09")
    ctx = build_ctx()
    if arm == "r09":
        if _handoff_file(BEST) is None:     # Eq. 24 once, on the GPU (the operator's PC has none); handed off to F
            lam = dict(_lambda_file(BEST_R08))
            lam["J"] = centre_current(ctx, prev_models(0), 0)
            steps._write(out / BEST, {"lambda": lam, "rule": "r09 Eq. 24 at theta_prev^r08 seed 0, GRID stages",
                                      "lambda_r08": _lambda_file(BEST_R08)})
            print(f"[e6h] Eq. 24: lambda_J = {lam['J']:.4e}", flush=True)
        lam = _lambda_star()
    else:
        lam = _lambda_file(BEST_R08)
    return smoke_arm(ctx, arm, lam, out, deadline)


def run_f(units: list | None = None, account: int | None = None) -> list:
    """The final runs: `pack@<h>h:e6-A-s0+e6-A-s1+e6-A-s2` tokens, one pack per GPU. `account` (v09) takes the
    packs of that account from final_plan(), every GPU equally loaded; a seed whose parts are all here is merged."""
    start = time.time()
    deadline = _deadline("f", start)
    out = results_dir()
    lam = _lambda_star()
    packs = []
    default = [f"pack@10.5h:{UNIT}0", f"pack@10.5h:{UNIT}1"]   # v09: one seed per GPU; other accounts pass their own
    if account is not None:
        units = ["+".join(p) for p in final_plan()[account]]
    for tok in ([f"pack@10.5h:{u}" for u in units] if units else None) or _tokens() or default:
        m = steps._PACK.match(str(tok))
        packs.append(m.group(2).split("+") if m else [str(tok)])
    ctx = build_ctx()
    _run_payloads([{"role": "final", "units": p, "lambda": lam, "out": str(out), "deadline": deadline}
                   for p in packs], ctx, deadline)
    for base in sorted({unit_spec(u, ctx.K)[2] for p in packs for u in p if ".p" in u}):
        merge_parts(out, ctx.prep, base)
    return [u for p in packs for u in p if steps.is_complete(out / f"{u}.json")]


def run_fcheck(account: int = 1, stage_limit: int = 2, adam_max: int = 50, lbfgs_max: int = 20,
               hours: float = 0.4) -> dict:
    """Short Kaggle check of the v09 F split (operator 2026-11-14): the packs of `account` from final_plan(),
    one per GPU, each part cut to its first `stage_limit` stages with few optimizer steps; then the split seed
    is merged. Writes e6r9-fcheck.json: GPUs seen, units completed, and the merged stages."""
    start = time.time()
    deadline = start + hours * 3600.0
    out = results_dir() / "fcheck"
    out.mkdir(parents=True, exist_ok=True)
    # plumbing check: the r09 lambda when it was handed off, else r08's with lambda_J = 1 (the values are not judged)
    lam = _lambda_star() if _handoff_file(BEST) is not None else {**_lambda_file(BEST_R08), "J": 1.0}
    packs = final_plan()[account]
    ctx = build_ctx()
    lines = []

    def report(line):
        lines.append(line)
        print(line, flush=True)

    payloads = [{"role": "final", "units": p, "lambda": lam, "out": str(out), "deadline": deadline,
                 "stage_limit": stage_limit, "adam_max": adam_max, "lbfgs_max": lbfgs_max} for p in packs]
    if _n_gpu() >= 2:
        path = _save_ctx(ctx)
        for p in payloads:
            p["ctx"] = path
        _spawn(payloads, deadline, report=report)
    else:
        _run_payloads(payloads, ctx, deadline)
    units = {u: steps.is_complete(out / f"{u}.json") for p in packs for u in p}
    merged = {}
    for base in sorted({unit_spec(u, ctx.K)[2] for p in packs for u in p}):
        rec = merge_parts(out, ctx.prep, base)
        merged[base] = None if rec is None else {"stages_done": rec["stages_done"],
                                                 "stages": [r["stage"] for r in rec["stage_progress"]]}
    gpus_seen = sorted({l.split("]")[0][1:] for l in lines if l.startswith("[gpu")})
    rec = {"unit": "e6r9-fcheck", "complete": True, "created": steps._now(), "account": account,
           "n_gpu": _n_gpu(), "gpus_seen": gpus_seen, "packs": packs, "units_complete": units, "merged": merged,
           "seconds": time.time() - start,
           "pass": _n_gpu() >= 2 and len(gpus_seen) == 2 and all(units.values())}
    steps._write(results_dir() / "e6r9-fcheck.json", rec)
    print(f"[fcheck] {json.dumps({k: rec[k] for k in ('n_gpu', 'gpus_seen', 'units_complete', 'merged', 'pass')})}",
          flush=True)
    return rec


def run_from_environment(stage: str):
    return {"p": run_p, "q": run_q, "h": run_h, "f": run_f}[stage]()
