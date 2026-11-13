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
from . import data, diagnostics, pce, steps, training, weights
from .config import Config
from .units import UnitSpec

__provenance__ = {
    "revision": "experiments-protocolo-experimental-de-la-pinn-por-etapas-sobre-dpv-v04.md",
    "sections": ["E6"],
    "experiments": ["E6"],
    "equations": [12, 13, 18, 19, 20],
    "invariants": [],
}
# r06 (proposal): Eq. 13 scores a trial by the terminal-current error (normalized by J_sc) instead of the field
# error of Eq. 18, and both the study and the final training start from theta_prev, the final networks of the r05
# run (one per seed and stage, handed over as handoff-e6/prev-s<seed>.pt).

# --- the parameters table of E6 (experiments v04) ---------------------------------------------------------
PRE_STEPS = 500             # pretraining Adam steps per stage
LR0, LR_END = 1e-3, 5e-5    # Adam, cosine from LR0 to LR_END
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
BUDGET_HOURS = {"p": 2.2, "q": 2.2, "h": 0.15, "f": 10.5}   # h: smoke capped at 9 min (operator, 2026-11-13)

PRE_DEVICES, PRE_NODES = 64, 32   # data batch of one pretraining step (same size as the physics batch of E1)
LOSS_MODE = "relative"            # per-node relative residual (r04 Eq. 10), the control arm of E1
TERMS = training.TERMS + ("data",)
STUDY = "e6r7"                    # r07 study: Eq. 13 read through the Eq. 20 current, so no earlier journal is continued
JOURNAL = "optuna_journal_r07.log"
BEST = "best_lambda_r07.json"
UNIT = "e6r7-A-s"                 # final units of r07; r06 units were e6r6-A-s<seed>, r05 units e6-A-s<seed>


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
    currents = {s: _currents(prep.refs[s], K, prep.scales) for s in ("val", "test")}
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
def stage_batch(ctx: Ctx, split: str, stage: int, key, n_dev: int, n_nodes: int) -> dict:
    """`training.make_batch`'s selection (same rng recipe) plus the stored outputs at the chosen nodes."""
    bank = getattr(ctx.prep, split)
    eligible = np.nonzero(bank.n_V > stage)[0]
    rng = np.random.default_rng(key)
    pick = np.sort(rng.choice(eligible, size=min(n_dev, eligible.size), replace=False))
    N = bank.x_hat.shape[1]
    k = min(n_nodes, N - 2)
    idx = np.argsort(rng.random((pick.size, N - 2)), axis=1)[:, :k] + 1
    b = training._gather(bank, pick, idx, ctx.dev)
    tgt = ctx.fields[split][pick[:, None], stage, :, idx]            # (P, k, 3)
    b["target"] = training._t(tgt.reshape(-1, 3).astype(np.float64), ctx.dev)
    return b


def physics_batch(ctx: Ctx, stage: int, seed: int) -> dict:
    """The fixed batch of a stage: exactly the nodes `training.make_batch` gives E1 for (seed, stage)."""
    return stage_batch(ctx, "train", stage, [int(seed), int(stage), 17], ctx.cfg.n_devices_per_batch,
                       ctx.cfg.n_nodes_per_device)


def data_loss(model, b: dict, V_tilde: float, scale: torch.Tensor) -> torch.Tensor:
    """Eq. 18 of r05 on a batch: < ((phi-phi*)/D_phi)^2 + sum_k ((Y_k-Y_k*)/(Y_k^max-Y_k^min))^2 >."""
    phi, pn, pp = model(b["x"], b["xi"], b["peq0"], b["peq1"], V_tilde)
    pred = torch.cat([phi, pn, pp], dim=1)
    return (((pred - b["target"]) / scale) ** 2).sum(dim=1).mean()


class PhasedObjective(training.Objective):
    """Eq. 11 of r05 over I' = I u {data}: E1's relative residual terms plus the data term on the same batch."""

    def __init__(self, ctx: Ctx, model, batch: dict, stage: int) -> None:
        super().__init__(model, batch, ctx.V(stage), ctx.sigma2, ctx.prep.material, ctx.prep.contacts,
                         mode=LOSS_MODE)
        self.scale = ctx.scale

    def _chunk_losses(self, b: dict) -> tuple:
        L, n = super()._chunk_losses(b)
        L["data"] = data_loss(self.model, b, self.V, self.scale)
        n["data"] = int(b["x"].shape[0])
        return L, n

    def _counts(self) -> list:
        out = super()._counts()
        for c, b in zip(out, self.chunks):
            c["data"] = int(b["x"].shape[0])
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
        for k in TERMS:
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
        for k in TERMS:
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


def open_study(journal: Path, worker: int = 0, create_centre: dict | None = None):
    import optuna
    sampler = optuna.samplers.TPESampler(seed=TPE_SEED + worker, n_startup_trials=TPE_STARTUP)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=TPE_STARTUP, n_warmup_steps=1)
    study = optuna.create_study(study_name=STUDY, storage=_storage(journal), direction="minimize",
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    if create_centre is not None and "centre" not in study.user_attrs:
        study.set_user_attr("centre", create_centre)
    return study


def full_lambda(phys: dict) -> dict:
    return {**{k: float(phys[k]) for k in training.TERMS}, "data": LAMBDA_DATA}


def run_trial(ctx: Ctx, start: dict, lam: dict, seed: int, deadline: float, trial=None, tel=None) -> float:
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
        obj = PhasedObjective(ctx, model, physics_batch(ctx, j, seed), j)
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


def tune_loop(ctx: Ctx, start: dict, journal: Path, worker: int, deadline: float, tel: Telemetry) -> None:
    import optuna
    from optuna.trial import TrialState
    study = open_study(journal, worker)
    c = study.user_attrs["centre"]
    longest = 0.0
    while True:
        if time.time() + 1.05 * longest > deadline:
            tel.line(f"wall budget: no room for another trial (longest {longest:.0f}s); stopping")
            return
        trial = study.ask()
        lam = full_lambda({k: 10.0 ** trial.suggest_float(f"log10_{k}", math.log10(c[k]) - BOX_DECADES,
                                                          math.log10(c[k]) + BOX_DECADES) for k in training.TERMS})
        t = time.time()
        try:
            value = run_trial(ctx, start, lam, 0, deadline, trial, tel)
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


def study_summary(journal: Path) -> dict:
    from optuna.trial import TrialState
    study = open_study(journal)
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
                   best_lambda=full_lambda({k: 10.0 ** b.params[f"log10_{k}"] for k in training.TERMS}))
    return out


# --- the final recipe -----------------------------------------------------------------------------------
def final_unit(ctx: Ctx, uid: str, seed: int, lam: dict, out: Path, deadline: float, n_stages: int | None = None,
               adam_max: int = ADAM_MAX, lbfgs_max: int = LBFGS_MAX, start: dict | None = None) -> dict | None:
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
    for j in range(len(st["rows"]), K):
        if st["diverged"]:
            break
        if time.time() + 1.05 * longest > deadline:
            tel.line(f"wall budget: stage {j} would not fit (longest {longest:.0f}s); checkpoint kept")
            save()
            return None
        ts = time.time()
        model.load_state_dict(st["pre"][j])
        obj = PhasedObjective(ctx, model, physics_batch(ctx, j, seed), j)
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
                frac = (j + (s + 1) / (adam_max + lbfgs_max)) / K
                eta = tel.progress(frac, phase="adam", stage=j, step=s)
                tel.line(f"stage {j + 1}/{K} · adam {s}/{adam_max} · loss {f:.3e} · ETA {eta}")
            if len(losses) > WINDOW and (losses[-WINDOW - 1] - losses[-1]) / abs(losses[-WINDOW - 1]) < TAU:
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
                frac = (j + (adam_max + min(k, lbfgs_max)) / (adam_max + lbfgs_max)) / K
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
        partial = np.array(list(training.errors_from_curves(st["J_test"], prep.test, j + 1, prep.scales).values()))
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
    errs = ({u: float("nan") for u in prep.test.uids} if st["diverged"]
            else training.errors_from_curves(st["J_test"], prep.test, done, prep.scales))
    seconds = st["seconds"] + time.time() - t0
    torch.save({"pretrained": st["pre"], "final": st["stage_models"]}, out / f"{uid}.models.pt")
    rec = {"unit": uid, "step": "e6", "complete": True, "created": steps._now(), "seed": seed,
           "config": ctx.cfg.to_dict(), "lambda": lam, "loss_mode": LOSS_MODE, "sigma2": ctx.sigma2,
           "recipe": {"pre_steps": PRE_STEPS, "adam_max": adam_max, "tau": TAU, "window": WINDOW,
                      "lbfgs_max": lbfgs_max, "lbfgs_history": LBFGS_HISTORY, "lr": [LR0, LR_END], "clip": CLIP},
           "stages_done": done, "stages_planned": K, "diverged": st["diverged"], "stage_progress": st["rows"],
           "per_device_error": errs, "seconds": seconds,
           "gpu_hours": seconds / 3600.0 if dev.type == "cuda" else 0.0, "device": str(dev)}
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
        tune_loop(ctx, start, Path(a["journal"]), a["worker"], a["deadline"], Telemetry(out, f"tune-w{a['worker']}"))
    else:
        lam = a["lambda"]
        for uid in a["units"]:
            if steps.is_complete(out / f"{uid}.json"):
                print(f"[{uid}] complete, skipped", flush=True)
                continue
            seed = int(uid.rsplit("s", 1)[1])
            if final_unit(ctx, uid, seed, lam, out, a["deadline"], start=prev_models(seed)) is None:
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
            tune_loop(ctx, start, Path(p["journal"]), p["worker"], deadline, Telemetry(Path(p["out"]), f"tune-w{p['worker']}"))
        else:
            for uid in p["units"]:
                if not steps.is_complete(Path(p["out"]) / f"{uid}.json"):
                    seed = int(uid.rsplit("s", 1)[1])
                    if final_unit(ctx, uid, seed, p["lambda"], Path(p["out"]), deadline,
                                  start=prev_models(seed)) is None:
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
    """theta_prev of `seed`: the final networks of the r05 run, one state per stage (handoff-e6/prev-s<seed>.pt)."""
    return _handoff_file(f"prev-s{seed}.pt")


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


def _lambda_star() -> dict:
    p = _handoff_file(BEST)
    if p is None:
        raise FileNotFoundError(f"{BEST} is neither in Results/e6 nor in handoff-e6/")
    return json.loads(p.read_text())["lambda"]


def run_h() -> dict | None:
    """The single smoke run of the final recipe: seed 0, stages 0-2, 100 Adam steps and 50 L-BFGS iterations."""
    start = time.time()
    deadline = _deadline("h", start)
    out = results_dir()
    ctx = build_ctx()
    return final_unit(ctx, f"{UNIT[:-2]}-smoke-s0", 0, _lambda_star(), out, deadline, n_stages=SMOKE["stages"],
                      adam_max=SMOKE["adam_max"], lbfgs_max=SMOKE["lbfgs_max"], start=prev_models(0))


def run_f() -> list:
    """The final runs: `pack@<h>h:e6-A-s0+e6-A-s1+e6-A-s2` tokens, one pack per GPU."""
    start = time.time()
    deadline = _deadline("f", start)
    out = results_dir()
    lam = _lambda_star()
    packs = []
    default = [f"pack@10.5h:{UNIT}0+{UNIT}1+{UNIT}2", f"pack@10.5h:{UNIT}3+{UNIT}4"]   # 3 seeds on GPU 0, 2 on GPU 1
    for tok in _tokens() or default:
        m = steps._PACK.match(str(tok))
        packs.append(m.group(2).split("+") if m else [str(tok)])
    ctx = build_ctx()
    _run_payloads([{"role": "final", "units": p, "lambda": lam, "out": str(out), "deadline": deadline}
                   for p in packs], ctx, deadline)
    return [u for p in packs for u in p if steps.is_complete(out / f"{u}.json")]


def run_from_environment(stage: str):
    return {"p": run_p, "q": run_q, "h": run_h, "f": run_f}[stage]()
