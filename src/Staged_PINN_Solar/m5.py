"""Network M5 of r13: five outputs and the scaled mixed system, plus the E8 smoke of experiments v14.

Outputs (r13 Eqs. 3 and 28): phi, phi_n, phi_p with the ansatz of Eq. 3, and the scaled carrier currents
    jn = (1 - x_hat) u_jn,   jp = x_hat u_jp,   J_n = J_gen jn,  J_p = J_gen jp,  J_gen = G_bar / delta,
so the minority contacts J_n(1) = J_p(0) = 0 hold by construction (S_n1 = S_p0 = 0 in the whole dpvB2 base).

Loss (r13 Eq. 32), every weight 1:
    P  = [delta^2 (eps phi')' - (n - p - Ndop)] / N*                  (Eq. 29, N* = max |Ndop| of the device)
    Cn = jn' - (R - G) / G_bar,        Cp = jp' + (R - G) / G_bar       (Eq. 29)
    Kn = [phi_n' - G_bar jn / (delta^2 mu_n n)] / s_K,  Kp likewise     (Eq. 29, s_K = rms slope over train)
    Bn = phi_n(0) - log(1 + jn(0) / beta_n0),  Bp = -(V + phi_p(1)) - log(1 + jp(1) / beta_p1)   (Eq. 30)
    data = Eq. 18 (fields at the nodes, scaled currents at the dPV interface midpoints)
    J    = ((J0 J_gen jn(0) - J_dPV) / J_sc_dPV)^2                        (Eqs. 23 and 31)
Collocation (Eq. 33): every node within W_J of the junction and within W_C of each contact, plus K_U uniform.

Smoke (operator 2026-11-15): pretraining PRE_STEPS per stage on the data term, then ADAM_STEPS Adam steps per
stage on Eq. 32 from that stage's theta_pre; every EVERY steps the log prints every term, the total and the
validation current error with its PCE-error bound; at the end the validation PCE of the final network and of
theta_pre (the data-only control) per material.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import carriers, data, data_m, e6, network, pce, residuals, scratch_m, steps, training

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r13.md",
    "sections": ["2", "3", "4", "5", "6"],
    "equations": ["18", "23", "28", "29", "30", "31", "32", "33"],
    "invariants": [
        "m5_has_267781_parameters_and_five_outputs",
        "minority_contact_currents_vanish_by_construction",
        "scaled_system_closes_on_dpv_references",
        "terminal_current_matches_the_gr_identity_on_dpv",
        "collocation_windows_hold_every_junction_and_contact_node",
        "smoke_runs_on_a_tiny_cpu_context",
    ],
}

W_J, W_C, K_U = 0.05, 0.05, 32       # Eq. 33 windows (experiments v13 E8)
PRE_STEPS, ADAM_STEPS, EVERY = 150, 500, 25    # smoke (operator 2026-11-15)
PRE_LR = 1e-3                        # measured locally: at 1e-4, 150 steps barely move the data term
N_DEV = 64                           # devices per stage batch
SEED = 0
HOURS = 9.0
TERMS = ("P", "Cn", "Cp", "Kn", "Kp", "Bn", "Bp", "data_f", "data_j", "J")   # data = data_f + data_j (Eq. 18)
TAG = "e8s"


def results_dir() -> Path:
    return steps.results_root() / TAG


# --- network ---------------------------------------------------------------------------------------------
class Arch5(torch.nn.Module):
    """One MLP (network M's width, depth and activation) with five raw outputs: u_phi = tanh, u_n, u_p = sigmoid,
    u_jn, u_jp linear (the currents change sign along the sweep)."""

    def __init__(self, n_xi: int, width: int, depth: int, seed: int) -> None:
        super().__init__()
        state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        try:
            self.net = network._mlp(1 + n_xi, 5, width, depth)
        finally:
            torch.random.set_rng_state(state)

    def forward(self, x_hat, xi_hat):
        r = self.net(torch.cat([x_hat, xi_hat], dim=1))
        return torch.cat([torch.tanh(r[:, 0:1]), torch.sigmoid(r[:, 1:3]), r[:, 3:5]], dim=1)


def equilibrium_start(arch: Arch5, bounds) -> Arch5:
    """network.equilibrium_start for five outputs: u_phi = 0, phi_n = phi_p = 0, jn = jp = 0 at the start."""
    biases = [0.0]
    for lo, hi in (bounds.phi_n, bounds.phi_p):
        if not lo < 0.0 < hi:
            raise ValueError(f"equilibrium start needs Y_min < 0 < Y_max, got ({lo}, {hi})")
        biases.append(float(torch.logit(torch.tensor(-lo / (hi - lo), dtype=torch.float64))))
    with torch.no_grad():
        last = arch.net[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor(biases + [0.0, 0.0], dtype=last.bias.dtype))
    return arch


class Ansatz5(torch.nn.Module):
    """Eq. 3 for (phi, phi_n, phi_p) and Eq. 28 for (jn, jp)."""

    def __init__(self, arch: Arch5, bounds) -> None:
        super().__init__()
        self.arch, self.bounds = arch, bounds

    def forward(self, x_hat, xi_hat, phi_eq0, phi_eq1, V_tilde):
        u = self.arch(x_hat, xi_hat)
        b = self.bounds
        phi = phi_eq0 + (phi_eq1 + V_tilde - phi_eq0) * x_hat + 4.0 * x_hat * (1.0 - x_hat) * b.D_phi * u[:, 0:1]
        pn = b.phi_n[0] + (b.phi_n[1] - b.phi_n[0]) * u[:, 1:2]
        pp = b.phi_p[0] + (b.phi_p[1] - b.phi_p[0]) * u[:, 2:3]
        return phi, pn, pp, (1.0 - x_hat) * u[:, 3:4], x_hat * u[:, 4:5]


def new_model(ctx, seed: int) -> Ansatz5:
    arch = Arch5(ctx.prep.n_xi, ctx.cfg.width, ctx.cfg.depth, seed)
    if ctx.cfg.start == "equilibrium":
        equilibrium_start(arch, ctx.prep.bounds)
    return Ansatz5(arch, ctx.prep.bounds).to(ctx.dev)


def n_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --- per-device scales and the stored interface currents ------------------------------------------------
def device_scales(bank, contacts) -> dict:
    """G_bar, J_gen, N*, beta_n0, beta_p1 per device (r13 Eqs. 28-30), from the bank alone."""
    x, G = bank.x_hat, bank.G
    Gbar = np.sum(0.5 * (G[:, 1:] + G[:, :-1]) * np.diff(x, axis=1), axis=1)
    Jgen = Gbar / bank.delta
    mat = bank.mat
    n_eq0 = mat["Nc"] * np.exp(bank.phi_eq0 + mat["chi"])                         # Eq. 4, phi_n = 0, x = 0
    p_eq1 = mat["Nv"] * np.exp(-bank.phi_eq1 - mat["chi"] - mat["Eg"])            # Eq. 4, phi_p = 0, x = 1
    return {"Gbar": Gbar, "Jgen": Jgen, "Nstar": np.max(np.abs(bank.Ndop), axis=1),
            "beta_n0": contacts.S_n0 * n_eq0 / Jgen, "beta_p1": contacts.S_p1 * p_eq1 / Jgen}


def interface_currents(refs, K: int, Jgen: np.ndarray) -> np.ndarray:
    """(D, K, 2, N-1) float32 scaled dPV currents J_n / J_gen, J_p / J_gen at the interface midpoints."""
    out = None
    for d, ref in enumerate(refs):
        dev = data_m.load_device(ref)
        Jn, Jp = np.asarray(dev.J_n, float), np.asarray(dev.J_p, float)
        if out is None:
            out = np.full((len(refs), K, 2, Jn.shape[-1]), np.nan, dtype=np.float32)
        n = min(Jn.shape[0], K)
        out[d, :n, 0], out[d, :n, 1] = Jn[:n] / Jgen[d], Jp[:n] / Jgen[d]
    return out


def slope_scale(fields: np.ndarray, x_hat: np.ndarray) -> float:
    """s_K: rms of d phi_n / dx_hat and d phi_p / dx_hat over every training device and stage (r13 premise xvi)."""
    h = np.diff(x_hat, axis=1)[:, None, :]
    total, count = 0.0, 0
    for c in (1, 2):
        s = np.diff(fields[:, :, c, :].astype(np.float64), axis=-1) / h
        ok = np.isfinite(s)
        total += float(np.sum(s[ok] ** 2))
        count += int(ok.sum())
    return math.sqrt(total / count)


def attach(ctx) -> None:
    """Adds ctx.r13 (per split: scales and scaled interface currents) and ctx.sK; reads each device's currents."""
    contacts = data.device_contacts(ctx.prep.scales)
    if contacts.S_n1 != 0 or contacts.S_p0 != 0:
        raise ValueError("Eq. 28 needs blocking minority contacts (S_n1 = S_p0 = 0)")
    ctx.r13 = {}
    for split in ("train", "val", "test"):
        bank = getattr(ctx.prep, split)
        sc = device_scales(bank, contacts)
        sc["jref"] = interface_currents(ctx.prep.refs[split], ctx.K, sc["Jgen"])
        ctx.r13[split] = sc
    ctx.sK = slope_scale(ctx.fields["train"], ctx.prep.train.x_hat)


def build_ctx(limit: int | None = None, root=None, cfg=None):
    ctx = scratch_m.build_ctx(cfg, limit=limit, root=root)
    attach(ctx)
    return ctx


# --- batches ---------------------------------------------------------------------------------------------
def batch(ctx, split: str, stage: int, key, n_dev: int | None = None) -> dict:
    """Eq. 33 nodes of `n_dev` devices whose curve reaches `stage`, with targets at the nodes (fields) and at the
    interface midpoint to the right of each node (scaled currents), and the per-device scales."""
    n_dev = N_DEV if n_dev is None else n_dev
    bank, sc = getattr(ctx.prep, split), ctx.r13[split]
    eligible = np.nonzero(bank.n_V > stage)[0]
    rng = np.random.default_rng(key)
    pick = np.sort(rng.choice(eligible, size=min(n_dev, eligible.size), replace=False))
    N = bank.x_hat.shape[1]
    idx = e6.junction_nodes(bank.x_hat[pick], bank.Ndop[pick], min(K_U, N - 2), rng, half_width=W_J, edge_width=W_C)
    k = idx.shape[1]
    b = training._gather(bank, pick, idx, ctx.dev)
    t = lambda a: training._t(np.asarray(a, dtype=np.float64), ctx.dev)
    b["target"] = t(ctx.fields[split][pick[:, None], stage, :, idx].reshape(-1, 3))
    im = np.minimum(idx, N - 2)
    rows = pick[:, None]
    b["x_mid"] = t((0.5 * (bank.x_hat[rows, im] + bank.x_hat[rows, im + 1])).reshape(-1, 1))
    b["jtarget"] = t(sc["jref"][rows, stage, :, im].reshape(-1, 2))
    for f in ("Gbar", "Nstar"):
        b[f] = t(np.repeat(sc[f][pick], k)[:, None])
    for f in ("Jgen", "beta_n0", "beta_p1"):
        b[f] = t(sc[f][pick])
    ref = ctx.currents[split]
    b["J_ref"], b["J_sc"] = t(ref[pick, stage]), t(ref[pick, 0])
    b["n_dev"], b["k"] = int(pick.size), int(k)
    return b


# --- loss ------------------------------------------------------------------------------------------------
def data_terms(model, b: dict, V: float, scale: torch.Tensor) -> tuple:
    """Eq. 18 of r13: (fields at the nodes, scaled currents at the interface midpoints)."""
    phi, pn, pp, _, _ = model(b["x"], b["xi"], b["peq0"], b["peq1"], V)
    fields = (((torch.cat([phi, pn, pp], dim=1) - b["target"]) / scale) ** 2).sum(dim=1).mean()
    _, _, _, jn, jp = model(b["x_mid"], b["xi"], b["peq0"], b["peq1"], V)
    currents = ((torch.cat([jn, jp], dim=1) - b["jtarget"]) ** 2).sum(dim=1).mean()
    return fields, currents


def contact_terms(model, b: dict, V: float, ctx) -> tuple:
    """Eq. 30 residuals per device and the terminal current of Eq. 31 [A/cm^2]."""
    phi, pn, pp, jn, jp = model(b["xc"], b["xic"], b["peq0c"], b["peq1c"], V)
    two = lambda a: a.reshape(b["n_dev"], 2)
    pn, pp, jn, jp = two(pn), two(pp), two(jn), two(jp)
    Bn = pn[:, 0] - torch.log1p(jn[:, 0] / b["beta_n0"])
    Bp = -(V + pp[:, 1]) - torch.log1p(jp[:, 1] / b["beta_p1"])
    J = ctx.prep.scales.J0 * b["Jgen"] * jn[:, 0]
    return Bn, Bp, J


def residual_terms(model, b: dict, V: float, sK: float) -> dict:
    """Eq. 29 at the collocation nodes, by autograd."""
    x = b["x"].clone().requires_grad_(True)
    phi, pn, pp, jn, jp = model(x, b["xi"], b["peq0"], b["peq1"], V)
    m = training.batch_material(b, "mat", None, b["ndop"]).like(phi)
    n, p = carriers.electron_density(phi, pn, m), carriers.hole_density(phi, pp, m)
    src = (carriers.recombination(n, p, m) - b["G"]) / b["Gbar"]
    d = residuals.autograd_derivative
    dl2 = b["delta"] ** 2
    return {"P": (dl2 * d(m.eps_r * d(phi, x), x) - (n - p - m.Ndop)) / b["Nstar"],
            "Cn": d(jn, x) - src, "Cp": d(jp, x) + src,
            "Kn": (d(pn, x) - b["Gbar"] * jn / (dl2 * m.mu_n * n)) / sK,
            "Kp": (d(pp, x) - b["Gbar"] * jp / (dl2 * m.mu_p * p)) / sK}


def loss_terms(model, b: dict, V: float, ctx) -> dict:
    """Every term of Eq. 32 (each a mean of squares)."""
    L = {k: (r ** 2).mean() for k, r in residual_terms(model, b, V, ctx.sK).items()}
    Bn, Bp, J = contact_terms(model, b, V, ctx)
    L["Bn"], L["Bp"] = (Bn ** 2).mean(), (Bp ** 2).mean()
    L["data_f"], L["data_j"] = data_terms(model, b, V, ctx.scale)
    L["J"] = (((J - b["J_ref"]) / b["J_sc"]) ** 2).mean()
    return L


# --- validation metrics ----------------------------------------------------------------------------------
@torch.no_grad()
def val_currents(ctx, model, stage: int, split: str = "val") -> tuple:
    """(devices reaching `stage`, J(V_j; theta) of Eq. 31 [A/cm^2]) on `split`."""
    bank, sc = getattr(ctx.prep, split), ctx.r13[split]
    ids = np.nonzero(bank.n_V > stage)[0]
    t = lambda a: training._t(np.asarray(a, dtype=np.float64), ctx.dev)
    zero = t(np.zeros((ids.size, 1)))
    _, _, _, jn, _ = model(zero, t(bank.xi_hat[ids]), t(bank.phi_eq0[ids][:, None]), t(bank.phi_eq1[ids][:, None]),
                           ctx.V(stage))
    return ids, ctx.prep.scales.J0 * sc["Jgen"][ids] * jn[:, 0].cpu().numpy()


def current_metrics(ctx, model, stage: int, split: str = "val") -> dict:
    """Median |J - J_dPV| / J_sc and the median PCE-error bound V_j |J - J_dPV| / P_in in percentage points
    (the maximum is 1-Lipschitz in the sup norm, so this bounds the PCE error this stage can cause)."""
    bank = getattr(ctx.prep, split)
    ids, J = val_currents(ctx, model, stage, split)
    ref = ctx.currents[split]
    err = np.abs(J - ref[ids, stage])
    V = stage * ctx.cfg.voltage_step
    return {"current_rel": float(np.median(err / np.abs(ref[ids, 0]))),
            "pce_bound_pp": float(np.median(100.0 * V * err * 1e4 / bank.p_in_std[ids]))}


@torch.no_grad()
def field_error(ctx, model, stage: int, split: str = "val", chunk: int = 32) -> float:
    """Eq. 18 fields part over every node of every device of `split` reaching `stage`."""
    bank, F = getattr(ctx.prep, split), ctx.fields[split]
    ids = np.nonzero(bank.n_V > stage)[0]
    N = bank.x_hat.shape[1]
    tot, cnt = 0.0, 0
    for s in range(0, ids.size, chunk):
        pick = ids[s:s + chunk]
        b = training._gather(bank, pick, np.tile(np.arange(N), (pick.size, 1)), ctx.dev)
        tgt = training._t(F[pick[:, None], stage, :, np.tile(np.arange(N), (pick.size, 1))].reshape(-1, 3).astype(np.float64), ctx.dev)
        phi, pn, pp, _, _ = model(b["x"], b["xi"], b["peq0"], b["peq1"], ctx.V(stage))
        e = (((torch.cat([phi, pn, pp], dim=1) - tgt) / ctx.scale) ** 2).sum(dim=1)
        tot, cnt = tot + float(e.sum()), cnt + int(e.numel())
    return tot / cnt if cnt else float("nan")


# --- training --------------------------------------------------------------------------------------------
def _terms_line(L: dict) -> str:
    return " ".join(f"{k} {float(v):.2e}" for k, v in L.items())


def pretrain(ctx, seed: int, n_stages: int, tel, deadline: float | None = None, n_steps: int = PRE_STEPS) -> dict:
    """theta_pre: Adam on Eq. 18 alone (fields and scaled currents), n_steps per stage, stages in continuation,
    a fresh batch per step. Every EVERY steps the log shows both data parts and the total."""
    model = new_model(ctx, seed)
    params = list(model.parameters())
    states = {}
    t0 = time.time()
    for j in range(n_stages):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"wall budget reached during pretraining at stage {j}")
        opt = torch.optim.Adam(params, lr=PRE_LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps, 1), eta_min=e6.LR_END)
        for s in range(n_steps):
            b = batch(ctx, "train", j, [int(seed), j, s, 29])
            f, c = data_terms(model, b, ctx.V(j), ctx.scale)
            L = f + c
            if not torch.isfinite(L):
                raise FloatingPointError(f"pretraining diverged at stage {j} step {s}")
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, e6.CLIP)
            opt.step()
            sched.step()
            if s % EVERY == 0 or s == n_steps - 1:
                f, c, L = f.item(), c.item(), L.item()
                tel.log(phase="pretrain", stage=j, step=s, total=L, terms={"data_f": f, "data_j": c})
                tel.line(f"pretrain · stage {j} ({j + 1}/{n_stages}) · step {s + 1}/{n_steps} | data_fields "
                         f"{f:.3e} data_currents {c:.3e} | total {L:.3e}")
        states[j] = e6._cpu_state(model)
        m = current_metrics(ctx, model, j)
        el = time.time() - t0
        tel.line(f"pretrain stage {j} done · val field err {field_error(ctx, model, j):.3e} · val |dJ|/Jsc "
                 f"{m['current_rel']:.3e} · PCE bound {m['pce_bound_pp']:.3f} pp · ETA "
                 f"{e6._hms(el / (j + 1) * (n_stages - j - 1))}")
    return states


def adam_stage(ctx, start: dict, stage: int, seed: int, tel, n_steps: int = ADAM_STEPS, pos: str = "") -> dict:
    """ADAM_STEPS steps on Eq. 32 from theta_pre of the stage, on the stage's fixed Eq. 33 batch."""
    model = new_model(ctx, seed)
    model.load_state_dict(start)
    params = list(model.parameters())
    b = batch(ctx, "train", stage, [int(seed), int(stage), 17])
    opt, sched = e6._adam(params, n_steps)
    V = ctx.V(stage)
    hist, first, t0, diverged = [], None, time.time(), False
    for s in range(n_steps + 1):
        L = loss_terms(model, b, V, ctx)
        total = sum(L.values())
        if not torch.isfinite(total):
            diverged = True
            tel.line(f"stage {stage}: non-finite loss at step {s}: {_terms_line(L)}")
            break
        if s % EVERY == 0 or s == n_steps:
            m = current_metrics(ctx, model, stage)
            rec = {"step": s, "total": float(total), "terms": {k: float(v) for k, v in L.items()}, **m}
            hist.append(rec)
            first = first or rec
            tel.log(phase="adam", stage=stage, **rec)
            el = time.time() - t0
            eta = e6._hms(el / max(s, 1) * (n_steps - s)) if s else "?"
            tel.line(f"adam · stage {stage}{pos} · step {s}/{n_steps} | {_terms_line(L)} | total {float(total):.3e} | "
                     f"val |dJ|/Jsc {m['current_rel']:.3e} · PCE bound {m['pce_bound_pp']:.3f} pp | stage ETA {eta}")
        if s == n_steps:
            break
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, e6.CLIP)
        opt.step()
        sched.step()
    ids, J = val_currents(ctx, model, stage)
    return {"stage": stage, "steps": n_steps, "diverged": diverged, "history": hist, "start": first,
            "end": hist[-1] if hist else None, "field_error_val": field_error(ctx, model, stage),
            "val_ids": ids.tolist(), "J_val": J.tolist(), "seconds": time.time() - t0, "model": e6._cpu_state(model)}


def _stage_file(out: Path, j: int) -> Path:
    return out / f"{TAG}-j{j:02d}.json"


def smoke_worker(ctx, a: dict) -> None:
    """Role "call": the Adam stages of one GPU, each written as soon as it ends (a rerun skips written ones)."""
    out = Path(a["out"])
    pre = torch.load(a["start_file"], map_location="cpu", weights_only=False)
    tel = e6.Telemetry(out, f"{TAG}-w{a['worker']}")
    for i, j in enumerate(a["stages"]):
        if _stage_file(out, j).exists():
            continue
        if time.time() > a["deadline"] - 900:
            tel.line(f"wall budget: stage {j} not started")
            return
        rec = adam_stage(ctx, pre[j], j, a["seed"], tel, a.get("steps", ADAM_STEPS), f" ({i + 1}/{len(a['stages'])})")
        torch.save(rec.pop("model"), out / f"{TAG}-j{j:02d}.pt")
        steps._write(_stage_file(out, j), rec)
        s, e = rec["start"], rec["end"]
        tel.line(f"stage {j} done in {rec['seconds'] / 60:.1f} min: total {s['total']:.2e}->{e['total']:.2e} | "
                 + " ".join(f"{k} {s['terms'][k]:.1e}->{e['terms'][k]:.1e}" for k in e["terms"])
                 + f" | val |dJ|/Jsc {s['current_rel']:.2e}->{e['current_rel']:.2e}")


def pce_table(ctx, curves: dict, split: str = "val") -> dict:
    """Per material: devices with every stage of their curve, median |PCE - PCE_dPV| (pp) and relative (%)."""
    bank, ref = getattr(ctx.prep, split), ctx.currents[split]
    rows = {}
    for d in range(len(bank)):
        nv = int(min(bank.n_V[d], ctx.K))
        J = [curves.get(j, {}).get(d) for j in range(nv)]
        if any(v is None for v in J):
            continue
        V = ctx.cfg.voltage_step * np.arange(nv)
        p = pce.jv_summary(V, np.asarray(J), bank.p_in_std[d])["pce"]
        p0 = pce.jv_summary(V, ref[d, :nv], bank.p_in_std[d])["pce"]
        mat = bank.uids[d].split("/")[0]
        rows.setdefault(mat, []).append((100 * abs(p - p0), 100 * abs(p - p0) / p0 if p0 else math.nan))
    return {m: {"devices": len(v), "median_abs_pp": float(np.nanmedian([a for a, _ in v])),
                "median_rel_pct": float(np.nanmedian([r for _, r in v]))} for m, v in sorted(rows.items())}


def summarize(ctx, out: Path, pre: dict) -> dict:
    """Per-stage start/end of every term, and the validation PCE of the final networks and of theta_pre."""
    recs = {json.loads(p.read_text())["stage"]: json.loads(p.read_text()) for p in sorted(out.glob(f"{TAG}-j*.json"))}
    final = {j: dict(zip(r["val_ids"], r["J_val"])) for j, r in recs.items()}
    ctrl = {}
    for j in recs:
        model = new_model(ctx, SEED)
        model.load_state_dict(pre[j])
        ids, J = val_currents(ctx, model, j)
        ctrl[j] = dict(zip(ids.tolist(), J.tolist()))
    ratio = lambda r, k: r["end"]["terms"][k] / r["start"]["terms"][k] if r["start"]["terms"][k] > 0 else math.nan
    keys = TERMS
    return {"stages": sorted(recs), "diverged": [j for j, r in recs.items() if r["diverged"]],
            "median_term_ratio_end_over_start": {k: float(np.nanmedian([ratio(r, k) for r in recs.values()])) for k in keys},
            "stages_where_term_fell_10x": {k: int(sum(ratio(r, k) <= 0.1 for r in recs.values())) for k in keys},
            "val_pce_final": pce_table(ctx, final), "val_pce_pretrained_control": pce_table(ctx, ctrl),
            "sK": ctx.sK}


def run_smoke(pre_steps: int = PRE_STEPS, adam_steps: int = ADAM_STEPS, parts: int | None = None,
              ctx=None) -> dict:
    """E8 smoke: theta_pre (pre_steps per stage, all K stages), then adam_steps per stage, stages over the GPUs."""
    start = time.time()
    deadline = start + HOURS * 3600.0
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    ctx = ctx or build_ctx()
    tel = e6.Telemetry(out, TAG)
    tel.line(f"M5: {n_parameters(new_model(ctx, SEED))} parameters · K = {ctx.K} stages · s_K = {ctx.sK:.2f} · "
             f"train/val/test {len(ctx.prep.train)}/{len(ctx.prep.val)}/{len(ctx.prep.test)}")
    pre_path = out / "pretrained_m5.pt"
    if pre_path.exists():
        pre = torch.load(pre_path, map_location="cpu", weights_only=False)
    else:
        pre = pretrain(ctx, SEED, ctx.K, tel, deadline, pre_steps)
        torch.save(pre, pre_path)
    n = parts or max(e6._n_gpu(), 1)
    order = list(range(ctx.K))
    pays = [{"role": "call", "fn": "Staged_PINN_Solar.m5:smoke_worker", "worker": w, "seed": SEED,
             "stages": [j for i, j in enumerate(order) if i % n == w], "steps": adam_steps,
             "start_file": str(pre_path), "out": str(out), "deadline": deadline} for w in range(n)]
    e6._run_payloads(pays, ctx, deadline)
    s = summarize(ctx, out, pre)
    steps._write(out / f"{TAG}.summary.json", s)
    tel.line("SUMMARY " + json.dumps(s, default=float))
    return s
