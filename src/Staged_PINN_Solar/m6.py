"""Network M6 of r14 Section 8 and experiment E9 of experiments v15.

One network N_theta(s, xi_hat, V / V_max) -> (N_phi, N_n, N_p, u_n, u_p); the output transform (Eq. 35) imposes the
six dPV boundary conditions for every theta; the residuals are Eq. 36; the data term is Eq. 37 (potentials at the
nodes, currents at the interface midpoints, every field in dPV's own dimensionless units); the loss is Eq. 38 with the
group switches (a, b, c) of the arm; the weights are fixed once at theta_pre by Eq. 39.

Run of one seed (one GPU): pretraining of PRE_EPOCHS epochs on the data term; Eq. 39 on a fixed batch; then each
arm from theta_pre: ADAM_STEPS Adam steps and L-BFGS of at most LBFGS_ITERS iterations on the same loss. Every EVERY
steps the log prints every term unweighted (the five data components, P, Kn, Kp, Cn, Cp), the weighted total and
the validation J-V error. Metrics are absolute errors only.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from . import carriers, m6_data, pce, steps

__provenance__ = {
    "revision": "research-concept-entrenamiento-por-etapas-de-una-pinn-no-lineal-para-el-model-r14.md",
    "sections": ["8"],
    "equations": ["34", "35", "36", "37", "38", "39"],
    "invariants": [
        "m6_output_transform_imposes_the_six_contact_conditions",
        "m6_residuals_close_on_dpv_fields",
        "m6_lambdas_follow_the_order_of_magnitude_rule",
        "m6_run_completes_on_a_tiny_cpu_context",
    ],
}

TAG = "e9b"
WIDTH, DEPTH = 256, 5
PRE_EPOCHS, PRE_LR, N_DEV = 300, 1e-3, 64
ADAM_STEPS, ADAM_LR, LR_MIN, CLIP = 3000, 1e-4, 5e-6, 1.0
LBFGS_ITERS = 500
N_DEV_PHYS = 16                        # T4 OOM at 64 devices x 32 nodes with the second derivative (smoke 2026-11-15)
NODES_DATA, NODES_PHYS = 64, 32        # nodes per (device, voltage) drawn per step from the 250 fixed ones
EVERY = 25
CKPT_SECONDS = 15 * 60
HOURS = 11.0
ARMS = {"PKC": (1, 1, 1), "PK": (1, 1, 0), "PC": (1, 0, 1), "Control": (0, 0, 0)}   # (a, b, c) of Eq. 38
DATA_TERMS = ("phi", "phi_n", "phi_p", "J_n", "J_p")
PHYS_TERMS = ("P", "Kn", "Kp", "Cn", "Cp")
GROUP = {"P": 0, "Kn": 1, "Kp": 1, "Cn": 2, "Cp": 2}
SPLIT_ID = {"train": 0, "val": 1, "test": 2}


def results_dir() -> Path:
    return steps.results_root() / TAG


# --- context ------------------------------------------------------------------------------------------------
class Ctx:
    """The reduced base on one torch device: per-device constants, fixed nodes and float32 targets."""

    def __init__(self, raw: dict, dev: torch.device, limit: int | None = None):
        self.dev, self.attrs = dev, raw["attrs"]
        split = raw["split"]
        keep = np.concatenate([np.nonzero(split == s)[0][:limit] if limit else np.nonzero(split == s)[0]
                               for s in range(3)])
        self.ids = {name: np.nonzero(split[keep] == s)[0] for name, s in SPLIT_ID.items()}
        self.uid = [raw["uid"][i] for i in keep]
        self.material = np.array([raw["material"][i] for i in keep])
        self.n_V = raw["n_V"][keep]
        self.V_np, self.J_np = raw["V"][keep], raw["J"][keep]
        self.pce_ref = raw["dev"]["pce"][keep]
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=dev)
        d = {k: v[keep] for k, v in raw["dev"].items()}
        self.devc = {k: t(v) for k, v in d.items()}
        V_th = self.attrs["V_th"]
        n_eq0 = d["Nc"] * np.exp(d["chi"] + d["phi_eq0"])                       # Eq. 34
        p_eq1 = d["Nv"] * np.exp(-d["chi"] - d["Eg"] - d["phi_eq1"])
        self.devc["A_n"] = t(self.attrs["S_n0"] * n_eq0)
        self.devc["A_p"] = t(self.attrs["S_p1"] * p_eq1)
        self.devc["Jgen"] = t(d["Gbar"] / d["delta"])                          # r13 Eq. 28: J_gen = G_bar / delta
        self.xi = t(raw["xi_hat"][keep])
        self.V = t(np.nan_to_num(self.V_np))
        self.Vt = self.V / V_th
        tr = self.ids["train"]
        self.V_max = float(np.nanmax(self.V_np[tr]))
        self.nodes = {k: t(v[keep]) for k, v in raw["nodes"].items()}
        self.fields = torch.as_tensor(raw["fields"][keep], device=dev)       # (D, K, 3, M) float32
        self.currents = torch.as_tensor(raw["currents"][keep], device=dev)   # (D, K, 2, M) float32
        self.M = self.fields.shape[-1]
        f = raw["fields"][keep][tr]
        c = raw["currents"][keep][tr]
        self.out_scale = torch.tensor([np.nanstd(f[:, :, 0]), np.nanstd(f[:, :, 1]), np.nanstd(f[:, :, 2]),
                                       1.0, 1.0], dtype=torch.float64)    # u_n, u_p are scaled currents j = J / J_gen

    @classmethod
    def load(cls, root=None, dev=None, limit=None):
        dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(m6_data.load(root), dev, limit)


# --- network and output transform (Eq. 35) ------------------------------------------------------------------
class M6(torch.nn.Module):
    def __init__(self, n_xi: int, out_scale: torch.Tensor, width: int = WIDTH, depth: int = DEPTH):
        super().__init__()
        layers, n = [], n_xi + 2
        for _ in range(depth):
            layers += [torch.nn.Linear(n, width), torch.nn.Tanh()]
            n = width
        layers.append(torch.nn.Linear(n, 5))
        self.net = torch.nn.Sequential(*layers).double()
        self.register_buffer("scale", out_scale.clone().double())

    def raw(self, s, xi, v):
        return self.net(torch.cat([s, xi, v], dim=1)) * self.scale

    def forward(self, s, q: dict, r0=None, r1=None):
        """(phi, phi_n, phi_p, J_n, J_p) at rows (s, q); q holds per-row xi, v, Vt, phi_eq0, phi_eq1, A_n, A_p.
        r0, r1: the raw outputs at s = 0 and s = 1 for the same rows (computed here when absent)."""
        r = self.raw(s, q["xi"], q["v"])
        if r0 is None:
            r0 = self.raw(torch.zeros_like(s), q["xi"], q["v"])
            r1 = self.raw(torch.ones_like(s), q["xi"], q["v"])
        col = lambda a, k: a[:, k:k + 1]
        phi = (1 - s) * q["phi_eq0"] + s * (q["phi_eq1"] + q["Vt"]) + s * (1 - s) * col(r, 0)
        a_n, b_p = col(r0, 3), col(r1, 4)                     # scaled contact currents j_n(0), j_p(1)
        # r15: the ohmic contacts solved for the potential, phi_n(0) = ln(1 + J_gen a_n / A_n) and
        # phi_p(1) = -V - ln(1 + J_gen b_p / A_p), so J_n(0) = A_n (e^{phi_n(0)} - 1) and J_p(1) likewise hold exactly
        ln = lambda z: torch.log(torch.clamp(1 + z, min=1e-300))
        pn = ln(q["Jgen"] * a_n / q["A_n"]) + s * col(r, 1)
        pp = -q["Vt"] - ln(q["Jgen"] * b_p / q["A_p"]) - (1 - s) * col(r, 2)
        Jn = q["Jgen"] * (1 - s) * (a_n + s * col(r, 3))
        Jp = q["Jgen"] * s * (b_p + (1 - s) * col(r, 4))
        return phi, pn, pp, Jn, Jp


def new_model(ctx: Ctx, seed: int) -> M6:
    torch.manual_seed(seed)
    return M6(ctx.xi.shape[1], ctx.out_scale).to(ctx.dev)


# --- batches ------------------------------------------------------------------------------------------------
def batch(ctx: Ctx, split: str, rng, n_dev: int, n_nodes: int) -> dict:
    """`n_dev` devices of `split` with ALL their voltages and `n_nodes` of each device's fixed nodes."""
    pool = ctx.ids[split]
    devs = np.sort(rng.choice(pool, size=min(n_dev, pool.size), replace=False))
    pd, pv = [], []
    for d in devs:
        pd += [d] * int(ctx.n_V[d])
        pv += list(range(int(ctx.n_V[d])))
    pd, pv = np.array(pd), np.array(pv)
    k = min(n_nodes, ctx.M)
    nodes = np.stack([np.sort(rng.choice(ctx.M, size=k, replace=False)) for _ in devs])
    node_of = {d: nodes[i] for i, d in enumerate(devs)}
    rd = np.repeat(pd, k)
    rv = np.repeat(pv, k)
    rp = np.repeat(np.arange(pd.size), k)
    rn = np.concatenate([node_of[d] for d in pd])
    return gather(ctx, pd, pv, rd, rv, rp, rn)


def gather(ctx: Ctx, pd, pv, rd, rv, rp, rn) -> dict:
    T = lambda a: torch.as_tensor(a, device=ctx.dev)
    pd_t, pv_t, rd_t, rv_t, rp_t, rn_t = map(T, (pd, pv, rd, rv, rp, rn))
    c = lambda a, i: a[i][:, None]
    pair = {"xi": ctx.xi[pd_t], "v": (ctx.V[pd_t, pv_t] / ctx.V_max)[:, None]}
    q = {"xi": ctx.xi[rd_t], "v": (ctx.V[rd_t, rv_t] / ctx.V_max)[:, None], "Vt": ctx.Vt[rd_t, rv_t][:, None]}
    for k in ("phi_eq0", "phi_eq1", "A_n", "A_p", "delta", "Jgen", "Gbar") + m6_data.MAT_KEYS:
        q[k] = c(ctx.devc[k], rd_t)
    for k in ("s", "s_mid", "G", "Ndop"):
        q[k] = ctx.nodes[k][rd_t, rn_t][:, None]
    q["target"] = ctx.fields[rd_t, rv_t, :, rn_t].double()
    q["jtarget"] = ctx.currents[rd_t, rv_t, :, rn_t].double()
    return {"pair": pair, "q": q, "rp": rp_t, "n_pairs": int(pd.size), "n_rows": int(rd.size)}


def _ends(model, b):
    s0 = torch.zeros(b["n_pairs"], 1, dtype=torch.float64, device=b["rp"].device)
    p = b["pair"]
    r0, r1 = model.raw(s0, p["xi"], p["v"]), model.raw(s0 + 1, p["xi"], p["v"])
    return r0[b["rp"]], r1[b["rp"]]


def _material(q) -> carriers.Material:
    return carriers.Material(Nc=q["Nc"], Nv=q["Nv"], chi=q["chi"], Eg=q["Eg"], eps_r=q["eps_r"], mu_n=q["mu_n"],
                             mu_p=q["mu_p"], tau_n=q["tau_n"], tau_p=q["tau_p"], Et=q["Et"], Br=q["Br"],
                             Ndop=q["Ndop"])


# --- loss terms ---------------------------------------------------------------------------------------------
def data_terms(model, b) -> dict:
    """Eq. 37 by component: the three potentials at the nodes, the two currents at the interface midpoints."""
    q = b["q"]
    r0, r1 = _ends(model, b)
    phi, pn, pp, _, _ = model(q["s"], q, r0, r1)
    _, _, _, Jn, Jp = model(q["s_mid"], q, r0, r1)
    e = torch.cat([phi, pn, pp], 1) - q["target"]
    j = (torch.cat([Jn, Jp], 1) - q["jtarget"]) / q["Jgen"]          # scaled currents j = J / J_gen (r15)
    return {"phi": (e[:, 0] ** 2).mean(), "phi_n": (e[:, 1] ** 2).mean(), "phi_p": (e[:, 2] ** 2).mean(),
            "J_n": (j[:, 0] ** 2).mean(), "J_p": (j[:, 1] ** 2).mean()}


def residuals(model, b, want=PHYS_TERMS) -> dict:
    """Eq. 36 at the batch nodes, by autograd on s."""
    q = b["q"]
    s = q["s"].clone().requires_grad_(True)
    r0, r1 = _ends(model, b)
    phi, pn, pp, Jn, Jp = model(s, q, r0, r1)
    d = lambda y: torch.autograd.grad(y, s, torch.ones_like(y), create_graph=True)[0]
    m = _material(q)
    n, p = carriers.electron_density(phi, pn, m), carriers.hole_density(phi, pp, m)
    dl = q["delta"]
    out = {}
    if "P" in want:
        out["P"] = q["eps_r"] * dl ** 2 * d(d(phi)) - (n - p - q["Ndop"])
    if "Kn" in want:
        gn = q["mu_n"] * n * dl / q["Jgen"]                 # r15: conductance over J_gen, spans ~1e-4..1e5
        out["Kn"] = (Jn / q["Jgen"] - gn * d(pn)) / (1 + gn)
    if "Kp" in want:
        gp = q["mu_p"] * p * dl / q["Jgen"]
        out["Kp"] = (Jp / q["Jgen"] - gp * d(pp)) / (1 + gp)
    if "Cn" in want or "Cp" in want:
        RG = carriers.recombination(n, p, m) - q["G"]
        out["Cn"] = (dl * d(Jn) - RG) / q["Gbar"]
        out["Cp"] = (dl * d(Jp) + RG) / q["Gbar"]
    return out


def terms(model, b, abc=(1, 1, 1)) -> dict:
    """Every unweighted term the arm needs (the data components always)."""
    L = data_terms(model, b)
    want = [k for k in PHYS_TERMS if abc[GROUP[k]]]
    if want:
        L.update({k: (r ** 2).mean() for k, r in residuals(model, b, want).items() if k in want})
    return L


def total(L: dict, lam: dict, abc) -> torch.Tensor:
    """Eq. 38: lambda_d L_d + the switched physics groups."""
    out = lam["data"] * sum(L[k] for k in DATA_TERMS)
    for k in PHYS_TERMS:
        if abc[GROUP[k]]:
            out = out + lam[k] * L[k]
    return out


def lambdas(values: dict) -> dict:
    """Eq. 39: lambda_i = 10^-floor(log10 L_i); lambda_d = 10 * 10^-floor(log10 L_d)."""
    mag = lambda v: 10.0 ** (-math.floor(math.log10(max(float(v), 1e-300))))
    lam = {k: mag(values[k]) for k in PHYS_TERMS if k in values}
    lam["data"] = 10.0 * mag(sum(float(values[k]) for k in DATA_TERMS))
    return lam


# --- validation: J-V curve and its figures, absolute errors -------------------------------------------------
@torch.no_grad()
def predicted_currents(ctx: Ctx, model, split: str = "val") -> dict:
    """J(V) = J0 Jn_hat(0) [A/cm^2] on the dPV voltage grid of every device of `split`."""
    out = {}
    ids = ctx.ids[split]
    pd = np.concatenate([[d] * int(ctx.n_V[d]) for d in ids]).astype(int)
    pv = np.concatenate([np.arange(int(ctx.n_V[d])) for d in ids]).astype(int)
    T = lambda a: torch.as_tensor(a, device=ctx.dev)
    pd_t, pv_t = T(pd), T(pv)
    s0 = torch.zeros(pd.size, 1, dtype=torch.float64, device=ctx.dev)
    r0 = model.raw(s0, ctx.xi[pd_t], (ctx.V[pd_t, pv_t] / ctx.V_max)[:, None])
    J = (ctx.attrs["J0"] * ctx.devc["Jgen"][pd_t] * r0[:, 3]).cpu().numpy()      # J = J0 J_gen j_n(0)
    for d in ids:
        out[int(d)] = J[pd == d]
    return out


def jv_metrics(ctx: Ctx, model, split: str = "val") -> dict:
    """Per material: median absolute error of the curve (RMSE, mA/cm^2), Jsc (mA/cm^2), Voc (V), FF, PCE (pp)."""
    Jp = predicted_currents(ctx, model, split)
    rows = []
    for d, j in Jp.items():
        n = int(ctx.n_V[d])
        v, jr = ctx.V_np[d, :n], ctx.J_np[d, :n]
        a, b = pce.jv_summary(v, j, ctx.attrs["P_in"]), pce.jv_summary(v, jr, ctx.attrs["P_in"])
        rows.append((ctx.material[d], 1e3 * float(np.sqrt(np.mean((j - jr) ** 2))),
                     0.1 * abs(a["jsc"] - b["jsc"]), abs(a["voc"] - b["voc"]), 100 * abs(a["ff"] - b["ff"]),
                     100 * abs(a["pce"] - b["pce"])))
    out = {}
    for mat in sorted(set(r[0] for r in rows)) + ["all"]:
        sel = np.array([r[1:] for r in rows if mat == "all" or r[0] == mat], dtype=float)
        out[mat] = {k: float(np.nanmedian(sel[:, i])) for i, k in enumerate(("curve", "Jsc", "Voc", "FF", "PCE"))}
        out[mat]["n"] = int(sel.shape[0])
    return out


# --- telemetry ----------------------------------------------------------------------------------------------
class Log:
    def __init__(self, out: Path, name: str):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.hist = open(out / f"{name}.history.jsonl", "a")
        self.txt = open(out / f"{name}.log", "a")

    def line(self, s: str):
        s = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(s, flush=True)
        self.txt.write(s + "\n")
        self.txt.flush()

    def rec(self, **kw):
        self.hist.write(json.dumps(kw, default=float) + "\n")
        self.hist.flush()


def _fmt(L: dict) -> str:
    return " ".join(f"{k}={float(v):.3e}" for k, v in L.items())


def _cosine(opt, n_steps, lr0, lr_min):
    f = lambda i: (lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * min(i, n_steps) / n_steps))) / lr0
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --- phases -------------------------------------------------------------------------------------------------
def pretrain(ctx, model, seed, log, epochs=PRE_EPOCHS, deadline=None) -> dict:
    """Data term only, `epochs` passes over the training devices in batches of N_DEV with all their voltages."""
    per_epoch = max(1, math.ceil(ctx.ids["train"].size / N_DEV))
    n_steps = epochs * per_epoch
    opt = torch.optim.Adam(model.parameters(), lr=PRE_LR)
    sch = _cosine(opt, n_steps, PRE_LR, LR_MIN)
    rng = np.random.default_rng(1000 + seed)
    first = None
    t0 = time.time()
    for i in range(n_steps):
        b = batch(ctx, "train", rng, N_DEV, NODES_DATA)
        L = data_terms(model, b)
        loss = sum(L.values())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sch.step()
        first = first if first is not None else float(loss)
        if i % EVERY == 0 or i == n_steps - 1:
            el = time.time() - t0
            log.line(f"PRE seed={seed} step {i + 1}/{n_steps} (epoch {i // per_epoch + 1}) data={float(loss):.3e} "
                     f"{_fmt(L)} lr={sch.get_last_lr()[0]:.2e} eta={el / (i + 1) * (n_steps - i - 1) / 60:.1f}min")
            log.rec(phase="pre", seed=seed, step=i + 1, total=float(loss), **{k: float(v) for k, v in L.items()},
                    lr=sch.get_last_lr()[0], t=el)
        if deadline and time.time() > deadline:
            log.line("PRE stopped by the deadline")
            break
    return {"first": first, "last": float(loss), "steps": i + 1, "seconds": time.time() - t0}


def measure(ctx, model, seed, n_batches: int = 16) -> dict:
    """Mean of every term at theta_pre over fixed batches (seeded), for Eq. 39."""
    rng = np.random.default_rng(2000 + seed)
    acc = {}
    for _ in range(n_batches):
        b = batch(ctx, "train", rng, N_DEV_PHYS, NODES_PHYS)
        L = terms(model, b, (1, 1, 1))
        for k, v in L.items():
            acc[k] = acc.get(k, 0.0) + float(v) / n_batches
    return acc


def train_arm(ctx, model, arm, lam, seed, log, out, adam_steps=ADAM_STEPS, lbfgs_iters=LBFGS_ITERS,
              deadline=None) -> dict:
    abc = ARMS[arm]
    nodes = NODES_PHYS if any(abc) else NODES_DATA
    n_dev = N_DEV_PHYS if any(abc) else N_DEV
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    sch = _cosine(opt, adam_steps, ADAM_LR, LR_MIN)
    rng = np.random.default_rng(3000 + seed)
    t0, last_ck = time.time(), time.time()
    start = None
    for i in range(adam_steps):
        b = batch(ctx, "train", rng, n_dev, nodes)
        L = terms(model, b, abc)
        loss = total(L, lam, abc)
        if not torch.isfinite(loss):
            log.line(f"ADAM {arm} seed={seed} step {i + 1}: non-finite loss, stopping the arm")
            return {"status": "nonfinite", "step": i + 1}
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sch.step()
        start = start or {k: float(v) for k, v in L.items()}
        if i % EVERY == 0 or i == adam_steps - 1:
            el = time.time() - t0
            jv = jv_metrics(ctx, model)["all"]
            log.line(f"ADAM {arm} seed={seed} step {i + 1}/{adam_steps} total={float(loss):.3e} {_fmt(L)} "
                     f"| val curve={jv['curve']:.3f}mA/cm2 PCE={jv['PCE']:.3f}pp "
                     f"eta={el / (i + 1) * (adam_steps - i - 1) / 60:.1f}min")
            log.rec(phase="adam", arm=arm, seed=seed, step=i + 1, total=float(loss),
                    **{k: float(v) for k, v in L.items()}, val=jv, lr=sch.get_last_lr()[0], t=el)
        if time.time() - last_ck > CKPT_SECONDS:
            torch.save({"state": _state(model), "step": i + 1}, out / f"{arm}.adam.ckpt.pt")
            last_ck = time.time()
        if deadline and time.time() > deadline:
            log.line(f"ADAM {arm} stopped by the deadline at step {i + 1}")
            break
    end_adam = {k: float(v) for k, v in L.items()}
    torch.save({"state": _state(model)}, out / f"{arm}.adam.pt")
    # L-BFGS on one fixed batch with the same loss (Eq. 38, same lambda)
    lb = batch(ctx, "train", np.random.default_rng(4000 + seed), n_dev, nodes)
    lopt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters, max_eval=int(1.25 * lbfgs_iters),
                             history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15,
                             line_search_fn="strong_wolfe")
    count = {"n": 0}
    t1 = time.time()

    def closure():
        lopt.zero_grad()
        L = terms(model, lb, abc)
        loss = total(L, lam, abc)
        loss.backward()
        count["n"] += 1
        if count["n"] % EVERY == 1:
            log.line(f"LBFGS {arm} seed={seed} eval {count['n']} total={float(loss):.3e} {_fmt(L)}")
            log.rec(phase="lbfgs", arm=arm, seed=seed, step=count["n"], total=float(loss),
                    **{k: float(v) for k, v in L.items()}, t=time.time() - t1)
        return loss

    lbfgs_status = "ok"
    if not deadline or time.time() < deadline:
        backup = _state(model)
        lopt.step(closure)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            model.load_state_dict(backup)
            lbfgs_status = "nonfinite-reverted"
    else:
        lbfgs_status = "skipped-deadline"
    final = {k: float(v) for k, v in terms(model, lb, abc).items()}
    torch.save({"state": _state(model)}, out / f"{arm}.final.pt")
    jv = jv_metrics(ctx, model)
    res = {"status": "ok", "arm": arm, "abc": abc, "start": start, "end_adam": end_adam, "final": final,
           "lbfgs": {"status": lbfgs_status, "evals": count["n"], "seconds": time.time() - t1},
           "val": jv, "seconds": time.time() - t0}
    log.line(f"END {arm} seed={seed} " + json.dumps(res["val"]["all"]))
    return res


def run_seed(seed: int, gpu: int = 0, root=None, limit=None, pre_epochs=PRE_EPOCHS, adam_steps=ADAM_STEPS,
             lbfgs_iters=LBFGS_ITERS, arms=tuple(ARMS), hours=HOURS) -> dict:
    """One seed on one device: pretraining, Eq. 39, then every arm from theta_pre."""
    t_start = time.time()
    deadline = t_start + hours * 3600
    dev = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    out = results_dir() / f"seed{seed}"
    log = Log(out, f"seed{seed}")
    ctx = Ctx.load(root, dev, limit)
    model = new_model(ctx, seed)
    log.line(f"M6 seed={seed} on {dev}: {sum(p.numel() for p in model.parameters())} parameters, "
             f"train/val/test {[ctx.ids[k].size for k in SPLIT_ID]}, M={ctx.M}, V_max={ctx.V_max:.2f} V, "
             f"out_scale={ctx.out_scale.tolist()}")
    pre_path = out / "theta_pre.pt"
    if pre_path.exists():
        model.load_state_dict(torch.load(pre_path, map_location=dev)["state"])
        pre = torch.load(pre_path, map_location="cpu")["pre"]
    else:
        pre = pretrain(ctx, model, seed, log, pre_epochs, deadline)
        torch.save({"state": _state(model), "pre": pre}, pre_path)
    theta_pre = _state(model)
    pre_val = jv_metrics(ctx, model)
    values = measure(ctx, model, seed)
    lam = lambdas(values)
    log.line("THETA_PRE terms " + _fmt(values) + " | lambda " + json.dumps(lam) + " | val " + json.dumps(pre_val["all"]))
    summary = {"seed": seed, "pre": pre, "theta_pre_terms": values, "lambda": lam, "theta_pre_val": pre_val,
               "arms": {}}
    steps._write(out / "summary.json", summary)
    for arm in arms:
        done = out / f"{arm}.result.json"
        if done.exists():
            summary["arms"][arm] = json.loads(done.read_text())
            continue
        model.load_state_dict(theta_pre)
        res = train_arm(ctx, model, arm, lam, seed, log, out, adam_steps, lbfgs_iters, deadline)
        steps._write(done, res)
        summary["arms"][arm] = res
        steps._write(out / "summary.json", summary)
    summary["seconds"] = time.time() - t_start
    steps._write(out / "summary.json", summary)
    log.line("SUMMARY " + json.dumps({a: r.get("val", {}).get("all") for a, r in summary["arms"].items()}))
    return summary


def _worker(gpu: int, seeds: list, kw: dict):
    run_seed(seeds[gpu], gpu=gpu, **kw)


def run_kernel(seeds=(0, 1), **kw) -> dict:
    """Kaggle entry: one seed per GPU, in parallel processes; on a single device the seeds run in turn."""
    seeds = list(seeds)
    kw.setdefault("root", str(m6_data.ensure(kw.get("root"))))    # build once, before the GPU processes start
    n = torch.cuda.device_count()
    if n >= len(seeds) and len(seeds) > 1:
        torch.multiprocessing.spawn(_worker, args=(seeds, kw), nprocs=len(seeds), join=True)
    else:
        for s in seeds:
            run_seed(s, gpu=0, **kw)
    return {"seeds": seeds, "out": str(results_dir())}


def run_from_env() -> dict:
    """Seeds from M6_SEEDS (comma separated, default 0,1)."""
    seeds = [int(x) for x in os.environ.get("M6_SEEDS", "0,1").split(",")]
    return run_kernel(seeds)


def run_seeds_01() -> dict:
    return run_kernel((0, 1))


def run_seeds_23() -> dict:
    return run_kernel((2, 3))


def run_seeds_45() -> dict:
    return run_kernel((4, 5))


def run_smoke() -> dict:
    """Kaggle smoke (~5 min): both GPUs, seeds 0 and 1, every arm, a few steps of each phase."""
    return run_kernel((0, 1), pre_epochs=1, adam_steps=20, lbfgs_iters=10, hours=0.15)
