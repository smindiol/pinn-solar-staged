"""Stochastic 254-device batches, 150 stratified nodes, all-train L-BFGS.
Logs every 50 steps; two seeds on two GPUs. Original m6 remains unchanged.
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

from . import carriers, m6_data, pce, steps, m6_sampling

TAG = "e9c_254x150_pre300"
WIDTH, DEPTH = 256, 5
PRE_STEPS, PRE_LR, N_DEV = 300, 1e-3, 254
ADAM_STEPS, ADAM_LR, LR_MIN, CLIP = 2000, 1e-4, 5e-6, 1.0
LBFGS_ITERS = 500
N_DEV_PHYS = 254
NODES_DATA, NODES_PHYS = 150, 150
EVERY = 50
CKPT_SECONDS = 15 * 60
HOURS = 11.0
ARMS = {"PKC": (1, 1, 1), "Control": (0, 0, 0)}   # (a, b, c) of Eq. 38
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
        from .scaling import Scales
        scales = Scales()
        coeff = {"InP": (9e-31, 9e-31), "Si": (1.1e-30, 3e-31), "CdTe": (0., 0.), "E1gen": (0., 0.)}
        self.devc["auger_n"] = t([coeff[x][0] * scales.t0 * scales.C0**2 for x in self.material])
        self.devc["auger_p"] = t([coeff[x][1] * scales.t0 * scales.C0**2 for x in self.material])
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
        ln = torch.log1p
        pn = ln(q["Jgen"] * a_n / q["A_n"]) + s * col(r, 1)
        pp = -q["Vt"] - ln(q["Jgen"] * b_p / q["A_p"]) - (1 - s) * col(r, 2)
        Jn = q["Jgen"] * (1 - s) * col(r, 3)
        Jp = q["Jgen"] * s * col(r, 4)
        return phi, pn, pp, Jn, Jp


def new_model(ctx: Ctx, seed: int) -> M6:
    torch.manual_seed(seed)
    return M6(ctx.xi.shape[1], ctx.out_scale).to(ctx.dev)


# --- batches ------------------------------------------------------------------------------------------------
def batch(ctx, split, rng, n_dev, n_nodes):
    return m6_sampling.SpatialPlan(ctx, split, rng, n_dev)


def gather(ctx: Ctx, pd, pv, rd, rv, rp, rn) -> dict:
    T = lambda a: torch.as_tensor(a, device=ctx.dev)
    pd_t, pv_t, rd_t, rv_t, rp_t, rn_t = map(T, (pd, pv, rd, rv, rp, rn))
    c = lambda a, i: a[i][:, None]
    pair = {"xi": ctx.xi[pd_t], "v": (ctx.V[pd_t, pv_t] / ctx.V_max)[:, None]}
    q = {"xi": ctx.xi[rd_t], "v": (ctx.V[rd_t, rv_t] / ctx.V_max)[:, None], "Vt": ctx.Vt[rd_t, rv_t][:, None]}
    for k in ("phi_eq0", "phi_eq1", "A_n", "A_p", "delta", "Jgen", "Gbar", "auger_n", "auger_p") + m6_data.MAT_KEYS:
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
        RG = carriers.recombination(n, p, m) + (q["auger_n"] * n + q["auger_p"] * p) * (n * p - carriers.intrinsic_density_squared(m)) - q["G"]
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
        out[mat]["invalid_Voc"] = int((~np.isfinite(sel[:, 2])).sum())
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
def pretrain(ctx, model, seed, log, n_steps=PRE_STEPS, deadline=None) -> dict:
    """Exactly n_steps stochastic data updates, drawing a fresh batch each time."""
    if n_steps < 1:
        raise ValueError("pretraining steps must be positive")
    opt = torch.optim.Adam(model.parameters(), lr=PRE_LR)
    sch = _cosine(opt, n_steps, PRE_LR, LR_MIN)
    rng = np.random.default_rng(1000 + seed)
    first = None
    t0 = time.time()
    for i in range(n_steps):
        b = batch(ctx, "train", rng, N_DEV, NODES_DATA)
        opt.zero_grad(set_to_none=True)
        L, loss = accumulated(model, b, {"data": 1.}, (0, 0, 0), backward=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sch.step()
        first = first if first is not None else float(loss)
        if i == 0 or (i + 1) % EVERY == 0 or i == n_steps - 1:
            el = time.time() - t0
            log.line(f"PRE seed={seed} step {i + 1}/{n_steps} data={float(loss):.3e} "
                     f"{_fmt(L)} lr={sch.get_last_lr()[0]:.2e} eta={el / (i + 1) * (n_steps - i - 1) / 60:.1f}min")
            log.rec(phase="pre", seed=seed, step=i + 1, total=float(loss), **{k: float(v) for k, v in L.items()},
                    lr=sch.get_last_lr()[0], t=el)
        if deadline and time.time() > deadline:
            log.line("PRE stopped by the deadline")
            break
    return {"first": first, "last": float(loss), "steps": i + 1, "seconds": time.time() - t0}


def measure(ctx, model, seed, n_batches: int = 4) -> dict:
    """Mean of every term at theta_pre over fixed batches (seeded), for Eq. 39."""
    rng = np.random.default_rng(2000 + seed)
    acc = {}
    for _ in range(n_batches):
        b = batch(ctx, "train", rng, N_DEV_PHYS, NODES_PHYS)
        L, _ = accumulated(model, b, {"data": 1., **{k: 1. for k in PHYS_TERMS}}, (1, 1, 1))
        for k, v in L.items():
            acc[k] = acc.get(k, 0.0) + float(v) / n_batches
    return acc


def accumulated(model, b, lam, abc, backward=False):
    return m6_sampling.objective(b.ctx, model, b, lam, abc, b.ctx.chunk_rows, backward)


def measure_split(ctx, model, seed, split):
    b = batch(ctx, split, np.random.default_rng(7000+seed), len(ctx.ids[split]), NODES_PHYS)
    L, _ = accumulated(model, b, {"data": 1., **{k: 1. for k in PHYS_TERMS}}, (1,1,1))
    return {k: float(v) for k,v in L.items()}


def train_arm(ctx, model, arm, lam, seed, log, out, adam_steps=ADAM_STEPS, lbfgs_iters=LBFGS_ITERS,
              deadline=None) -> dict:
    abc = ARMS[arm]
    nodes = NODES_PHYS
    n_dev = 254
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    sch = _cosine(opt, adam_steps, ADAM_LR, LR_MIN)
    rng = np.random.default_rng(3000 + seed)
    t0, last_ck = time.time(), time.time()
    start = None
    for i in range(adam_steps):
        b = batch(ctx, "train", rng, n_dev, nodes)
        opt.zero_grad()
        L, loss = accumulated(model, b, lam, abc, backward=True)
        if not torch.isfinite(loss):
            log.line(f"ADAM {arm} seed={seed} step {i + 1}: non-finite loss, stopping the arm")
            return {"status": "nonfinite", "step": i + 1}
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sch.step()
        start = start or {k: float(v) for k, v in L.items()}
        if i == 0 or (i + 1) % EVERY == 0 or i == adam_steps - 1:
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
    lb = batch(ctx, "train", np.random.default_rng(4000 + seed), len(ctx.ids["train"]), nodes)
    lopt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters, max_eval=int(1.25 * lbfgs_iters),
                             history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15,
                             line_search_fn="strong_wolfe")
    count = {"n": 0}
    t1 = time.time()

    def closure():
        lopt.zero_grad()
        L, loss = accumulated(model, lb, lam, abc, backward=True)
        count["n"] += 1
        if count["n"] == 1 or count["n"] % EVERY == 0:
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
    final, _ = accumulated(model, lb, lam, abc)
    final = {k: float(v) for k, v in final.items()}
    validation_physics = measure_split(ctx, model, seed, "val")
    torch.save({"state": _state(model)}, out / f"{arm}.final.pt")
    jv = jv_metrics(ctx, model)
    res = {"status": "ok", "arm": arm, "abc": abc, "start": start, "end_adam": end_adam, "final": final,
           "lbfgs": {"status": lbfgs_status, "evals": count["n"], "seconds": time.time() - t1},
           "val": jv, "validation_terms": validation_physics, "train_terms": measure_split(ctx, model, seed, "train"), "seconds": time.time() - t0}
    log.line(f"END {arm} seed={seed} " + json.dumps(res["val"]["all"]))
    return res


def run_seed(seed: int, gpu: int = 0, root=None, limit=None, pre_steps=PRE_STEPS, adam_steps=ADAM_STEPS,
             lbfgs_iters=LBFGS_ITERS, arms=tuple(ARMS), hours=HOURS, run_label=None) -> dict:
    """One seed on one device: pretraining, Eq. 39, then every arm from theta_pre."""
    t_start = time.time()
    deadline = t_start + hours * 3600
    dev = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    out = results_dir() / (run_label or f"seed{seed}")
    log = Log(out, f"seed{seed}")
    ctx = Ctx.load(root, dev, limit)
    model = new_model(ctx, seed)
    probe = batch(ctx, "train", np.random.default_rng(8000 + seed), 254, 150)
    calibration = m6_sampling.calibrate(ctx, model, probe, log)
    ctx.chunk_rows = calibration["rows"]
    steps._write(out / "memory_calibration.json", calibration)
    log.line(f"SAMPLING devices=254 nodes=150 (15/90/15/30), all voltages; chunk_rows={ctx.chunk_rows}; LBFGS=all train")
    log.line(f"M6 seed={seed} on {dev}: {sum(p.numel() for p in model.parameters())} parameters, "
             f"train/val/test {[ctx.ids[k].size for k in SPLIT_ID]}, M={ctx.M}, V_max={ctx.V_max:.2f} V, "
             f"out_scale={ctx.out_scale.tolist()}")
    pre_path = out / "theta_pre.pt"
    if pre_path.exists():
        model.load_state_dict(torch.load(pre_path, map_location=dev)["state"])
        pre = torch.load(pre_path, map_location="cpu")["pre"]
    else:
        safe_rows = ctx.chunk_rows
        ctx.chunk_rows = min(1024, safe_rows)
        pre = pretrain(ctx, model, seed, log, pre_steps, deadline)
        ctx.chunk_rows = safe_rows
        torch.save({"state": _state(model), "pre": pre}, pre_path)
    theta_pre = _state(model)
    pre_val = jv_metrics(ctx, model)
    pre_validation_terms = measure_split(ctx, model, seed, "val")
    pre_train_terms = measure_split(ctx, model, seed, "train")
    values = measure(ctx, model, seed)
    lam = lambdas(values)
    log.line("THETA_PRE terms " + _fmt(values) + " | lambda " + json.dumps(lam) + " | val " + json.dumps(pre_val["all"]))
    summary = {"seed": seed, "pre": pre, "theta_pre_terms": values, "lambda": lam, "theta_pre_val": pre_val, "pre_validation_terms": pre_validation_terms, "pre_train_terms": pre_train_terms,
               "arms": {}}
    steps._write(out / "summary.json", summary)
    per_arm_seconds = max(0., deadline - time.time()) / len(arms)
    for arm in arms:
        done = out / f"{arm}.result.json"
        if done.exists():
            summary["arms"][arm] = json.loads(done.read_text())
            continue
        model.load_state_dict(theta_pre)
        res = train_arm(ctx, model, arm, lam, seed, log, out, adam_steps, lbfgs_iters, time.time() + per_arm_seconds)
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
    return run_kernel((0, 1), pre_steps=1, adam_steps=20, lbfgs_iters=10, hours=0.15)


def _arm_worker(gpu, kw):
    arm = ("Control", "PKC")[gpu]
    run_seed(0, gpu=gpu, arms=(arm,), run_label=f"seed0_{arm}", **kw)


def run_training(**kw):
    """One seed, one arm per T4; 2000 Adam updates and <=500 LBFGS iterations."""
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This comparison requires two GPUs")
    kw.setdefault("root", str(m6_data.ensure(kw.get("root"))))
    torch.multiprocessing.spawn(_arm_worker, args=(kw,), nprocs=2, join=True)
    return {"seed": 0, "arms": ["Control", "PKC"], "out": str(results_dir())}
