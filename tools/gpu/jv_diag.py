"""Per-device J-V diagnosis of a saved checkpoint (CPU only, safe while a GPU job runs).

For every validation device it reports where the predicted curve is right and
where it is wrong, and why Voc can be undefined:

* ``voc_ref``      reference Voc (V) from the dPV curve.
* ``voc_grid``     predicted Voc on the dPV voltage grid (NaN = no sign change there).
* ``voc_fine``     predicted Voc on a fine grid extended to 1.5 x the last dPV voltage,
                   so the true crossing is found even beyond the dPV grid.
* ``j_last``       predicted J at the last dPV voltage (mA/cm^2); > 0 means the
                   prediction has not reached zero where dPV already did.
* ``err_lo/mid/hi`` RMSE (mA/cm^2) over the first, middle and last third of the
                   curve, to see which part of the curve carries the error.

Usage:
    python tools/gpu/jv_diag.py <checkpoint.pt> <out_dir> [--split val]
Environment: DPV_BASE_M6 as for training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from Staged_PINN_Solar import m6c as base, m6_data, pce  # noqa: E402


def predict(ctx, model, d, volts):
    """Predicted J (A/cm^2) of device d at arbitrary voltages."""
    v = torch.as_tensor(volts, dtype=torch.float64)
    n = v.numel()
    s0 = torch.zeros(n, 1, dtype=torch.float64)
    xi = ctx.xi[d].reshape(1, -1).expand(n, -1)
    r0 = model.raw(s0, xi, (v / ctx.V_max)[:, None])
    return (ctx.attrs["J0"] * ctx.devc["Jgen"][d] * r0[:, 3]).numpy()


def crossing(v, j):
    """First voltage where J changes sign (linear interpolation), else NaN."""
    idx = np.nonzero(j[:-1] * j[1:] <= 0)[0]
    if idx.size == 0:
        return float("nan")
    k = idx[0]
    return float(v[k] + j[k] * (v[k + 1] - v[k]) / (j[k] - j[k + 1])) if j[k] != j[k + 1] else float(v[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = torch.device("cpu")
    ctx = base.Ctx.load(str(m6_data.ensure(None)), dev, None)
    model = base.new_model(ctx, 0)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev)["state"])
    model.eval()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in ctx.ids[args.split]:
        d = int(d)
        n = int(ctx.n_V[d])
        v, jr = ctx.V_np[d, :n], ctx.J_np[d, :n]
        jp = predict(ctx, model, d, v)
        a, b = pce.jv_summary(v, jp, ctx.attrs["P_in"]), pce.jv_summary(v, jr, ctx.attrs["P_in"])
        vf = np.linspace(0.0, 1.5 * float(v[-1]), 600)
        jf = predict(ctx, model, d, vf)
        thirds = np.array_split(np.arange(n), 3)
        err = [1e3 * float(np.sqrt(np.mean((jp[t] - jr[t]) ** 2))) for t in thirds]
        rows.append(dict(
            device=d, uid=ctx.uid[d], material=str(ctx.material[d]), n_V=n,
            voc_ref=b["voc"], voc_grid=a["voc"], voc_fine=crossing(vf, jf),
            j_last=1e3 * float(jp[-1]), jref_last=1e3 * float(jr[-1]),
            jsc_ref=b["jsc"] * 0.1, jsc_err=0.1 * (a["jsc"] - b["jsc"]),
            ff_err=100 * (a["ff"] - b["ff"]), pce_err=100 * (a["pce"] - b["pce"]),
            curve=1e3 * float(np.sqrt(np.mean((jp - jr) ** 2))),
            err_lo=err[0], err_mid=err[1], err_hi=err[2]))
        np.savez(out / f"curve_{d}.npz", v=v, jr=jr, jp=jp, vf=vf, jf=jf)

    import pandas as pd
    df = pd.DataFrame(rows)
    df["invalid"] = ~np.isfinite(df.voc_grid)
    df["voc_fine_err_mV"] = 1e3 * (df.voc_fine - df.voc_ref)
    df.to_csv(out / "jv_diag.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"devices={len(df)} invalid_Voc={int(df.invalid.sum())}")
    print("\n== per material (medians; invalid count)")
    g = df.groupby("material")
    print(pd.DataFrame({
        "n": g.size(), "invalid": g.invalid.sum(),
        "curve": g.curve.median(), "err_lo": g.err_lo.median(), "err_mid": g.err_mid.median(),
        "err_hi": g.err_hi.median(), "|Jsc|": g.jsc_err.apply(lambda s: s.abs().median()),
        "|FF|": g.ff_err.apply(lambda s: s.abs().median()),
        "|PCE|": g.pce_err.apply(lambda s: s.abs().median()),
        "Voc_fine_mV": g.voc_fine_err_mV.apply(lambda s: s.abs().median()),
    }).round(3).to_string())
    inv = df[df.invalid]
    if len(inv):
        print("\n== invalid devices: where does the predicted curve cross zero?")
        print(inv[["device", "material", "voc_ref", "voc_fine", "voc_fine_err_mV", "j_last",
                   "jref_last", "curve", "err_hi"]].round(4).to_string(index=False))
        print(f"\nsign of j_last on invalid devices: >0 (not yet at zero) = {(inv.j_last > 0).sum()}, "
              f"<0 = {(inv.j_last < 0).sum()}; no crossing even on 1.5x grid = "
              f"{int((~np.isfinite(inv.voc_fine)).sum())}")
    print(f"\nvalid devices: signed Voc error mV median={df[~df.invalid].voc_fine_err_mV.median():.2f} "
          f"(positive = predicted Voc too high)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    worst = list(inv.device[:6]) + list(df[~df.invalid].sort_values("curve").device[-3:]) \
        + list(df.sort_values("curve").device[:3])
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, d in zip(axes.flat, worst):
        z = np.load(out / f"curve_{d}.npz")
        r = df[df.device == d].iloc[0]
        ax.plot(z["v"], 1e3 * z["jr"], "k.-", label="dPV")
        ax.plot(z["vf"], 1e3 * z["jf"], "r-", lw=1, label="PINN (fine)")
        ax.plot(z["v"], 1e3 * z["jp"], "rx", ms=4)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylim(min(-2, 1e3 * z["jr"].min() - 1), 1e3 * z["jr"].max() * 1.15)
        tag = "INVALID" if r.invalid else f"curve={r.curve:.2f}"
        ax.set_title(f"{r.material} dev{d} {tag}", fontsize=9)
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("V (V)")
    fig.supylabel("J (mA/cm²)")
    fig.tight_layout()
    fig.savefig(out / "jv_diag.png", dpi=110)
    print(f"\nsaved {out/'jv_diag.csv'} and {out/'jv_diag.png'}")


if __name__ == "__main__":
    main()
