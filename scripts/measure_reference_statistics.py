"""Re-measure sigma_i^2, f_i and rho_i (Eqs. 10-11) on the training devices of the 4096-device set.

Usage: PYTHONPATH=src .venv/bin/python scripts/measure_reference_statistics.py OUT.json [--limit N]
Streams one device at a time; seed 42 fixes the device-level split. Three readings are measured in the same pass:
'dpv' (default: dPV interface currents, every term a separate summand), 'continuous' (Eq. 6 formed
on the nodes) and 'dpv_as_written' (each parenthesis of Eqs. 5, 7, 9 one summand).
"""
import json
import sys
import time

from Staged_PINN_Solar import data, loss
from Staged_PINN_Solar.scaling import Scales


def main() -> None:
    out = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    s = Scales()
    split = data.split_devices(data.inventory(), seed=data.SEED)
    train = split.train[:limit] if limit else split.train
    readings = {"dpv": dict(currents="dpv", grouping="expanded"),
                "continuous": dict(currents="continuous", grouping="expanded"),
                "dpv_as_written": dict(currents="dpv", grouping="as_written")}
    stats = {k: loss.ReferenceStatistics() for k in readings}
    t0 = time.time()
    for k, ref in enumerate(train):
        dev = data.load_device(ref)
        for mode, st in stats.items():
            st.add(loss.reference_terms(dev, s, **readings[mode]))
        if (k + 1) % 250 == 0:
            print(f"{k + 1}/{len(train)} {time.time() - t0:.0f}s", flush=True)
    result = {"n_train_devices": len(train), "seed": data.SEED, "C0": s.C0, "mu0": s.mu0,
              "seconds": time.time() - t0}
    for mode, st in stats.items():
        result[mode] = {"sigma2": st.sigma2(), "f": st.f(), "rho": st.rho(), "count": st.count()}
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
