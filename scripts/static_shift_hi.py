#!/usr/bin/env python3
"""High-statistics correlated-gas static shift (paired, N=3000).

Resolves the sign of the equilibrated-minus-ideal shift in the static
diquark-formation probability.  Writes runs/static_shift_hi.json.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

os.environ.setdefault("NUMBA_NUM_THREADS", "2")

N_SEEDS = 64
N_PAIRS = 1500


def one_seed(seed):
    import numpy as np

    from diquark_md.io import load_config
    from diquark_md.observables import lebed_static_mc
    from diquark_md.sim import Simulation

    cfg = load_config("configs/default.toml")
    cfg["n_pairs"] = N_PAIRS
    sim = Simulation(cfg, seed=1000 + seed)
    s = sim.state
    x0 = s["x"].copy()
    sim.equilibrate()
    out = {"seed": seed}
    for mode in ("lebed", "discrete"):
        import numpy as np

        p_i, _ = lebed_static_mc(x0, s["labels"], sim.L, R=1.0, mode=mode,
                                 rng=np.random.default_rng(seed))
        p_e, _ = lebed_static_mc(s["x"], s["labels"], sim.L, R=1.0, mode=mode,
                                 rng=np.random.default_rng(seed))
        out[mode] = {"ideal": p_i, "equilibrated": p_e, "shift": p_e - p_i}
    return out


def main():
    import numpy as np

    rows = []
    with ProcessPoolExecutor(max_workers=8,
                             mp_context=get_context("spawn")) as pool:
        futs = [pool.submit(one_seed, s) for s in range(N_SEEDS)]
        for fut in as_completed(futs):
            rows.append(fut.result())
            print(f"[{len(rows)}/{N_SEEDS}]", flush=True)

    summary = {}
    for mode in ("lebed", "discrete"):
        d = np.array([r[mode]["shift"] for r in rows])
        summary[mode] = {
            "mean_shift": float(d.mean()),
            "err": float(d.std(ddof=1) / np.sqrt(len(d))),
        }
        print(f"{mode}: shift = {d.mean():+.4f} +- "
              f"{d.std(ddof=1)/np.sqrt(len(d)):.4f}")
    os.makedirs("runs", exist_ok=True)
    with open("runs/static_shift_hi.json", "w") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=2)
    print("saved runs/static_shift_hi.json")


if __name__ == "__main__":
    main()
