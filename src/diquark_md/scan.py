"""Ensemble scan runner: base config x parameter points x seeds, in parallel.

Scan TOML format:

    mode = "scan"
    base = "configs/default.toml"
    seeds = 24
    max_workers = 8

    [[points]]
    name = "H0_0.05"
    h0 = 0.05

Results stream to a JSONL file (one record per completed run) so partial
scans survive interruption.
"""

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path


def _run_one(base_cfg, overrides, seed):
    """Worker: run one event, return a summary record (JSON-serializable)."""
    from .sim import Simulation

    cfg = dict(base_cfg)
    cfg.update({k: v for k, v in overrides.items() if k != "name"})
    if isinstance(cfg.get("tau_bound"), str):
        cfg["tau_bound"] = float(cfg["tau_bound"])
    t_start = time.time()
    sim = Simulation(cfg, seed=seed)
    sim.equilibrate()
    res = sim.run_production()
    dq = res["diquark_first_fraction"]
    return {
        "point": overrides.get("name", "base"),
        "overrides": {k: (str(v) if v == math.inf else v)
                      for k, v in overrides.items() if k != "name"},
        "seed": seed,
        "species_counts": res["species_counts"],
        "exotic_counts": res["exotic_counts"],
        "diquark_first_fraction": None if dq != dq else dq,
        "n_pathway_baryons": res["n_pathway_baryons"],
        "n_alive_final": res["n_alive_final"],
        "n_free_final": res["n_free_final"],
        "n_photons": len(res["photon_ledger"]),
        "n_gluon_events": len(res["gluon_events"]),
        "t_final": res["t_final"],
        "walltime_s": round(time.time() - t_start, 2),
    }


def run_scan(scan_cfg, base_cfg, out_path):
    points = scan_cfg.get("points", [{"name": "base"}])
    seeds = int(scan_cfg.get("seeds", 8))
    max_workers = int(scan_cfg.get("max_workers", 8))

    # cores used ~= max_workers x numba_threads; RAM ~= 300 MB / worker
    os.environ["NUMBA_NUM_THREADS"] = str(scan_cfg.get("numba_threads", 2))
    jobs = [(pt, seed) for pt in points for seed in range(seeds)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    t0 = time.time()
    ctx = get_context("spawn")
    with out_path.open("w") as fh, ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx
    ) as pool:
        futures = {
            pool.submit(_run_one, base_cfg, pt, seed): (pt, seed)
            for pt, seed in jobs
        }
        for fut in as_completed(futures):
            pt, seed = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # keep the scan alive, record failure
                rec = {"point": pt.get("name", "base"), "seed": seed,
                       "error": repr(exc)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done += 1
            print(f"[{n_done}/{len(jobs)}] {rec.get('point')} seed={seed} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"scan complete: {out_path}")
    return out_path
