"""Command-line interface: python -m diquark_md run <config.toml> [options]."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _run_dynamic(cfg, seed, out):
    from .io import save_results
    from .sim import Simulation

    t_start = time.time()
    sim = Simulation(cfg, seed=seed)
    n0 = int(sim.state["alive"].sum())
    print(f"[init] N={n0}  L={sim.L:.2f} fm  T0={cfg['t0']} GeV  "
          f"seed={seed}")
    sim.equilibrate()
    print(f"[equil] T_eff={sim.t_eff():.4f} GeV  "
          f"({time.time() - t_start:.1f}s)")

    def progress(s):
        rec = s.history[-1]
        spec = rec.get("species", {})
        print(f"[t={rec['t']:7.2f}] a={rec['a']:6.2f} T={rec['t_eff']:.4f} "
              f"N={rec['n_alive']} ann={rec['n_annihilations']} "
              f"gamma={rec['n_photons']} species={spec}")

    results = sim.run_production(progress=progress)
    print(f"[done] t={results['t_final']:.1f} fm/c  "
          f"({time.time() - t_start:.1f}s)")
    print("final species:", results["species_counts"])
    print("exotic:", results["exotic_counts"])
    print("diquark-first fraction:",
          f"{results['diquark_first_fraction']:.3f} "
          f"(n={results['n_pathway_baryons']})")
    path = save_results(out, cfg, seed, results)
    print(f"[saved] {path}")
    return results


def _run_static(cfg, seed, out):
    """Static Lebed validation mode: ideal-gas snapshot, both estimators."""
    from .observables import lebed_analytic, lebed_static_mc

    rng = np.random.default_rng(seed)
    n0 = cfg["static_n0"]
    R = cfg["static_R"]
    L = 12.0 * R
    n_each = int(n0 * L**3)
    rows = []
    for _ in range(cfg["static_n_realizations"]):
        x = rng.uniform(0, L, (2 * n_each, 3))
        labels = np.concatenate([
            np.repeat(np.arange(3), n_each // 3 + 1)[:n_each],
            np.repeat(np.arange(3, 6), n_each // 3 + 1)[:n_each],
        ]).astype(np.int64)
        p_leb, _ = lebed_static_mc(x, labels, L, R=R, mode="lebed", rng=rng)
        p_dis, _ = lebed_static_mc(x, labels, L, R=R, mode="discrete", rng=rng)
        rows.append((p_leb, p_dis))
    rows = np.array(rows)
    ana_leb = lebed_analytic(n0, R=R)
    ana_dis = lebed_analytic(n0, R=R, frac1=2 / 3, frac2=1 / 3)
    print(f"n0 = {n0} / R^3   (Lebed PRD 94, 034039 hard-wall convention)")
    print(f"  Lebed mode:    MC = {rows[:,0].mean():.4f} +- "
          f"{rows[:,0].std(ddof=1)/np.sqrt(len(rows)):.4f}   "
          f"analytic = {ana_leb:.4f}")
    print(f"  discrete mode: MC = {rows[:,1].mean():.4f} +- "
          f"{rows[:,1].std(ddof=1)/np.sqrt(len(rows)):.4f}   "
          f"analytic = {ana_dis:.4f}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({
            "n0": n0, "R": R,
            "mc_lebed": rows[:, 0].tolist(),
            "mc_discrete": rows[:, 1].tolist(),
            "analytic_lebed": ana_leb,
            "analytic_discrete": ana_dis,
        }, indent=2))
        print(f"[saved] {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="diquark-md")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run a simulation from a TOML config")
    p_run.add_argument("config")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from .io import load_config

    cfg = load_config(args.config)
    stem = Path(args.config).stem
    if cfg["mode"] == "static_lebed":
        out = args.out or f"runs/{stem}_seed{args.seed}.json"
        _run_static(cfg, args.seed, out)
    elif cfg["mode"] == "scan":
        import tomllib

        from .scan import run_scan

        with open(args.config, "rb") as f:
            scan_cfg = tomllib.load(f)
        base_cfg = load_config(scan_cfg["base"])
        out = args.out or f"runs/{stem}.jsonl"
        run_scan(scan_cfg, base_cfg, out)
    else:
        out = args.out or f"runs/{stem}_seed{args.seed}.h5"
        _run_dynamic(cfg, args.seed, out)


if __name__ == "__main__":
    main()
