"""Config loading (TOML) and result output (HDF5) with provenance."""

import json
import math
import tomllib
from pathlib import Path

import h5py
import numpy as np

DEFAULTS = {
    # composition
    "n_pairs": 300,          # light+strange q-qbar pairs
    "n_ccbar": 0,
    "density": 2.0,          # total fm^-3 at t=0
    "t0": 0.20,              # GeV
    # interaction
    "alpha_s": 0.4,
    "lam_d0": 0.4,           # fm
    "r0": 0.2,               # fm
    "color_mode": "mean_channel",
    # dynamics
    "dt": 0.005,             # fm/c
    "t_equil": 10.0,         # fm/c
    "thermostat_nu": 1.0,
    "h0": 0.1,               # c/fm
    "a_max": 20.0,
    # reactions
    "lambda_scatt": 1.0,     # c/fm
    "tau_bound": math.inf,   # fm/c ("inf" in TOML -> parsed below)
    "p_gamma": 0.1,
    "t_chem": 0.16,          # GeV
    "d_ann": 0.2,            # fm
    "react_every": 1,
    # measurement
    "snap_interval": 1.0,    # fm/c
    "measure_below_t": 0.16, # GeV
    "r_cl_factor": 3.0,
    "delta_e_th": 0.0,
    "n_persist": 3,
    "pathway_window": 1.0,
    # mode: "dynamic" or "static_lebed"
    "mode": "dynamic",
    # static-Lebed-mode options
    "static_R": 1.0,
    "static_n0": 8.0,
    "static_n_realizations": 5,
}


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    merged = dict(DEFAULTS)
    merged.update(cfg)
    if isinstance(merged.get("tau_bound"), str):
        merged["tau_bound"] = float(merged["tau_bound"])  # "inf" -> inf
    return merged


def save_results(path, cfg, seed, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["config_json"] = json.dumps(
            {k: (str(v) if v == math.inf else v) for k, v in cfg.items()}
        )
        f.attrs["seed"] = seed
        from . import __version__

        f.attrs["version"] = __version__

        g = f.create_group("final")
        g.attrs["species_counts"] = json.dumps(results["species_counts"])
        g.attrs["exotic_counts"] = json.dumps(results["exotic_counts"])
        g.attrs["diquark_first_fraction"] = (
            results["diquark_first_fraction"]
            if results["diquark_first_fraction"] == results["diquark_first_fraction"]
            else -1.0
        )
        g.attrs["n_pathway_baryons"] = results["n_pathway_baryons"]
        g.attrs["n_alive_final"] = results["n_alive_final"]
        g.attrs["n_free_final"] = results["n_free_final"]
        g.attrs["t_final"] = results["t_final"]
        g.attrs["a_final"] = results["a_final"]

        if results["photon_ledger"]:
            f.create_dataset("photon_ledger",
                             data=np.array(results["photon_ledger"]))
        hist = results["history"]
        if hist:
            keys = ["t", "a", "t_eff", "L", "n_alive", "n_annihilations",
                    "n_reinjections", "n_photons"]
            arr = np.array([[h.get(k, np.nan) for k in keys] for h in hist])
            ds = f.create_dataset("history", data=arr)
            ds.attrs["columns"] = json.dumps(keys)
            f.attrs["species_history_json"] = json.dumps(
                [{**{"t": h["t"]}, **h.get("species", {})} for h in hist]
            )
    return path
