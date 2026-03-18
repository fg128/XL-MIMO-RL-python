"""Run all three baselines and report mean secrecy rate over random trials.

Usage:
    python -m baselines.run_baselines
    python -m baselines.run_baselines --n_trials 1000 --seed 42
    python -m baselines.run_baselines --n_trials 100 --seed 0 --no_save
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classes.config import Config
from baselines.exhaustive_search import exhaustive_search
from baselines.fixed_power_allocation import fixed_power_allocation
from baselines.mrt import mrt_no_an


def sample_locations(config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Sample random Bob and Eve locations matching reset_function.py exactly."""
    bx = (np.random.rand() - 0.5) * 2 * config.max_x
    bz = 20 + np.random.rand() * (config.max_z - 20)
    bob_loc = np.array([bx, 0.0, bz])

    ex = bx + (np.random.rand() - 0.5) * 20
    ez = bz + (np.random.rand() - 0.5) * 20
    eve_loc = np.array([ex, 0.0, ez])

    return bob_loc, eve_loc


def run_baselines(
    n_trials: int = 1000,
    seed: int = 42,
    psf_fpa: tuple = (0.5, 0.8),
    save_results: bool = True,
    yaml_path: str = "config.yaml",
) -> dict:
    """Run all baselines over n_trials random Bob/Eve configurations.

    Args:
        n_trials:     Number of random location trials.
        seed:         Random seed for reproducibility.
        psf_fpa:      PSF values for the Fixed Power Allocation baseline.
        save_results: Whether to save results to a JSON file.
        yaml_path:    Path to config.yaml.

    Returns:
        results: Dict with lists of secrecy rates per method.
    """
    np.random.seed(seed)
    config = Config(yaml_path)

    results = {
        "exhaustive": [],
        **{f"fpa_{p}": [] for p in psf_fpa},
        "mrt": [],
    }

    print(f"Running {n_trials} trials (seed={seed})...")
    t_start = time.time()

    for trial in range(n_trials):
        if trial % max(1, n_trials // 10) == 0:
            elapsed = time.time() - t_start
            print(f"  Trial {trial}/{n_trials}  ({elapsed:.1f}s elapsed)")

        bob_loc, eve_loc = sample_locations(config)

        # Exhaustive search
        sr_es, _, _ = exhaustive_search(config, bob_loc, eve_loc)
        results["exhaustive"].append(sr_es)

        # Fixed power allocation
        fpa_srs = fixed_power_allocation(config, bob_loc, eve_loc, psf_values=psf_fpa)
        for p in psf_fpa:
            results[f"fpa_{p}"].append(fpa_srs[p])

        # MRT / no AN
        results["mrt"].append(mrt_no_an(config, bob_loc, eve_loc))

    total_time = time.time() - t_start

    # Compute summary statistics
    summary = {}
    for method, values in results.items():
        arr = np.array(values)
        summary[method] = {"mean": float(arr.mean()), "std": float(arr.std())}

    # Print results table
    print()
    print(f"{'=' * 60}")
    print(f"  Baseline Results  (N={n_trials} trials, seed={seed})")
    print(f"{'=' * 60}")
    col_w = 28
    for method, stats in summary.items():
        if method == "exhaustive":
            label = "Exhaustive Search (upper bound)"
        elif method == "mrt":
            label = "MRT / No AN (lower bound)"
        else:
            label = f"FPA (psf={method.split('_')[1]})"
        print(f"  {label:<{col_w}}  {stats['mean']:.4f} +/- {stats['std']:.4f} bps/Hz")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time:.1f}s  ({total_time / n_trials * 1000:.1f} ms/trial)")
    print()

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"baselines_results_{timestamp}.json"
        payload = {
            "n_trials": n_trials,
            "seed": seed,
            "psf_fpa": list(psf_fpa),
            "summary": summary,
            "raw": {k: [float(v) for v in vals] for k, vals in results.items()},
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XL-MIMO baselines")
    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--config", default="config.yaml", dest="yaml_path")
    args = parser.parse_args()

    run_baselines(
        n_trials=args.n_trials,
        seed=args.seed,
        save_results=not args.no_save,
        yaml_path=args.yaml_path,
    )
