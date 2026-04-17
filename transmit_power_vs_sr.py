"""
Secrecy Rate & Phi vs Transmit Power Evaluation (dBm)
========================================
Loads the trained SAC agent and evaluates it alongside MRT, FPA, and Convex baselines
across a sweep of total transmit powers in dBm.

Generates TWO plots:
1. Mean secrecy rate vs power (3x3 grid)
2. Mean Power Splitting Factor (phi) vs power (3x3 grid)

Also saves all results to a CSV and prints a distance-to-SCA+GSS ranking.

Run from the project root:
    python evaluate_power.py
"""
import os
import sys
import time
import warnings
import csv

# Suppress Mean of Empty Slice warnings if SAC/Convex are commented out
warnings.filterwarnings("ignore", category=RuntimeWarning)

from SCA_GCC_convex import XLMIMO_System
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from classes.config import Config
from classes.logged_signals import LoggedSignals
from XL_MIMO_Enviroment import XLMIMOEnv
import baselines.mrt as mrt_baseline
import baselines.fpa as fpa_baseline

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
FILE_NAME       = "checkpoint_120219"
DEFAULT_MODEL   = 'training_results/SAC_41_20260416_220817/checkpoints/checkpoint_120219.zip'
POWER_MIN_DBM   = 0.0          # minimum Transmit Power (dBm) -> equals 0.001 W
POWER_MAX_DBM   = 30.0         # maximum Transmit Power (dBm) -> equals 1.0 W
N_POWERS        = 5            # number of points along the x-axis
N_TRIALS        = 10           # random scenarios averaged at each power step
WARMUP_FRAC     = 0.75         # fraction of episode steps discarded as warm-up

GRID_ROWS       = 3
GRID_COLS       = 3

FPA_PHIS        = {            # FPA variants to compare
    r'FPA ($\phi$=0.9)': 0.9,
    r'FPA ($\phi$=0.7)': 0.7,
    r'FPA ($\phi$=0.5)': 0.5,
    r'FPA ($\phi$=0.3)': 0.3,
    r'FPA ($\phi$=0.1)': 0.1,
}

# --------------------------------------------------------------------------- #
# Global store for all results (used for CSV export and distance analysis)
# --------------------------------------------------------------------------- #
# Structure: all_results[eve_dist]['sr' | 'phi']['method'] = [values over powers_dbm]
all_results = {}

# --------------------------------------------------------------------------- #
# Global store for per-decision inference times (ms), pooled across all
# (distance, power, trial) combinations. One list per method.
# --------------------------------------------------------------------------- #
inference_times: dict[str, list[float]] = {'SAC': [], 'MRT': [], 'SCA+GSS': []}
inference_times.update({lbl: [] for lbl in FPA_PHIS})

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def random_bob(config):
    """Draw a random Bob location inside the grid."""
    bx = (np.random.rand() - 0.5) * 2 * config.max_x
    bz = 20 + np.random.rand() * (config.max_z - 20)
    return np.array([bx, 0.0, bz])


def eve_at_distance(bob_loc, dist):
    """Place Eve exactly `dist` metres from Bob in a random 2-D direction."""
    angle = np.random.uniform(0, 2 * np.pi)
    ex = bob_loc[0] + dist * np.cos(angle)
    ez = bob_loc[2] + dist * np.sin(angle)
    return np.array([ex, 0.0, ez])


def inject_locations(raw_env, config, bob_loc, eve_loc):
    """Overwrite the environment's logged_signals with specific locations."""
    bx, bz = bob_loc[0], bob_loc[2]
    ex, ez = eve_loc[0], eve_loc[2]

    r_beam     = np.sqrt(bx**2 + bz**2)
    theta_beam = np.arctan2(bx, bz)
    start_psf  = np.random.rand()

    raw_env.logged_signals = LoggedSignals(
        bob_loc=bob_loc,
        eve_loc=eve_loc,
        ideal_r=r_beam,
        ideal_theta=theta_beam,
        ideal_psf=start_psf,
    )

    max_r           = np.sqrt(config.max_x**2 + config.max_z**2)
    max_theta_sweep = np.pi / 2

    r_bob        = np.sqrt(bx**2 + bz**2)
    theta_bob    = np.arctan2(bx, bz)
    r_eve        = np.sqrt(ex**2 + ez**2)
    theta_eve    = np.arctan2(ex, ez)
    bob_eve_diff = np.linalg.norm(bob_loc - eve_loc)

    obs = np.array([
        r_beam                        / max_r,
        theta_beam                    / max_theta_sweep,
        start_psf,
        (r_bob   - r_beam)            / max_r,
        (theta_bob - theta_beam)      / max_theta_sweep,
        (r_eve   - r_beam)            / max_r,
        (theta_eve - theta_beam)      / max_theta_sweep,
        np.exp(-bob_eve_diff)
    ], dtype=np.float32)

    return obs


def run_sac_episode(model, raw_env, config, bob_loc, eve_loc):
    """Run one episode of the SAC policy with fixed locations and fading.

    Returns:
        mean_sr:          average secrecy rate over post-warmup steps
        mean_phi:         average phi over post-warmup steps
        predict_times_ms: list of per-step `model.predict` wall-clock times (ms)
    """
    obs = inject_locations(raw_env, config, bob_loc, eve_loc)

    T      = config.max_steps_per_episode
    warmup = int(T * WARMUP_FRAC)
    rates            = []
    phis             = []
    predict_times_ms = []

    for t in range(T):
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        t1 = time.perf_counter()
        predict_times_ms.append((t1 - t0) * 1000.0)

        obs, _, terminated, truncated, info = raw_env.step(action)

        if t >= warmup:
            rates.append(info.get('secrecy_rate', 0.0))
            phis.append(raw_env.logged_signals.ideal_psf)

        if terminated or truncated:
            break

    mean_sr  = float(np.mean(rates)) if rates else 0.0
    mean_phi = float(np.mean(phis))  if phis  else 0.0
    return mean_sr, mean_phi, predict_times_ms


# --------------------------------------------------------------------------- #
# Plotting style map
# --------------------------------------------------------------------------- #
STYLES = {
    'SAC':                   dict(color='tab:blue',   lw=2.5, ls='-',   marker='o', ms=5, zorder=5),
    'MRT':                   dict(color='tab:red',    lw=1.8, ls='--',  marker='s', ms=4),
    'SCA+GSS':               dict(color='tab:cyan',   lw=1.8, ls='-',   marker='X', ms=4),
    r'FPA ($\phi$=0.9)':     dict(color='tab:green',  lw=1.5, ls='-.',  marker='^', ms=4, alpha=0.6),
    r'FPA ($\phi$=0.7)':     dict(color='tab:purple', lw=1.5, ls='-.',  marker='D', ms=4, alpha=0.6),
    r'FPA ($\phi$=0.5)':     dict(color='tab:brown',  lw=1.5, ls='-.',  marker='P', ms=4, alpha=0.6),
    r'FPA ($\phi$=0.3)':     dict(color='tab:pink',   lw=1.5, ls='-.',  marker='X', ms=4, alpha=0.6),
    r'FPA ($\phi$=0.1)':     dict(color='tab:gray',   lw=1.5, ls='-.',  marker='h', ms=4, alpha=0.6),
}


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #
def evaluate_on_axis(model_path, ax_sr, ax_phi, eve_dist, row, col):
    """
    Run the full power sweep for a single Eve distance and populate both axes.
    row / col are 0-based grid positions, used for axis label placement.
    """
    config  = Config(yaml_path='config.yaml')
    raw_env = XLMIMOEnv(config=config)
    model   = SAC.load(model_path, env=raw_env)

    powers_dbm = np.linspace(POWER_MIN_DBM, POWER_MAX_DBM, N_POWERS)

    # Initialise result dicts — insertion order preserved (Python 3.7+)
    results_sr  = {'SAC': [], 'MRT': [], 'SCA+GSS': []}
    results_phi = {'SAC': [], 'MRT': [], 'SCA+GSS': []}
    results_sr.update( {lbl: [] for lbl in FPA_PHIS})
    results_phi.update({lbl: [] for lbl in FPA_PHIS})

    print(f"\nEvaluating for Eve Distance = {eve_dist}m ...")

    for power_dbm in powers_dbm:
        power_watts          = 10 ** ((power_dbm - 30) / 10)
        config.P_total_watts = power_watts

        sac_sr_trial, mrt_sr_trial, convex_sr_trial   = [], [], []
        sac_phi_trial, mrt_phi_trial, convex_phi_trial = [], [], []
        fpa_sr_trials  = {lbl: [] for lbl in FPA_PHIS}
        fpa_phi_trials = {lbl: [] for lbl in FPA_PHIS}

        for _ in range(N_TRIALS):
            bob_loc = random_bob(config)
            eve_loc = eve_at_distance(bob_loc, eve_dist)

            # --- SAC ---
            sac_sr, sac_phi = run_sac_episode(model, raw_env, config, bob_loc, eve_loc)
            sac_sr_trial.append(sac_sr)
            sac_phi_trial.append(sac_phi)

            # --- MRT ---
            mrt_sr = mrt_baseline.compute_secrecy_rate(config, bob_loc, eve_loc)
            mrt_sr_trial.append(mrt_sr)
            mrt_phi_trial.append(1.0)          # MRT uses 100% data power

            # --- FPA ---
            for lbl, phi in FPA_PHIS.items():
                fpa_sr = fpa_baseline.compute_secrecy_rate(config, bob_loc, eve_loc, phi=phi)
                fpa_sr_trials[lbl].append(fpa_sr)
                fpa_phi_trials[lbl].append(phi)

            # --- Convex (SCA+GSS) ---
            xl_mimo = XLMIMO_System(K=1, E=1, bob_loc=bob_loc, eve_loc=eve_loc, config=config)
            W_t, phi_t, H_LUE, H_EUE, lue_coords, eue_coords = xl_mimo.run_alternating_optimization()
            convex_sr = xl_mimo.calculate_secrecy_rate(H_LUE, H_EUE, W_t, phi_t)
            convex_sr_trial.append(convex_sr)
            convex_phi_trial.append(phi_t)

        # ---- Store trial averages ----
        if sac_sr_trial:
            results_sr['SAC'].append(np.mean(sac_sr_trial))
            results_phi['SAC'].append(np.mean(sac_phi_trial))
        if mrt_sr_trial:
            results_sr['MRT'].append(np.mean(mrt_sr_trial))
            results_phi['MRT'].append(np.mean(mrt_phi_trial))
        if convex_sr_trial:
            results_sr['SCA+GSS'].append(np.mean(convex_sr_trial))
            results_phi['SCA+GSS'].append(np.mean(convex_phi_trial))  # no extra list wrap

        for lbl in FPA_PHIS:
            if fpa_sr_trials[lbl]:
                results_sr[lbl].append(np.mean(fpa_sr_trials[lbl]))
                results_phi[lbl].append(np.mean(fpa_phi_trials[lbl]))

        print(
            f"  P={power_dbm:5.1f} dBm | "
            f"MRT_SR={results_sr['MRT'][-1]:.2f} | "
            f"SAC_PHI={results_phi['SAC'][-1]:.2f} | "
            f"CONVEX_PHI={results_phi['SCA+GSS'][-1]:.2f}"
        )

    # ---- Persist for CSV / distance analysis ----
    all_results[eve_dist] = {
        'powers_dbm': powers_dbm.tolist(),
        'sr':  {k: list(v) for k, v in results_sr.items()},
        'phi': {k: list(v) for k, v in results_phi.items()},
    }

    # ---- Populate SR axis ----
    for label, values in results_sr.items():
        if values:
            ax_sr.plot(powers_dbm[:len(values)], values, label=label, **STYLES[label])
    ax_sr.set_title(f"Eve Distance = {eve_dist} m", fontsize=11)
    ax_sr.grid(True, alpha=0.3)
    ax_sr.set_xlim(POWER_MIN_DBM, POWER_MAX_DBM)
    ax_sr.set_ylim(0, 25)
    if row == GRID_ROWS - 1:                    # bottom row only
        ax_sr.set_xlabel("Total Transmit Power (dBm)", fontsize=10)
    if col == 0:                                 # left column only
        ax_sr.set_ylabel("Secrecy Rate (bits/s/Hz)", fontsize=10)

    # ---- Populate Phi axis ----
    for label, values in results_phi.items():
        if values:
            ax_phi.plot(powers_dbm[:len(values)], values, label=label, **STYLES[label])
    ax_phi.set_title(f"Eve Distance = {eve_dist} m", fontsize=11)
    ax_phi.grid(True, alpha=0.3)
    ax_phi.set_xlim(POWER_MIN_DBM, POWER_MAX_DBM)
    ax_phi.set_ylim(0.0, 1.05)
    if row == GRID_ROWS - 1:
        ax_phi.set_xlabel("Total Transmit Power (dBm)", fontsize=10)
    if col == 0:
        ax_phi.set_ylabel(r"Allocated Power Factor ($\phi$)", fontsize=10)


# --------------------------------------------------------------------------- #
# CSV Export
# --------------------------------------------------------------------------- #
def save_results_to_csv(config, file_name):
    """Write all collected results to a flat CSV for later reconstruction."""
    out_path = (
        f'evaluation_plots/{config.Nt}/'
        f'eval_data_{config.Nt}_{config.fc}_{file_name}.csv'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eve_dist_m', 'metric', 'method', 'power_dbm', 'value'])

        for eve_dist, data in all_results.items():
            powers = data['powers_dbm']
            for metric in ('sr', 'phi'):
                for method, values in data[metric].items():
                    for power_dbm, val in zip(powers, values):
                        writer.writerow([
                            eve_dist, metric, method,
                            round(power_dbm, 4), round(val, 6)
                        ])

    print(f"\nAll results saved to '{out_path}'")
    return out_path


# --------------------------------------------------------------------------- #
# Distance-to-SCA+GSS Analysis
# --------------------------------------------------------------------------- #
def compute_and_print_distances():
    """
    For each method (excluding SCA+GSS), compute the L2 norm of the difference
    between its SR curve and the SCA+GSS SR curve at each Eve-distance subplot.
    Sum across all subplots for a single overall score, then rank methods.
    """
    print("\n" + "=" * 90)
    print("  DISTANCE TO SCA+GSS BENCHMARK  (Secrecy Rate curves, L2 norm per subplot)")
    print("=" * 90)

    sorted_dists = sorted(all_results.keys())
    sample_data  = all_results[sorted_dists[0]]
    all_methods  = [m for m in sample_data['sr'].keys() if m != 'SCA+GSS']

    dist_table = {m: [] for m in all_methods}

    col_w   = 24
    d_col_w = 9
    header  = (
        f"{'Method':<{col_w}}"
        + "".join(f"{'d='+str(d)+'m':>{d_col_w}}" for d in sorted_dists)
        + f"{'TOTAL':>{d_col_w + 2}}"
    )
    print(header)
    print("-" * len(header))

    for method in all_methods:
        row_vals = []
        for eve_dist in sorted_dists:
            data        = all_results[eve_dist]
            sca_vals    = np.array(data['sr'].get('SCA+GSS', []))
            method_vals = np.array(data['sr'].get(method,    []))

            if len(sca_vals) == 0 or len(method_vals) == 0:
                row_vals.append(float('nan'))
                continue

            n    = min(len(sca_vals), len(method_vals))
            dist = float(np.linalg.norm(method_vals[:n] - sca_vals[:n]))
            row_vals.append(dist)

        dist_table[method] = row_vals
        total   = float(np.nansum(row_vals))
        row_str = (
            f"{method:<{col_w}}"
            + "".join(f"{v:>{d_col_w}.4f}" for v in row_vals)
            + f"{total:>{d_col_w + 2}.4f}"
        )
        print(row_str)

    # Rank by total distance ascending (closest = best)
    totals = {m: float(np.nansum(v)) for m, v in dist_table.items()}
    ranked = sorted(totals.items(), key=lambda x: x[1])

    print("\n" + "=" * 90)
    print("  RANKING  (smallest total L2 distance to SCA+GSS = best approximation)")
    print("=" * 90)
    for rank, (method, total) in enumerate(ranked, start=1):
        marker = "  <-- BEST" if rank == 1 else ""
        print(f"  {rank:>2}. {method:<{col_w}}  total L2 = {total:.4f}{marker}")

    print("=" * 90)

    best_method = ranked[0][0]
    print(f"\n  >> Best approximation of SCA+GSS: '{best_method}'")
    if best_method == 'SAC':
        print("  >> SAC IS the closest method to SCA+GSS across all distances and powers.\n")
    else:
        sac_rank = next(r for r, (m, _) in enumerate(ranked, 1) if m == 'SAC')
        print(f"  >> SAC ranks #{sac_rank} — it is NOT the closest overall.\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    eve_distances = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    assert len(eve_distances) == GRID_ROWS * GRID_COLS, (
        f"eve_distances must have exactly {GRID_ROWS * GRID_COLS} entries "
        f"for a {GRID_ROWS}x{GRID_COLS} grid."
    )

    fig_sr,  axes_sr  = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(16, 12))
    fig_phi, axes_phi = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(16, 12))

    for idx, dist in enumerate(eve_distances):
        row, col = divmod(idx, GRID_COLS)
        evaluate_on_axis(
            DEFAULT_MODEL,
            axes_sr[row, col],
            axes_phi[row, col],
            dist,
            row, col,
        )

    config = Config()

    # ---------- Save CSV ----------
    save_results_to_csv(config, FILE_NAME)

    # ---------- Distance analysis ----------
    compute_and_print_distances()

    # ---------- Finalise SR figure ----------
    handles, labels = axes_sr[0, 0].get_legend_handles_labels()
    fig_sr.legend(handles, labels, loc='upper center', ncol=len(labels),
                  fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig_sr.suptitle("Secrecy Rate vs Transmit Power", fontsize=13, y=1.01)
    fig_sr.tight_layout(rect=[0, 0, 1, 0.96])
    out_path_sr = (
        f'evaluation_plots/{config.Nt}/'
        f'eval_SR_subplots_{config.Nt}_{config.fc}_{FILE_NAME}.png'
    )
    os.makedirs(os.path.dirname(out_path_sr), exist_ok=True)
    fig_sr.savefig(out_path_sr, dpi=200, bbox_inches='tight')
    print(f"Secrecy Rate plot saved to '{out_path_sr}'")

    # ---------- Finalise Phi figure ----------
    handles_phi, labels_phi = axes_phi[0, 0].get_legend_handles_labels()
    fig_phi.legend(handles_phi, labels_phi, loc='upper center', ncol=len(labels_phi),
                   fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig_phi.suptitle(r"Power Splitting Factor ($\phi$) vs Transmit Power", fontsize=13, y=1.01)
    fig_phi.tight_layout(rect=[0, 0, 1, 0.96])
    out_path_phi = (
        f'evaluation_plots/{config.Nt}/'
        f'eval_PHI_subplots_{config.Nt}_{config.fc}_{FILE_NAME}.png'
    )
    fig_phi.savefig(out_path_phi, dpi=200, bbox_inches='tight')
    print(f"Phi Allocation plot saved to '{out_path_phi}'")

    plt.show()
