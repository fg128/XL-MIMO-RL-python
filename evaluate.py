"""
Secrecy Rate vs Eve Distance Evaluation
========================================
Loads the trained SAC agent and evaluates it alongside MRT and FPA baselines
across a sweep of Eve-to-Bob distances. Plots mean secrecy rate vs distance
to show whether SAC is surfing the performance envelope.

Run from the project root:
    python evaluate.py
    python evaluate.py --model path/to/model.zip
"""
import os
import sys

from SCA_GCC_convex import XLMIMO_System
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit

from classes.config import Config
from classes.logged_signals import LoggedSignals
from XL_MIMO_Enviroment import XLMIMOEnv
import baselines.mrt as mrt_baseline
import baselines.fpa as fpa_baseline

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
DEFAULT_MODEL   = 'trained_agent.zip'
DIST_MIN        = 0.01          # minimum Eve-Bob separation (m)
DIST_MAX        = 10.0         # maximum Eve-Bob separation (m)
N_DISTANCES     = 5           # number of distance points along x-axis
N_TRIALS        = 10           # random scenarios averaged at each distance
WARMUP_FRAC     = 0.5          # fraction of episode steps discarded as warm-up
FPA_PHIS        = {            # FPA variants to compare
    r'FPA ($\phi$=0.9)': 0.9,
    r'FPA ($\phi$=0.8)': 0.8,
    r'FPA ($\phi$=0.7)': 0.7,
    r'FPA ($\phi$=0.5)': 0.5,
    r'FPA ($\phi$=0.3)': 0.3,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def random_bob(config):
    """Draw a random Bob location inside the grid (same as reset_function)."""
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
    """Overwrite the environment's logged_signals with specific locations and
    fading vectors, returning the matching initial observation."""
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

    r_bob      = np.sqrt(bx**2 + bz**2)
    theta_bob  = np.arctan2(bx, bz)
    r_eve      = np.sqrt(ex**2 + ez**2)
    theta_eve  = np.arctan2(ex, ez)

    obs = np.array([
        r_beam              / max_r,
        theta_beam          / max_theta_sweep,
        start_psf,
        (r_bob - r_beam)    / max_r,
        (theta_bob - theta_beam) / max_theta_sweep,
        (r_eve - r_beam)    / max_r,
        (theta_eve - theta_beam) / max_theta_sweep,
    ], dtype=np.float32)

    return obs


def run_sac_episode(model, raw_env, config, bob_loc, eve_loc):
    """Run one episode of the SAC policy with fixed locations and fading.
    Returns the mean secrecy rate over the post-warmup portion of the episode."""
    obs = inject_locations(raw_env, config, bob_loc, eve_loc)

    T          = config.max_steps_per_episode
    warmup     = int(T * WARMUP_FRAC)
    rates      = []

    for t in range(T):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = raw_env.step(action)
        if t >= warmup:
            rates.append(info.get('secrecy_rate', 0.0))
        if terminated or truncated:
            break

    return float(np.mean(rates)) if rates else 0.0


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #
def evaluate(model_path):
    config  = Config(yaml_path='config.yaml')
    raw_env = XLMIMOEnv(config=config)

    print(f"Loading SAC model from '{model_path}' ...")
    model = SAC.load(model_path, env=raw_env)

    distances = np.linspace(DIST_MIN, DIST_MAX, N_DISTANCES)

    # Accumulators: {label: list of per-distance mean values}
    results = {'SAC': [], 'MRT': [], 'Convex': []}
    results.update({lbl: [] for lbl in FPA_PHIS})

    for dist in distances:
        sac_trial, mrt_trial = [], []
        fpa_trials = {lbl: [] for lbl in FPA_PHIS}

        for _ in range(N_TRIALS):
            bob_loc    = random_bob(config)
            eve_loc    = eve_at_distance(bob_loc, dist)

            # SAC
            sac_sr = run_sac_episode(model, raw_env, config, bob_loc, eve_loc)
            sac_trial.append(sac_sr)

            # MRT
            mrt_sr = mrt_baseline.compute_secrecy_rate(config, bob_loc, eve_loc)
            mrt_trial.append(mrt_sr)

            # FPA variants
            for lbl, phi in FPA_PHIS.items():
                fpa_sr = fpa_baseline.compute_secrecy_rate(config, bob_loc, eve_loc,
                                                            phi=phi)
                fpa_trials[lbl].append(fpa_sr)

            print(bob_loc, eve_loc)
            xl_mimo = XLMIMO_System(K=1, E=1, bob_loc=bob_loc, eve_loc=eve_loc, config=config)
            W_t, epsilon_t, H_LUE, H_EUE, lue_coords, eue_coords = xl_mimo.run_alternating_optimization()
            convex_sr = xl_mimo.calculate_secrecy_rate(H_LUE, H_EUE, W_t, epsilon_t, debug=True)

        results['SAC'].append(np.mean(sac_trial))
        results['MRT'].append(np.mean(mrt_trial))
        results['Convex'].append(np.mean(convex_sr))
        for lbl in FPA_PHIS:
            results[lbl].append(np.mean(fpa_trials[lbl]))

        print(f"  d={dist:5.1f}m | SAC={results['SAC'][-1]:.3f} | "
              f"MRT={results['MRT'][-1]:.3f} | "
              + " | ".join(f"{lbl}={results[lbl][-1]:.3f}" for lbl in FPA_PHIS))

    # ----------------------------------------------------------------------- #
    # Plot
    # ----------------------------------------------------------------------- #
    # Save data as csv
    with open('evaluation_data.csv', 'w', newline='') as f:
        header = 'Distance,' + ','.join(results.keys()) + '\n'
        f.write(header)
        for i in range(len(distances)):
            row = f"{distances[i]:.2f}," + ','.join(f"{results[label][i]:.6f}" for label in results) + '\n'
            f.write(row)
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        'SAC':                       dict(color='tab:blue',   lw=2.5, ls='-',  marker='o', ms=5, zorder=5),
        'MRT':                       dict(color='tab:red',    lw=1.8, ls='--', marker='s', ms=4),
        'Convex':                    dict(color='tab:cyan',   lw=1.8, ls='-.', marker='X', ms=4),
        r'FPA ($\phi$=0.9)':         dict(color='tab:green',  lw=1.5, ls=':',  marker='^', ms=4),
        r'FPA ($\phi$=0.8)':         dict(color='tab:orange', lw=1.5, ls=':',  marker='v', ms=4),
        r'FPA ($\phi$=0.7)':         dict(color='tab:purple', lw=1.5, ls=':',  marker='D', ms=4),
        r'FPA ($\phi$=0.5)':         dict(color='tab:brown',  lw=1.5, ls=':',  marker='P', ms=4),
        r'FPA ($\phi$=0.3)':         dict(color='tab:pink',   lw=1.5, ls=':',  marker='X', ms=4),
    }

    for label, values in results.items():
        ax.plot(distances, values, label=label, **styles[label])

    ax.set_xlabel("Eve Distance from Bob (m)", fontsize=13)
    ax.set_ylabel("Mean Secrecy Rate (bits/s/Hz)", fontsize=13)
    ax.set_title("Secrecy Rate vs Eve–Bob Separation", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(DIST_MIN, DIST_MAX)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path = 'evaluation_secrecy_vs_eve_dist.png'
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to '{out_path}'")
    plt.show()


if __name__ == '__main__':
    model_file = 'checkpoints/SAC_11_79578.zip'
    evaluate(model_file)
