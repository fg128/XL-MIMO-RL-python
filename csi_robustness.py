"""
CSI Robustness Evaluation
=========================
Evaluates SAC, SCA+GSS, MRT, and FPA under imperfect channel state information.
Each method designs its beamformer (W), AN nullspace (V), and phi using a NOISY
channel estimate h_hat = h_true + e, while the actual secrecy rate is computed
from the TRUE propagation channels.

Key insight: SAC's beamformer W is geometry-based (USW focal-point model), NOT
built from h_bob. So SAC's W is immune to channel estimation error. Only the AN
nullspace projector V is affected. MRT, FPA, and SCA+GSS all derive W from h_bob,
so their beamformer alignment degrades under noisy CSI.

Plot: Mean SR vs Pilot SNR (dB) at fixed P=30 dBm, Eve=1 m.

Run from the project root:
    python csi_robustness.py
"""
import os
import sys
import csv
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from classes.config import Config
from classes.logged_signals import LoggedSignals
from XL_MIMO_Enviroment import XLMIMOEnv
from functions.get_channel import get_channel
from SCA_GCC_convex import XLMIMO_System

# Reuse helpers from mobility_scenario
from mobility_scenario import (
    build_W_from_focal, build_V, instantaneous_sr,
    SACController, build_sac_obs,
)

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
FILE_NAME     = "checkpoint_120219"
DEFAULT_MODEL = 'training_results/SAC_41_20260416_220817/checkpoints/checkpoint_120219.zip'
POWER_DBM     = 30.0
EVE_DISTS     = [1.0, 10.0]   # two panels: close Eve (AN-heavy) vs far Eve (signal-heavy)
N_TRIALS      = 10
SAC_WARMUP    = 50

PILOT_SNRS_DB = [0, 5, 10, 15, 20, 25, 30]   # finite values
# np.inf is appended at the end for perfect CSI

FPA_PHIS = {
    r'FPA ($\phi$=0.9)': 0.9,
    r'FPA ($\phi$=0.5)': 0.5,
}

METHODS = ['SAC', 'SCA+GSS', 'MRT'] + list(FPA_PHIS.keys())

STYLES = {
    'SAC':                dict(color='tab:blue',   lw=2.5, ls='-',   marker='o', ms=6, zorder=5),
    'SCA+GSS':            dict(color='tab:cyan',   lw=1.8, ls='-',   marker='X', ms=6),
    'MRT':                dict(color='tab:red',    lw=1.6, ls='--',  marker='s', ms=5),
    r'FPA ($\phi$=0.9)':  dict(color='tab:green',  lw=1.4, ls='-.',  marker='^', ms=5, alpha=0.7),
    r'FPA ($\phi$=0.5)':  dict(color='tab:brown',  lw=1.4, ls='-.',  marker='P', ms=5, alpha=0.7),
}


# --------------------------------------------------------------------------- #
# Channel noise model
# --------------------------------------------------------------------------- #
def pilot_snr_to_nmse(pilot_snr_db):
    """MMSE channel estimation NMSE = 1 / (1 + SNR_linear).
    Returns 0 for infinite pilot SNR (perfect CSI).
    """
    if not np.isfinite(pilot_snr_db):
        return 0.0
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    return 1.0 / (1.0 + snr_lin)


def add_channel_noise(h_true, nmse):
    """Add estimation error: h_hat = h_true + e.

    e ~ CN(0, sigma_e^2 * I), with sigma_e^2 chosen so that
    E[||e||^2] / ||h_true||^2 = nmse.

    Args:
        h_true: (Nt, 1) complex channel vector.
        nmse:   normalised mean square error (0 = perfect CSI).

    Returns:
        h_hat: (Nt, 1) noisy channel estimate.
    """
    if nmse <= 0:
        return h_true.copy()
    Nt = h_true.shape[0]
    sigma_e_sq = nmse * float(np.linalg.norm(h_true) ** 2) / Nt
    e = np.sqrt(sigma_e_sq / 2) * (
        np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)
    )
    return h_true + e


def add_noise_to_H(H, nmse):
    """Add per-row estimation noise to a (K, Nt) channel matrix."""
    if nmse <= 0:
        return H.copy()
    K, Nt = H.shape
    H_noisy = H.copy()
    for k in range(K):
        h_k = H[k, :]
        sigma_e_sq = nmse * float(np.linalg.norm(h_k) ** 2) / Nt
        e = np.sqrt(sigma_e_sq / 2) * (
            np.random.randn(Nt) + 1j * np.random.randn(Nt)
        )
        H_noisy[k, :] = h_k + e
    return H_noisy


# --------------------------------------------------------------------------- #
# Per-method update functions under noisy CSI
# --------------------------------------------------------------------------- #
def update_mrt_noisy(config, bob_loc, nmse):
    h_bob = get_channel(config, bob_loc)
    h_hat = add_channel_noise(h_bob, nmse)
    W = h_hat / np.linalg.norm(h_hat)
    # phi=1 means no AN, V irrelevant
    V = np.zeros((config.Nt, config.Nt), dtype=complex)
    return W, 1.0, V


def update_fpa_noisy(config, bob_loc, phi_c, nmse):
    h_bob = get_channel(config, bob_loc)
    h_hat = add_channel_noise(h_bob, nmse)
    W = h_hat / np.linalg.norm(h_hat)
    V = build_V(h_hat, config.Nt)
    return W, phi_c, V


def update_sca_noisy(config, bob_loc, eve_loc, nmse):
    """Run SCA+GSS optimisation on noisy channel estimates."""
    xl = XLMIMO_System(K=1, E=1, bob_loc=bob_loc, eve_loc=eve_loc, config=config)
    H_LUE, H_EUE, _, _ = xl.generate_channels()

    # Add estimation noise
    H_LUE_noisy = add_noise_to_H(H_LUE, nmse)
    H_EUE_noisy = add_noise_to_H(H_EUE, nmse)

    # Reproduce run_alternating_optimization with noisy channels
    H_b_conj = H_LUE_noisy.conj().T
    W_t = H_b_conj @ np.linalg.inv(
        H_LUE_noisy @ H_b_conj + xl.sigma2 * np.eye(xl.K)
    )
    W_t = W_t * np.sqrt(xl.K / np.trace(W_t.conj().T @ W_t).real)

    epsilon_t = 0.5
    for _ in range(14):
        W_next = xl.optimize_beamfocusing_SCA(H_LUE_noisy, H_EUE_noisy, W_t, epsilon_t)
        if W_next is not None:
            W_t = W_next
        epsilon_t = xl.optimize_power_allocation_GSS(H_LUE_noisy, H_EUE_noisy, W_t)

    # Convention fix: conjugate W (same as mobility_scenario.update_sca)
    W_aligned = W_t[:, 0:1].conj()
    h_bob_noisy = H_LUE_noisy[0, :].reshape(-1, 1)
    V = build_V(h_bob_noisy, config.Nt)
    return W_aligned, float(epsilon_t), V


def update_sac_noisy(sac_ctrl, config, bob_loc, nmse):
    """SAC's W is geometry-based (immune to CSI noise). Only V is affected."""
    W = build_W_from_focal(config, sac_ctrl.ideal_r, sac_ctrl.ideal_theta)
    phi = sac_ctrl.ideal_psf
    h_bob = get_channel(config, bob_loc)
    h_hat = add_channel_noise(h_bob, nmse)
    V = build_V(h_hat, config.Nt)
    return W, phi, V


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def random_bob(config):
    bx = (np.random.rand() - 0.5) * 2 * config.max_x
    bz = 20 + np.random.rand() * (config.max_z - 20)
    return np.array([bx, 0.0, bz])


def eve_at_distance(bob_loc, dist):
    angle = np.random.uniform(0, 2 * np.pi)
    ex = bob_loc[0] + dist * np.cos(angle)
    ez = bob_loc[2] + dist * np.sin(angle)
    return np.array([ex, 0.0, ez])


# --------------------------------------------------------------------------- #
# Evaluate one Eve distance — returns {method: [sr_per_snr_point]}
# --------------------------------------------------------------------------- #
def evaluate_eve_dist(config, model, eve_dist, all_snrs):
    """Run the full pilot-SNR sweep for a single Eve distance."""
    mean_sr = {m: [] for m in METHODS}

    print(f"\n  Eve = {eve_dist} m")
    for snr_db in all_snrs:
        nmse = pilot_snr_to_nmse(snr_db)
        snr_label = f"{snr_db}" if np.isfinite(snr_db) else "inf"
        trial_srs = {m: [] for m in METHODS}

        for trial in range(N_TRIALS):
            np.random.seed(1000 + trial)
            bob_loc = random_bob(config)
            eve_loc = eve_at_distance(bob_loc, eve_dist)

            # --- SAC: warm up beam on Bob (true obs), then extract controls ---
            sac_ctrl = SACController(model, config)
            sac_ctrl.reset_to(bob_loc)
            for _ in range(SAC_WARMUP):
                sac_ctrl.step(bob_loc, eve_loc)
            W_sac, phi_sac, V_sac = update_sac_noisy(sac_ctrl, config, bob_loc, nmse)
            trial_srs['SAC'].append(
                instantaneous_sr(config, bob_loc, eve_loc, W_sac, phi_sac, V_sac)
            )

            # --- MRT ---
            W_mrt, phi_mrt, V_mrt = update_mrt_noisy(config, bob_loc, nmse)
            trial_srs['MRT'].append(
                instantaneous_sr(config, bob_loc, eve_loc, W_mrt, phi_mrt, V_mrt)
            )

            # --- FPA ---
            for lbl, phi_c in FPA_PHIS.items():
                W_f, phi_f, V_f = update_fpa_noisy(config, bob_loc, phi_c, nmse)
                trial_srs[lbl].append(
                    instantaneous_sr(config, bob_loc, eve_loc, W_f, phi_f, V_f)
                )

            # --- SCA+GSS ---
            W_sca, phi_sca, V_sca = update_sca_noisy(config, bob_loc, eve_loc, nmse)
            trial_srs['SCA+GSS'].append(
                instantaneous_sr(config, bob_loc, eve_loc, W_sca, phi_sca, V_sca)
            )

        for m in METHODS:
            mean_sr[m].append(float(np.mean(trial_srs[m])))

        print(
            f"    SNR={snr_label:>3s} dB | "
            + " | ".join(f"{m}={mean_sr[m][-1]:.2f}" for m in METHODS)
        )

    return mean_sr


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    config = Config(yaml_path='config.yaml')
    config.P_total_watts = 10 ** ((POWER_DBM - 30) / 10)

    print(f"\nLoading SAC model: {DEFAULT_MODEL}")
    env = XLMIMOEnv(config=config)
    model = SAC.load(DEFAULT_MODEL, env=env)

    all_snrs = PILOT_SNRS_DB + [np.inf]
    x_labels = [str(s) for s in PILOT_SNRS_DB] + [r'$\infty$']
    x_pos    = np.arange(len(all_snrs))

    print(f"\nSweeping pilot SNR with P={POWER_DBM} dBm, N_TRIALS={N_TRIALS}")

    # Collect results per Eve distance
    all_results = {}
    for eve_dist in EVE_DISTS:
        all_results[eve_dist] = evaluate_eve_dist(config, model, eve_dist, all_snrs)

    # ----------------------------------------------------------------------- #
    # Plot: one panel per Eve distance
    # ----------------------------------------------------------------------- #
    n_panels = len(EVE_DISTS)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    for idx, eve_dist in enumerate(EVE_DISTS):
        ax = axes[0, idx]
        mean_sr = all_results[eve_dist]
        for m in METHODS:
            ax.plot(x_pos, mean_sr[m], label=m, **STYLES[m])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Pilot SNR (dB)")
        if idx == 0:
            ax.set_ylabel("Mean Secrecy Rate (bits/s/Hz)")
        ax.set_title(f"Eve = {eve_dist} m")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"CSI Robustness: {config.Nt} ant, {config.fc/1e9:.1f} GHz, P={POWER_DBM} dBm",
        fontsize=12, y=0.96,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir = f'csi_plots/{config.Nt}'
    os.makedirs(out_dir, exist_ok=True)
    png_path = f'{out_dir}/csi_robustness_{config.Nt}_{config.fc}_{FILE_NAME}.png'
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved to {png_path}")

    # ----------------------------------------------------------------------- #
    # CSV dump
    # ----------------------------------------------------------------------- #
    csv_path = f'{out_dir}/csi_robustness_{config.Nt}_{config.fc}_{FILE_NAME}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eve_dist_m', 'pilot_snr_db', 'nmse', 'method', 'mean_sr'])
        for eve_dist in EVE_DISTS:
            mean_sr = all_results[eve_dist]
            for i, snr_db in enumerate(all_snrs):
                nmse = pilot_snr_to_nmse(snr_db)
                for m in METHODS:
                    writer.writerow([
                        eve_dist,
                        snr_db if np.isfinite(snr_db) else 'inf',
                        round(nmse, 6), m, round(mean_sr[m][i], 6)
                    ])
    print(f"Raw data saved to {csv_path}")

    plt.show()


if __name__ == '__main__':
    main()
