"""
Mobility Scenario Evaluation
============================
Compares SAC, SCA+GSS, MRT, and FPA under a moving Bob. Each method recomputes
its beamformer (W) and power-split (phi) at its natural control-loop rate; between
updates, the held W/phi drift out of alignment as Bob moves. This highlights SAC's
combination of (a) adaptive phi and (b) low per-decision latency.

Two panels:
    A. Time series at a single velocity (shows SCA+GSS stair-step vs SAC smooth)
    B. Mean SR vs Bob velocity (shows crossover)

Run from the project root:
    python mobility_scenario.py
"""
import os
import sys
import time
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

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
FILE_NAME     = "checkpoint_120219"
DEFAULT_MODEL = 'training_results/SAC_41_20260416_220817/checkpoints/checkpoint_120219.zip'
POWER_DBM     = 30.0
EVE_DIST      = 1.0
VELOCITIES    = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
N_TRIALS      = 5
SIM_DT        = 5e-3          # 5 ms simulation grid
T_TOTAL       = 4.5           # seconds per trajectory
V_TIMESERIES  = 5.0           # velocity used for Panel A
SAC_WARMUP    = 50            # SAC steps with Bob static before mobility begins

# FPA variants shown in the plot (bracketing high/low phi)
FPA_PHIS = {
    r'FPA ($\phi$=0.9)': 0.9,
    r'FPA ($\phi$=0.5)': 0.5,
}

METHODS = ['SAC', 'SCA+GSS', 'MRT'] + list(FPA_PHIS.keys())

STYLES = {
    'SAC':                dict(color='tab:blue',   lw=2.5, ls='-',   marker='o', ms=4, zorder=5),
    'SCA+GSS':            dict(color='tab:cyan',   lw=1.8, ls='-',   marker='X', ms=4),
    'MRT':                dict(color='tab:red',    lw=1.6, ls='--',  marker='s', ms=3),
    r'FPA ($\phi$=0.9)':  dict(color='tab:green',  lw=1.4, ls='-.',  marker='^', ms=3, alpha=0.7),
    r'FPA ($\phi$=0.5)':  dict(color='tab:brown',  lw=1.4, ls='-.',  marker='P', ms=3, alpha=0.7),
}


# --------------------------------------------------------------------------- #
# Low-level helpers (W, V, SR)
# --------------------------------------------------------------------------- #
def build_W_from_focal(config, ideal_r, ideal_theta):
    """Build the USW-model beamformer W for a focal point (r, theta)."""
    ideal_x = ideal_r * np.sin(ideal_theta)
    ideal_z = ideal_r * np.cos(ideal_theta)
    current_focus_point = np.array([ideal_x, 0, ideal_z])
    dist_from_target_to_antennas = np.sqrt(
        np.sum((config.pos - current_focus_point[:, np.newaxis]) ** 2, axis=0)
    )
    response_vector = np.exp(-1j * config.k * dist_from_target_to_antennas)
    W = response_vector / np.linalg.norm(response_vector)
    return W.reshape(-1, 1)


def build_V(h_bob, Nt):
    """AN nullspace projector for Bob: V = I - h h^H / ||h||^2."""
    return np.eye(Nt) - (h_bob @ h_bob.conj().T) / (np.linalg.norm(h_bob) ** 2)


def instantaneous_sr(config, bob_loc, eve_loc, W, phi, V):
    """Secrecy rate given the held (W, phi, V) and current Bob/Eve locations."""
    h_bob = get_channel(config, bob_loc)
    h_eve = get_channel(config, eve_loc)

    P_s  = config.P_total_watts * phi
    P_an = config.P_total_watts * (1.0 - phi)

    W_col = W.reshape(-1, 1) if W.ndim == 1 else W

    sig_pwr_bob = P_s * abs((h_bob.conj().T @ W_col).item()) ** 2
    sig_pwr_eve = P_s * abs((h_eve.conj().T @ W_col).item()) ** 2

    an_leak_bob = P_an * float(np.linalg.norm(h_bob.conj().T @ V) ** 2)
    an_leak_eve = P_an * float(np.linalg.norm(h_eve.conj().T @ V) ** 2)

    SINR_bob = sig_pwr_bob / (config.noise_power_watts + an_leak_bob)
    SINR_eve = sig_pwr_eve / (config.noise_power_watts + an_leak_eve)

    rate_bob = np.log2(1 + SINR_bob)
    rate_eve = np.log2(1 + SINR_eve)
    return max(0.0, float(rate_bob - rate_eve))


# --------------------------------------------------------------------------- #
# Per-method update functions: (bob_loc, eve_loc) -> (W, phi, V)
# --------------------------------------------------------------------------- #
def update_mrt(config, bob_loc, eve_loc):
    h_bob = get_channel(config, bob_loc)
    W = h_bob / np.linalg.norm(h_bob)
    # phi=1 means no AN; V never used, but provide a valid matrix
    V = np.zeros((config.Nt, config.Nt), dtype=complex)
    return W, 1.0, V


def update_fpa(config, bob_loc, eve_loc, phi_c):
    h_bob = get_channel(config, bob_loc)
    W = h_bob / np.linalg.norm(h_bob)
    V = build_V(h_bob, config.Nt)
    return W, phi_c, V


def update_sca(config, bob_loc, eve_loc):
    xl = XLMIMO_System(K=1, E=1, bob_loc=bob_loc, eve_loc=eve_loc, config=config)
    W_t, phi_t, H_LUE, H_EUE, _, _ = xl.run_alternating_optimization()
    # CONVENTION FIX: SCA+GSS's cvxpy formulation scores the signal term as
    # |H @ W|^2 (SCA_GCC_convex.py:64). Everywhere else in this repo — step_function,
    # MRT, FPA, and instantaneous_sr below — it is |h.conj().T @ W|^2. The optimal
    # W under one convention is the complex conjugate of the optimal W under the
    # other (e.g. MRT uses W = h/||h|| while SCA returns W ~ conj(h)/||h||). So we
    # conjugate here to bring SCA's output into the repo's convention. V is safe
    # without conjugation because |h.conj().T @ V|^2 = |h.T @ V_SCA|^2.
    W_aligned = W_t[:, 0:1].conj()
    h_bob_solve = H_LUE[0, :].reshape(-1, 1)
    V = build_V(h_bob_solve, config.Nt)
    return W_aligned, float(phi_t), V


# --------------------------------------------------------------------------- #
# SAC handling (maintains its own state between updates)
# --------------------------------------------------------------------------- #
def build_sac_obs(config, ideal_r, ideal_theta, ideal_psf, bob_loc, eve_loc):
    """Same obs layout used everywhere else (reset_function / step_function)."""
    bx, bz = bob_loc[0], bob_loc[2]
    ex, ez = eve_loc[0], eve_loc[2]

    max_r = np.sqrt(config.max_x ** 2 + config.max_z ** 2)
    max_theta_sweep = np.pi / 2

    r_bob = np.sqrt(bx ** 2 + bz ** 2)
    theta_bob = np.arctan2(bx, bz)
    r_eve = np.sqrt(ex ** 2 + ez ** 2)
    theta_eve = np.arctan2(ex, ez)

    bob_eve_diff = np.linalg.norm(bob_loc - eve_loc)

    # NOTE: this script uses np.exp(-bob_eve_diff) to match the currently loaded
    # checkpoint's training distribution. If the agent is retrained with the
    # linear (bob_eve_diff / max_r) encoding, switch this line accordingly.
    return np.array([
        ideal_r / max_r,
        ideal_theta / max_theta_sweep,
        ideal_psf,
        (r_bob   - ideal_r)     / max_r,
        (theta_bob - ideal_theta) / max_theta_sweep,
        (r_eve   - ideal_r)     / max_r,
        (theta_eve - ideal_theta) / max_theta_sweep,
        np.exp(-bob_eve_diff),
    ], dtype=np.float32)


class SACController:
    """Wraps the SAC policy with its held beam state (ideal_r, ideal_theta, ideal_psf)."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.ideal_r = 40.0
        self.ideal_theta = 0.0
        self.ideal_psf = 0.5
        self.max_r = np.sqrt(config.max_x ** 2 + config.max_z ** 2)

    def reset_to(self, bob_loc):
        """Initialize beam pointing at Bob, phi at 0.5."""
        bx, bz = bob_loc[0], bob_loc[2]
        self.ideal_r = float(np.sqrt(bx ** 2 + bz ** 2))
        self.ideal_theta = float(np.arctan2(bx, bz))
        self.ideal_psf = 0.5

    def step(self, bob_loc, eve_loc):
        """One SAC decision: predict action and apply to internal state."""
        obs = build_sac_obs(
            self.config, self.ideal_r, self.ideal_theta, self.ideal_psf,
            bob_loc, eve_loc,
        )
        action, _ = self.model.predict(obs, deterministic=True)

        delta_r     = float(action[0]) * self.config.max_r_step
        delta_theta = float(action[1]) * self.config.max_theta_step

        self.ideal_r     = float(np.clip(self.ideal_r + delta_r, 0.1, self.max_r))
        self.ideal_theta = float(np.clip(self.ideal_theta + delta_theta, -np.pi / 2, np.pi / 2))
        self.ideal_psf   = float(np.clip((float(action[2]) + 1.0) / 2.0, 0.0, 1.0))

        V = build_V(get_channel(self.config, bob_loc), self.config.Nt)
        W = build_W_from_focal(self.config, self.ideal_r, self.ideal_theta)

    def get_control(self, bob_loc):
        """Return (W, phi, V) given current held focal point and current Bob loc."""
        W = build_W_from_focal(self.config, self.ideal_r, self.ideal_theta)
        h_bob = get_channel(self.config, bob_loc)
        V = build_V(h_bob, self.config.Nt)
        return W, self.ideal_psf, V


# --------------------------------------------------------------------------- #
# Latency measurement
# --------------------------------------------------------------------------- #
def measure_latencies(config, model, bob_loc, eve_loc, n_warmup=20):
    """Measure per-decision latencies for SAC and SCA+GSS."""
    # SAC
    sac_ctrl = SACController(model, config)
    sac_ctrl.reset_to(bob_loc)
    sac_times = []
    for _ in range(n_warmup):
        t0 = time.perf_counter()
        sac_ctrl.step(bob_loc, eve_loc)
        sac_times.append(time.perf_counter() - t0)
    tau_sac = float(np.mean(sac_times[5:]))  # drop first 5 (warmup noise)

    # SCA+GSS
    sca_times = []
    for _ in range(n_warmup):
        t0 = time.perf_counter()
        update_sca(config, bob_loc, eve_loc)
        sca_times.append(time.perf_counter() - t0)
    tau_sca = float(np.mean(sca_times[1:]))

    # Measure MRT and FPA for reference (should be very fast, essentially just channel estimation)
    mrt_times = []
    for _ in range(n_warmup):
        t0 = time.perf_counter()
        update_mrt(config, bob_loc, eve_loc)
        mrt_times.append(time.perf_counter() - t0)
    tau_mrt = float(np.mean(mrt_times[1:]))

    fpa_times = []
    for _ in range(n_warmup):
        t0 = time.perf_counter()
        update_fpa(config, bob_loc, eve_loc, phi_c=0.5)
        fpa_times.append(time.perf_counter() - t0)
    tau_fpa = float(np.mean(fpa_times[1:]))

    return tau_sac, tau_sca, tau_mrt, tau_fpa


# --------------------------------------------------------------------------- #
# Main simulation: move Bob along a linear trajectory
# --------------------------------------------------------------------------- #
def simulate_trajectory(config, model, bob_start, eve_loc, velocity,
                        tau_sac, tau_sca, dt=SIM_DT, T=T_TOTAL):
    """
    Simulate one mobility trajectory. Returns dict {method: sr_per_step (np.ndarray)}.
    """
    n_steps = int(np.ceil(T / dt))
    times = np.arange(n_steps) * dt

    # --- SAC warmup so beam is already tracking Bob at t=0 ---
    sac_ctrl = SACController(model, config)
    sac_ctrl.reset_to(bob_start)
    for _ in range(SAC_WARMUP):
        sac_ctrl.step(bob_start, eve_loc)

    # --- Initial solve for each method at t=0 ---
    W_mrt, phi_mrt, V_mrt = update_mrt(config, bob_start, eve_loc)
    fpa_state = {lbl: update_fpa(config, bob_start, eve_loc, phi) for lbl, phi in FPA_PHIS.items()}
    W_sca, phi_sca, V_sca = update_sca(config, bob_start, eve_loc)
    W_sac, phi_sac, V_sac = sac_ctrl.get_control(bob_start)

    # --- Staleness bookkeeping (in seconds) ---
    t_last_sca = 0.0
    t_last_sac = 0.0
    # MRT and FPA treated as "always fresh" — updated every sim step.

    sr_traces = {m: np.zeros(n_steps) for m in METHODS}

    for i, t in enumerate(times):
        bob_loc = bob_start + np.array([velocity * t, 0.0, 0.0])

        # --- Refresh non-stale methods every step ---
        W_mrt, phi_mrt, V_mrt = update_mrt(config, bob_loc, eve_loc)
        for lbl in FPA_PHIS:
            fpa_state[lbl] = update_fpa(config, bob_loc, eve_loc, FPA_PHIS[lbl])

        # --- Conditionally update SAC ---
        if t - t_last_sac >= tau_sac:
            sac_ctrl.step(bob_loc, eve_loc)
            W_sac, phi_sac, V_sac = sac_ctrl.get_control(bob_loc)
            t_last_sac = t

        # --- Conditionally update SCA+GSS ---
        if t - t_last_sca >= tau_sca:
            W_sca, phi_sca, V_sca = update_sca(config, bob_loc, eve_loc)
            t_last_sca = t

        # --- Instantaneous SR per method using held controls ---
        sr_traces['SAC'][i]     = instantaneous_sr(config, bob_loc, eve_loc, W_sac, phi_sac, V_sac)
        sr_traces['SCA+GSS'][i] = instantaneous_sr(config, bob_loc, eve_loc, W_sca, phi_sca, V_sca)
        sr_traces['MRT'][i]     = instantaneous_sr(config, bob_loc, eve_loc, W_mrt, phi_mrt, V_mrt)
        for lbl in FPA_PHIS:
            W_f, phi_f, V_f = fpa_state[lbl]
            sr_traces[lbl][i] = instantaneous_sr(config, bob_loc, eve_loc, W_f, phi_f, V_f)

    return times, sr_traces


def random_start(config, r0=40.0):
    """Bob starts on the left half (negative x) so +x motion has room inside
    the array's visible region even at 20 m/s over T=4.5 s (~90 m travel).
    """
    # Negative starting angle keeps bx negative; motion in +x moves him into view
    angle = np.random.uniform(-np.pi / 3, -np.pi / 18)
    bx = r0 * np.sin(angle)
    bz = r0 * np.cos(angle)
    return np.array([bx, 0.0, bz])


def random_eve_offset(bob_loc, dist):
    """Eve placed at fixed `dist` metres from Bob's starting location."""
    ang = np.random.uniform(0, 2 * np.pi)
    ex = bob_loc[0] + dist * np.cos(ang)
    ez = bob_loc[2] + dist * np.sin(ang)
    return np.array([ex, 0.0, ez])


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def main():
    config = Config(yaml_path='config.yaml')
    config.P_total_watts = 10 ** ((POWER_DBM - 30) / 10)   # set requested power

    print(f"\nLoading SAC model: {DEFAULT_MODEL}")
    env = XLMIMOEnv(config=config)
    model = SAC.load(DEFAULT_MODEL, env=env)

    # -------- Measure latencies (once) --------
    np.random.seed(0)
    bench_bob = random_start(config)
    bench_eve = random_eve_offset(bench_bob, EVE_DIST)
    print("\nMeasuring control-loop latencies ...")
    tau_sac, tau_sca, tau_mrt, tau_fpa = measure_latencies(config, model, bench_bob, bench_eve)

    # -------- Sanity check: SCA's internal SR vs our instantaneous_sr on the
    # same fresh solve. These used to disagree by >10 bits due to the H@W vs
    # h.conj().T@W convention mismatch; with update_sca conjugating W they
    # should now agree to within ~0.5 bits/s/Hz.
    _W_mob, _phi_mob, _V_mob = update_sca(config, bench_bob, bench_eve)
    _sr_mob = instantaneous_sr(config, bench_bob, bench_eve, _W_mob, _phi_mob, _V_mob)
    _xl = XLMIMO_System(K=1, E=1, bob_loc=bench_bob, eve_loc=bench_eve, config=config)
    _W_int, _phi_int, _H_LUE, _H_EUE, _, _ = _xl.run_alternating_optimization()
    _sr_int = _xl.calculate_secrecy_rate(_H_LUE, _H_EUE, _W_int, _phi_int)
    print(f"\nSCA+GSS SR sanity check (fresh solve): internal={_sr_int:.3f}  mobility={_sr_mob:.3f} bits/s/Hz")
    if abs(_sr_int - _sr_mob) > 1.0:
        print("  WARNING: convention mismatch between SCA's internal SR and instantaneous_sr.")

    print(f"  tau_SAC    = {tau_sac*1000:7.2f} ms  ({1.0/tau_sac:7.1f} Hz)")
    print(f"  tau_SCA+GSS= {tau_sca*1000:7.2f} ms  ({1.0/tau_sca:7.1f} Hz)")
    print(f"  tau_MRT    = {tau_mrt*1000:7.2f} ms  ( {1.0/tau_mrt:7.1f} Hz)")
    print(f"  tau_FPA    = {tau_fpa*1000:7.2f} ms  ( {1.0/tau_fpa:7.1f} Hz)")
    tau_mrt = SIM_DT
    tau_fpa = SIM_DT

    # ----------------------------------------------------------------------- #
    # Panel A: single-trial time series at V_TIMESERIES
    # ----------------------------------------------------------------------- #
    print(f"\n--- Panel A: time series at v = {V_TIMESERIES} m/s ---")
    np.random.seed(1)
    ts_bob = random_start(config)
    ts_eve = random_eve_offset(ts_bob, EVE_DIST)
    times_A, sr_A = simulate_trajectory(
        config, model, ts_bob, ts_eve, V_TIMESERIES, tau_sac, tau_sca
    )
    for m in METHODS:
        print(f"  {m:<22s} mean SR = {float(np.mean(sr_A[m])):.3f} bits/s/Hz")

    # ----------------------------------------------------------------------- #
    # Panel B: velocity sweep, N_TRIALS per velocity
    # ----------------------------------------------------------------------- #
    print(f"\n--- Panel B: mean SR vs velocity (N_TRIALS={N_TRIALS}) ---")
    mean_sr = {m: [] for m in METHODS}

    for v in VELOCITIES:
        trial_means = {m: [] for m in METHODS}
        for trial in range(N_TRIALS):
            np.random.seed(100 * int(v * 10) + trial)
            bob0 = random_start(config)
            eve = random_eve_offset(bob0, EVE_DIST)
            _, sr_t = simulate_trajectory(
                config, model, bob0, eve, v, tau_sac, tau_sca
            )
            for m in METHODS:
                trial_means[m].append(float(np.mean(sr_t[m])))
        for m in METHODS:
            mean_sr[m].append(float(np.mean(trial_means[m])))
        print(
            f"  v={v:5.1f} m/s | "
            + " | ".join(f"{m}={mean_sr[m][-1]:.2f}" for m in METHODS)
        )

    # ----------------------------------------------------------------------- #
    # Plot
    # ----------------------------------------------------------------------- #
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A
    for m in METHODS:
        axA.plot(times_A, sr_A[m], label=m, **STYLES[m])
    axA.set_xlabel("Time (s)")
    axA.set_ylabel("Instantaneous Secrecy Rate (bits/s/Hz)")
    axA.set_title(f"Time series at v = {V_TIMESERIES} m/s, Eve = {EVE_DIST} m")
    axA.grid(True, alpha=0.3)

    # Panel B
    for m in METHODS:
        axB.plot(VELOCITIES, mean_sr[m], label=m, **STYLES[m])
    axB.set_xscale('log')
    axB.set_xlabel("Bob velocity (m/s)")
    axB.set_ylabel("Mean Secrecy Rate (bits/s/Hz)")
    axB.set_title(f"Mean SR vs velocity, Eve = {EVE_DIST} m, P = {POWER_DBM} dBm")
    axB.grid(True, alpha=0.3, which='both')

    handles, labels = axB.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=9)
    # fig.suptitle(
    #     f"Mobility Scenario: {config.Nt} ant, {config.fc/1e9:.1f} GHz   "
    #     f"(tau_SAC={tau_sac*1000:.1f} ms, tau_SCA={tau_sca*1000:.1f} ms)",
    #     fontsize=11, y=0.97,
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = f'mobility_plots/{config.Nt}'
    os.makedirs(out_dir, exist_ok=True)
    png_path = f'{out_dir}/mobility_{config.Nt}_{config.fc}_{FILE_NAME}.png'
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved to {png_path}")

    # ----------------------------------------------------------------------- #
    # CSV dump
    # ----------------------------------------------------------------------- #
    csv_path = f'{out_dir}/mobility_{config.Nt}_{config.fc}_{FILE_NAME}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Time-series block
        writer.writerow(['panel', 'method', 'x', 'value'])
        for m in METHODS:
            for t, sr in zip(times_A, sr_A[m]):
                writer.writerow(['A_timeseries', m, round(float(t), 4), round(float(sr), 6)])
        # Velocity sweep block
        for m in METHODS:
            for v, sr in zip(VELOCITIES, mean_sr[m]):
                writer.writerow(['B_velocity', m, v, round(sr, 6)])
        # Latencies row
        writer.writerow(['latency_ms', 'SAC', 0, round(tau_sac * 1000, 4)])
        writer.writerow(['latency_ms', 'SCA+GSS', 0, round(tau_sca * 1000, 4)])
    print(f"Raw data saved to {csv_path}")

    plt.show()


if __name__ == '__main__':
    main()
