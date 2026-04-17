"""
Remake mobility plots from saved CSV data without re-running the simulation.

Usage:
    python mobility_plots/256/make_mobility_plot.py
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CSV_PATH = os.path.join(os.path.dirname(__file__),
                        'mobility_256_3500000000.0_checkpoint_120219.csv')

METHODS = ['SAC', 'SCA+GSS', 'MRT',
           r'FPA ($\phi$=0.9)', r'FPA ($\phi$=0.5)']

STYLES = {
    'SAC':                dict(color='tab:blue',   lw=2.5, ls='-',   marker='o', ms=4, zorder=5),
    'SCA+GSS':            dict(color='tab:cyan',   lw=1.8, ls='-',   marker='X', ms=4),
    'MRT':                dict(color='tab:red',    lw=1.6, ls='--',  marker='s', ms=3),
    r'FPA ($\phi$=0.9)':  dict(color='tab:green',  lw=1.4, ls='-.',  marker='^', ms=3, alpha=0.7),
    r'FPA ($\phi$=0.5)':  dict(color='tab:brown',  lw=1.4, ls='-.',  marker='P', ms=3, alpha=0.7),
}

EVE_DIST = 1.0
POWER_DBM = 30.0
V_TIMESERIES = 5.0

# --------------------------------------------------------------------------- #
# Load CSV
# --------------------------------------------------------------------------- #
timeseries = {m: {'t': [], 'sr': []} for m in METHODS}
velocity   = {m: {'v': [], 'sr': []} for m in METHODS}
latencies  = {}

with open(CSV_PATH, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        panel, method, x, value = row[0], row[1], row[2], row[3]
        if panel == 'A_timeseries' and method in timeseries:
            timeseries[method]['t'].append(float(x))
            timeseries[method]['sr'].append(float(value))
        elif panel == 'B_velocity' and method in velocity:
            velocity[method]['v'].append(float(x))
            velocity[method]['sr'].append(float(value))
        elif panel == 'latency_ms':
            latencies[method] = float(x)  # x column holds 0, value holds ms

# Convert to arrays
for m in METHODS:
    timeseries[m]['t']  = np.array(timeseries[m]['t'])
    timeseries[m]['sr'] = np.array(timeseries[m]['sr'])
    velocity[m]['v']    = np.array(velocity[m]['v'])
    velocity[m]['sr']   = np.array(velocity[m]['sr'])

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A — time series
for m in METHODS:
    axA.plot(timeseries[m]['t'], timeseries[m]['sr'], label=m, **STYLES[m])
axA.set_xlabel("Time (s)")
axA.set_ylabel("Instantaneous Secrecy Rate (bits/s/Hz)")
axA.set_title(f"Time series at v = {V_TIMESERIES} m/s, Eve = {EVE_DIST} m")
axA.grid(True, alpha=0.3)

# Panel B — velocity sweep
for m in METHODS:
    axB.plot(velocity[m]['v'], velocity[m]['sr'], label=m, **STYLES[m])
axB.set_xscale('log')
axB.set_xlabel("Bob velocity (m/s)")
axB.set_ylabel("Mean Secrecy Rate (bits/s/Hz)")
axB.set_title(f"Mean SR vs velocity, Eve = {EVE_DIST} m, P = {POWER_DBM} dBm")
axB.grid(True, alpha=0.3, which='both')

handles, labels = axB.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(os.path.dirname(CSV_PATH),
                        'mobility_plot.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Plot saved to {out_path}")

plt.show()
