"""
Plot Secrecy Rate vs Transmit Power from CSV
=============================================
Reads the saved evaluation_power_data.csv file and reconstructs
the evaluation plot, renaming the 'Convex' baseline to 'SCA & GSS'.

Run from the project root:
    python plot_csv_data.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_from_csv(csv_path='evaluation_power_data.csv'):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'. Make sure you've run the evaluation script first.")
        return

    # 1. Load the data
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    powers_dbm = df['Transmit_Power_dBm']

    # 2. Setup the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Same styles as the original script
    styles = {
        'SAC':                       dict(color='tab:blue',   lw=2.5, ls='-',  marker='o', ms=5, zorder=5),
        'MRT':                       dict(color='tab:red',    lw=1.8, ls='--', marker='s', ms=4),
        'Convex':                    dict(color='tab:cyan',   lw=1.8, ls='-.', marker='X', ms=4),
        r'FPA ($\phi$=0.9)':         dict(color='tab:green',  lw=1.5, ls=':',  marker='^', ms=4),
        r'FPA ($\phi$=0.8)':         dict(color='tab:orange', lw=1.5, ls=':',  marker='v', ms=4),
        r'FPA ($\phi$=0.7)':         dict(color='tab:purple', lw=1.5, ls=':',  marker='D', ms=4),
    }

    # 3. Loop through columns and plot
    for col in df.columns:
        if col == 'Transmit_Power_dBm':
            continue  # Skip the x-axis column

        # Rename 'Convex' to 'SCA & GSS' for the legend
        plot_label = 'SCA & GSS' if col == 'Convex' else col

        # Apply styles if they exist, otherwise just plot default
        if col in styles:
            ax.plot(powers_dbm, df[col], label=plot_label, **styles[col])
        else:
            ax.plot(powers_dbm, df[col], label=plot_label)

    # 4. Format the graph
    ax.set_xlabel("Total Transmit Power (dBm)", fontsize=13)
    ax.set_ylabel("Mean Secrecy Rate (bits/s/Hz)", fontsize=13)
    ax.set_title("Secrecy Rate vs Transmit Power (Eve distance = 5.0m)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(powers_dbm.min(), powers_dbm.max())
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    # 5. Save and show
    out_path = 'reconstructed_secrecy_vs_power.png'
    fig.savefig(out_path, dpi=150)
    print(f"Plot successfully saved to '{out_path}'")

    plt.show()

if __name__ == '__main__':
    plot_from_csv()
