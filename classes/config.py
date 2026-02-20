import os
from typing import Any, cast

import numpy as np
import yaml
from scipy.constants import speed_of_light, Boltzmann

from functions.codebook import codebook


class Config:
    """Stores all configuration parameters as attributes."""

    def __init__(self, yaml_path=None):
        """Loads configuration from a YAML file and computes derived parameters.

        Args:
            yaml_path: Path to the YAML config file. Defaults to ../config.yaml
                       relative to this file.
        """
        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

        with open(yaml_path, 'r') as f:
            raw = cast(dict[str, Any], yaml.safe_load(f))

        # -----------------------------------------------------------------
        # 1. GENERAL CONFIG (from YAML + physical constants)
        # -----------------------------------------------------------------
        self.c = speed_of_light
        self.fc = raw['fc']
        self.lam = self.c / self.fc                    # Wavelength (m)
        self.k = 2 * np.pi / self.lam                  # Wavenumber (m^-1)
        self.Nt = raw['Nt']
        self.max_x = raw['max_x']
        self.max_z = raw['max_z']
        self.resolution = raw['resolution']
        self.P_total_watts = raw['P_total_watts']
        self.show_plot_every_nth_steps = raw['show_plot_every_nth_steps']
        self.max_episodes = raw['max_episodes']
        self.max_steps_per_episode = raw['max_steps_per_episode']


        # -----------------------------------------------------------------
        # 2. NOISE CONFIG
        # -----------------------------------------------------------------
        self.k_B = Boltzmann
        self.BW = raw['BW']
        self.NF_dB = raw['NF_dB']
        self.T_temp = raw['T_temp']
        self.noise_power_watts = (self.k_B * self.T_temp * self.BW
                                  * 10 ** (self.NF_dB / 10))

        # -----------------------------------------------------------------
        # 3. CODEBOOK CONFIG
        # -----------------------------------------------------------------
        self.psf_N = raw['psf_N']
        self.psf_codebook = np.linspace(0, 1, self.psf_N + 1)

        self.size_cb = raw['size_cb']
        Nt = self.Nt
        d = 0.5 * self.lam
        x_positions = (np.arange(Nt) - (Nt - 1) / 2) * d
        y_positions = np.zeros(Nt)
        z_positions = np.zeros(Nt)
        self.pos = np.vstack([x_positions, y_positions, z_positions])  # (3, Nt)

        w_beam_codebook, grid_coords = codebook(
            self.Nt, self.pos, self.k,
            self.size_cb, self.max_x, self.max_z
        )
        self.w_beam_codebook = w_beam_codebook
        self.beam_focal_locs = grid_coords
