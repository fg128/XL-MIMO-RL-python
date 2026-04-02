import time

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


if __name__ == "__main__":
    xl_mimo = XLMIMO_System(N_bx=256, N_bz=1, K=1, E=1, bob_loc=np.array([-20, 0, 30]), eve_loc=np.array([20, 0, 10]))
    W_t, epsilon_t, secrecy_rate = xl_mimo.run_alternating_optimization()
    xl_mimo.visualize_beamfocusing(W_t, epsilon_t, lue_coords=np.array([(xl_mimo.bob_theta, xl_mimo.bob_r)]), eue_coords=np.array([(xl_mimo.eve_theta, xl_mimo.eve_r)]))
    # print(f"Final Achieved Secrecy Rate: {secrecy_rate:.4f} bps/Hz")
