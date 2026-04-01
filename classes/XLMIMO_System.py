import time
import os
# Set path to root of the project for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from classes.config import Config
from functions.get_channel import get_channel
from functions.visualise import visualise

class XLMIMO_System:
    def __init__(self, bob_loc, eve_loc, config: Config):
        """
        Initialize the XL-MIMO System Model.
        Note: config.Ntz is set to 1 (Uniform Linear Array) to easily visualize the 2D plane.
        """
        self.config : Config= config
        self.bob_loc = bob_loc
        self.bx, self.bz = bob_loc[0], bob_loc[2]
        self.bob_r = np.sqrt(self.bx**2 + self.bz**2)
        self.bob_theta = np.arctan2(self.bz, self.bx)

        self.eve_loc = eve_loc
        self.ex, self.ez = eve_loc[0], eve_loc[2]
        self.eve_r = np.sqrt(self.ex**2 + self.ez**2)
        self.eve_theta = np.arctan2(self.ez, self.ex)

        self.K = 1
        self.E = 1

    def generate_channels(self):
        # 4. Get channels for Bob and Eve (use per-episode NLOS vectors for block fading)
        h_bob = get_channel(self.config, self.bob_loc)
        h_eve = get_channel(self.config, self.eve_loc)
        return h_bob, h_eve

    def calculate_secrecy_rate(self, h_bob, h_eve, W, psf):

        P_s = self.config.P_total_watts * psf
        P_an = self.config.P_total_watts * (1 - psf)
        Nt = self.config.Nt

        # 5. Transmitted signal sent
        V = np.eye(Nt) - (h_bob @ h_bob.conj().T) / (np.linalg.norm(h_bob)**2) # Null space of Bob

        # Analytical signal power
        sig_pwr_bob = P_s * np.abs((h_bob.conj().T @ W).item())**2
        sig_pwr_eve = P_s * np.abs((h_eve.conj().T @ W).item())**2

        # Analytical interference power (AN leakage)
        # Bob: h_bob' * V = 0 by construction, so leakage is 0
        an_leakage_bob = P_an * (np.linalg.norm(h_bob.conj().T @ V) ** 2).item()
        an_leakage_eve = P_an * (np.linalg.norm(h_eve.conj().T @ V) ** 2).item()

        # 8. SINR (Signal to Interference plus Noise Ratio)
        SINR_bob = sig_pwr_bob / (self.config.noise_power_watts + an_leakage_bob)
        SINR_eve = sig_pwr_eve / (self.config.noise_power_watts + an_leakage_eve)

        # 9. Secrecy rate
        rate_bob = np.log2(1 + SINR_bob)
        rate_eve = np.log2(1 + SINR_eve)
        secrecy_rate = max(0.0, rate_bob - rate_eve)

        return secrecy_rate
    
    def optimize_power_allocation_GSS(self, h_bob, h_eve, W):
        phi_gss = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi_gss
        a, b = 0.001, 0.999
        tol = 0.05 * (b - a)

        c = a + resphi * (b - a)
        d = b - resphi * (b - a)
        fc = self.calculate_secrecy_rate(h_bob, h_eve, W, c)
        fd = self.calculate_secrecy_rate(h_bob, h_eve, W, d)

        while abs(b - a) > tol:
            if fc > fd:
                b, d, fd = d, c, fc
                c = a + resphi * (b - a)
                fc = self.calculate_secrecy_rate(h_bob, h_eve, W, c)
            else:
                a, c, fc = c, d, fd
                d = b - resphi * (b - a)
                fd = self.calculate_secrecy_rate(h_bob, h_eve, W, d)
        return (b + a) / 2

    def optimize_beamfocusing_SCA(self, h_bob, h_eve, W_prev, epsilon):
        h_bob, h_eve, W_prev = h_bob.reshape(1, -1), h_eve.reshape(1, -1), W_prev.reshape(-1, 1) # Ensure correct shapes for matrix operations
        
        W = cp.Variable((self.config.Nt, self.K), complex=True)
        xi = cp.Variable(nonneg=True)
        constraints = [cp.sum(cp.sum_squares(cp.abs(W))) <= self.K]

        P_W = epsilon * self.config.P_total_watts
        P_AN = (1 - epsilon) * self.config.P_total_watts
        H_b_pinv = np.linalg.pinv(h_bob)
        V_0 = np.eye(self.config.Nt) - H_b_pinv @ h_bob

        for k in range(self.K):
            # ---------------------------------------------------------
            # 1. SETUP LEGITIMATE USER (Bob) CONSTANTS
            # ---------------------------------------------------------
            h_b_k = np.sqrt(P_W) * h_bob[k, :]
            sigma_LUE = self.config.noise_power_watts # Bob is in the null-space, so AN leakage is 0

            # Precompute power components at W_prev (iteration t)
            h_b_w_t_k = h_b_k.conj() @ W_prev[:, k]
            signal_b_t = np.abs(h_b_w_t_k)**2
            interf_b_t = sum(np.abs(h_b_k.conj() @ W_prev[:, j])**2 for j in range(self.K) if j != k)
            total_b_t = signal_b_t + interf_b_t

            # Base rate R_{k,k}(W^{(t)}) using natural log (ln)
            R_kk_t = np.log(1 + signal_b_t / (interf_b_t + sigma_LUE))

            # ---------------------------------------------------------
            # 2. CONCAVE LOWER BOUND (f1) FOR BOB
            # ---------------------------------------------------------
            T1_b = R_kk_t
            T2_b = - (signal_b_t + sigma_LUE) / (interf_b_t + sigma_LUE)
            T3_b = sigma_LUE / (total_b_t + sigma_LUE)

            # Linear (Affine) term
            T4_b = (2 * cp.real(np.conj(h_b_w_t_k) * (h_b_k.conj() @ W[:, k]))) / (interf_b_t + sigma_LUE)

            # Concave term (negative sum of squares)
            current_total_b_W = cp.sum([cp.sum_squares(h_b_k.conj() @ W[:, j]) for j in range(self.K)])
            T5_b = - (signal_b_t * current_total_b_W) / ((total_b_t + sigma_LUE) * (interf_b_t + sigma_LUE))

            f1_LB = T1_b + T2_b + T3_b + T4_b + T5_b

            for e in range(self.E):
                # ---------------------------------------------------------
                # 3. SETUP EAVESDROPPER (Eve) CONSTANTS
                # ---------------------------------------------------------
                h_e_e = np.sqrt(P_W) * h_eve[e, :]

                # Eve suffers from Artificial Noise leakage + background noise
                sigma_EUE = P_AN * np.linalg.norm(h_eve[e, :] @ V_0)**2 + self.config.noise_power_watts

                # Precompute power components at W_prev (iteration t)
                h_e_w_t_k = h_e_e.conj() @ W_prev[:, k]
                signal_e_t = np.abs(h_e_w_t_k)**2
                interf_e_t = sum(np.abs(h_e_e.conj() @ W_prev[:, j])**2 for j in range(self.K) if j != k)
                total_e_t = signal_e_t + interf_e_t

                # Base rate R_{e,k}(W^{(t)}) using natural log (ln)
                R_ek_t = np.log(1 + signal_e_t / (interf_e_t + sigma_EUE))

                # ---------------------------------------------------------
                # 4. CONVEX UPPER BOUND (f2) FOR EVE
                # ---------------------------------------------------------
                T1_e = R_ek_t
                T2_e = interf_e_t / sigma_EUE
                T3_e = - (signal_e_t * sigma_EUE) / ((total_e_t + sigma_EUE) * (interf_e_t + sigma_EUE))

                # Convex term 1 (Total power W)
                current_total_e_W = cp.sum([cp.sum_squares(h_e_e.conj() @ W[:, j]) for j in range(self.K)])
                T4_e = current_total_e_W / (total_e_t + sigma_EUE)

                # Linear (Affine) term (Trace expansion)
                trace_expansion = cp.sum([np.conj(h_e_e.conj() @ W_prev[:, j]) * (h_e_e.conj() @ W[:, j]) for j in range(self.K) if j != k])
                T5_e = - (2 * cp.real(trace_expansion)) / sigma_EUE

                # Convex term 2 (Interference power W_k)
                current_interf_e_W = cp.sum([cp.sum_squares(h_e_e.conj() @ W[:, j]) for j in range(self.K) if j != k])
                T6_e = (interf_e_t * current_interf_e_W) / ((interf_e_t + sigma_EUE) * sigma_EUE)

                f2_UB = T1_e + T2_e + T3_e + T4_e + T5_e + T6_e

                # ---------------------------------------------------------
                # 5. SECRECY RATE CONSTRAINT
                # ---------------------------------------------------------
                # Ensure the difference is divided by ln(2) if the final objective needs to be in bits/s/Hz
                # If the paper strictly leaves it in nats, you can remove the division by np.log(2).
                LB_R_k = f1_LB / np.log(2)
                UB_R_ek = f2_UB / np.log(2)

                constraints.append(LB_R_k - UB_R_ek >= xi)

        prob = cp.Problem(cp.Maximize(xi), constraints)
        try:
            prob.solve(solver=cp.CLARABEL, max_iters=2000)
            return W.value
        except Exception as e:
            # print("SCA Step Failed:", e)
            return W_prev

    def run_alternating_optimization(self):
        h_bob, h_eve = self.generate_channels()

        H_b_conj = h_bob.conj().T
        W_t = H_b_conj @ np.linalg.inv(h_bob @ H_b_conj + self.config.noise_power_watts * np.eye(self.K))
        W_t = W_t * np.sqrt(self.K / np.trace(W_t.conj().T @ W_t).real)

        psf_t = 0.5
        eta_out = 1e-3
        xi_prev = 0

        t0 = time.perf_counter()
        for t in range(1, 15):
            W_next = self.optimize_beamfocusing_SCA(h_bob, h_eve, W_t, psf_t)
            if W_next is not None: W_t = W_next

            psf_t = self.optimize_power_allocation_GSS(h_bob, h_eve, W_t)
            current_rate = self.calculate_secrecy_rate(h_bob, h_eve, W_t, psf_t)

            print(f"Iteration {t:02d} | Epsilon: {psf_t:.3f} | Min Secrecy Rate: {current_rate:.4f} bps/Hz")

            if abs(current_rate - xi_prev) / (xi_prev + 1e-8) < eta_out:
                print("--> Convergence Reached!")
                break
            xi_prev = current_rate
        t1 = time.perf_counter()
        print(f"Optimization completed in {t1 - t0:.2f} seconds.")

        return W_t, psf_t, current_rate
    
    def visualize(self, W_t, psf_t):
       visualise(W_t, psf_t, self.bx, self.bz, self.ex, self.ez, self.config, step=69)


if __name__ == "__main__":
    config = Config()
    xl_mimo = XLMIMO_System(bob_loc=np.array([-20, 0, 30]), eve_loc=np.array([20, 0, 10]), config=config)
    W_t, psf_t, secrecy_rate = xl_mimo.run_alternating_optimization()
    xl_mimo.visualize(W_t, psf_t)
    print(f"Final Achieved Secrecy Rate: {secrecy_rate:.4f} bps/Hz")
