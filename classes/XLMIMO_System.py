import numpy as np
import cvxpy as cp
from stable_baselines3 import SAC

from functions.get_channel import get_channel
from functions.visualise import visualise
from classes.config import Config

class XLMIMO_System:
    def __init__(self, bob_loc, eve_loc,K=1, E=1, config : Config = Config()):
        self.N_bx = config.Nt
        self.N_bz = 1
        self.N_b = self.N_bx * self.N_bz
        self.K = K
        self.E = E
        self.bob_r = np.sqrt(bob_loc[0]**2 + bob_loc[2]**2)
        self.bob_theta = np.arctan2(bob_loc[2], bob_loc[0])
        self.eve_r = np.sqrt(eve_loc[0]**2 + eve_loc[2]**2)
        self.eve_theta = np.arctan2(eve_loc[2], eve_loc[0])

        self.c = config.c 
        self.freq = config.fc
        self.lambda_c = self.c / self.freq
        self.d_ant = self.lambda_c / 2

        self.P_b = config.P_total_watts
        self.sigma2 = config.noise_power_watts

        self.config = config
        self.bx = bob_loc[0]
        self.bz = bob_loc[2]
        self.ex = eve_loc[0]
        self.ez = eve_loc[2]

    def generate_channels(self):
        H_LUE = np.zeros((self.K, self.N_b), dtype=complex)
        H_EUE = np.zeros((self.E, self.N_b), dtype=complex)
        lue_coords, eue_coords = [], []

        # 3. Generate channels for Bob
        for k in range(self.K):
            # Formulate Bob's 3D coordinates based on initialization
            bob_loc = [self.bx, 0, self.bz] 
            H_LUE[k, :] = get_channel(self.config, bob_loc).flatten()
            lue_coords.append((self.bob_theta, self.bob_r))

        # 4. Generate channels for Eve
        for e in range(self.E):
            # Formulate Eve's 3D coordinates based on initialization
            eve_loc = [self.ex, 0, self.ez]
            H_EUE[e, :] = get_channel(self.config, eve_loc).flatten()
            eue_coords.append((self.eve_theta, self.eve_r))

        return H_LUE, H_EUE, lue_coords, eue_coords

    def calculate_secrecy_rate(self, H_LUE, H_EUE, W, epsilon, debug=False):
        min_rate = np.inf
        H_b_pinv = np.linalg.pinv(H_LUE)
        V_0 = np.eye(self.N_b) - H_b_pinv @ H_LUE
        AN_power = (1 - epsilon) * self.P_b

        for k in range(self.K):
            for e in range(self.E):
                signal_k = epsilon * self.P_b * np.abs(H_LUE[k, :] @ W[:, k])**2
                interf_k = epsilon * self.P_b * sum(np.abs(H_LUE[k, :] @ W[:, j])**2 for j in range(self.K) if j != k)
                if debug:
                    print(f"Signal Power at Bob: {signal_k:.4e} W | IAN Leakage at Bob: {interf_k:.4e} W | N")
                R_k = np.log2(1 + signal_k / (interf_k + self.sigma2))

                signal_e = epsilon * self.P_b * np.abs(H_EUE[e, :] @ W[:, k])**2
                interf_e = epsilon * self.P_b * sum(np.abs(H_EUE[e, :] @ W[:, j])**2 for j in range(self.K) if j != k)
                noise_an_e = AN_power * np.linalg.norm(H_EUE[e, :] @ V_0)**2
                R_e = np.log2(1 + signal_e / (interf_e + noise_an_e + self.sigma2))

                if debug:
                    print(f"Bob Rate: {R_k:.4f} bps/Hz | Eve Rate: {R_e:.4f} bps/Hz")

                secrecy_rate = max(0, R_k - R_e)
                if secrecy_rate < min_rate:
                    min_rate = secrecy_rate
        return min_rate

    def optimize_power_allocation_GSS(self, H_LUE, H_EUE, W):
        phi_gss = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi_gss
        a, b = 0.001, 0.999
        tol = 0.05 * (b - a)
        c, d = a + resphi * (b - a), b - resphi * (b - a)
        fc = self.calculate_secrecy_rate(H_LUE, H_EUE, W, c)
        fd = self.calculate_secrecy_rate(H_LUE, H_EUE, W, d)

        while abs(b - a) > tol:
            if fc > fd:
                b, d, fd = d, c, fc
                c = a + resphi * (b - a)
                fc = self.calculate_secrecy_rate(H_LUE, H_EUE, W, c)
            else:
                a, c, fc = c, d, fd
                d = b - resphi * (b - a)
                fd = self.calculate_secrecy_rate(H_LUE, H_EUE, W, d)
        return (b + a) / 2

    def optimize_beamfocusing_SCA(self, H_LUE, H_EUE, W_prev, epsilon):
        W = cp.Variable((self.N_b, self.K), complex=True)
        xi = cp.Variable(nonneg=True)
        constraints = [cp.sum(cp.sum_squares(cp.abs(W))) <= self.K]

        P_W = epsilon * self.P_b
        P_AN = (1 - epsilon) * self.P_b / self.N_b
        H_b_pinv = np.linalg.pinv(H_LUE)
        V_0 = np.eye(self.N_b) - H_b_pinv @ H_LUE

        for k in range(self.K):
            h_b_k = np.sqrt(P_W) * H_LUE[k, :]
            sigma_LUE = self.sigma2 

            h_b_w_t_k = h_b_k @ W_prev[:, k]
            signal_b_t = np.abs(h_b_w_t_k)**2
            interf_b_t = sum(np.abs(h_b_k @ W_prev[:, j])**2 for j in range(self.K) if j != k)
            total_b_t = signal_b_t + interf_b_t
            R_kk_t = np.log(1 + signal_b_t / (interf_b_t + sigma_LUE))

            T1_b = R_kk_t
            T2_b = - (signal_b_t + sigma_LUE) / (interf_b_t + sigma_LUE)
            T3_b = sigma_LUE / (total_b_t + sigma_LUE)
            T4_b = (2 * cp.real(np.conj(h_b_w_t_k) * (h_b_k @ W[:, k]))) / (interf_b_t + sigma_LUE)
            current_total_b_W = cp.sum([cp.sum_squares(h_b_k @ W[:, j]) for j in range(self.K)])
            T5_b = - (signal_b_t * current_total_b_W) / ((total_b_t + sigma_LUE) * (interf_b_t + sigma_LUE))
            f1_LB = T1_b + T2_b + T3_b + T4_b + T5_b

            for e in range(self.E):
                h_e_e = np.sqrt(P_W) * H_EUE[e, :]
                sigma_EUE = P_AN * np.linalg.norm(H_EUE[e, :] @ V_0)**2 + self.sigma2

                h_e_w_t_k = h_e_e @ W_prev[:, k]
                signal_e_t = np.abs(h_e_w_t_k)**2
                interf_e_t = sum(np.abs(h_e_e @ W_prev[:, j])**2 for j in range(self.K) if j != k)
                total_e_t = signal_e_t + interf_e_t
                R_ek_t = np.log(1 + signal_e_t / (interf_e_t + sigma_EUE))

                T1_e = R_ek_t
                T2_e = interf_e_t / sigma_EUE
                T3_e = - (signal_e_t * sigma_EUE) / ((total_e_t + sigma_EUE) * (interf_e_t + sigma_EUE))
                current_total_e_W = cp.sum([cp.sum_squares(h_e_e @ W[:, j]) for j in range(self.K)])
                T4_e = current_total_e_W / (total_e_t + sigma_EUE)
                trace_expansion = cp.sum([np.conj(h_e_e @ W_prev[:, j]) * (h_e_e @ W[:, j]) for j in range(self.K) if j != k])
                T5_e = - (2 * cp.real(trace_expansion)) / sigma_EUE
                current_interf_e_W = cp.sum([cp.sum_squares(h_e_e @ W[:, j]) for j in range(self.K) if j != k])
                T6_e = (interf_e_t * current_interf_e_W) / ((interf_e_t + sigma_EUE) * sigma_EUE)
                f2_UB = T1_e + T2_e + T3_e + T4_e + T5_e + T6_e

                LB_R_k = f1_LB / np.log(2)
                UB_R_ek = f2_UB / np.log(2)
                constraints.append(LB_R_k - UB_R_ek >= xi)

        prob = cp.Problem(cp.Maximize(xi), constraints)
        try:
            prob.solve(solver=cp.CLARABEL, max_iters=2000)
            return W.value
        except Exception as e:
            return W_prev

    def run_alternating_optimization(self):
        """Now strictly returns W and epsilon without calculating internal secrecy rate."""
        H_LUE, H_EUE, lue_coords, eue_coords = self.generate_channels()
        H_b_conj = H_LUE.conj().T
        W_t = H_b_conj @ np.linalg.inv(H_LUE @ H_b_conj + self.sigma2 * np.eye(self.K))
        W_t = W_t * np.sqrt(self.K / np.trace(W_t.conj().T @ W_t).real)

        epsilon_t = 0.5
        for t in range(1, 15):
            W_next = self.optimize_beamfocusing_SCA(H_LUE, H_EUE, W_t, epsilon_t)
            if W_next is not None: W_t = W_next
            epsilon_t = self.optimize_power_allocation_GSS(H_LUE, H_EUE, W_t)

        secrecy_rate = self.calculate_secrecy_rate(H_LUE, H_EUE, W_t, epsilon_t, debug=False)
        return W_t, epsilon_t, H_LUE, H_EUE, lue_coords, eue_coords

    def visualize_beamfocusing(self, W, epsilon, step=0):
        visualise(W.conj(), epsilon, self.bx, self.bz, self.ex, self.ez, self.config, step=step)