import time

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

class XLMIMO_System:
    def __init__(self, bob_loc, eve_loc, N_bx=32, N_bz=1, K=1, E=1, freq=2e9, P_b_dBm=20, noise_dBm=-96):
        """
        Initialize the XL-MIMO System Model.
        Note: N_bz is set to 1 (Uniform Linear Array) to easily visualize the 2D plane.
        """
        self.N_bx = N_bx
        self.N_bz = N_bz
        self.N_b = N_bx * N_bz
        self.K = K
        self.E = E
        self.bob_r = np.sqrt(bob_loc[0]**2 + bob_loc[2]**2)
        self.bob_theta = np.arctan2(bob_loc[2], bob_loc[0])
        self.eve_r = np.sqrt(eve_loc[0]**2 + eve_loc[2]**2)
        self.eve_theta = np.arctan2(eve_loc[2], eve_loc[0])

        # print(f"Bob={self.bob_r:.2f} m, {np.degrees(self.bob_theta):.2f}° | Eve={self.eve_r:.2f} m, {np.degrees(self.eve_theta):.2f}°")

        self.c = 299_792_458  # Speed of light (m/s)
        self.freq = freq
        self.lambda_c = self.c / self.freq
        self.d_ant = self.lambda_c / 2

        self.P_b = 10 ** ((P_b_dBm - 30) / 10)
        self.sigma2 = 10 ** ((noise_dBm - 30) / 10)

    def near_field_array_response(self, theta, phi, r):
        a = np.zeros(self.N_b, dtype=complex)
        idx = 0
        for nx in range(-(self.N_bx//2), self.N_bx//2 + 1 if self.N_bx%2!=0 else self.N_bx//2):
            for nz in range(-(self.N_bz//2), self.N_bz//2 + 1 if self.N_bz%2!=0 else self.N_bz//2):
                dist_shift = (nx * self.d_ant * np.cos(theta) * np.sin(phi) +
                              nz * self.d_ant * np.cos(phi))
                phase = - (2 * np.pi / self.lambda_c) * (r - dist_shift)
                a[idx] = np.exp(1j * phase)
                idx += 1
        return a

    def generate_channels(self):
        # np.random.seed(42)
        H_LUE = np.zeros((self.K, self.N_b), dtype=complex)
        H_EUE = np.zeros((self.E, self.N_b), dtype=complex)

        lue_coords = []
        eue_coords = []

        # Fixed phi = pi/2 to keep users on a 2D plane for clean visualization
        phi = np.pi / 2

        for k in range(self.K):
            theta, r = self.bob_theta, self.bob_r
            H_LUE[k, :] = self.near_field_array_response(theta, phi, r) * np.sqrt(1e-7)
            lue_coords.append((theta, r))

        for e in range(self.E):
            theta, r = self.eve_theta, self.eve_r
            H_EUE[e, :] = self.near_field_array_response(theta, phi, r) * np.sqrt(1e-7)
            eue_coords.append((theta, r))

        return H_LUE, H_EUE, lue_coords, eue_coords

    def calculate_secrecy_rate(self, H_LUE, H_EUE, W, epsilon):
        min_rate = np.inf
        H_b_pinv = np.linalg.pinv(H_LUE)
        V_0 = np.eye(self.N_b) - H_b_pinv @ H_LUE
        AN_power = (1 - epsilon) * self.P_b / self.N_b

        for k in range(self.K):
            for e in range(self.E):
                signal_k = epsilon * self.P_b * np.abs(H_LUE[k, :] @ W[:, k])**2
                interf_k = epsilon * self.P_b * sum(np.abs(H_LUE[k, :] @ W[:, j])**2 for j in range(self.K) if j != k)
                R_k = np.log2(1 + signal_k / (interf_k + self.sigma2))

                signal_e = epsilon * self.P_b * np.abs(H_EUE[e, :] @ W[:, k])**2
                interf_e = epsilon * self.P_b * sum(np.abs(H_EUE[e, :] @ W[:, j])**2 for j in range(self.K) if j != k)
                noise_an_e = AN_power * np.linalg.norm(H_EUE[e, :] @ V_0)**2
                R_e = np.log2(1 + signal_e / (interf_e + noise_an_e + self.sigma2))

                secrecy_rate = max(0, R_k - R_e)
                if secrecy_rate < min_rate:
                    min_rate = secrecy_rate
        return min_rate

    def optimize_power_allocation_GSS(self, H_LUE, H_EUE, W):
        phi_gss = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi_gss
        a, b = 0.001, 0.999
        tol = 0.05 * (b - a)

        c = a + resphi * (b - a)
        d = b - resphi * (b - a)
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
        print(H_LUE.shape, H_EUE.shape, W_prev.shape)
        W = cp.Variable((self.N_b, self.K), complex=True)
        xi = cp.Variable(nonneg=True)
        constraints = [cp.sum(cp.sum_squares(cp.abs(W))) <= self.K]

        P_W = epsilon * self.P_b
        P_AN = (1 - epsilon) * self.P_b / self.N_b
        H_b_pinv = np.linalg.pinv(H_LUE)
        V_0 = np.eye(self.N_b) - H_b_pinv @ H_LUE

        for k in range(self.K):
            # ---------------------------------------------------------
            # 1. SETUP LEGITIMATE USER (Bob) CONSTANTS
            # ---------------------------------------------------------
            h_b_k = np.sqrt(P_W) * H_LUE[k, :]
            sigma_LUE = self.sigma2 # Bob is in the null-space, so AN leakage is 0

            # Precompute power components at W_prev (iteration t)
            h_b_w_t_k = h_b_k @ W_prev[:, k]
            signal_b_t = np.abs(h_b_w_t_k)**2
            interf_b_t = sum(np.abs(h_b_k @ W_prev[:, j])**2 for j in range(self.K) if j != k)
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
            T4_b = (2 * cp.real(np.conj(h_b_w_t_k) * (h_b_k @ W[:, k]))) / (interf_b_t + sigma_LUE)

            # Concave term (negative sum of squares)
            current_total_b_W = cp.sum([cp.sum_squares(h_b_k @ W[:, j]) for j in range(self.K)])
            T5_b = - (signal_b_t * current_total_b_W) / ((total_b_t + sigma_LUE) * (interf_b_t + sigma_LUE))

            f1_LB = T1_b + T2_b + T3_b + T4_b + T5_b

            for e in range(self.E):
                # ---------------------------------------------------------
                # 3. SETUP EAVESDROPPER (Eve) CONSTANTS
                # ---------------------------------------------------------
                h_e_e = np.sqrt(P_W) * H_EUE[e, :]

                # Eve suffers from Artificial Noise leakage + background noise
                sigma_EUE = P_AN * np.linalg.norm(H_EUE[e, :] @ V_0)**2 + self.sigma2

                # Precompute power components at W_prev (iteration t)
                h_e_w_t_k = h_e_e @ W_prev[:, k]
                signal_e_t = np.abs(h_e_w_t_k)**2
                interf_e_t = sum(np.abs(h_e_e @ W_prev[:, j])**2 for j in range(self.K) if j != k)
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
                current_total_e_W = cp.sum([cp.sum_squares(h_e_e @ W[:, j]) for j in range(self.K)])
                T4_e = current_total_e_W / (total_e_t + sigma_EUE)

                # Linear (Affine) term (Trace expansion)
                trace_expansion = cp.sum([np.conj(h_e_e @ W_prev[:, j]) * (h_e_e @ W[:, j]) for j in range(self.K) if j != k])
                T5_e = - (2 * cp.real(trace_expansion)) / sigma_EUE

                # Convex term 2 (Interference power W_k)
                current_interf_e_W = cp.sum([cp.sum_squares(h_e_e @ W[:, j]) for j in range(self.K) if j != k])
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
        H_LUE, H_EUE, lue_coords, eue_coords = self.generate_channels()

        H_b_conj = H_LUE.conj().T
        W_t = H_b_conj @ np.linalg.inv(H_LUE @ H_b_conj + self.sigma2 * np.eye(self.K))
        W_t = W_t * np.sqrt(self.K / np.trace(W_t.conj().T @ W_t).real)

        epsilon_t = 0.5
        eta_out = 1e-3
        xi_prev = 0

        # print(f"Starting Joint Optimization | Users: {self.K}, Eavesdroppers: {self.E}, Antennas: {self.N_b}")
        # print("-" * 65)
        t0 = time.perf_counter()
        for t in range(1, 15):
            W_next = self.optimize_beamfocusing_SCA(H_LUE, H_EUE, W_t, epsilon_t)
            if W_next is not None: W_t = W_next

            epsilon_t = self.optimize_power_allocation_GSS(H_LUE, H_EUE, W_t)
            current_rate = self.calculate_secrecy_rate(H_LUE, H_EUE, W_t, epsilon_t)

            # print(f"Iteration {t:02d} | Epsilon: {epsilon_t:.3f} | Min Secrecy Rate: {current_rate:.4f} bps/Hz")

            # if abs(current_rate - xi_prev) / (xi_prev + 1e-8) < eta_out:
            #     print("--> Convergence Reached!")
            #     break
            xi_prev = current_rate
        t1 = time.perf_counter()
        # print(f"Optimization completed in {t1 - t0:.2f} seconds.")

        # self.visualize_beamfocusing(W_t, epsilon_t, lue_coords, eue_coords)
        return W_t, epsilon_t, current_rate

    def visualize_beamfocusing(self, W, epsilon, lue_coords, eue_coords):
        """Generates a 2D Spatial Heatmap of the Signal Power."""
        print("Generating visualization... (this may take a moment)")
        x_min, x_max = -15, 15
        y_min, y_max = 2, 20
        grid_res = 50

        X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                           np.linspace(y_min, y_max, grid_res))
        Power_Map = np.zeros_like(X, dtype=float)

        P_W = epsilon * self.P_b

        # Calculate signal power across the spatial grid
        for i in range(grid_res):
            for j in range(grid_res):
                x, y = X[i, j], Y[i, j]
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                phi = np.pi / 2 # Evaluated on the 2D plane

                h_test = self.near_field_array_response(theta, phi, r) * np.sqrt(1e-7)

                # Total signal power from all beams at this location
                p_signal = sum(P_W * np.abs(h_test @ W[:, k])**2 for k in range(self.K))
                Power_Map[i, j] = p_signal

        # Plotting
        plt.figure(figsize=(10, 7))
        plt.pcolormesh(X, Y, 10*np.log10(Power_Map + 1e-12), shading='auto', cmap='viridis')
        plt.colorbar(label='Signal Power (dB)')

        # Plot LUEs (Legitimate Users)
        for idx, (theta, r) in enumerate(lue_coords):
            plt.scatter(r * np.cos(theta), r * np.sin(theta),
                        c='lime', marker='^', s=150, edgecolors='black',
                        label='Bob (LUE)' if idx == 0 else "")

        # Plot EUEs (Eavesdroppers)
        for idx, (theta, r) in enumerate(eue_coords):
            plt.scatter(r * np.cos(theta), r * np.sin(theta),
                        c='red', marker='X', s=150, edgecolors='black',
                        label='Eve (EUE)' if idx == 0 else "")

        # Plot the Base Station (XL-MIMO Array)
        plt.scatter(0, 0, c='cyan', marker='s', s=200, edgecolors='black', label='Base Station')

        plt.title(f"Near-Field Optimal Beamfocusing Heatmap\nEpsilon (Power Split) = {epsilon:.3f}")
        plt.xlabel("X coordinate (m)")
        plt.ylabel("Y/Depth coordinate (m)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("beamfocusing_visualization.png", dpi=300)
        # plt.show()

if __name__ == "__main__":
    xl_mimo = XLMIMO_System(N_bx=256, N_bz=1, K=1, E=1, bob_loc=np.array([-20, 0, 30]), eve_loc=np.array([20, 0, 10]))
    W_t, epsilon_t, secrecy_rate = xl_mimo.run_alternating_optimization()
    xl_mimo.visualize_beamfocusing(W_t, epsilon_t, lue_coords=np.array([(xl_mimo.bob_theta, xl_mimo.bob_r)]), eue_coords=np.array([(xl_mimo.eve_theta, xl_mimo.eve_r)]))
    # print(f"Final Achieved Secrecy Rate: {secrecy_rate:.4f} bps/Hz")
