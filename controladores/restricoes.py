import numpy as np
import random

class Controlador_33_Restricoes:
    def __init__(self, u_lim=5.0):
        self.u_lim    = u_lim   # limite mínimo de excitação
        self.k        = 0
        self.I_ante_1 = 0.0
        self.I_histo  = np.zeros(100)

    def calcular_controle(self, P_ref, P_k_medido, teta_est, d_est, m_est, inovacao_k, P_cov):
        self.k += 1
        a1_est, b1_est, bm_est = teta_est
        b1_ctrl = max(b1_est, 0.01)

        if self.k <= 10:
            PRBS = 10.0
            if self.k > 5:
                PRBS = 60 - (60 / 3.8) * abs(b1_est)
            I_calculado = PRBS if random.random() > 0.5 else 0.0
            I_calculado = np.clip(I_calculado, 0, 180)
            self.I_histo = np.roll(self.I_histo, 1);  self.I_histo[0] = I_calculado
            self.I_ante_1 = I_calculado
            return I_calculado

        k0 = (-a1_est) ** d_est * P_k_medido
        for i in range(1, d_est):
            t1 = b1_ctrl * self.I_histo[i]         if i         < len(self.I_histo) else 0.0
            t2 = bm_est  * self.I_histo[m_est + i] if m_est + i < len(self.I_histo) else 0.0
            k0 += (-a1_est) ** i * (t1 + t2)
        if m_est < len(self.I_histo):
            k0 += bm_est * self.I_histo[m_est]

        # Base: cauteloso real (Eq. 2.23)
        p_b1    = P_cov[1, 1]
        IP_teta = P_cov[1, :] @ teta_est

        num_ca = b1_ctrl * (P_ref - k0) + IP_teta + p_b1 * self.I_ante_1
        den_ca = b1_ctrl ** 2 + p_b1
        u_ca   = num_ca / max(den_ca, 1e-6)

        # Restrição de Jacobs e Hughes (Eq. 3.2):
        # se o cauteloso for fraco demais, força u_lim para manter excitação
        if abs(u_ca) < self.u_lim:
            I_calculado = self.u_lim * np.sign(u_ca) if u_ca != 0 else self.u_lim
        else:
            I_calculado = u_ca

        I_calculado = np.clip(I_calculado, 0, 180)
        self.I_histo = np.roll(self.I_histo, 1);  self.I_histo[0] = I_calculado
        self.I_ante_1 = I_calculado
        return I_calculado