import numpy as np
import random

class Controlador_GPC_Adaptativo:
    def __init__(self, ro=1.5):
        self.ro = ro
        self.k  = 0
        self.I_ante_1 = 0.0
        self.I_histo  = np.zeros(100)

    def calcular_controle(self, P_ref, P_k_medido, teta_est, d_est, m_est, inovacao_k, P_cov):
        # P_cov ignorado — usa RLS mas sem dualidade via covariância
        self.k += 1
        a1_est, b1_est, bm_est = teta_est
        b1_ctrl = max(b1_est, 0.01)

        if self.k <= 10:
            PRBS = 10.0
            if self.k > 5:
                PRBS = 60 - (60 / 3.8) * abs(b1_est)
            I_calculado = PRBS if random.random() > 0.5 else 0.0
            I_calculado = np.clip(I_calculado, 0, 180)
            self.I_histo = np.roll(self.I_histo, 1)
            self.I_histo[0] = I_calculado
            self.I_ante_1 = I_calculado
            return I_calculado

        k0 = (-a1_est) ** d_est * P_k_medido
        for i in range(1, d_est):
            t1 = b1_ctrl * self.I_histo[i]         if i         < len(self.I_histo) else 0.0
            t2 = bm_est  * self.I_histo[m_est + i] if m_est + i < len(self.I_histo) else 0.0
            k0 += (-a1_est) ** i * (t1 + t2)
        if m_est < len(self.I_histo):
            k0 += bm_est * self.I_histo[m_est]

        num = b1_ctrl * (P_ref - k0) + self.ro * self.I_ante_1
        den = b1_ctrl ** 2 + self.ro
        I_calculado = np.clip(num / max(den, 1e-6), 0, 180)

        self.I_histo = np.roll(self.I_histo, 1)
        self.I_histo[0] = I_calculado
        self.I_ante_1 = I_calculado
        return I_calculado