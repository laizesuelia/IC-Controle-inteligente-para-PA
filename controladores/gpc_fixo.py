import numpy as np
import random

class Controlador_GPC_Fixo:
    def __init__(self, teta_fixo, d_fixo, m_fixo, ro=1.5):
        self.a1, self.b1, self.bm = teta_fixo
        self.d  = int(d_fixo)
        self.m  = int(m_fixo)
        self.ro = ro
        self.k  = 0
        self.I_ante_1 = 0.0
        self.I_histo  = np.zeros(100)

    def calcular_controle(self, P_ref, P_k_medido, teta_est, d_est, m_est, inovacao_k, P_cov):
        # P_cov ignorado — parâmetros congelados, não usa RLS nem covariância
        self.k += 1
        b1_ctrl = max(self.b1, 0.01)

        if self.k <= 10:
            PRBS = 10.0
            if self.k > 5:
                PRBS = 60 - (60 / 3.8) * abs(self.b1)
            I_calculado = PRBS if random.random() > 0.5 else 0.0
            I_calculado = np.clip(I_calculado, 0, 180)
            self.I_histo = np.roll(self.I_histo, 1)
            self.I_histo[0] = I_calculado
            self.I_ante_1 = I_calculado
            return I_calculado

        k0 = (-self.a1) ** self.d * P_k_medido
        for i in range(1, self.d):
            t1 = b1_ctrl * self.I_histo[i]          if i          < len(self.I_histo) else 0.0
            t2 = self.bm * self.I_histo[self.m + i] if self.m + i < len(self.I_histo) else 0.0
            k0 += (-self.a1) ** i * (t1 + t2)
        if self.m < len(self.I_histo):
            k0 += self.bm * self.I_histo[self.m]

        num = b1_ctrl * (P_ref - k0) + self.ro * self.I_ante_1
        den = b1_ctrl ** 2 + self.ro
        I_calculado = np.clip(num / max(den, 1e-6), 0, 180)

        self.I_histo = np.roll(self.I_histo, 1)
        self.I_histo[0] = I_calculado
        self.I_ante_1 = I_calculado
        return I_calculado