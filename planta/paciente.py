import numpy as np
import random

class Paciente:
    def __init__(self, teta_real, d_real, m_real, P0, sigma):
        self.teta_real = np.array(teta_real)
        self.d = int(d_real)
        self.m = int(m_real)
        self.P0 = P0
        self.sigma = sigma
        self.P_histo = np.zeros(200)
        self.I_histo = np.zeros(200)

    def step(self, I_atual_k_menos_1):
        self.I_histo = np.roll(self.I_histo, 1)
        self.I_histo[0] = I_atual_k_menos_1
        P_ante_1 = self.P_histo[0]
        idx_d = self.d - 1
        I_ante_d  = self.I_histo[idx_d]  if idx_d  < len(self.I_histo) else 0.0
        idx_dm    = self.d + self.m - 1
        I_ante_dm = self.I_histo[idx_dm] if idx_dm < len(self.I_histo) else 0.0
        fi = np.array([-P_ante_1, I_ante_d, I_ante_dm])
        ruido_e = random.gauss(0, self.sigma)
        P_k = self.teta_real @ fi + ruido_e
        MAP = self.P0 - P_k
        self.P_histo = np.roll(self.P_histo, 1)
        self.P_histo[0] = P_k
        return MAP, P_k, ruido_e
