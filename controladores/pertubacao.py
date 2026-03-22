import numpy as np
import random

class Controlador_Perturbacao:
    def __init__(self, amplitude=8.0, taxa_decaimento=0.997, I_min=0.0, I_max=180.0):
        self.amp = amplitude
        self.decay = taxa_decaimento
        self.I_min = I_min
        self.I_max = I_max

    def calcular_K0(self, theta_hat, d0, P_k, I_hist):
        a1_hat, b1_hat, bm1_hat = theta_hat
        K0 = ((-a1_hat)**d0) * P_k
        K0 += bm1_hat * I_hist[d0]
        for i in range(1, d0):
            K0 += ((-a1_hat)**i) * b1_hat * I_hist[i]
            K0 += ((-a1_hat)**i) * bm1_hat * I_hist[d0 + i]
        return K0

    def calcular_controle(self, K0, theta_hat, P_ref):
        b1_ctrl = max(theta_hat[1], 0.01)
        I_mvc = (P_ref - K0) / b1_ctrl
        bit = 1 if np.random.rand() > 0.5 else -1
        up = self.amp * bit
        self.amp *= self.decay
        I_k = I_mvc + up
        return max(self.I_min, min(I_k, self.I_max))