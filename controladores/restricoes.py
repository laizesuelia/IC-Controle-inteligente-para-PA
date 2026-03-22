import numpy as np

# CONTROLADOR 3.3 — COM RESTRIÇÕES
import random 
random.seed(42)

class ControladorGPC_Restricao:
    def __init__(self, rho=1.5, I_min=0.0, I_max=180.0):
        self.rho = rho
        self.I_min = I_min
        self.I_max = I_max
        self.I_k_minus_1 = 0.0

    def calcular_K0(self, theta_hat, d0, P_k, I_hist):
        a1_hat, b1_hat, bm1_hat = theta_hat
        K0 = ((-a1_hat)**d0) * P_k
        K0 += bm1_hat * I_hist[d0]
        for i in range(1, d0):
            K0 += ((-a1_hat)**i) * b1_hat * I_hist[i]
            K0 += ((-a1_hat)**i) * bm1_hat * I_hist[d0 + i]
        return K0

    def calcular_controle(self, K0, theta_hat, P_ref, MAP_medido):
        b1_ctrl = max(theta_hat[1], 0.01)
        numerador = b1_ctrl * (P_ref - K0) + self.rho * self.I_k_minus_1
        denominador = (b1_ctrl**2) + self.rho
        I_k = numerador / denominador
        I_k = max(self.I_min, min(I_k, self.I_max))
        self.I_k_minus_1 = I_k
        return I_k