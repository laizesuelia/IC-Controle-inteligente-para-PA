import numpy as np
import random

class ControladorGPC:
    
    def __init__(self, rho=0.5, I_min=0.0, I_max=180.0):
        self.rho = rho
        self.I_min = I_min
        self.I_max = I_max
        self.I_k_minus_1 = 0.0

    # --- cálculo do termo determinístico da predição ---
    def calcular_K0(self, theta_hat, d0, P_k, I_hist):
        
        a1_hat, b1_hat, bm1_hat = theta_hat
        
        K0 = ((-a1_hat)**d0) * P_k
        K0 += bm1_hat * I_hist[d0]

        for i in range(1, d0):
            K0 += ((-a1_hat)**i) * b1_hat * I_hist[i]
            K0 += ((-a1_hat)**i) * bm1_hat * I_hist[d0 + i]

        return K0

    # --- lei de controle GPC ---
    def calcular_controle(self, K0, theta_hat, P_ref):
        
        b1_hat = theta_hat[1]

        numerador = b1_hat * (P_ref - K0) + self.rho * self.I_k_minus_1
        denominador = max((b1_hat**2) + self.rho, 1e-6)

        I_k = numerador / denominador

        # Saturação fisiológica
        I_k = max(self.I_min, min(I_k, self.I_max))

        # guarda para próxima iteração
        self.I_k_minus_1 = I_k

        return I_k