import numpy as np
import random

class ControladorIDC:
    def __init__(self, lambda_idc=1, I_min=0.0, I_max=180.0):
        self.lambda_idc = lambda_idc
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

    def calcular_controle(self, K0, theta_hat, P_cov, P_ref):
        b1_hat = theta_hat[1]
        p_b1 = P_cov[1, 1]
        I_P_theta = np.dot(P_cov[1, :], theta_hat)
        numerador = (b1_hat * (P_ref - K0) + (1 - self.lambda_idc) * I_P_theta)
        denominador = max((b1_hat**2) + (1 - self.lambda_idc) * p_b1, 1e-6)
        I_k = numerador / denominador
        I_k = max(self.I_min, min(I_k, self.I_max))
        return I_k