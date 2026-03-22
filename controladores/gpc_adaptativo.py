import numpy as np
import random

class ControladorGPCAdaptativo:
    def __init__(self, alpha_rho=5.0, beta_rho=0.90, rho_min=0.5, I_min=0.0, I_max=180.0):
        self.alpha_rho = alpha_rho
        self.beta_rho = beta_rho
        self.rho_min = rho_min
        self.I_min = I_min
        self.I_max = I_max
        self.I_k_minus_1 = 0.0
        self.energia_inovacao = 1.0
        self.rho = rho_min

    def calcular_K0(self, theta_hat, d0, P_k, I_hist):
        a1_hat, b1_hat, bm1_hat = theta_hat
        K0 = ((-a1_hat)**d0) * P_k
        K0 += bm1_hat * I_hist[d0]
        for i in range(1, d0):
            K0 += ((-a1_hat)**i) * b1_hat * I_hist[i]
            K0 += ((-a1_hat)**i) * bm1_hat * I_hist[d0 + i]
        return K0

    def calcular_controle(self, K0, theta_hat, P_ref):
        b1_hat = theta_hat[1]
        numerador = b1_hat * (P_ref - K0) + self.rho * self.I_k_minus_1
        denominador = max((b1_hat**2) + self.rho, 1e-6)
        I_k = numerador / denominador
        I_k = max(self.I_min, min(I_k, self.I_max))
        self.I_k_minus_1 = I_k
        return I_k

    def atualizar_rho(self, erro):
        self.energia_inovacao = (self.beta_rho * self.energia_inovacao) + ((1 - self.beta_rho) * (erro**2))
        rho_calculado = self.alpha_rho / (self.energia_inovacao + 1e-6)
        self.rho = max(self.rho_min, rho_calculado)
        return self.rho