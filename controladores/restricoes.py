import numpy as np

# CONTROLADOR 3.3 — COM RESTRIÇÕES
import random 
random.seed(42)

class ControladorGPC_Restricao:
    # ro=0.5 (GPC Clássico padrão) ou ro=1.5 (GPC 3.3 mais conservador)
    def __init__(self, rho=1.5, I_min=0.0, I_max=180.0):
        self.rho = rho
        self.I_min = I_min
        self.I_max = I_max
        self.I_k_minus_1 = 0.0

    # O cálculo de K0 continua separado e limpo, usando a memória da planta
    def calcular_K0(self, theta_hat, d0, P_k, I_hist):
        a1_hat, b1_hat, bm1_hat = theta_hat
        K0 = ((-a1_hat)**d0) * P_k
        K0 += bm1_hat * I_hist[d0]
        for i in range(1, d0):
            K0 += ((-a1_hat)**i) * b1_hat * I_hist[i]
            K0 += ((-a1_hat)**i) * bm1_hat * I_hist[d0 + i]
        return K0

    def calcular_controle(self, K0, theta_hat, P_ref, MAP_medido):
        # A Trava de Segurança Clínica do seu Controlador 3.3!
        # Impede ganho zero ou negativo (inversão de ação)
        b1_ctrl = max(theta_hat[1], 0.01)

        # Lei de Controle (Eq. 14)
        numerador = b1_ctrl * (P_ref - K0) + self.rho * self.I_k_minus_1
        denominador = (b1_ctrl**2) + self.rho

        I_k = numerador / denominador

        # Saturação fisiológica
        I_k = max(self.I_min, min(I_k, self.I_max))

        # Guarda para a próxima iteração
        self.I_k_minus_1 = I_k

        return I_k
