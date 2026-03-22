import numpy as np
import random

class ControladorGPC:
    def __init__(self, rho=0.5, I_min=0.0, I_max=180.0):
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

    def calcular_controle(self, K0, theta_hat, P_ref):
        b1_hat = theta_hat[1]
        numerador = b1_hat * (P_ref - K0) + self.rho * self.I_k_minus_1
        denominador = max((b1_hat**2) + self.rho, 1e-6)
        I_k = numerador / denominador
        I_k = max(self.I_min, min(I_k, self.I_max))
        self.I_k_minus_1 = I_k
        return I_k

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

class Controlador_PID_Adaptativo:
    def __init__(self, Kp0=0.4, Ki0=0.05, Kd0=0.2, Kaw=0.5, b0=0.187, alpha_d=0.8):
        self.Kp0 = Kp0
        self.Ki0 = Ki0
        self.Kd0 = Kd0
        self.Kaw = Kaw
        self.b0 = b0
        self.alpha_d = alpha_d
        self.I_integral = 0.0
        self.y_ant = 150.0
        self.dy_f = 0.0
        self.u_ant = 0.0
        self.u_min = 0.0
        self.u_max = 180.0
        self.du_max = 20.0
        self.Kp = Kp0
        self.Ki = Ki0
        self.Kd = Kd0

    def calcular_controle(self, MAP_alvo, MAP_medida, b_hat):
        b_hat_seguro = max(b_hat, 0.02)
        fator_ganho = self.b0 / b_hat_seguro
        fator_ganho = np.clip(fator_ganho, 0.3, 3.0)
        self.Kp = 0.9 * self.Kp + 0.1 * (self.Kp0 * fator_ganho)
        self.Ki = 0.9 * self.Ki + 0.1 * (self.Ki0 * fator_ganho)
        self.Kd = 0.9 * self.Kd + 0.1 * (self.Kd0 * fator_ganho)
        
        e_k = MAP_medida - MAP_alvo
        dy = MAP_medida - self.y_ant
        self.dy_f = self.alpha_d * self.dy_f + (1 - self.alpha_d) * dy
        self.y_ant = MAP_medida
        
        u_raw = self.Kp * e_k + self.I_integral + self.Kd * self.dy_f
        du = u_raw - self.u_ant
        du_sat = np.clip(du, -self.du_max, self.du_max)
        u_sat = self.u_ant + du_sat
        u_final = np.clip(u_sat, self.u_min, self.u_max)
        
        self.I_integral = self.I_integral + self.Ki * e_k + self.Kaw * (u_final - u_raw)
        self.u_ant = u_final
        return u_final

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