import numpy as np
import random


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