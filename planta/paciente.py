import numpy as np

class Paciente:
    def __init__(self, vetor_ruidos, P0=150, a1=-0.741, b1=0.187, bm1=0.075, d=3, m=3):
        self.P0 = P0
        self.a1 = a1
        self.b1 = b1
        self.b1_base = 0.187
        self.bm1 = bm1
        self.d = d
        self.m = m
        self.vetor_ruidos = vetor_ruidos
        
        self.P_hist = np.zeros(20)
        self.I_hist = np.zeros(20)

    def atualizar(self, I_k, k):
        self.I_hist = np.roll(self.I_hist, 1)
        self.I_hist[0] = I_k
        self.b1 = self.b1_base + 0.1 * np.sin(2 * np.pi * 1/1200 * k + 15)
        
        e_k = self.vetor_ruidos[k]
        
        P_k = self.a1 * self.P_hist[0] + \
              self.b1 * self.I_hist[self.d] + \
              self.bm1 * self.I_hist[self.d + self.m] + e_k
              
        self.P_hist = np.roll(self.P_hist, 1)
        self.P_hist[0] = P_k
        MAP = self.P0 - P_k
        
        return MAP, P_k