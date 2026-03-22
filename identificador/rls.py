import numpy as np

class Identificador:
    def __init__(self, lambda_val, d_vetor, m_vetor, N_sim):
        self.lambda_val  = lambda_val
        self.d_vetor     = np.array(d_vetor)
        self.m_vetor     = np.array(m_vetor)
        self.k           = 0
        self.teta_ea     = np.zeros(12)
        self.Pant        = np.kron(np.eye(4), 10 * np.eye(3))
        self.histo_erro  = np.zeros((N_sim, 4))
        self.fator       = np.zeros(4)
        self.P_histo     = np.zeros(100)
        self.I_histo     = np.zeros(100)
        self.ind_melhor  = 0
        self.teta_melhor = np.zeros(3)

    def RLS(self, P_k_medido, I_k_anterior):
        self.k += 1
        self.P_histo = np.roll(self.P_histo, 1);  self.P_histo[0] = P_k_medido
        self.I_histo = np.roll(self.I_histo, 1);  self.I_histo[0] = I_k_anterior

        fia = np.zeros(12)
        for j in range(4):
            ind  = 3 * j
            d_j, m_j = self.d_vetor[j], self.m_vetor[j]
            fia[ind] = -self.P_histo[1]
            if d_j - 1       < len(self.I_histo): fia[ind + 1] = self.I_histo[d_j - 1]
            if d_j + m_j - 1 < len(self.I_histo): fia[ind + 2] = self.I_histo[d_j + m_j - 1]

        Soma_mi = 0
        for j in range(4):
            sl    = slice(3*j, 3*j+3)
            fia_j = fia[sl];  Pant_j = self.Pant[sl, sl]
            den   = self.lambda_val + fia_j @ Pant_j @ fia_j
            if den == 0: den = 1e-9
            kk_j  = (Pant_j @ fia_j) / den
            teta_ant = self.teta_ea[sl]
            self.teta_ea[sl] = teta_ant + kk_j * (P_k_medido - teta_ant @ fia_j)
            self.Pant[sl, sl] = (np.eye(3) - np.outer(kk_j, fia_j)) @ Pant_j / self.lambda_val
            ep = (P_k_medido - self.teta_ea[sl] @ fia_j)**2
            self.histo_erro[self.k-1, j] = ep
            janela = 20
            ini = max(0, self.k - janela)
            self.fator[j] = np.sum(self.histo_erro[ini:self.k, j])
            Soma_mi += 1.0 / (self.fator[j] + 1e-9)

        Med_adeq = np.array([1.0/(self.fator[j]+1e-9) for j in range(4)]) / (Soma_mi + 1e-9)
        ind_cand = np.argmax(Med_adeq)
        if Med_adeq[ind_cand] > 1.35 * Med_adeq[self.ind_melhor]:
            self.ind_melhor = ind_cand

        sl_best = slice(3*self.ind_melhor, 3*self.ind_melhor+3)
        self.teta_melhor = self.teta_ea[sl_best]
        incertezas = np.diagonal(self.Pant)[[1, 4, 7, 10]]
        P_melhor = self.Pant[sl_best, sl_best].copy()
        return self.teta_melhor, self.d_vetor[self.ind_melhor], self.m_vetor[self.ind_melhor], incertezas, self.fator.copy(), P_melhor

