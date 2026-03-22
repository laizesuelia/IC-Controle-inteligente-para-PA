import numpy as np
import random

class Controlador_32_Perturbacao:
    def __init__(self, amplitude=8.0, taxa_decaimento=0.997, seed=None):
        self.k        = 0
        self.I_ante_1 = 0.0
        self.I_histo  = np.zeros(100)
        self._amp     = amplitude
        self._decay   = taxa_decaimento
        self._rng     = np.random.default_rng(seed)

    def calcular_controle(self, P_ref, P_k_medido, teta_est, d_est, m_est, inovacao_k, P_cov):
        self.k += 1
        a1_est, b1_est, bm_est = teta_est
        b1_ctrl = max(b1_est, 0.01)

        if self.k <= 10:
            I_calculado = 30.0 if self._rng.integers(0, 2) == 1 else 0.0
            I_calculado = np.clip(I_calculado, 0, 180)
            self.I_histo = np.roll(self.I_histo, 1);  self.I_histo[0] = I_calculado
            self.I_ante_1 = I_calculado
            return I_calculado

        # K0 padrão
        k0 = (-a1_est) ** d_est * P_k_medido
        for i in range(1, d_est):
            t1 = b1_ctrl * self.I_histo[i]         if i         < len(self.I_histo) else 0.0
            t2 = bm_est  * self.I_histo[m_est + i] if m_est + i < len(self.I_histo) else 0.0
            k0 += (-a1_est) ** i * (t1 + t2)
        if m_est < len(self.I_histo):
            k0 += bm_est * self.I_histo[m_est]

        # --- Controlador cauteloso (Eq. 2.23) como base ---
        # p_b1: variância de b1 da matriz de covariância
        # Cauteloso (Eq. 2.23) — base do 3.2
        p_b1 = P_cov[1, 1]  # limita para não zerar o controle no início
        IP_teta = P_cov[1, :] @ teta_est

        num_ca = b1_ctrl * (P_ref - k0) + IP_teta + p_b1 * self.I_ante_1
        den_ca = b1_ctrl ** 2 + p_b1
        u_ca   = num_ca / max(den_ca, 1e-6)

        # --- Perturbação PRBS — dualidade explícita (Seção 3.2) ---
        u_p       = self._amp * (self._rng.integers(0, 2) * 2 - 1)
        self._amp *= self._decay

        I_calculado = np.clip(u_ca + u_p, 0, 180)
        self.I_histo = np.roll(self.I_histo, 1);  self.I_histo[0] = I_calculado
        self.I_ante_1 = I_calculado
        return I_calculado