{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8605ffd7",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Controlador_GPC_Adaptativo:\n",
    "    def __init__(self, ro=1.5):\n",
    "        self.ro = ro\n",
    "        self.k  = 0\n",
    "        self.I_ante_1 = 0.0\n",
    "        self.I_histo  = np.zeros(100)\n",
    "\n",
    "    def calcular_controle(self, P_ref, P_k_medido, teta_est, d_est, m_est, inovacao_k, P_cov):\n",
    "        # P_cov ignorado — usa RLS mas sem dualidade via covariância\n",
    "        self.k += 1\n",
    "        a1_est, b1_est, bm_est = teta_est\n",
    "        b1_ctrl = max(b1_est, 0.01)\n",
    "\n",
    "        if self.k <= 10:\n",
    "            PRBS = 10.0\n",
    "            if self.k > 5:\n",
    "                PRBS = 60 - (60 / 3.8) * abs(b1_est)\n",
    "            I_calculado = PRBS if random.random() > 0.5 else 0.0\n",
    "            I_calculado = np.clip(I_calculado, 0, 180)\n",
    "            self.I_histo = np.roll(self.I_histo, 1)\n",
    "            self.I_histo[0] = I_calculado\n",
    "            self.I_ante_1 = I_calculado\n",
    "            return I_calculado\n",
    "\n",
    "        k0 = (-a1_est) ** d_est * P_k_medido\n",
    "        for i in range(1, d_est):\n",
    "            t1 = b1_ctrl * self.I_histo[i]         if i         < len(self.I_histo) else 0.0\n",
    "            t2 = bm_est  * self.I_histo[m_est + i] if m_est + i < len(self.I_histo) else 0.0\n",
    "            k0 += (-a1_est) ** i * (t1 + t2)\n",
    "        if m_est < len(self.I_histo):\n",
    "            k0 += bm_est * self.I_histo[m_est]\n",
    "\n",
    "        num = b1_ctrl * (P_ref - k0) + self.ro * self.I_ante_1\n",
    "        den = b1_ctrl ** 2 + self.ro\n",
    "        I_calculado = np.clip(num / max(den, 1e-6), 0, 180)\n",
    "\n",
    "        self.I_histo = np.roll(self.I_histo, 1)\n",
    "        self.I_histo[0] = I_calculado\n",
    "        self.I_ante_1 = I_calculado\n",
    "        return I_calculado"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
