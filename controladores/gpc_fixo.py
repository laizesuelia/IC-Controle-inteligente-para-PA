{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a6e3675c",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Controlador_GPC_Fixo:\n",
    "    def __init__(self, teta_fixo, d_fixo, m_fixo, ro=1.5):\n",
    "        self.a1, self.b1, self.bm = teta_fixo\n",
    "        self.d  = int(d_fixo)\n",
    "        self.m  = int(m_fixo)\n",
    "        self.ro = ro\n",
    "        self.k  = 0\n",
    "        self.I_ante_1 = 0.0\n",
    "        self.I_histo  = np.zeros(100)\n",
    "\n",
    "    def calcular_controle(self, P_ref, P_k_medido, teta_est, d_est, m_est, inovacao_k, P_cov):\n",
    "        # P_cov ignorado — parâmetros congelados, não usa RLS nem covariância\n",
    "        self.k += 1\n",
    "        b1_ctrl = max(self.b1, 0.01)\n",
    "\n",
    "        if self.k <= 10:\n",
    "            PRBS = 10.0\n",
    "            if self.k > 5:\n",
    "                PRBS = 60 - (60 / 3.8) * abs(self.b1)\n",
    "            I_calculado = PRBS if random.random() > 0.5 else 0.0\n",
    "            I_calculado = np.clip(I_calculado, 0, 180)\n",
    "            self.I_histo = np.roll(self.I_histo, 1)\n",
    "            self.I_histo[0] = I_calculado\n",
    "            self.I_ante_1 = I_calculado\n",
    "            return I_calculado\n",
    "\n",
    "        k0 = (-self.a1) ** self.d * P_k_medido\n",
    "        for i in range(1, self.d):\n",
    "            t1 = b1_ctrl * self.I_histo[i]          if i          < len(self.I_histo) else 0.0\n",
    "            t2 = self.bm * self.I_histo[self.m + i] if self.m + i < len(self.I_histo) else 0.0\n",
    "            k0 += (-self.a1) ** i * (t1 + t2)\n",
    "        if self.m < len(self.I_histo):\n",
    "            k0 += self.bm * self.I_histo[self.m]\n",
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
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
