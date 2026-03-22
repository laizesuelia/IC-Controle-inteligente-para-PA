{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0310229",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "class Paciente:\n",
    "    def __init__(self, teta_real, P0, vetor_ruido):\n",
    "        self.teta_real = np.array(teta_real)\n",
    "        self.P0 = P0\n",
    "        self.ruido = vetor_ruido\n",
    "\n",
    "        self.P_hist = np.zeros(100)\n",
    "        self.I_hist = np.zeros(100)\n",
    "\n",
    "    def step(self, I_k, k):\n",
    "        self.I_hist = np.roll(self.I_hist, 1)\n",
    "        self.I_hist[0] = I_k\n",
    "\n",
    "        a1, b1, bm = self.teta_real\n",
    "\n",
    "        P_k = -a1 * self.P_hist[0] + b1 * self.I_hist[1] + bm * self.I_hist[2] + self.ruido[k]\n",
    "\n",
    "        self.P_hist = np.roll(self.P_hist, 1)\n",
    "        self.P_hist[0] = P_k\n",
    "\n",
    "        MAP = self.P0 - P_k\n",
    "        return MAP, P_k"
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
