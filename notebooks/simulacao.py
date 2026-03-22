import numpy as np
import random

# Importando os Modelos, Identificador e Controladores
from planta.paciente import Paciente
from identificador.rls import Identificador
from controladores.gpc_fixo import Controlador_GPC_Fixo
from controladores.gpc_adaptativo import Controlador_GPC_Adaptativo
from controladores.pertubacao import Controlador_32_Perturbacao
from controladores.restricoes import Controlador_33_Restricoes
from controladores.idc import Controlador_35_Inovacoes

# Importando nossos utilitários visuais e de métricas
from utils.graficos import plotar_resultados
from utils.metricas import exibir_tabela_desempenho

def run_simulation(ro, lambda_val, P0, map_ref, sigma,
                   a1_ini, b1_ini, bm_ini, d_ini, m_ini,
                   k_change, a1_new, b1_new, bm_new, d_new, m_new,
                   N_MC=50):

    N = 500
    yref = P0 - map_ref
    d_vetor = [2, 3, 4, 5]
    m_vetor = [2, 3, 4, 5]
    teta_real_ini = [a1_ini, b1_ini, bm_ini]
    t = np.arange(N)
    pref = np.full(N, map_ref)

    cfg = [
        ('GPC Fixo',        'seagreen',     lambda: Controlador_GPC_Fixo([a1_ini, b1_ini, bm_ini], d_ini, m_ini, ro)),
        ('GPC Adaptativo',  'mediumpurple', lambda: Controlador_GPC_Adaptativo(ro)),
        ('3.2 Perturbação', 'royalblue',    lambda: Controlador_32_Perturbacao(amplitude=8.0, taxa_decaimento=0.997, seed=None)),
        ('3.3 Restrições',  'crimson',      lambda: Controlador_33_Restricoes(u_lim=15.0)),
        ('3.5 IDC',         'darkorange',   lambda: Controlador_35_Inovacoes(lambda_idc=0.5)),
    ]

    res = {}
    for nome, cor, _ in cfg:
        res[nome] = dict(
            cor          = cor,
            all_map      = np.zeros((N_MC, N)),
            all_i        = np.zeros((N_MC, N)),
            delay_h      = np.zeros(N),
            delay_real   = np.zeros(N),
            teta_all     = np.zeros((N, 12)),
            teta_real    = np.zeros((N, 3)),
            custos       = np.zeros((N, 4)),
            incerteza    = np.zeros((N, 4)),
        )

    # Loop Monte Carlo (mantém a mesma lógica da sua simulação)
    for nome, _, make_ctrl in cfg:
        r = res[nome]
        for mc in range(N_MC):
            paciente = Paciente(teta_real_ini, d_ini, m_ini, P0, sigma)
            identificador = Identificador(lambda_val, d_vetor, m_vetor, N)
            ctrl = make_ctrl()

            map_h, i_h = np.zeros(N), np.zeros(N)
            d_h, dr_h  = np.zeros(N), np.zeros(N)
            ta_h, tr_h = np.zeros((N, 12)), np.zeros((N, 3))
            cu_h, inc_h = np.zeros((N, 4)), np.zeros((N, 4))

            I_pac = 0.0
            RAMPA = 50 

            for k in range(N):
                if k_change <= k < k_change + RAMPA:
                    alpha = (k - k_change) / RAMPA
                    paciente.teta_real = np.array([
                        a1_ini + alpha * (a1_new - a1_ini),
                        b1_ini + alpha * (b1_new - b1_ini),
                        bm_ini + alpha * (bm_new - bm_ini),
                    ])
                elif k == k_change + RAMPA:
                    paciente.teta_real = np.array([a1_new, b1_new, bm_new])
                    paciente.d = int(d_new)
                    paciente.m = int(m_new)
                    identificador.Pant = np.kron(np.eye(4), 10 * np.eye(3))

                tr_h[k] = paciente.teta_real
                dr_h[k] = paciente.d

                MAP, P_med, _ = paciente.step(I_pac)
                teta_est, d_est, m_est, vetor_inc, custos_step, P_melhor = identificador.RLS(P_med, I_pac)
                var_b1 = vetor_inc[identificador.ind_melhor]
                I_atual = ctrl.calcular_controle(yref, P_med, teta_est, d_est, m_est, var_b1, P_melhor)

                I_pac = I_atual
                map_h[k], i_h[k] = MAP, I_atual
                d_h[k], ta_h[k] = d_est, identificador.teta_ea
                cu_h[k], inc_h[k] = custos_step, vetor_inc

            r['all_map'][mc] = map_h
            r['all_i'][mc] = i_h
            r['delay_h'] = d_h
            r['delay_real'] = dr_h
            r['teta_all'] = ta_h
            r['teta_real'] = tr_h
            r['custos'] = cu_h
            r['incerteza'] = inc_h

    # ======= A MAGIA ACONTECE AQUI =======
    # Todo aquele código enorme de plotagem e print virou apenas 2 linhas!
    plotar_resultados(res, cfg, t, pref, map_ref, k_change, N_MC)
    exibir_tabela_desempenho(res, cfg, map_ref, N, N_MC)