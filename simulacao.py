import numpy as np
import random
from planta.paciente import Paciente
from identificador.rls import IdentificadorMultiModelo
from controladores import (
    ControladorGPC, ControladorGPCAdaptativo, ControladorIDC, 
    Controlador_PID_Adaptativo, Controlador_Perturbacao, ControladorGPC_Restricao
)

def run_simulacao(passos_totais=500, sigma_e=1.5, MAP_desejado=100.0, semente=120):
    np.random.seed(semente)
    vetor_ruido_fixo = np.random.normal(0, sigma_e, passos_totais)
    P_ref = 150.0 - MAP_desejado
    passos_prbs = 15
    nomes_modelos = ['IDC', 'GPC_Fixo', 'GPC_Adaptativo', 'PID_Adaptativo', 'Controlador_Perturbacao', 'Controlador_Restricao']
    
    pacientes = {nome: Paciente(vetor_ruido_fixo) for nome in nomes_modelos}
    ids = {nome: IdentificadorMultiModelo(sigma_e=sigma_e) for nome in nomes_modelos}
    controladores = {
        'IDC': ControladorIDC(lambda_idc=0.1),
        'GPC_Fixo': ControladorGPC(rho=0.5),
        'GPC_Adaptativo': ControladorGPCAdaptativo(alpha_rho=5.0, beta_rho=0.90, rho_min=0.5),
        'PID_Adaptativo': Controlador_PID_Adaptativo(),
        'Controlador_Perturbacao': Controlador_Perturbacao(amplitude=8.0, taxa_decaimento=0.997),
        'Controlador_Restricao': ControladorGPC_Restricao(rho=1.5, I_min=0.0, I_max=180.0)
    }
    
    resultados = {nome: {
        'historico_MAP': [], 'historico_I': [], 'hist_rho': [],
        'hist_a1': [], 'hist_b1': [], 'hist_bm1': [],
        'real_a1': [], 'real_b1': [], 'real_bm1': [],
        'd0_hist': [], 'd0_real': [], 'hist_var_b1': [],
        'hist_custo': {2: [], 3: [], 4: [], 5: []}
    } for nome in nomes_modelos}
    
    for k in range(passos_totais):
        for p in pacientes.values():
            if 200 <= k < 250:
                p.b1 += (0.300 - 0.187) / 50.0
                p.d = 2
            elif k == 350:
                p.d = 4
                
        for nome in nomes_modelos:
            p = pacientes[nome]
            ident = ids[nome]
            ctrl = controladores[nome]
            res = resultados[nome]
            
            melhor_d0 = min(ident.erros_quadraticos, key=ident.erros_quadraticos.get)
            theta_hat = ident.theta_hat[melhor_d0]
            P_cov = ident.P_cov[melhor_d0]
            
            if k < passos_prbs:
                I_k = 10.0 if np.random.rand() > 0.5 else 0.0
                if nome == 'GPC_Adaptativo':
                    res['hist_rho'].append(ctrl.rho)
            else:
                if nome == 'PID_Adaptativo':
                    MAP_atual = p.P0 - p.P_hist[0]
                    I_k = ctrl.calcular_controle(MAP_desejado, MAP_atual, theta_hat[1])
                else:
                    K0 = ctrl.calcular_KO(theta_hat, melhor_d0, p.P_hist[0], p.I_hist)
                    if nome == 'IDC':
                        I_k = ctrl.calcular_controle(K0, theta_hat, P_cov, P_ref)
                    elif nome == 'GPC_Fixo':
                        I_k = ctrl.calcular_controle(K0, theta_hat, P_ref)
                    elif nome == 'GPC_Adaptativo':
                        I_k = ctrl.calcular_controle(K0, theta_hat, P_ref)
                        res['hist_rho'].append(ctrl.rho)
                    elif nome == 'Controlador_Perturbacao':
                        I_k = ctrl.calcular_controle(K0, theta_hat, P_ref)
                    elif nome == 'Controlador_Restricao':
                        MAP_atual = p.P0 - p.P_hist[0]
                        I_k = ctrl.calcular_controle(K0, theta_hat, P_ref, MAP_atual)
                        
            MAP_medida, P_k_medida = p.atualizar(I_k, k)
            res['historico_I'].append(I_k)
            res['historico_MAP'].append(MAP_medida)
            
            P_ant = p.P_hist[1] if k > 0 else 0.0
            if nome == 'GPC_Adaptativo':
                phi_pred = np.array([-P_ant, p.I_hist[melhor_d0], p.I_hist[2*melhor_d0]])
                erro_preditivo = P_k_medida - np.dot(theta_hat, phi_pred)
                ctrl.atualizar_rho(erro_preditivo)
                
            d_opt, th_opt, cov_opt = ident.estimar(P_k_medida, P_ant, p.I_hist)
            
            res['hist_a1'].append(th_opt[0])
            res['hist_b1'].append(th_opt[1])
            res['hist_bm1'].append(th_opt[2])
            res['real_a1'].append(p.a1)
            res['real_b1'].append(p.b1)
            res['real_bm1'].append(p.bm1)
            res['d0_hist'].append(d_opt)
            res['d0_real'].append(p.d)
            res['hist_var_b1'].append(cov_opt[1, 1])
            
            for d in ident.atrasos:
                res['hist_custo'][d].append(ident.erros_quadraticos[d])
                
    return resultados