import numpy as np

class Identificador:
    def __init__(self, atrasos_possiveis=[2, 3, 4, 5], sigma_e=1.5, lambda_rls=0.995):
        
        self.atrasos = atrasos_possiveis
        self.sigma_e2 = sigma_e ** 2
        
        # λ FIXO
        self.lambda_rls = lambda_rls
        
        self.theta_hat = {d0: np.array([0.0, 0.1, 0.05]) for d0 in self.atrasos}
        self.P_cov = {d0: np.eye(3) * 10.0 for d0 in self.atrasos}
        self.erros_quadraticos = {d0: 0.0 for d0 in self.atrasos}

    def estimar(self, P_k, P_k_minus_1, I_hist):
        
        melhor_d0 = self.atrasos[0]
        menor_erro = float('inf')
        erro_opt = 0.0
        
        for d0 in self.atrasos:
            
            # Vetor de regressão
            phi = np.array([-P_k_minus_1, I_hist[d0], I_hist[2*d0]])
            
            # Predição
            P_hat = np.dot(self.theta_hat[d0], phi)
            erro = P_k - P_hat

            # Índice de adequabilidade
            self.erros_quadraticos[d0] = 0.98 * self.erros_quadraticos[d0] + erro**2
            
            if self.erros_quadraticos[d0] < menor_erro:
                menor_erro = self.erros_quadraticos[d0]
                melhor_d0 = d0
                erro_opt = erro
            
            # --- RLS com λ fixo ---
            P = self.P_cov[d0]
            
            num = np.dot(P, phi)
            den = max(self.lambda_rls + np.dot(phi, np.dot(P, phi)), 1e-6)
            
            K = num / den
            
            # Atualização dos parâmetros
            self.theta_hat[d0] = self.theta_hat[d0] + K * erro
            
            # Atualização da covariância
            self.P_cov[d0] = (P - np.outer(K, np.dot(phi, P))) / self.lambda_rls
        
        return melhor_d0, self.theta_hat[melhor_d0], self.P_cov[melhor_d0]