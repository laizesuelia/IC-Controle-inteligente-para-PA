

def exibir_tabela_desempenho(res, cfg, map_ref, N, N_MC):
    """
    Calcula V_bar, sigma_v e pb1_bar e imprime a tabela de desempenho.
    """
    INICIO = 40  # passos descartados (fase de inicialização PRBS)

    print("\n" + "="*85)
    print(f"  TABELA DE DESEMPENHO — Monte Carlo  (N_MC={N_MC}, passos avaliados={N-INICIO})")
    print("="*85)
    print(f"  {'Controlador':<22} | {'V̄ (Perda Média)':<18} | {'σᵥ (Desvio Padrão)':<20} | {'p̄_b1 (Var. Média b1)'}")
    print("-" * 85)

    for nome, _, _ in cfg:
        erros_mc = ((res[nome]['all_map'][:, INICIO:] - map_ref) ** 2).mean(axis=1)
        V_bar    = erros_mc.mean()               
        
        std_mc   = (res[nome]['all_map'][:, INICIO:] - map_ref).std(axis=1)
        sigma_v  = std_mc.mean()

        pb1_bar  = res[nome]['incerteza'][:, :].mean()
        
        print(f"  {nome:<22} | {V_bar:<18.4f} | {sigma_v:<20.4f} | {pb1_bar:.6f}")

    print("="*85)
    print("  V̄: menor = melhor rastreamento")
    print("  σᵥ: menor = mais robusto ao ruído")
    print("  p̄_b1: menor = RLS mais confiante nos parâmetros")
    print("="*85 + "\n")