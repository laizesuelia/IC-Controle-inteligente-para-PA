import matplotlib.pyplot as plt
import numpy as np

def plotar_resultados(res, cfg, t, pref, map_ref, k_change, N_MC):
    """
    Gera e exibe as Figuras 1, 2 e 3 comparando os controladores.
    """
    # Funções auxiliares de plot
    def _vline(ax):
        ax.axvline(x=k_change, color='black', linestyle=':', linewidth=1.2,
                   label='Mudança paciente')

    def _envelope(ax, nome, arr_mc):
        cor   = res[nome]['cor']
        media = arr_mc.mean(axis=0)
        ax.plot(t, media, color=cor, linewidth=2, label=nome)

    # FIGURA 1 — MAP, Infusão, Atraso, a1, b1, bm
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Comparativo Monte Carlo — 5 Controladores  (N_MC={N_MC})',
                 fontsize=14, fontweight='bold')

    # MAP
    ax = axes[0, 0]
    for nome, _, _ in cfg:
        _envelope(ax, nome, res[nome]['all_map'])
    ax.plot(t, pref, 'r--', linewidth=1.5, label=f'Ref ({map_ref} mmHg)')
    _vline(ax)
    ax.set_ylim(50, 200);  ax.set_title('Pressão Arterial — mmHg')
    ax.legend(fontsize=8, loc='upper right');  ax.grid(True)

    # Infusão
    ax = axes[1, 0]
    for nome, _, _ in cfg:
        _envelope(ax, nome, res[nome]['all_i'])
    _vline(ax)
    ax.set_ylim(-10, 190);  ax.set_title('Taxa de Infusão de SNP — ml/h')
    ax.legend(fontsize=8, loc='upper right');  ax.grid(True)

    # Parâmetro a1
    ax = axes[0, 1]
    nome_ref = 'GPC Fixo'
    for nome, _, _ in cfg:
        ax.plot(t, res[nome]['teta_all'][:, 0], color=res[nome]['cor'],
                linewidth=1.2, alpha=0.85, label=nome)
    ax.plot(t, res[nome_ref]['teta_real'][:, 0], 'k--', linewidth=2, label='a1 Real')
    _vline(ax)
    ax.set_ylim(-1.5, 0.5);  ax.set_title('Parâmetro a1')
    ax.legend(fontsize=8);  ax.grid(True)

    # Parâmetro b1
    ax = axes[1, 1]
    for nome, _, _ in cfg:
        ax.plot(t, res[nome]['teta_all'][:, 1], color=res[nome]['cor'],
                linewidth=1.2, alpha=0.85, label=nome)
    ax.plot(t, res[nome_ref]['teta_real'][:, 1], 'k--', linewidth=2, label='b1 Real')
    _vline(ax)
    ax.set_ylim(-0.5, 1.5);  ax.set_title('Parâmetro b1')
    ax.legend(fontsize=8);  ax.grid(True)

    # Parâmetro bm
    ax = axes[2, 1]
    for nome, _, _ in cfg:
        ax.plot(t, res[nome]['teta_all'][:, 2], color=res[nome]['cor'],
                linewidth=1.2, alpha=0.85, label=nome)
    ax.plot(t, res[nome_ref]['teta_real'][:, 2], 'k--', linewidth=2, label='bm Real')
    _vline(ax)
    ax.set_ylim(-1.0, 1.5);  ax.set_title('Parâmetro bm')
    ax.legend(fontsize=8);  ax.grid(True)

    # Atraso
    ax = axes[2, 0]
    for nome, _, _ in cfg:
        ax.plot(t, res[nome]['delay_h'], color=res[nome]['cor'],
                linewidth=1.5, label=nome)
    ax.plot(t, res[nome_ref]['delay_real'], 'k--', linewidth=2, label='Atraso Real')
    _vline(ax)
    ax.set_ylim(1, 6);  ax.set_title('Atraso Estimado vs Real')
    ax.legend(fontsize=8);  ax.grid(True)

    plt.tight_layout()
    plt.show()

    # FIGURA 2 — Custo Acumulado
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Custo Acumulado por Modelo — última run', fontsize=13, fontweight='bold')
    labels_d = ['d=2', 'd=3', 'd=4', 'd=5']
    cores_d  = ['gray', 'blue', 'green', 'red']
    axs_flat = axes.flat
    for nome, _, _ in cfg:
        ax = next(axs_flat)
        for j, (lbl, cd) in enumerate(zip(labels_d, cores_d)):
            ax.semilogy(t, res[nome]['custos'][:, j], color=cd,
                        label=lbl, alpha=0.8, linewidth=1.3)
        ax.axvline(x=k_change, color='k', linestyle='--', linewidth=1)
        ax.set_title(nome, color=res[nome]['cor'], fontweight='bold', fontsize=11)
        ax.set_ylabel('Custo (log)');  ax.legend(fontsize=8);  ax.grid(True)
    next(axs_flat).set_visible(False) 
    plt.tight_layout()
    plt.show()

    # FIGURA 3 — Variância de b1 (Incerteza)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Variância do Parâmetro b1 (Incerteza) por Modelo — última run',
                 fontsize=13, fontweight='bold')
    labels_m = ['Atraso 2', 'Atraso 3', 'Atraso 4', 'Atraso 5']
    cores_m  = ['blue', 'orange', 'green', 'red']
    axs_flat = axes.flat
    for nome, _, _ in cfg:
        ax = next(axs_flat)
        for i, (lbl, cm) in enumerate(zip(labels_m, cores_m)):
            ax.plot(t, res[nome]['incerteza'][:, i], color=cm,
                    label=lbl, linewidth=1.5)
        ax.axvline(x=k_change, color='k', linestyle='--', linewidth=1)
        ax.set_title(nome, color=res[nome]['cor'], fontweight='bold', fontsize=11)
        ax.set_yscale('log');  ax.set_ylabel('Variância')
        ax.legend(fontsize=8);  ax.grid(True, which='both', ls='-', alpha=0.4)
    next(axs_flat).set_visible(False)
    plt.tight_layout()
    plt.show()