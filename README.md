#  Controle Inteligente para Pressão Arterial (PA)

Projeto de simulação e comparação de controladores para regulação da pressão arterial (MAP), utilizando técnicas de controle preditivo e identificação adaptativa.

---

##  Objetivo

Avaliar o desempenho de diferentes controladores diante de:
incertezas no modelo
variações nos parâmetros do paciente
presença de ruído

---

##  Componentes

- **Paciente (planta):** simulação da dinâmica da pressão arterial  
- **Identificador (RLS):** estima parâmetros e incertezas  
- **Controladores:**
  - GPC Fixo
  - GPC Adaptativo
  - Perturbação
  - Restrições
  - IDC (Inovações)

cd IC-Controle-inteligente-para-PA
pip install numpy matplotlib

No Python:

from simulacao import run_simulation
