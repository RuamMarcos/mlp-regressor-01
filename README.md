# Previsão de Preços de Casas com MLP

Treinamento de uma rede neural MLP aplicada a um problema real de regressão: previsão do valor mediano de casas na Califórnia.  
O projeto realiza carregamento dos dados, divisão por **hold-out**, normalização, treinamento de `MLPRegressor` e avaliação com métricas clássicas de regressão: **MAE**, **MSE**, **RMSE**, **MAPE** e **R²**.

---

## Estrutura do repositório

- `main.py` — script principal: prepara os dados, treina o modelo, gera previsões e imprime/mostra métricas e gráficos.
- `README.md` — este arquivo.
- 
---

## Requisitos

- Python 3.8+
- Bibliotecas (instale com `pip`):

```bash
pip install numpy pandas scikit-learn matplotlib
