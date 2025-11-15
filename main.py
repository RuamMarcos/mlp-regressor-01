import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Carregar base de dados (California Housing)
# ==========================================
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="median_house_value")

# ==========================================
# 2. Dividir dados – Hold-out (80% treino, 20% teste)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 3. Normalização
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. Treinar MLP Regressor
# ==========================================
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# ==========================================
# 5. Previsões
# ==========================================
y_pred = mlp.predict(X_test_scaled)

# ==========================================
# 6. Métricas de Regressão
# ==========================================
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
R2 = r2_score(y_test, y_pred)

print("===== MÉTRICAS DE AVALIAÇÃO =====")
print(f"MAE:  {MAE:.4f}")
print(f"MSE:  {MSE:.4f}")
print(f"RMSE: {RMSE:.4f}")
print(f"MAPE: {MAPE:.2f}%")
print(f"R²:   {R2:.4f}\n\n")

df_compare = pd.DataFrame({
    "Real": y_test.values[:10],
    "Previsto": y_pred[:10]
})
print(df_compare)
