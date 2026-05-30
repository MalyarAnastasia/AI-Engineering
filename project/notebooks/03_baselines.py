#!/usr/bin/env python
# coding: utf-8

# # Baseline модели

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))

# Создание папки для графиков
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(notebook_dir)
plots_dir = os.path.join(project_dir, "artifacts", "plots")
os.makedirs(plots_dir, exist_ok=True)


# ## 1. Подготовка данных

# In[2]:


import os
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(notebook_dir)
data_path = os.path.join(project_dir, "data", "cleaned_data.csv")

df = pd.read_csv(data_path)

# Используем выборку для ускорения
sample_df = df.sample(100000, random_state=42)

X = sample_df[['area', 'kitchen_area', 'rooms', 'geo_lat', 'geo_lon']]
y = sample_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")


# ## 2. Linear Regression

# In[3]:


lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression:")
print(f"  MAE: {mae_lr:.0f}")
print(f"  RMSE: {rmse_lr:.0f}")
print(f"  R²: {r2_lr:.3f}")


# ## 3. Decision Tree

# In[4]:


dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree:")
print(f"  MAE: {mae_dt:.0f}")
print(f"  RMSE: {rmse_dt:.0f}")
print(f"  R²: {r2_dt:.3f}")


# ## 4. Сравнение моделей

# In[5]:


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree'],
    'MAE': [mae_lr, mae_dt],
    'RMSE': [rmse_lr, rmse_dt],
    'R²': [r2_lr, r2_dt]
})

print(results)

# Визуализация результатов
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

results.plot(x='Model', y='MAE', kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Сравнение MAE')
axes[0].set_ylabel('MAE (руб)')

results.plot(x='Model', y='RMSE', kind='bar', ax=axes[1], color='lightcoral')
axes[1].set_title('Сравнение RMSE')
axes[1].set_ylabel('RMSE (руб)')

results.plot(x='Model', y='R²', kind='bar', ax=axes[2], color='lightgreen')
axes[2].set_title('Сравнение R²')
axes[2].set_ylabel('R²')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_baseline_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

