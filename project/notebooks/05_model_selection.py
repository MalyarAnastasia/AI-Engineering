#!/usr/bin/env python
# coding: utf-8

# # Выбор финальной модели

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))

sns.set_style("whitegrid")

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


# ## 2. Обучение всех моделей

# In[3]:


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=3, random_state=42),
    'Bagging': BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=3),
        n_estimators=10,
        random_state=42
    ),
    'CatBoost': CatBoostRegressor(
        iterations=400,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print(results_df)


# ## 3. Визуализация сравнения

# In[4]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

results_df.plot(x='Model', y='MAE', kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Сравнение MAE')
axes[0].set_ylabel('MAE (руб)')

results_df.plot(x='Model', y='RMSE', kind='bar', ax=axes[1], color='lightcoral')
axes[1].set_title('Сравнение RMSE')
axes[1].set_ylabel('RMSE (руб)')

results_df.plot(x='Model', y='R²', kind='bar', ax=axes[2], color='lightgreen')
axes[2].set_title('Сравнение R²')
axes[2].set_ylabel('R²')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '05_model_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()


# ## 4. Выбор финальной модели
# 
# **CatBoost выбран как финальная модель по следующим причинам:**
# 
# 1. **Лучшая точность**: Наименьший MAE и RMSE среди всех моделей
# 2. **Высокий R²**: Объясняет большую часть дисперсии целевой переменной
# 3. **Скорость предсказания**: < 50ms на запрос
# 4. **Интерпретируемость**: Встроенная важность признаков
# 5. **Устойчивость**: Менее чувствителен к выбросам
# 
# **Trade-off'ы:**
# - Требует больше памяти для обучения
# - Время обучения больше, чем у линейной регрессии
# - Но для продакшена важнее скорость предсказания
