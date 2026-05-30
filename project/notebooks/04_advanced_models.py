#!/usr/bin/env python
# coding: utf-8

# # Продвинутые модели (CatBoost)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
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


# ## 2. Bagging Regressor

# In[3]:


bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=10,
    random_state=42
)
bagging.fit(X_train, y_train)

y_pred_bag = bagging.predict(X_test)

mae_bag = mean_absolute_error(y_test, y_pred_bag)
rmse_bag = np.sqrt(mean_squared_error(y_test, y_pred_bag))
r2_bag = r2_score(y_test, y_pred_bag)

print(f"Bagging Regressor:")
print(f"  MAE: {mae_bag:.0f}")
print(f"  RMSE: {rmse_bag:.0f}")
print(f"  R²: {r2_bag:.3f}")


# ## 3. CatBoost Regressor

# In[4]:


catboost = CatBoostRegressor(
    iterations=400,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=False
)
catboost.fit(X_train, y_train)

y_pred_cb = catboost.predict(X_test)

mae_cb = mean_absolute_error(y_test, y_pred_cb)
rmse_cb = np.sqrt(mean_squared_error(y_test, y_pred_cb))
r2_cb = r2_score(y_test, y_pred_cb)

print(f"CatBoost:")
print(f"  MAE: {mae_cb:.0f}")
print(f"  RMSE: {rmse_cb:.0f}")
print(f"  R²: {r2_cb:.3f}")


# ## 4. Важность признаков

# In[5]:


feature_importance = catboost.get_feature_importance()
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Важность')
plt.title('Важность признаков (CatBoost)')
plt.savefig(os.path.join(plots_dir, '04_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.show()

