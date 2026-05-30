#!/usr/bin/env python
# coding: utf-8

# # Кластеризация и сегментация рынка

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))

sns.set_style("whitegrid")

# Создание папки для графиков
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(notebook_dir)
plots_dir = os.path.join(project_dir, "artifacts", "plots")
os.makedirs(plots_dir, exist_ok=True)


# ## 1. Загрузка данных

# In[2]:


import os
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(notebook_dir)
data_path = os.path.join(project_dir, "data", "cleaned_data.csv")

df = pd.read_csv(data_path)
print(f"Размер датасета: {df.shape}")
df.head()


# ## 2. Подготовка данных для кластеризации

# In[3]:


# Используем выборку для ускорения
sample_df = df.sample(100000, random_state=42)
X = sample_df[['geo_lat', 'geo_lon', 'price']]

print(f"Размер выборки: {X.shape}")


# ## 3. Метод локтя для определения оптимального k

# In[4]:


inertia_values = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, marker='o', linestyle='--')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Инерция')
plt.xticks(k_range)
plt.grid(True)
plt.savefig(os.path.join(plots_dir, '02_elbow_method.png'), dpi=300, bbox_inches='tight')
plt.show()


# ## 4. Кластеризация с k=4

# In[5]:


kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
sample_df['cluster'] = kmeans.fit_predict(X)

silhouette = silhouette_score(X, sample_df['cluster'])
print(f"Silhouette Score: {silhouette:.3f}")


# ## 5. Анализ кластеров

# In[6]:


cluster_stats = sample_df.groupby('cluster').agg({
    'price': ['mean', 'count'],
    'area': 'mean',
    'rooms': 'mean'
}).round(2)

print(cluster_stats)


# ## 6. Визуализация кластеров

# In[7]:


plt.figure(figsize=(12, 8))
scatter = plt.scatter(sample_df['geo_lon'], sample_df['geo_lat'], c=sample_df['cluster'], cmap='viridis', s=1, alpha=0.5)
plt.colorbar(scatter, label='Кластер')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title('Географическая кластеризация')
plt.savefig(os.path.join(plots_dir, '02_clustering_map.png'), dpi=300, bbox_inches='tight')
plt.show()

