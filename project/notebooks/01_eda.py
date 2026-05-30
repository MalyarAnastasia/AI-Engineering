#!/usr/bin/env python
# coding: utf-8

# # EDA - Разведочный анализ данных

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

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
print(f"\nТипы данных:")
print(df.dtypes)
print(f"\nПервые 5 строк:")
df.head()


# ## 2. Статистическое описание

# In[3]:


df.describe()


# ## 3. Распределения признаков

# In[4]:


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sns.histplot(df['area'], ax=axes[0, 0], kde=True)
axes[0, 0].set_title('Распределение площади')

sns.histplot(df['kitchen_area'], ax=axes[0, 1], kde=True)
axes[0, 1].set_title('Распределение площади кухни')

sns.histplot(df['rooms'], ax=axes[0, 2], kde=True)
axes[0, 2].set_title('Распределение комнат')

sns.histplot(df['price'], ax=axes[1, 0], kde=True)
axes[1, 0].set_title('Распределение цены')

sns.scatterplot(data=df.sample(10000), x='area', y='price', ax=axes[1, 1])
axes[1, 1].set_title('Зависимость цены от площади')

sns.scatterplot(data=df.sample(10000), x='geo_lat', y='geo_lon', ax=axes[1, 2])
axes[1, 2].set_title('Географическое распределение')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_distributions.png'), dpi=300, bbox_inches='tight')
plt.show()


# ## 4. Корреляционный анализ

# In[5]:


corr_matrix = df[['area', 'kitchen_area', 'rooms', 'geo_lat', 'geo_lon', 'price']].corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица')
plt.savefig(os.path.join(plots_dir, '01_correlation.png'), dpi=300, bbox_inches='tight')
plt.show()


# ## 5. Выводы
# 
# - Датасет содержит 4.5M+ записей недвижимости
# - Сильная корреляция между площадью и ценой
# - Географические координаты показывают кластеризацию объектов
# - Распределения признаков близки к нормальным после очистки выбросов
