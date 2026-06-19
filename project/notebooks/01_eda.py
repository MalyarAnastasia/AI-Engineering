#!/usr/bin/env python
# coding: utf-8

# # EDA - Разведочный анализ данных

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.notebook_helpers import get_project_paths

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

project_dir, data_path, plots_dir = get_project_paths()

# ## 1. Загрузка данных

df = pd.read_csv(data_path)

print(f"Размер датасета: {df.shape}")
print(f"\nТипы данных:")
print(df.dtypes)
print(f"\nПервые 5 строк:")
print(df.head())

# ## 2. Статистическое описание

print(df.describe())

# ## 3. Распределения признаков

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sns.histplot(df["area"], ax=axes[0, 0], kde=True)
axes[0, 0].set_title("Распределение площади")

sns.histplot(df["kitchen_area"], ax=axes[0, 1], kde=True)
axes[0, 1].set_title("Распределение площади кухни")

sns.histplot(df["rooms"], ax=axes[0, 2], kde=True)
axes[0, 2].set_title("Распределение комнат")

sns.histplot(df["price"], ax=axes[1, 0], kde=True)
axes[1, 0].set_title("Распределение цены")

sns.scatterplot(data=df.sample(min(10000, len(df))), x="area", y="price", ax=axes[1, 1])
axes[1, 1].set_title("Зависимость цены от площади")

sns.scatterplot(data=df.sample(min(10000, len(df))), x="geo_lat", y="geo_lon", ax=axes[1, 2])
axes[1, 2].set_title("Географическое распределение")

plt.tight_layout()
plt.savefig(f"{plots_dir}/01_distributions.png", dpi=300, bbox_inches="tight")
plt.show()

# ## 4. Корреляционный анализ

corr_matrix = df[["area", "kitchen_area", "rooms", "geo_lat", "geo_lon", "price"]].corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляционная матрица")
plt.savefig(f"{plots_dir}/01_correlation.png", dpi=300, bbox_inches="tight")
plt.show()

# ## 5. Выводы
# - После очистки и фильтра по Москве/МО: 641 362 записи
# - Сильная корреляция между площадью и ценой
# - Географические координаты показывают кластеризацию объектов
# - Распределения признаков близки к нормальным после очистки выбросов
