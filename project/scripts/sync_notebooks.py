"""Приводит .ipynb в соответствие с рабочими .py-скриптами."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"

BOOTSTRAP = """import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _find_project():
    for p in [Path.cwd(), Path.cwd().parent]:
        if (p / "src" / "models" / "train.py").exists():
            return p
    raise FileNotFoundError("Запускайте ноутбук из project/ или project/notebooks/")

PROJECT = _find_project()
sys.path.insert(0, str(PROJECT))
from src.data.notebook_helpers import get_project_paths

project_dir, data_path, plots_dir = get_project_paths()
"""

DATA_CELLS = {
    "01_eda.ipynb": """df = pd.read_csv(data_path)

print(f"Размер датасета: {df.shape}")
print(df.dtypes)
df.head()
""",
    "02_clustering.ipynb": """import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.data.notebook_helpers import load_sample

sns.set_style("whitegrid")
sample_df = load_sample(100000, random_state=42)
X_cluster = sample_df[["price", "area", "kitchen_area", "geo_lat", "geo_lon"]]
print(f"Размер выборки: {X_cluster.shape}")
""",
    "03_baselines.ipynb": """from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.notebook_helpers import load_sample, get_xy

df = load_sample(100000, random_state=42)
X, y = get_xy(df, with_cluster=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
""",
    "04_advanced_models.ipynb": """from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.notebook_helpers import load_sample, add_kmeans_cluster, get_xy

df = load_sample(100000, random_state=42)
df, _ = add_kmeans_cluster(df)
X, y = get_xy(df, with_cluster=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""",
    "05_model_selection.ipynb": """import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.notebook_helpers import load_sample, add_kmeans_cluster, get_xy

sns.set_style("whitegrid")
df = load_sample(100000, random_state=42)
df, _ = add_kmeans_cluster(df)
X, y = get_xy(df, with_cluster=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""",
}


def to_source(code: str) -> list[str]:
    lines = code.strip().split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])


def fix_notebook(name: str):
    path = NOTEBOOKS / name
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    code_indices = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    nb["cells"][code_indices[0]]["source"] = to_source(BOOTSTRAP)
    nb["cells"][code_indices[0]]["outputs"] = []
    nb["cells"][code_indices[0]]["execution_count"] = None

    if name in DATA_CELLS and len(code_indices) > 1:
        nb["cells"][code_indices[1]]["source"] = to_source(DATA_CELLS[name])
        nb["cells"][code_indices[1]]["outputs"] = []
        nb["cells"][code_indices[1]]["execution_count"] = None

    # Fix R² -> R2 in plot cells for Windows
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"]).replace("R²", "R2")
            cell["source"] = to_source(src) if src.strip() else cell["source"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("fixed", name)


for nb_name in DATA_CELLS:
    fix_notebook(nb_name)
