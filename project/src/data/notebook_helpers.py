import os
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from src.models.train import CLUSTER_FEATURES, MODEL_FEATURES

FEATURE_COLUMNS = [c for c in MODEL_FEATURES if c != "kmeans_cluster"]


def find_project_dir() -> Path:
    """Корень project/ — работает из Jupyter (notebooks/) и при запуске .py."""
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parents[2],
    ]
    for path in candidates:
        if (path / "src" / "models" / "train.py").exists():
            return path
    raise FileNotFoundError(
        "Не найден корень project/. Запускайте ноутбук из project/ или project/notebooks/."
    )


def get_project_paths():
    project_dir = find_project_dir()
    data_path = project_dir / "data" / "cleaned_data.csv"
    plots_dir = project_dir / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return str(project_dir), str(data_path), str(plots_dir)


def load_sample(sample_size: int = 100_000, random_state: int = 42) -> pd.DataFrame:
    _, data_path, _ = get_project_paths()
    df = pd.read_csv(data_path)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)
    return df.reset_index(drop=True)


def add_kmeans_cluster(df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, KMeans]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    result = df.copy()
    result["kmeans_cluster"] = kmeans.fit_predict(result[CLUSTER_FEATURES])
    return result, kmeans


def get_xy(df: pd.DataFrame, with_cluster: bool = True):
    feature_cols = MODEL_FEATURES if with_cluster else FEATURE_COLUMNS
    X = df[feature_cols]
    y = df["price"]
    return X, y
