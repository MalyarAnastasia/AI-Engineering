import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    from src.models.train import load_data, train_model, train_clustering
    from src.service.app import app
    from src.data.preprocessing import EstatePreprocessing, MOSCOW_REGION_CODES
    assert app is not None
    assert "77" in MOSCOW_REGION_CODES


def test_config_exists():
    assert os.path.exists("configs/.env.example")


def test_requirements_exists():
    assert os.path.exists("requirements.txt")


def test_service_module_exists():
    assert os.path.exists("src/service/app.py")
    assert os.path.exists("src/models/train.py")
    assert os.path.exists("src/data/preprocessing.py")
    assert os.path.exists("src/data/prepare.py")
    assert os.path.exists("artifacts/feature_columns.joblib")


def test_moscow_region_filter():
    from src.data.preprocessing import EstatePreprocessing

    df = pd.DataFrame({
        "price": [5_000_000, 6_000_000, 7_000_000],
        "area": [50.0, 60.0, 70.0],
        "kitchen_area": [10.0, 12.0, 14.0],
        "rooms": [1, 2, 3],
        "geo_lat": [55.75, 55.76, 59.93],
        "geo_lon": [37.61, 37.62, 30.33],
        "region": ["77", "50", "78"],
    })
    processor = EstatePreprocessing.__new__(EstatePreprocessing)
    processor.df = df.copy()
    processor.filter_moscow_region()
    assert len(processor.df) == 2
    assert set(processor.df["region"]) == {"77", "50"}


def test_assign_cluster():
    from sklearn.cluster import KMeans
    from src.models.train import assign_cluster, CLUSTER_FEATURES

    df = pd.DataFrame({
        "price": [5_000_000, 6_000_000, 7_000_000, 8_000_000],
        "area": [50.0, 60.0, 70.0, 80.0],
        "kitchen_area": [10.0, 12.0, 14.0, 16.0],
        "geo_lat": [55.75, 55.76, 55.77, 55.78],
        "geo_lon": [37.61, 37.62, 37.63, 37.64],
    })
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(df[CLUSTER_FEATURES])

    cluster = assign_cluster(kmeans, 55.0, 10.0, 55.75, 37.61, 5_000_000)
    assert cluster in (0, 1)
