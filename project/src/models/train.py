import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data.preprocessing import EstatePreprocessing, MOSCOW_REGION_CODES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLUSTER_FEATURES = ["price", "area", "kitchen_area", "geo_lat", "geo_lon"]
MODEL_FEATURES = ["area", "kitchen_area", "rooms", "geo_lat", "geo_lon", "kmeans_cluster"]
REQUIRED_COLS = ["area", "kitchen_area", "rooms", "geo_lat", "geo_lon", "price", "region"]


def _maybe_sample(df: pd.DataFrame) -> pd.DataFrame:
    sample_size = os.getenv("TRAIN_SAMPLE_SIZE")
    if not sample_size:
        return df
    n = int(sample_size)
    if n < len(df):
        df = df.sample(n=n, random_state=42)
        logger.info("Using sample of %s rows (TRAIN_SAMPLE_SIZE)", n)
    return df


def load_data(data_path: str, raw_data_path: str | None = None) -> pd.DataFrame:
    """Загрузка и предобработка данных с фильтрацией по Москве и МО."""
    logger.info("Loading data from %s", data_path)

    if os.path.basename(data_path) == "all_v2.csv" or raw_data_path:
        source_path = raw_data_path or data_path
        processor = EstatePreprocessing(source_path)
        df = processor.run_full_pipeline()
    else:
        df = pd.read_csv(data_path)
        if "region" in df.columns:
            df["region"] = df["region"].astype(str)
            before = len(df)
            df = df[df["region"].isin(MOSCOW_REGION_CODES)]
            logger.info("Moscow/MO filter on cleaned file: %s -> %s rows", before, len(df))
        else:
            logger.warning("Column region missing in %s, Moscow filter skipped", data_path)

        missing = [c for c in REQUIRED_COLS if c != "region" and c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.dropna(subset=[c for c in REQUIRED_COLS if c in df.columns and c != "region"])

    df = df[[c for c in REQUIRED_COLS if c in df.columns]]
    df = _maybe_sample(df)
    logger.info("Data ready: %s rows", len(df))
    return df


def train_clustering(df: pd.DataFrame, n_clusters: int = 4) -> tuple[KMeans, pd.DataFrame]:
    """K-Means по цене, площади, кухне и координатам (как в real_estate_analytics)."""
    logger.info("Training K-Means clustering (k=%s)", n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["kmeans_cluster"] = kmeans.fit_predict(df[CLUSTER_FEATURES])

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(kmeans, "artifacts/kmeans.joblib")
    logger.info("K-Means saved to artifacts/kmeans.joblib")
    return kmeans, df


def assign_cluster(
    kmeans: KMeans,
    area: float,
    kitchen_area: float,
    geo_lat: float,
    geo_lon: float,
    price_hint: float,
) -> int:
    features = np.array([[price_hint, area, kitchen_area, geo_lat, geo_lon]])
    return int(kmeans.predict(features)[0])


def train_model(df: pd.DataFrame) -> tuple[CatBoostRegressor, dict]:
    """Обучение CatBoost с признаком kmeans_cluster."""
    logger.info("Training CatBoost model")

    X = df[MODEL_FEATURES].values
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    model = CatBoostRegressor(
        iterations=int(os.getenv("CATBOOST_ITERATIONS", "400")),
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    logger.info(
        "Model performance: MAE=%.0f, RMSE=%.0f, R²=%.3f",
        metrics["mae"], metrics["rmse"], metrics["r2"],
    )

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")
    joblib.dump(MODEL_FEATURES, "artifacts/feature_columns.joblib")
    logger.info("Model saved to artifacts/model.joblib")
    return model, metrics


def main():
    logger.info("Starting training pipeline")

    data_path = os.getenv("DATA_PATH", "data/cleaned_data.csv")
    raw_data_path = os.getenv("RAW_DATA_PATH")

    df = load_data(data_path, raw_data_path=raw_data_path)
    kmeans, df = train_clustering(df)
    model, metrics = train_model(df)

    logger.info("Training pipeline completed successfully")
    logger.info("Final metrics: %s", metrics)
    logger.info("Rows used: %s", len(df))


if __name__ == "__main__":
    main()
