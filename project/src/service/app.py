from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
from typing import Optional
import os

from src.models.train import assign_cluster, MODEL_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real Estate Valuation API", version="1.0.0")

model = None
kmeans = None
feature_columns = MODEL_FEATURES

PRICE_HINT_PER_SQM = float(os.getenv("PRICE_HINT_PER_SQM", "150000"))


class PropertyFeatures(BaseModel):
    area: float = Field(..., gt=0, description="Общая площадь в м²")
    kitchen_area: float = Field(..., gt=0, description="Площадь кухни в м²")
    rooms: int = Field(..., ge=0, description="Количество комнат (0 = студия)")
    geo_lat: float = Field(..., ge=55.0, le=56.0, description="Широта")
    geo_lon: float = Field(..., ge=37.0, le=38.0, description="Долгота")


class PredictionResponse(BaseModel):
    predicted_price: float
    investment_rating: str
    price_per_meter: float
    cluster: Optional[int] = None


def load_models():
    global model, kmeans, feature_columns
    try:
        model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
        kmeans_path = os.getenv("KMEANS_PATH", "artifacts/kmeans.joblib")
        features_path = os.getenv("FEATURE_COLUMNS_PATH", "artifacts/feature_columns.joblib")

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)
        else:
            logger.warning("Model not found at %s", model_path)

        if os.path.exists(kmeans_path):
            kmeans = joblib.load(kmeans_path)
            logger.info("KMeans loaded from %s", kmeans_path)
        else:
            logger.warning("KMeans not found at %s", kmeans_path)

        if os.path.exists(features_path):
            feature_columns = joblib.load(features_path)
            logger.info("Feature columns loaded: %s", feature_columns)

    except Exception as e:
        logger.error("Error loading models: %s", e)


@app.on_event("startup")
def startup_event():
    load_models()
    logger.info("Service started successfully")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "kmeans_loaded": kmeans is not None,
    }


def _predict_price(features: PropertyFeatures, cluster: int) -> float:
    if model is None:
        return features.area * PRICE_HINT_PER_SQM

    feature_values = np.array([[
        features.area,
        features.kitchen_area,
        features.rooms,
        features.geo_lat,
        features.geo_lon,
        cluster,
    ]])
    return float(model.predict(feature_values)[0])


def _investment_rating(predicted_price: float, area: float) -> str:
    price_per_meter = predicted_price / area
    if price_per_meter < 120000:
        return "undervalued"
    if price_per_meter > 180000:
        return "overvalued"
    return "fair"


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PropertyFeatures):
    try:
        logger.info(
            "Prediction request: area=%s, rooms=%s, lat=%s, lon=%s",
            features.area, features.rooms, features.geo_lat, features.geo_lon,
        )

        cluster = 0
        if kmeans is not None:
            price_hint = features.area * PRICE_HINT_PER_SQM
            cluster = assign_cluster(
                kmeans,
                features.area,
                features.kitchen_area,
                features.geo_lat,
                features.geo_lon,
                price_hint,
            )

        predicted_price = _predict_price(features, cluster)

        if kmeans is not None:
            cluster = assign_cluster(
                kmeans,
                features.area,
                features.kitchen_area,
                features.geo_lat,
                features.geo_lon,
                predicted_price,
            )
            predicted_price = _predict_price(features, cluster)

        price_per_meter = predicted_price / features.area
        investment_rating = _investment_rating(predicted_price, features.area)

        logger.info(
            "Prediction result: price=%.0f, rating=%s, cluster=%s",
            predicted_price, investment_rating, cluster,
        )

        return PredictionResponse(
            predicted_price=predicted_price,
            investment_rating=investment_rating,
            price_per_meter=price_per_meter,
            cluster=cluster,
        )

    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
