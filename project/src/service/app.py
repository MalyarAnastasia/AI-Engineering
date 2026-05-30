from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Optional
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real Estate Valuation API", version="1.0.0")

# Глобальные переменные для модели
model = None
kmeans = None

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
    """Загрузка модели и кластеризации"""
    global model, kmeans
    try:
        model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
        kmeans_path = os.getenv("KMEANS_PATH", "artifacts/kmeans.joblib")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}, using dummy model")
            
        if os.path.exists(kmeans_path):
            kmeans = joblib.load(kmeans_path)
            logger.info(f"KMeans loaded from {kmeans_path}")
        else:
            logger.warning(f"KMeans not found at {kmeans_path}")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@app.on_event("startup")
def startup_event():
    """Инициализация при запуске сервиса"""
    load_models()
    logger.info("Service started successfully")

@app.get("/health")
def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "kmeans_loaded": kmeans is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(features: PropertyFeatures):
    """Оценка стоимости недвижимости"""
    try:
        logger.info(f"Prediction request: area={features.area}, rooms={features.rooms}")
        
        # Подготовка признаков в том же порядке, что при обучении
        feature_values = np.array([[
            features.area,
            features.kitchen_area,
            features.rooms,
            features.geo_lat,
            features.geo_lon
        ]])
        
        # Предсказание
        if model is not None:
            predicted_price = model.predict(feature_values)[0]
        else:
            # Dummy предсказание если модель не загружена
            predicted_price = features.area * 150000  # 150k за м²
        
        # Расчёт дополнительных метрик
        price_per_meter = predicted_price / features.area
        
        # Инвестиционный рейтинг
        if predicted_price < features.area * 120000:
            investment_rating = "undervalued"
        elif predicted_price > features.area * 180000:
            investment_rating = "overvalued"
        else:
            investment_rating = "fair"
        
        # Кластеризация
        cluster = None
        if kmeans is not None:
            geo_features = np.array([[features.geo_lat, features.geo_lon, predicted_price]])
            cluster = int(kmeans.predict(geo_features)[0])
        
        logger.info(f"Prediction result: price={predicted_price:.0f}, rating={investment_rating}")
        
        return PredictionResponse(
            predicted_price=float(predicted_price),
            investment_rating=investment_rating,
            price_per_meter=float(price_per_meter),
            cluster=cluster
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
