import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Загрузка данных"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Базовая очистка
    required_cols = ['area', 'kitchen_area', 'rooms', 'geo_lat', 'geo_lon', 'price']
    df = df[required_cols].dropna()
    
    # Фильтрация
    df = df[df['area'] > 0]
    df = df[df['kitchen_area'] > 0]
    df = df[df['price'] > 0]
    df = df[df['rooms'] >= 0]
    
    # Удаление выбросов (IQR метод)
    for col in ['area', 'kitchen_area', 'price']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    logger.info(f"Data loaded: {len(df)} rows")
    return df

def train_clustering(df):
    """Обучение K-Means кластеризации"""
    logger.info("Training K-Means clustering")
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(df[['geo_lat', 'geo_lon', 'price']])
    
    # Сохранение кластеризации
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(kmeans, 'artifacts/kmeans.joblib')
    logger.info(f"K-Means saved to artifacts/kmeans.joblib")
    
    return kmeans

def train_model(df):
    """Обучение CatBoost модели"""
    logger.info("Training CatBoost model")
    
    # Подготовка данных - используем numpy массивы
    feature_cols = ['area', 'kitchen_area', 'rooms', 'geo_lat', 'geo_lon']
    X = df[feature_cols].values
    y = df['price'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42
    )
    
    # Обучение модели
    model = CatBoostRegressor(
        iterations=400,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model performance: MAE={mae:.0f}, RMSE={rmse:.0f}, R²={r2:.3f}")
    
    # Сохранение модели
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, 'artifacts/model.joblib')
    logger.info("Model saved to artifacts/model.joblib")
    
    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}

def main():
    """Главная функция"""
    logger.info("Starting training pipeline")
    
    # Загрузка данных
    data_path = os.getenv("DATA_PATH", "data/cleaned_data.csv")
    df = load_data(data_path)
    
    # Обучение модели
    model, metrics = train_model(df)
    
    # Обучение кластеризации
    kmeans = train_clustering(df)
    
    logger.info("Training pipeline completed successfully")
    logger.info(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()
