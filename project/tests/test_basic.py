import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Тест импортов основных модулей"""
    from src.data.preprocessing import EstatePreprocessing
    from src.models.modeling import EstateClustering, EstateRegression
    assert True

def test_config_exists():
    """Тест существования конфигурационных файлов"""
    assert os.path.exists("configs/config.yaml")
    assert os.path.exists("configs/.env.example")

def test_requirements_exists():
    """Тест существования requirements.txt"""
    assert os.path.exists("requirements.txt")

def test_service_module_exists():
    """Тест существования модуля сервиса"""
    assert os.path.exists("src/service/app.py")
    assert os.path.exists("src/models/train.py")
