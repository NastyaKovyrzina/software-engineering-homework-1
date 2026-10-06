import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_positive():
    response = client.post("/predict", json={"text": "Отличный продукт, очень доволен!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "POSITIVE"

def test_predict_negative():
    response = client.post("/predict", json={"text": "Ужасное качество, не рекомендую."})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "NEGATIVE"

def test_health_check():
    """Тест эндпоинта проверки здоровья сервиса"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_batch_mixed():
    """Тест пакетной обработки с разными тональностями"""
    payload = {
        "texts": [
            "Прекрасный сервис, всё быстро и качественно!",
            "Доставка задержалась на неделю, товар поврежден."
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_processed"] == 2
    assert data["results"][0]["sentiment"] == "POSITIVE"
    assert data["results"][1]["sentiment"] == "NEGATIVE"

def test_predict_batch_validation_empty_list():
    """Тест валидации: пустой список текстов должен отклоняться"""
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422 