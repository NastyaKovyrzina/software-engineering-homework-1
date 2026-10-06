import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

os.environ["HF_HOME"] = r"D:\ML_University_Task\hf_cache"

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

app = FastAPI(
    title="Multilingual Sentiment Analysis API",
    description="API для анализа тональности текста",
    version="2.1.0"
)

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device)
print("Модель успешно загружена!")

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Текст отзыва (1-512 символов)")

class SentimentResponse(BaseModel):
    original_text: str
    sentiment: str = Field(description="Нормализованная тональность: POSITIVE или NEGATIVE")
    confidence: float = Field(description="Уверенность модели от 0.0 до 1.0")
    raw_label: str = Field(description="Сырая метка от модели")

class BatchReviewRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=50, description="Список текстов для анализа (от 1 до 50 шт.)")

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processed: int

def normalize_sentiment(raw_label: str) -> str:
    raw_upper = raw_label.upper()
    if "POS" in raw_upper:
        return "POSITIVE"
    elif "NEG" in raw_upper:
        return "NEGATIVE"
    else:
        return "NEGATIVE" 

# Эндпоинты 
@app.get("/", summary="Корневой эндпоинт")
def root():
    return {"message": "API работает", "model": MODEL_NAME}

@app.get("/health", summary="Проверка состояния сервиса", tags=["Monitoring"])
def health_check():
    """Возвращает статус готовности модели к обработке запросов"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_NAME
    }

@app.post("/predict", response_model=SentimentResponse, summary="Анализ одного отзыва", tags=["Prediction"])
def predict_sentiment(request: ReviewRequest):
    try:
        result = classifier(request.text)[0]
        return SentimentResponse(
            original_text=request.text,
            sentiment=normalize_sentiment(result['label']),
            confidence=result['score'],
            raw_label=result['label'].upper()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка модели: {str(e)}")

@app.post("/predict/batch", response_model=BatchSentimentResponse, summary="Пакетный анализ отзывов", tags=["Prediction"])
def predict_batch_sentiment(request: BatchReviewRequest):
    """Принимает список текстов и возвращает анализ для каждого из них"""
    try:
        results = []
        predictions = classifier(request.texts)
        
        for text, pred in zip(request.texts, predictions):
            results.append(SentimentResponse(
                original_text=text,
                sentiment=normalize_sentiment(pred['label']),
                confidence=pred['score'],
                raw_label=pred['label'].upper()
            ))
            
        return BatchSentimentResponse(
            results=results,
            total_processed=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при пакетной обработке: {str(e)}")