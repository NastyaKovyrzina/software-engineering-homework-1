# Movie RecSys (Part I)

## Установка
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Обучение моделей (скачает MovieLens ml-latest-small автоматически)
python train.py

## Запуск UI
streamlit run app.py