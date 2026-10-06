import os
import time
import psutil
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ["HF_HOME"] = r"D:\ML_University_Task\hf_cache"

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

csv_path = r"D:\ML_University_Task\Part_1\task_1_text\rureviews.csv"

try:
    print(f"Чтение файла: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8', sep='\t', engine='python')
    
    if list(df.columns) == [0, 1] or ('Review' not in df.columns and 'Sentiment' not in df.columns):
        print("Заголовков в файле не найдено")
        df.columns = ['Review', 'Sentiment']
    
    print(f"Всего загружено строк: {len(df)}")
    print(f"Колонки в файле: {list(df.columns)}")
    
    df = df.rename(columns={'Review': 'review', 'Sentiment': 'label'})
    
    # Нормализация меток датасета
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].replace({
        'negative': 0, 'neg': 0, '0': 0, 
        'positive': 1, 'pos': 1, '1': 1
    })
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    
    # Оставляем только 0 и 1
    df_filtered = df[df['label'].isin([0, 1])].copy()
    
    # 1 = POSITIVE, 0 = NEGATIVE
    df_filtered['true_label'] = np.where(df_filtered['label'] == 1, 'POSITIVE', 'NEGATIVE')
    
    # Берем 500 случайных отзывов
    df_sample = df_filtered.sample(n=500, random_state=42).copy()
    df = df_sample[['review', 'true_label']].reset_index(drop=True)
    
    print(f"Отфильтровано и подготовлено {len(df)} отзывов.")
    print("Распределение классов:")
    print(df['true_label'].value_counts())
    
except FileNotFoundError:
    print(f"ОШИБКА: Файл не найден по пути {csv_path}")
    exit()
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    import traceback
    traceback.print_exc()
    exit()


true_labels = []
pred_labels = []

process = psutil.Process(os.getpid())
start_ram = process.memory_info().rss / (1024 * 1024)

try:
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME, device=-1)
    
    end_ram = process.memory_info().rss / (1024 * 1024)
    ram_used = end_ram - start_ram
    print(f"Модель загружена. Потребление RAM: +{ram_used:.0f} МБ")
    
    start_time = time.time()
    
    print("Обработка 500 отзывов...")
    for index, row in df.iterrows():
        text = str(row['review'])[:512]
        true_label = row['true_label']
        
        result = classifier(text)[0]
        
        # Приводим к верхнеу регистру для универсального поиска
        pred_label_raw = result['label'].upper()
        
        # нормализация 
        if "POS" in pred_label_raw:
            pred_label = "POSITIVE"
        elif "NEG" in pred_label_raw:
            pred_label = "NEGATIVE"
        else:
            pred_label = "NEGATIVE"
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        if (index + 1) % 100 == 0:
            print(f"  Обработано: {index + 1} / {len(df)}")
            
    end_time = time.time()
    
    # Расчет метрик
    acc = accuracy_score(true_labels, pred_labels) * 100
    prec = precision_score(true_labels, pred_labels, pos_label="POSITIVE", zero_division=0) * 100
    rec = recall_score(true_labels, pred_labels, pos_label="POSITIVE", zero_division=0) * 100
    f1 = f1_score(true_labels, pred_labels, pos_label="POSITIVE", zero_division=0) * 100
    
    total_time = end_time - start_time
    avg_time = total_time / len(df)
    
    results = [{
        "Модель": MODEL_NAME.split('/')[-1],
        "Accuracy": f"{acc:.1f}%",
        "Precision": f"{prec:.1f}%",
        "Recall": f"{rec:.1f}%",
        "F1-score": f"{f1:.1f}%",
        "Время на отзыв (сек)": f"{avg_time:.4f}",
        "Общее время (сек)": f"{total_time:.1f}",
        "RAM (МБ)": f"{ram_used:.0f}"
    }]
    
    print("Результаты")
    res_df = pd.DataFrame(results)
    
    try:
        print(res_df.to_markdown(index=False))
    except ImportError:
        print(res_df.to_string(index=False))
        
    
    print("Примеры ошибок модели:")
    errors_count = 0
    for index, row in df.iterrows():
        if true_labels[index] != pred_labels[index] and errors_count < 3:
            print(f"Отзыв: '{str(row['review'])[:120]}...'")
            print(f"Истинная метка: {true_labels[index]}, предсказание модели: {pred_labels[index]}")
            errors_count += 1
            
    if errors_count == 0:
        print("Ошибок не найдено! Модель отработала идеально на этой выборке.")
            
except Exception as e:
    print(f"Критическая ошибка при тестировании: {e}")
    import traceback
    traceback.print_exc()