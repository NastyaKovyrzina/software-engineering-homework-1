import os
import time
import psutil
from transformers import pipeline
from datasets import load_dataset
from jiwer import wer, cer 

os.environ["HF_HOME"] = r"D:\ML_University_Task\hf_cache"

MODEL_NAME = "openai/whisper-small"
DATASET_NAME = "fixie-ai/common_voice_17_0"
NUM_SAMPLES = 50 


# Замер RAM
process = psutil.Process(os.getpid())
start_ram = process.memory_info().rss / (1024 * 1024)

try:
    pipe = pipeline("automatic-speech-recognition", model=MODEL_NAME, device=-1)
    end_ram = process.memory_info().rss / (1024 * 1024)
    ram_used = end_ram - start_ram
    print(f"Модель загружена. Потребление RAM: +{ram_used:.0f} МБ")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit()

# Загрузка датасета
try:
    ds = load_dataset(DATASET_NAME, "ru", split="test", streaming=True)
    dataset_list = list(ds.take(NUM_SAMPLES))
    print(f"Успешно загружено {len(dataset_list)} аудиофайлов.")
except Exception as e:
    print(f"Не удалось загрузить сплит 'test' ({e})")
    try:
        ds = load_dataset(DATASET_NAME, "ru", split="validation", streaming=True)
        dataset_list = list(ds.take(NUM_SAMPLES))
        print(f"Успешно загружено {len(dataset_list)} аудиофайлов.")
    except Exception as e2:
        print(f"Критическая ошибка загрузки датасета: {e2}")
        exit()

wer_scores = []
cer_scores = [] 
time_scores = []
errors_examples = []

start_total_time = time.time()

for i, item in enumerate(dataset_list):
    audio_data = item["audio"]
    true_text = item["sentence"]
    
    start_time = time.time()
    result = pipe(audio_data)
    transcription = result["text"]
    end_time = time.time()
    
    processing_time = end_time - start_time
    time_scores.append(processing_time)
    
    # Очистка текста
    true_clean = "".join(c for c in true_text.lower() if c.isalnum() or c.isspace()).strip()
    pred_clean = "".join(c for c in transcription.lower() if c.isalnum() or c.isspace()).strip()
    
    #  WER (ошибка по словам) и CER (ошибка по символам)
    error_rate_wer = wer(true_clean, pred_clean)
    error_rate_cer = cer(true_clean, pred_clean)
    
    wer_scores.append(error_rate_wer)
    cer_scores.append(error_rate_cer)
    
    # Сохраняю примеры с высокими ошибками (WER > 0.3)
    if error_rate_wer > 0.3 and len(errors_examples) < 3:
        errors_examples.append({
            "index": i + 1,
            "true": true_clean,
            "pred": pred_clean,
            "wer": error_rate_wer,
            "cer": error_rate_cer
        })
    
    if (i + 1) % 10 == 0 or (i + 1) == len(dataset_list):
        print(f"  Обработано: {i + 1} / {len(dataset_list)}")

total_time = time.time() - start_total_time

print("Итоговые результаты:")

avg_wer = sum(wer_scores) / len(wer_scores) * 100
avg_cer = sum(cer_scores) / len(cer_scores) * 100
avg_time = sum(time_scores) / len(time_scores)

print(f"Количество обработанных файлов: {len(dataset_list)}")
print(f"Средний WER (ошибка на уровне слов): {avg_wer:.2f}%")
print(f"Средний CER (ошибка на уровне символов): {avg_cer:.2f}%")
print(f"Среднее время на файл: {avg_time:.2f} сек.")
print(f"Общее время обработки: {total_time:.1f} сек.")
print(f"Потребление RAM моделью: +{ram_used:.0f} МБ")

if errors_examples:
    print("Примеры ошибок:")
    for err in errors_examples:
        print(f"Пример #{err['index']}] WER: {err['wer']*100:.1f}% | CER: {err['cer']*100:.1f}%")
        print(f"Эталон:    {err['true'][:120]}...")
        print(f"Распознано: {err['pred'][:120]}...")
else:
    print("Грубых ошибок не найдено")