import os
import time
import psutil
import cv2
import torch
import torchvision
from huggingface_hub import list_repo_files, hf_hub_download
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    all_files = list_repo_files("nateraw/kinetics-mini", repo_type="dataset")
    
    video_files = [f for f in all_files if f.startswith("train/") and f.endswith(".mp4")]
    
    video_files = video_files[:50]
    print(f"Найдено {len(video_files)} видео для обработки.")
    
except Exception as e:
    print(f"Ошибка получения списка файлов: {e}")
    exit()

process = psutil.Process(os.getpid())
start_ram = process.memory_info().rss / (1024 * 1024)

weights = torchvision.models.video.R3D_18_Weights.KINETICS400_V1
model = torchvision.models.video.r3d_18(weights=weights)
model.eval()

classes = weights.meta["categories"]

end_ram = process.memory_info().rss / (1024 * 1024)
ram_used = end_ram - start_ram
print(f"Модель загружена. Потребление RAM: +{ram_used:.0f} МБ")

def prepare_video_tensor(video_path, num_frames=16, target_size=(112, 112)):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            return None
        
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, target_size)
            frames.append(frame_resized)
        
        cap.release()
        
        if len(frames) < num_frames:
            return None
        
        video_array = torch.as_tensor(frames, dtype=torch.float32)
        video_array = video_array.permute(3, 0, 1, 2) / 255.0
        
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        video_array = (video_array - mean) / std
        
        return video_array.unsqueeze(0)
        
    except Exception:
        return None

print(f"Обработка {len(video_files)} видео...")

true_labels = []
pred_labels = []
time_scores = []
confidences = []

for i, video_file in enumerate(video_files):
    try:
        local_path = hf_hub_download(
            repo_id="nateraw/kinetics-mini",
            filename=video_file,
            repo_type="dataset",
            cache_dir="./hf_cache"
        )
        
        parts = video_file.split("/")
        if len(parts) >= 2:
            class_name_from_path = parts[1]  
        else:
            continue
        
    
        true_label_idx = None
        for idx, cls in enumerate(classes):
            cls_lower = cls.lower()
            if class_name_from_path.lower() in cls_lower or cls_lower.replace(" ", "_") == class_name_from_path.lower():
                true_label_idx = idx
                break
        
        if true_label_idx is None:
            continue
        
        true_class_name_full = classes[true_label_idx].lower()
        
    except Exception as e:
        print(f"  Ошибка скачивания {video_file}: {e}")
        continue
    
    video_tensor = prepare_video_tensor(local_path)
    if video_tensor is None:
        continue
    
    true_labels.append(true_class_name_full)
    
    start_time = time.time()
    try:
        with torch.no_grad():
            output = model(video_tensor)
        
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()
        
        pred_class_name = classes[pred_idx].lower()
        pred_labels.append(pred_class_name)
        confidences.append(confidence)
        
    except Exception:
        true_labels.pop()
        continue
    
    end_time = time.time()
    time_scores.append(end_time - start_time)
    
    if (i + 1) % 10 == 0:
        print(f"Обработано: {i + 1} / {len(video_files)}")

if len(true_labels) == 0:
    print("Не удалось обработать ни одного видео")
    exit()

print(" Расчет итоговых метрик")

accuracy = accuracy_score(true_labels, pred_labels) * 100
precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0) * 100
recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0) * 100
f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0) * 100

avg_time = sum(time_scores) / len(time_scores)
avg_conf = sum(confidences) / len(confidences)

print("Итог")

print(f"Обработано видео:      {len(true_labels)} из {len(video_files)}")
print(f"Accuracy:            {accuracy:.2f}%")
print(f"Precision:           {precision:.2f}%")
print(f"Recall:              {recall:.2f}%")
print(f"F1-score:            {f1:.2f}%")
print(f"Среднее время:       {avg_time:.2f} сек. на видео")
print(f"Средняя уверенность: {avg_conf:.2%}")
print(f"Потребление RAM:     +{ram_used:.0f} МБ")
