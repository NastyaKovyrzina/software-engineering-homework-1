import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F
import time
import os

VOC_TO_COCO = {
    'aeroplane': 5, 'bicycle': 2, 'bird': 16, 'boat': 9, 'bottle': 44,
    'bus': 6, 'car': 3, 'cat': 17, 'chair': 62, 'cow': 21,
    'diningtable': 67, 'dog': 18, 'horse': 19, 'motorbike': 4, 'person': 1,
    'pottedplant': 64, 'sheep': 20, 'sofa': 63, 'train': 7, 'tvmonitor': 70
}

def calculate_iou(boxA, boxB):
    """Считает коэффициент пересечения рамок (IoU)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def parse_voc_target(target_dict):
    boxes = []
    labels = []
    
    objects = target_dict['annotation'].get('object', [])
    if isinstance(objects, dict):
        objects = [objects]
        
    for obj in objects:
        class_name = obj['name']
        if class_name in VOC_TO_COCO:
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin'])
            ymin = float(bndbox['ymin'])
            xmax = float(bndbox['xmax'])
            ymax = float(bndbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_TO_COCO[class_name])
            
    return boxes, labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Устройство: {device}")

    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights)
    model.to(device)
    model.eval()

    dataset = VOCDetection(root='./data_voc', year='2012', image_set='val', download=True)
    
    MAX_IMAGES = 200

    total_tp = 0  
    total_fp = 0  
    total_fn = 0  
    total_gt_objects = 0
    total_pred_objects = 0

    start_time = time.time()

    for i in range(MAX_IMAGES):
        image, target = dataset[i]
        
        # Получаем Ground Truth (что реально на фото)
        gt_boxes, gt_labels = parse_voc_target(target)
        total_gt_objects += len(gt_boxes)
        
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(image_tensor)[0]
            
        pred_boxes = predictions['boxes'].cpu().tolist()
        pred_labels = predictions['labels'].cpu().tolist()
        pred_scores = predictions['scores'].cpu().tolist()
        
        valid_preds = [(b, l, s) for b, l, s in zip(pred_boxes, pred_labels, pred_scores) if s > 0.5]
        total_pred_objects += len(valid_preds)
        
        matched_preds = set()
        for gt_idx, gt_box in enumerate(gt_boxes):
            gt_label = gt_labels[gt_idx]
            best_iou = 0
            best_pred_idx = -1
            
            for pred_idx, (p_box, p_label, p_score) in enumerate(valid_preds):
                if pred_idx in matched_preds:
                    continue
                if p_label != gt_label:
                    continue
                
                iou = calculate_iou(gt_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
                    
            if best_iou >= 0.5 and best_pred_idx != -1:
                total_tp += 1
                matched_preds.add(best_pred_idx)
            else:
                total_fn += 1 
                
        total_fp += len(valid_preds) - len(matched_preds)
        
        if (i + 1) % 50 == 0:
            print(f"Обработано {i + 1}/{MAX_IMAGES} изображений")

    total_time = time.time() - start_time

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Итоги")
    print(f"Время выполнения: {total_time:.1f} сек.")
    print(f"Всего реальных объектов (Ground Truth): {total_gt_objects}")
    print(f"Всего найдено моделью (Confidence > 0.5): {total_pred_objects}")
    print(f"True Positives (TP): {total_tp} (Модель нашла объект, и он есть на фото)")
    print(f"False Positives (FP): {total_fp} (Модель нашла 'левый' объект, галлюцинация)")
    print(f"False Negatives (FN): {total_fn} (Модель пропустила реальный объект)")
    print(f"Precision (Точность): {precision:.2%}")
    print(f"Recall (Полнота): {recall:.2%}")
    print(f"F1-Score (Accuracy детекции): {f1_score:.2%}")

if __name__ == "__main__":
    main()