import cv2
from ultralytics import YOLO
import torch
import os
import json
from PIL import Image
from torchvision import transforms as T
from copy import deepcopy

def classify(cropped, model, transform, device):
    image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        pred = (torch.sigmoid(logit) >= 0.5).int().cpu().numpy()[0][0]
    return int(pred)

def draw_annotations(frame, ann_list, class_names):
    for ann in ann_list:
        x1, y1, x2, y2 = ann["bbox"]
        label_idx = ann["label_idx"]
        label = f'{class_names[label_idx]} Potato'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        y_text = y1 + 20
        cv2.putText(frame, label, (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
    return frame

def draw_on_crop(crop, label, class_names):
    # Disegna un box e testo sul crop per debug
    h, w = crop.shape[:2]
    cv2.rectangle(crop, (0, 0), (w-1, h-1), (0, 255, 0), 2)
    cv2.putText(crop, f"{class_names[label]}",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
    return crop

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    YOLO_path    = "models/yolo_fined.pt"
    image_path   = "samples"
    results_path = "output"
    os.makedirs(results_path, exist_ok=True)

    final_classes = ["Good", "Bad"]

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        #T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    yolo_detector = YOLO(YOLO_path)
    classifier = torch.load("models/classifier.pt", weights_only=False, map_location=device)
    classifier.to(device).eval()

    all_annotations = {}

    for image_name in os.listdir(image_path):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue

        img_fp = os.path.join(image_path, image_name)
        frame = cv2.imread(img_fp)
        if frame is None:
            print(f"Immagine non leggibile: {img_fp}")
            continue

        results = yolo_detector(frame)[0]
        ann_list = []
        H, W = frame.shape[:2]

        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        clss = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

        for idx, ((x1, y1, x2, y2), det_cls, conf) in enumerate(zip(xyxy, clss, confs)):
            x1 = max(0, min(int(x1), W-1))
            y1 = max(0, min(int(y1), H-1))
            x2 = max(0, min(int(x2), W-1))
            y2 = max(0, min(int(y2), H-1))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped = deepcopy(frame[y1:y2, x1:x2])
            label_idx = classify(cropped, classifier, transform, device)

            # Salva crop grezzo
            cv2.imwrite(os.path.join(results_path,
                        f"crop_raw_{idx}_{image_name}"), cropped)

            # Salva crop con etichetta
            cropped_with_label = draw_on_crop(cropped.copy(), label_idx, final_classes)
            cv2.imwrite(os.path.join(results_path,
                        f"crop_labeled_{idx}_{image_name}"), cropped_with_label)

            ann_list.append({
                "bbox": [x1, y1, x2, y2],
                "label_idx": label_idx,
                "det_cls": int(det_cls),
                "conf": float(conf)
            })

        all_annotations[image_name] = ann_list

    for image_name, ann_list in all_annotations.items():
        img_fp = os.path.join(image_path, image_name)
        frame = cv2.imread(img_fp)
        if frame is None:
            continue

        frame_annot = draw_annotations(frame, ann_list, final_classes)
        out_fp = os.path.join(results_path, f"out_{image_name}")
        cv2.imwrite(out_fp, frame_annot)
        print(f"Salvato: {out_fp}")

if __name__ == "__main__":
    main()
