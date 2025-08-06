import cv2
from ultralytics import YOLO
import torch
import timm
import joblib
import os
from PIL import Image
from torchvision import transforms

def load_backbone(backbone_name='resnet50'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = timm.create_model(backbone_name, pretrained=True)
    # Freeze the model
    backbone.eval()  # disables dropout, batchnorm updates
    for param in backbone.parameters():
        param.requires_grad = False

    backbone.to(device)
    return backbone, device


def detect():
    pass

def classify(cropped, backbone, classifier, transform, device):
    image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
    # Crea un oggetto PIL.Image dalla matrice numpy
    image_pil = Image.fromarray(image_rgb)
    image_pil = transform(image_pil)
    image_pil = image_pil.unsqueeze(0) # Add batch dimension
    image_pil = image_pil.to(device)
    feats = backbone.forward_features(image_pil)
    pooled = feats.mean(dim=[2, 3])  # global average pooling to [B, 2048]

    prediction = classifier.predict(pooled.cpu().numpy())
    return prediction[0]

def main():

    YOLO_path = "models/best.pt"  # Path to the YOLO
    image_path = "dataset/rotten_healthy/test/"  # Path to the input image
    results_path = "output"  # Path to save the output image
    final_classes = ["Healthy", "Rotten"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    yolo_detector = YOLO(YOLO_path)
    backbone, device = load_backbone('resnet50')
    classifier = joblib.load("models/rotten_healthy_classifier.pkl")

    for image in os.listdir(image_path):
        if not image.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue

        frame = cv2.imread(os.path.join(image_path, image))
        results = yolo_detector(frame)[0]  # Get the first (and only) result

        for detection in results.boxes:  # Boxes object for bounding box outputs

            detected_cls = int(detection.cls[0])  # Ensure class is an integer
            coords = detection.xyxy.cpu().numpy()

            x1 = int(coords[0][0])
            y1 = int(coords[0][1])
            x2 = int(coords[0][2])
            y2 = int(coords[0][3])

            cropped = frame[y1:y2, x1:x2]

            prediction = classify(cropped, backbone, classifier, transform, device)
            label = f'{final_classes[prediction]}{yolo_detector.names[detected_cls]}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4) # Draw rectangle
            cv2.putText(frame, label, (x1, y1 +10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        os.makedirs(results_path, exist_ok=True)
        output_path = f"{results_path}/out_{image}"
        cv2.imwrite(output_path, frame)

if __name__ == "__main__":
    main()