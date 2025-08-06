from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("../models/yolo11n.pt")
results = model.train(data="../dataset/potatoes-v11/data.yaml", epochs=10, imgsz=640)