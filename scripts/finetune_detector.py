from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")
results = model.train(data="../dataset/potatoes-v11/data.yaml", epochs=1, imgsz=640)
