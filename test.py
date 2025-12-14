from ultralytics import YOLO
model = YOLO("/Users/dhananjaydeshpande/Desktop/Columbia EE DES/Digital Signal Processing/Project/saffron_flower_dataset-master/runs/detect/train4/weights/best.pt")

metrics_test = model.val(data="data.yaml", split="test")