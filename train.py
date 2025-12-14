import os
from ultralytics import YOLO

#model = YOLO("yolo11n.yaml")  # architecture only, no .pt file
#model.train(data="data.yaml", epochs=3, imgsz=640, device='cpu')

model = YOLO("/Users/dhananjaydeshpande/Desktop/Columbia EE DES/Digital Signal Processing/Project/saffron_flower_dataset-master/runs/detect/train4/weights/best.pt")
model.train(
    data="data.yaml",
    epochs=3,
    imgsz=640, 
    device='cpu',
    hsv_h=0.03,
    hsv_s=0.6,
    hsv_v=0.5,
    mosaic=0.7,
    mixup=0.2,
    fliplr=0.5,
    scale=0.5,
    degrees=10.0,
)
