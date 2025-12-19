import os
from ultralytics import YOLO

#model = YOLO("yolo11n.yaml")  # architecture only, no .pt file
#model.train(data="data.yaml", epochs=3, imgsz=640, device='cpu')

model = YOLO("yolo11n.yaml")
model.train(
    data="data.yaml",
    epochs=200,
    imgsz=640, 
    hsv_h=0.03,
    hsv_s=0.6,
    hsv_v=0.5,
    mosaic=0.7,
    mixup=0.2,
    fliplr=0.5,
    scale=0.5,
    degrees=10.0,
    translate=0.5,
    shear=10.0,
    perspective=0.001,
    flipud=0.5,
    bgr=0.5,
    cutmix=0.5,
    copy_paste=0.5,
    erasing=0.5,
)
