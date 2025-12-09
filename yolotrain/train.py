from ultralytics import YOLO
import torch

# GPU kontrol
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

# YOLO modelini başlat (pretrained)
model = YOLO("yolov8s.pt")  # küçük ama güçlü model

# Eğitim
model.train(
    data="data.yaml",        # Roboflow'dan gelen YAML
    epochs=100,              # 100 genelde yeterli
    imgsz=640,               # eğitim boyutu
    batch=8,                 # VRAM'e göre artırılabilir
    device=device,           # otomatik CUDA/CPU seçimi
    patience=20,             # 20 epoch gelişme yoksa durdur
    augment=True,            # güçlü augmentasyon
    hsv_h=0.015,             # renk varyasyonu
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=1.0,              # mozaik veri artırma
    mixup=0.2,
)

print("Training finished!")
print("Best model here: runs/detect/train/weights/best.pt")
