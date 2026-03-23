from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="drone-detection-2/data.yaml",
    epochs=3,
    imgsz=300,
    batch=16,
    name="drone_detector",
    device="mps",
    cache=True
)

print("Training complete!")