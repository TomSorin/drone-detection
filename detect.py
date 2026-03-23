import os
import shutil
import glob
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

output_dir = "detections"
os.makedirs(output_dir, exist_ok=True)

frame_dirs = [d for d in glob.glob("frames/*") if os.path.isdir(d)]

for frame_dir in sorted(frame_dirs):
    for filename in sorted(os.listdir(frame_dir)):
        if not filename.endswith(".jpg"):
            continue
        filepath = os.path.join(frame_dir, filename)
        results = model(filepath, conf=0.15, device="mps", verbose=False, classes=[4])
        for r in results:
            if len(r.boxes) > 0:
                shutil.copy(filepath, os.path.join(output_dir, f"{os.path.basename(frame_dir)}_{filename}"))
                print(f"Found in: {filename}")
                break

print(f"Done! Detection frames saved to '{output_dir}/'")