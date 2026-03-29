import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

def make_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,1,0],
                     [0,1,0,1],
                     [0,0,1,0],
                     [0,0,0,1]], dtype=float)
    kf.H = np.array([[1,0,0,0],
                     [0,1,0,0]], dtype=float)
    kf.R *= 10
    kf.P *= 100
    kf.Q *= 0.1
    return kf

def get_center(box):
    x1, y1, x2, y2 = box
    return [(x1+x2)/2, (y1+y2)/2]

model = YOLO("yolov8s.pt")
video_dirs = sorted(glob.glob("frames/video*"))
os.makedirs("output_videos", exist_ok=True)

for video_dir in video_dirs:
    video_name = os.path.basename(video_dir)
    frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".jpg")])

    kf = make_kalman()
    initialized = False
    trajectory = []
    missing = 0
    max_missing = 10

    first_frame = cv2.imread(os.path.join(video_dir, frames[0]))
    h, w = first_frame.shape[:2]
    out = cv2.VideoWriter(
        f"output_videos/{video_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        5, (w, h)
    )

    for fname in frames:
        filepath = os.path.join(video_dir, fname)
        img = cv2.imread(filepath)
        results = model(filepath, conf=0.15, device="mps", verbose=False, classes=[4])

        detected = False
        box = None
        for r in results:
            if len(r.boxes) > 0:
                b = r.boxes.xyxy[0].tolist()
                box = b
                center = get_center(b)
                if not initialized:
                    kf.x = np.array([[center[0]], [center[1]], [0], [0]], dtype=float)
                    initialized = True
                else:
                    kf.predict()
                    kf.update(np.array([[center[0]], [center[1]]], dtype=float))
                trajectory.append((int(kf.x[0][0]), int(kf.x[1][0])))
                missing = 0
                detected = True
                break

        if not detected and initialized and missing < max_missing:
            kf.predict()
            trajectory.append((int(kf.x[0][0]), int(kf.x[1][0])))
            missing += 1

        if initialized:
            if box:
                x1,y1,x2,y2 = [int(v) for v in box]
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, "drone", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            for i in range(1, len(trajectory)):
                cv2.line(img, trajectory[i-1], trajectory[i], (0,0,255), 2)
            out.write(img)

    out.release()
    print(f"Done: output_videos/{video_name}.mp4")

print("All tracking videos saved!")