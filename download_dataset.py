from roboflow import Roboflow

rf = Roboflow(api_key="hkAb6XLBZR34pFXYpCe5")
project = rf.workspace("roboflow-universe-projects").project("drone-detection-2-ckiqe")
version = project.version(3)
dataset = version.download("yolov8")