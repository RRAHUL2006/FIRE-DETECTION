"""from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0
    )"""
from ultralytics import YOLO

if __name__ == "__main__":

    # Resume from last trained checkpoint
    model = YOLO(
        r"D:\Downloads\Fire -smoke Detection.v1i.yolov8\runs\detect\train4\weights\last.pt "
    )

    model.train(
        resume=True,
        device=0,
        workers=0
    )