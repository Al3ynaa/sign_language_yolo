from ultralytics import YOLO


def main():

    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs",
        name="asl_sign_model",
        plots=True
    )


if __name__ == "__main__":
    main()
