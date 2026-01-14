from ultralytics import YOLO
import cv2
from pathlib import Path


MODEL_PATH = "runs/asl_sign_model3/weights/best.pt"


IMAGE_PATH = "dataset/test/images/A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg"


def main():
    # Dosyalar var mı kontrol
    if not Path(MODEL_PATH).exists():
        print(f" Model yok: {MODEL_PATH}")
        return
    if not Path(IMAGE_PATH).exists():
        print(f" Resim yok: {IMAGE_PATH}")
        return

    model = YOLO(MODEL_PATH)

    results = model.predict(source=IMAGE_PATH, conf=0.25, save=True)

    r = results[0]

    # Hiç obje yoksa
    if r.boxes is None or len(r.boxes) == 0:
        print(" Hiç harf tespit edemedim.")
    else:
        # En yüksek confidence'ı seç
        confs = r.boxes.conf.tolist()
        best_i = confs.index(max(confs))

        cls_id = int(r.boxes.cls[best_i].item())
        label = r.names[cls_id]
        confidence = float(r.boxes.conf[best_i].item())

        print(f" Bu harf: {label}  (confidence: {confidence:.2f})")

    annotated = r.plot()

    cv2.imshow("ASL Prediction (press any key)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
