import cv2
from collections import deque, Counter
from ultralytics import YOLO

MODEL_PATH = "runs/asl_sign_model3/weights/best.pt"
CONF_TH = 0.70               # düşükse "Unknown"
HISTORY = 12                 # son 12 tahminle oylama

model = YOLO(MODEL_PATH)
history = deque(maxlen=HISTORY)


def majority_vote(hist):
    if not hist:
        return None
    return Counter(hist).most_common(1)[0][0]


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    # ROI kutusu (elini buraya koy)
    roi_x1, roi_y1 = 120, 80
    roi_x2, roi_y2 = 520, 480

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ROI kırp
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Tahmin
        results = model.predict(source=roi, conf=0.25, verbose=False)
        r = results[0]

        label_to_show = "Unknown"
        conf_to_show = 0.0

        if r.boxes is not None and len(r.boxes) > 0:
            confs = r.boxes.conf.tolist()
            best_i = confs.index(max(confs))

            cls_id = int(r.boxes.cls[best_i].item())
            label = r.names[cls_id]
            conf = float(r.boxes.conf[best_i].item())

            # confidence yeterliyse geçmişe ekle
            if conf >= CONF_TH:
                history.append(label)

            voted = majority_vote(history)
            if voted is not None:
                label_to_show = voted
                conf_to_show = conf

        # Ekrana ROI kutusu çiz
        cv2.rectangle(frame, (roi_x1, roi_y1),
                      (roi_x2, roi_y2), (0, 255, 0), 2)

        # Üstte yazı
        cv2.putText(frame, f"ASL: {label_to_show}  (conf: {conf_to_show:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, "Elini yesil kutunun icine koy. Q ile cik.",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("ASL Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
