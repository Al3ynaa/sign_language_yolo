import cv2
from ultralytics import YOLO
import time

MODEL_PATH = "runs/asl_sign_model3/weights/best.pt"
model = YOLO(MODEL_PATH)

CONF_TH = 0.35
ADD_COOLDOWN = 0.35

# ROI ayarları (ekrandaki yeşil kutu)
ROI_SIZE = 320
ROI_MARGIN = 20

text = ""
last_add_time = 0


def get_roi_box(w, h):
    # sağ tarafta kutu
    x2 = w - ROI_MARGIN
    x1 = x2 - ROI_SIZE
    y1 = int(h/2 - ROI_SIZE/2)
    y2 = y1 + ROI_SIZE
    return x1, y1, x2, y2


def topk_from_result(r, k=3):
    if r.boxes is None or len(r.boxes) == 0:
        return []

    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)

    idxs = confs.argsort()[::-1][:k]
    out = []
    for i in idxs:
        out.append((r.names[clss[i]], float(confs[i])))
    return out


cap = cv2.VideoCapture(0)

print(
    "Kısayollar: [F]=harf ekle | [Space]=boşluk | [Backspace]=sil | [C]=temizle | [Q]=çık")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ayna gibi göster (sağ/sol rahat olsun)
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_roi_box(w, h)

    # ROI kes (sadece elin olduğu alanı modele ver)
    roi = frame[y1:y2, x1:x2].copy()

    # Tahmin
    results = model.predict(roi, conf=CONF_TH, verbose=False)
    r = results[0]

    # ROI üzerinde çizimli görüntü
    roi_annot = r.plot()

    # ROI'yi ana frame'e geri yapıştır (kutunun içine çizimli görünür)
    frame[y1:y2, x1:x2] = roi_annot

    # Top-3 tahmin al
    top3 = topk_from_result(r, k=3)

    if len(top3) == 0:
        main_pred = "-"
        main_conf = 0.0
        top_text = "Top3: -"
    else:
        main_pred, main_conf = top3[0]
        top_text = "Top3: " + " | ".join([f"{a}:{b:.2f}" for a, b in top3])

    # Yazılar
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Tahmin: {main_pred}  conf:{main_conf:.2f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, top_text, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"Yazi: {text}", (15, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("ASL Live Type", frame)

    key = cv2.waitKey(1) & 0xFF
    now = time.time()

    if key == ord('q'):
        break

    # F ile harf ekle
    if key == ord('f'):
        if now - last_add_time >= ADD_COOLDOWN:
            if main_pred != "-" and main_conf >= 0.25:
                text += main_pred
                last_add_time = now

    # boşluk
    if key == 32:
        text += " "

    # backspace
    if key == 8:
        text = text[:-1]

    # temizle
    if key == ord('c'):
        text = ""

cap.release()
cv2.destroyAllWindows()
