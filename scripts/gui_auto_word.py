import tkinter as tk
from tkinter import messagebox
import cv2
from ultralytics import YOLO
from collections import deque, Counter

MODEL_PATH = "runs/asl_sign_model3/weights/best.pt"

# --- Ayarlar ---
CONF_TH = 0.75          # çok yüksek olursa hiç yazmaz, çok düşük olursa yanlış yazar
REQ_CONSEC = 8          # aynı harfi art arda bu kadar kare görürse ekle
HISTORY = 10            # smoothing için

MIRROR = True
ROI_SIDE = "left"       # "left" / "right"
ROI_SIZE = 360
ROI_MARGIN = 40

SHOW_ROI_WINDOW = True  # ROI penceresi aç/kapat


def enhance_for_model(bgr):
    """Daha net tahmin için kontrast artır."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)
    out = cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)
    return out


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Live → Anında Kelime Yazma")
        self.root.geometry("680x260")

        self.model = YOLO(MODEL_PATH)

        self.pred_hist = deque(maxlen=HISTORY)
        self.current_candidate = None
        self.consec = 0
        self.last_committed = None

        self.text = ""

        # UI
        self.pred_var = tk.StringVar(value="Tahmin: -")
        self.text_var = tk.StringVar(value="Yazı: ")

        tk.Label(root, textvariable=self.pred_var,
                 font=("Arial", 14)).pack(pady=6)
        tk.Label(root, textvariable=self.text_var,
                 font=("Arial", 20, "bold")).pack(pady=6)

        row = tk.Frame(root)
        row.pack(pady=6)
        tk.Button(row, text="Boşluk", width=10, command=self.add_space).grid(
            row=0, column=0, padx=6)
        tk.Button(row, text="Sil", width=10, command=self.backspace).grid(
            row=0, column=1, padx=6)
        tk.Button(row, text="Temizle", width=10, command=self.clear_text).grid(
            row=0, column=2, padx=6)

        ctrl = tk.Frame(root)
        ctrl.pack(pady=6)
        tk.Button(ctrl, text="Kamerayı Başlat", width=16,
                  command=self.start).grid(row=0, column=0, padx=8)
        tk.Button(ctrl, text="Kamerayı Durdur", width=16,
                  command=self.stop).grid(row=0, column=1, padx=8)

        self.cap = None
        self.running = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def add_space(self):
        self.text += " "
        self.text_var.set(f"Yazı: {self.text}")
        self.last_committed = None

    def backspace(self):
        if self.text:
            self.text = self.text[:-1]
            self.text_var.set(f"Yazı: {self.text}")
        self.last_committed = None

    def clear_text(self):
        self.text = ""
        self.text_var.set("Yazı: ")
        self.last_committed = None

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Hata", "Kamera açılamadı.")
            return
        self.running = True
        self.loop()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def on_close(self):
        self.stop()
        self.root.destroy()

    def get_roi(self, frame):
        h, w = frame.shape[:2]
        size = min(ROI_SIZE, h - 10, w - 10)
        y1 = max(0, (h - size) // 2)
        y2 = y1 + size

        if ROI_SIDE == "left":
            x1 = ROI_MARGIN
            x2 = x1 + size
        else:
            x2 = w - ROI_MARGIN
            x1 = x2 - size

        return x1, y1, x2, y2

    def predict_roi(self, roi_bgr):
        # ROI’yi büyüt (el küçük kalmasın)
        roi_big = cv2.resize(roi_bgr, (640, 640),
                             interpolation=cv2.INTER_LINEAR)
        roi_big = enhance_for_model(roi_big)

        results = self.model.predict(roi_big, conf=CONF_TH, verbose=False)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return None, None, r.plot()

        confs = r.boxes.conf.tolist()
        best_i = confs.index(max(confs))

        cls_id = int(r.boxes.cls[best_i].item())
        label = r.names[cls_id]
        conf = float(r.boxes.conf[best_i].item())

        return label, conf, r.plot()

    def loop(self):
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.root.after(20, self.loop)
            return

        if MIRROR:
            frame = cv2.flip(frame, 1)

        x1, y1, x2, y2 = self.get_roi(frame)
        roi = frame[y1:y2, x1:x2].copy()

        label, conf, ann_roi_big = self.predict_roi(roi)

        # ROI penceresi (el gerçekten kutuda mı gör)
        if SHOW_ROI_WINDOW:
            cv2.imshow("ROI (Hand Box)", roi)

        # Tahmin history’ye sadece gerçek tespit varsa ekle
        if label is not None and conf is not None and conf >= CONF_TH:
            self.pred_hist.append(label)

        stable = Counter(self.pred_hist).most_common(1)[
            0][0] if self.pred_hist else None

        if stable is None:
            self.pred_var.set("Tahmin: -")
        else:
            self.pred_var.set(f"Tahmin: {stable} (conf: {conf:.2f})")

        # Ardışık kare teyidi
        if stable != self.current_candidate:
            self.current_candidate = stable
            self.consec = 0

        if stable is not None and conf is not None and conf >= CONF_TH:
            self.consec += 1
        else:
            self.consec = max(0, self.consec - 1)

        # Yeterli teyit → hemen ekle
        if stable is not None and self.consec >= REQ_CONSEC and stable != self.last_committed:
            self.text += stable
            self.text_var.set(f"Yazı: {self.text}")
            self.last_committed = stable
            self.consec = 0  # aynı harfi tekrar yazmasın diye

        # Ana görüntü
        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (80, 255, 80), 2)
        cv2.putText(display, "Elini yesil kutuya koy (q ile kapat)",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 80), 2)

        cv2.putText(display, self.text[-20:], (12, display.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)

        cv2.imshow("ASL Live", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.stop()
            return
        elif key == ord(" "):
            self.add_space()
        elif key == 8:
            self.backspace()
        elif key == ord("c"):
            self.clear_text()

        self.root.after(10, self.loop)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
