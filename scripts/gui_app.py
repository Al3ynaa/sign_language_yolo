import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from collections import deque, Counter
from ultralytics import YOLO

MODEL_PATH = "runs/asl_sign_model3/weights/best.pt"

model = YOLO(MODEL_PATH)

# --- Ayarlar ---
CONF_TH = 0.50          # webcam confidence eşiği
SMOOTH_N = 10           # son N tahminden çoğunluk
ROI_SIZE = 320          # merkez kare ROI
ADD_COOLDOWN_MS = 500

# Harfi ekleme tuşu: "Return" = Enter, "f" = F
ADD_KEY = "Return"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Harf Tanıma - YOLO (Resim + Kamera + Yazı)")
        self.root.geometry("820x640")

        # --- Durum ---
        self.running = False
        self.cap = None
        self.pred_history = deque(maxlen=SMOOTH_N)

        self.current_letter = None
        self.current_conf = None

        self.text = ""
        self.last_add_time = 0

        # --- Üst başlık ---
        self.title_text = tk.StringVar(value="Sonuç: -")
        tk.Label(root, textvariable=self.title_text,
                 font=("Arial", 16, "bold")).pack(pady=10)

        # --- Butonlar ---
        btns = tk.Frame(root)
        btns.pack(pady=10)

        tk.Button(btns, text="🖼️ Resim Seç ve Tahmin Et", font=("Arial", 12),
                  command=self.choose_image).grid(row=0, column=0, padx=10)

        self.cam_btn = tk.Button(btns, text="📷 Kamerayı Başlat", font=("Arial", 12),
                                 command=self.toggle_camera)
        self.cam_btn.grid(row=0, column=1, padx=10)

        # --- Yazı alanı (kelime yazma) ---
        word_frame = tk.Frame(root)
        word_frame.pack(pady=10)

        self.text_var = tk.StringVar(value="Yazı: ")
        tk.Label(word_frame, textvariable=self.text_var,
                 font=("Arial", 18, "bold")).grid(row=0, column=0, columnspan=4, pady=5)

        tk.Button(word_frame, text="➕ Harfi Ekle (Enter)", width=18,
                  command=self.add_letter).grid(row=1, column=0, padx=6, pady=5)
        tk.Button(word_frame, text="␣ Boşluk", width=12,
                  command=self.add_space).grid(row=1, column=1, padx=6, pady=5)
        tk.Button(word_frame, text="⌫ Sil", width=12,
                  command=self.backspace).grid(row=1, column=2, padx=6, pady=5)
        tk.Button(word_frame, text="🧹 Temizle", width=12,
                  command=self.clear_text).grid(row=1, column=3, padx=6, pady=5)

        help_text = (
            "Kısayollar: Enter=Harf Ekle | Space=Boşluk | Backspace=Sil | C=Temizle | Q=Kamera durdur\n"
            "İpucu: Elini yeşil kutunun içine koy, harf netleşince Enter'a bas."
        )
        tk.Label(root, text=help_text, font=("Arial", 10)).pack(pady=5)

        # --- Kamera görüntüsü ---
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # --- Kısayol tuşları ---
        self.root.bind("<Return>", lambda e: self.add_letter())
        self.root.bind("<space>", lambda e: self.add_space())
        self.root.bind("<BackSpace>", lambda e: self.backspace())
        self.root.bind("c", lambda e: self.clear_text())
        self.root.bind("C", lambda e: self.clear_text())
        self.root.bind("q", lambda e: self.stop_camera())
        self.root.bind("Q", lambda e: self.stop_camera())

    # ---------------- RESİM MODU ----------------
    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Bir resim seç",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        label, conf, annotated_bgr = self.predict_bgr_image(
            cv2.imread(file_path), conf_th=0.25)

        if annotated_bgr is None:
            messagebox.showerror("Hata", "Resim okunamadı.")
            return

        if label is None:
            self.title_text.set("Sonuç: Bulunamadı")
            messagebox.showinfo("Sonuç", "❌ Hiç harf tespit edemedim.")
        else:
            self.title_text.set(f"Sonuç: {label}  (confidence: {conf:.2f})")

        self.show_on_gui(annotated_bgr)

    # ---------------- KAMERA ----------------
    def toggle_camera(self):
        if not self.running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Hata", "Kamera açılamadı.")
            return

        self.running = True
        self.cam_btn.config(text="⏹️ Kamerayı Durdur")
        self.pred_history.clear()
        self.current_letter = None
        self.current_conf = None
        self.update_camera()

    def stop_camera(self):
        self.running = False
        self.cam_btn.config(text="📷 Kamerayı Başlat")
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_camera(self):
        if not self.running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)  # ayna

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        half = ROI_SIZE // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)

        roi = frame[y1:y2, x1:x2].copy()

        label, conf, annotated_roi = self.predict_bgr_image(
            roi, conf_th=CONF_TH)

        frame_show = frame.copy()

        # yeşil ROI kutusu
        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # talimat yazısı
        cv2.putText(frame_show, "Elini yesil kutunun icine koy",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # smoothing + ekranda gösterilecek harf
        if label is not None:
            self.pred_history.append(label)
            majority = Counter(self.pred_history).most_common(1)[0][0]
            self.current_letter = majority
            self.current_conf = conf

            self.title_text.set(
                f"Sonuç: {majority}  (anlik: {label}, conf: {conf:.2f})")

            # ROI içindeki çizimli görüntüyü ana frame'e bas
            if annotated_roi is not None:
                frame_show[y1:y2, x1:x2] = annotated_roi
        else:
            self.current_letter = None
            self.current_conf = None
            self.title_text.set("Sonuç: -")

        # Yazı overlay (ekranda da görünsün)
        cv2.putText(frame_show, f"Yazi: {self.text}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        self.show_on_gui(frame_show)

        self.root.after(30, self.update_camera)

    # ---------------- YAZI FONKSİYONLARI ----------------
    def add_letter(self):
        # Kamera açık değilse ekleme yapma
        if not self.running:
            messagebox.showinfo("Bilgi", "Önce kamerayı başlat.")
            return

        # Spam engeli
        # dummy call to avoid lint; not used
        now = self.root.winfo_fpixels('1i')
        ms = int(self.root.tk.call('clock', 'milliseconds'))
        if ms - self.last_add_time < ADD_COOLDOWN_MS:
            return

        if self.current_letter is None:
            return

        # İstersen conf kontrolü (çok düşükse ekleme)
        if self.current_conf is not None and self.current_conf < CONF_TH:
            return

        self.text += self.current_letter
        self.text_var.set(f"Yazı: {self.text}")
        self.last_add_time = ms

    def add_space(self):
        self.text += " "
        self.text_var.set(f"Yazı: {self.text}")

    def backspace(self):
        self.text = self.text[:-1]
        self.text_var.set(f"Yazı: {self.text}")

    def clear_text(self):
        self.text = ""
        self.text_var.set("Yazı: ")

    # ---------------- TAHMİN ----------------
    def predict_bgr_image(self, bgr_img, conf_th=0.25):
        if bgr_img is None:
            return None, None, None

        results = model.predict(source=bgr_img, conf=conf_th, verbose=False)
        r = results[0]
        annotated = r.plot()

        if r.boxes is None or len(r.boxes) == 0:
            return None, None, annotated

        confs = r.boxes.conf.tolist()
        best_i = confs.index(max(confs))
        cls_id = int(r.boxes.cls[best_i].item())
        label = r.names[cls_id]
        confidence = float(r.boxes.conf[best_i].item())

        return label, confidence, annotated

    # ---------------- GUI'DE GÖSTER ----------------
    def show_on_gui(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im = im.resize((720, 405))  # biraz daha büyük

        imgtk = ImageTk.PhotoImage(image=im)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
