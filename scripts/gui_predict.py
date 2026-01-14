import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO

import cv2
from PIL import Image, ImageTk

MODEL_PATH = "runs/asl_sign_model3/weights/best.pt"

# Modeli 1 kere yükle
model = YOLO(MODEL_PATH)


def predict_image(img_path: str):
    results = model.predict(source=img_path, conf=0.25, save=False)
    r = results[0]

    # hiç tespit yoksa
    if r.boxes is None or len(r.boxes) == 0:
        return None, None, r.plot()

    confs = r.boxes.conf.tolist()
    best_i = confs.index(max(confs))

    cls_id = int(r.boxes.cls[best_i].item())
    label = r.names[cls_id]
    confidence = float(r.boxes.conf[best_i].item())

    return label, confidence, r.plot()


def show_image_on_tk(annotated_bgr):
    # BGR -> RGB
    rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    # pencereye sığdır
    img = img.resize((420, 320))

    imgtk = ImageTk.PhotoImage(image=img)
    image_label.config(image=imgtk)
    image_label.image = imgtk


def choose_and_predict():
    file_path = filedialog.askopenfilename(
        title="Bir resim seç",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    label, conf, annotated = predict_image(file_path)

    if label is None:
        messagebox.showinfo("Sonuç", "❌ Hiç harf tespit edemedim.")
        title_text.set("Sonuç: Bulunamadı")
    else:
        title_text.set(f"Sonuç: {label}  (confidence: {conf:.2f})")

    show_image_on_tk(annotated)


# ---------------- GUI ----------------
root = tk.Tk()
root.title("ASL Harf Tanıma - YOLO")
root.geometry("480x520")

title_text = tk.StringVar(value="Sonuç: -")
lbl = tk.Label(root, textvariable=title_text, font=("Arial", 16))
lbl.pack(pady=15)

btn = tk.Button(root, text="Resim Seç ve Tahmin Et",
                font=("Arial", 12), command=choose_and_predict)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

root.mainloop()
