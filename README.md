# ASL Sign Language Recognition

Real-time American Sign Language (A–Z) letter detection using a custom YOLO model and a simple GUI (Tkinter + OpenCV). The user can detect letters from webcam or images and convert them into text.

## Features
- Webcam detection
- Image detection
- Confidence scores
- Text output
- Simple GUI

## Dataset
- ASL letters (A–Z)
- Source: Roboflow
## Model Results

Training curves:
![results](https://github.com/Al3ynaa/sign_language_yolo/blob/master/media/results.png%20%20(loss%20%2B%20mAP).png)

Confusion matrix:
![confusion](https://github.com/Al3ynaa/sign_language_yolo/blob/master/media/confusion_matrix.png.png)


## Run
```bash
pip install -r requirements.txt
python scripts/gui_app.py

Shortcuts

Enter = add letter

Space = space

Backspace = delete

C = clear

Q = stop camera
