# Realtime Posture Detection — YOLOv8 (Proyek Akhir Pemrosesan Data)

Streamlit app that watches your webcam and flags slouching ("Bungkuk") vs
good sitting posture ("Baik") in real time, using a YOLOv8 object-detection
model trained on a custom 2-class dataset.

## How it works

1. Webcam frame → YOLOv8 (`best.pt`) → bounding box + class (`sitting_good` /
   `sitting_bad`).
2. Frame is annotated with a status banner, bounding box, and confidence bar.
3. If "Bungkuk" persists for N consecutive frames (sensitivity slider), an
   MP3 alert plays, rate-limited to once every 8 seconds.

## Project layout

```
Deteksi Postur Tubuh (Streamlit)/
├── app.py                  # Streamlit app — run this
├── best.pt                 # trained YOLOv8 weights used by app.py
├── fahh.mp3                # alert sound
└── training/
    ├── posturedetection_training.ipynb   # training notebook
    ├── dataset/                          # Roboflow export (train/valid/test), 2 classes
    ├── yolov8n.pt, yolov8s.pt            # base weights
    └── runs/                             # training runs, metrics, confusion matrices
```

## Run the app

```bash
pip install streamlit ultralytics opencv-python-headless numpy
cd "Deteksi Postur Tubuh (Streamlit)"
streamlit run app.py
```

⚠️ Known issue: `app.py` plays `alert.mp3`, but the file in this repo is
`fahh.mp3` — rename one to match or the alert sound silently won't play.

⚠️ The webcam index is hardcoded (`cv2.VideoCapture(1)`) — on most laptops
the built-in camera is index `0`. Change it in `app.py` if the app can't
find your camera.

## Dataset & training

- Source: Roboflow project `abdullah-5vy0d`, 2 classes — `sitting_bad`,
  `sitting_good` (see `training/dataset/data.yaml`).
- Training notebook: `training/posturedetection_training.ipynb`.
- Both a `nano` and `small` YOLOv8 variant were trained (detection and
  classification runs under `training/runs/`), with metrics/plots
  (confusion matrix, PR curves, results.csv) saved per run.

## Requirements

- Python 3.9+
- `streamlit`, `ultralytics`, `opencv-python-headless`, `numpy`
- A webcam
