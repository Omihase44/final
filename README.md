# NeuroDetect AI

NeuroDetect AI is a Flask-based medical imaging platform for brain tumor and Alzheimer-related screening workflows. It includes patient and doctor dashboards, scan upload and analysis flows, report management, appointment scheduling, and realtime chat.

## Features

- Brain tumor classification and staging
- Alzheimer stage prediction
- MRI enhancement and segmentation support
- Patient and doctor web portals
- Medical report generation
- Appointment booking and calendar views
- Realtime chat and status updates

## Tech Stack

- Python
- Flask
- Flask-SocketIO
- TensorFlow / Keras
- OpenCV
- SQLite
- Bootstrap

## Project Structure

```text
app.py                 Flask application entry point
routes/                API and feature routes
services/              Core business logic
models/                Model loading and inference helpers
training/              Training utilities and scripts
templates/             Patient/doctor HTML templates
static/                Static assets
tests/                 Test suite
```

## Getting Started

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the application

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

## Training

Tumor and Alzheimer training scripts live in `training/`.

Example:

```powershell
.\.venv-fresh\Scripts\python.exe training\train_tumor_model.py --backbone vgg16 --image-size 160 --batch-size 2 --epochs 30 --validation-split 0.25 --fine-tune-layers 4 --learning-rate 1e-5
```

## Notes

- Runtime files, uploads, datasets, logs, and local training outputs are excluded from Git.
- The repository currently includes baseline `.h5` model files used by the app at runtime.
