# Project Structure

This project is a Flask-based medical AI platform with patient, doctor, chat, appointment, and report flows.

## Top-Level Layout

```text
brain tumor and alzheimer/
|-- app.py
|-- requirements.txt
|-- users.json
|-- reports.json
|-- patient_details.json
|-- chat_store.sqlite3
|-- appointment_store.sqlite3
|-- brain_model_new.h5
|-- alz_model_new.h5
|-- routes/
|-- services/
|-- utils/
|-- templates/
|-- static/
|-- uploads/
|-- media/
|-- assets/
|-- models/
|-- dataset/
|-- b_tumor/
|-- .venv/
```

## Main Files And Folders

`app.py`
Main Flask application entry point. Handles auth, patient upload, reports, dashboard APIs, blueprint registration, and app startup.

`routes/`
Feature-specific Flask route modules.

`routes/analysis_routes.py`
Medical image analysis and `/analyze` handling.

`routes/chat_routes.py`
Real-time chat APIs, media upload, and Socket.IO chat integration.

`routes/appointment_routes.py`
Appointment booking, slot listing, status updates, and realtime appointment sync.

`services/`
Business logic and backend helpers used by the routes.

`services/classification.py`
Tumor and Alzheimer's classification coordination.

`services/enhancement.py`
MRI enhancement logic.

`services/segmentation.py`
Segmentation generation logic.

`services/volume_calc.py`
Tumor volume calculation helpers.

`services/chat_service.py`
Chat user resolution, permissions, and message payload helpers.

`services/report_generator.py`
PDF report creation and report image rendering.

`utils/`
Low-level shared helper modules.

`utils/image_processing.py`
Image decoding, preprocessing, and voxel metadata handling.

`utils/tensorflow_compat.py`
TensorFlow model loading compatibility helpers.

`utils/volume_calculation.py`
Volume-related utility functions.

`templates/`
HTML templates for patient and doctor dashboards.

`static/`
Static frontend assets.

`uploads/`
Uploaded medical scans and chat media runtime storage.

`users.json`
Patient and doctor account storage.

`reports.json`
Generated report records and analysis output history.

`patient_details.json`
Extra patient profile details and report mapping.

`chat_store.sqlite3`
Persistent chat message storage.

`appointment_store.sqlite3`
Persistent appointment storage.

## Runtime Flow

1. `app.py` starts Flask and registers blueprints.
2. Patient uploads a scan through the patient dashboard.
3. Analysis routes and services process the image.
4. Results are saved into `reports.json`.
5. Doctor dashboard reads reports and reviews them.
6. Chat and appointment modules use SQLite for realtime data sync.

## Active Core Structure

If you continue developing this app, the most important folders are:

- `routes/`
- `services/`
- `utils/`
- `templates/`
- `static/`
- `uploads/`

The main persistent app data files are:

- `users.json`
- `reports.json`
- `patient_details.json`
- `chat_store.sqlite3`
- `appointment_store.sqlite3`
