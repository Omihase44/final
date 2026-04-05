from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file, send_from_directory
from flask_cors import CORS
from functools import wraps
import os
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import json
import hashlib
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image as PILImage

from routes.analysis_routes import analysis_bp, analyze_medical_image
from routes.appointment_routes import create_appointment_blueprint, register_appointment_socketio
from routes.chat_routes import create_chat_blueprint, register_chat_socketio
from services.report_generator import build_medical_report_pdf
from utils.tensorflow_compat import safe_load_keras_model

try:
    from flask_socketio import SocketIO
except ImportError:  # pragma: no cover - startup fallback until dependency is installed
    class SocketIO:  # type: ignore
        def __init__(self, app=None, **kwargs):
            self.app = app

        def on(self, event_name):
            def decorator(func):
                return func
            return decorator

        def emit(self, *args, **kwargs):
            return None

        def run(self, app, *args, **kwargs):
            return app.run(*args, **kwargs)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _resolve_runtime_path(env_key, default_name):
    configured_value = os.environ.get(env_key)
    if configured_value:
        return configured_value if os.path.isabs(configured_value) else os.path.join(BASE_DIR, configured_value)
    return os.path.join(BASE_DIR, default_name)


def _ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _ensure_json_file(path, default_value):
    _ensure_parent_dir(path)
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as file_handle:
            json.dump(default_value, file_handle, indent=2)


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-only-secret-key')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

UPLOAD_FOLDER = _resolve_runtime_path("UPLOAD_FOLDER", "uploads")
CHAT_MEDIA_FOLDER = os.path.join(UPLOAD_FOLDER, "chat_media")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'dcm', 'dicom'}
USERS_FILE = _resolve_runtime_path("USERS_FILE", "users.json")
REPORTS_FILE = _resolve_runtime_path("REPORTS_FILE", "reports.json")
PATIENT_DETAILS_FILE = _resolve_runtime_path("PATIENT_DETAILS_FILE", "patient_details.json")
CHAT_DB_PATH = _resolve_runtime_path("CHAT_DB_PATH", "chat_store.sqlite3")
APPOINTMENT_DB_PATH = _resolve_runtime_path("APPOINTMENT_DB_PATH", "appointment_store.sqlite3")
DATA_SEED_FILES = (
    (USERS_FILE, {"doctors": [], "patients": []}),
    (REPORTS_FILE, []),
    (PATIENT_DETAILS_FILE, {}),
)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['CHAT_DB_PATH'] = CHAT_DB_PATH
app.config['APPOINTMENT_DB_PATH'] = APPOINTMENT_DB_PATH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHAT_MEDIA_FOLDER, exist_ok=True)
for seed_path, default_value in DATA_SEED_FILES:
    _ensure_json_file(seed_path, default_value)

# Global variables for models
brain_model = None
alz_model = None

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def usernames_match(left, right):
    return normalize_text(left) == normalize_text(right)


def format_confidence_percentage(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith('%'):
            return normalized
        try:
            value = float(normalized)
        except ValueError:
            return normalized

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if numeric_value <= 1:
        numeric_value *= 100
    formatted = f"{numeric_value:.2f}".rstrip('0').rstrip('.')
    return f"{formatted}%"


def _first_non_empty(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _extract_image_payload(value):
    if isinstance(value, dict):
        return (
            value.get('base64')
            or value.get('image')
            or value.get('image_base64')
            or value.get('data')
        )
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def safe_dict(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped:
            try:
                obj = json.loads(stripped)
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
            return obj if isinstance(obj, dict) else {}
    return {}


def _coerce_dict(value):
    return safe_dict(value)


def _coerce_list(value):
    return value if isinstance(value, list) else []


def _coerce_dict_list(value):
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _get_request_json_dict(error_message='Invalid JSON format'):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raw_data = request.get_data(cache=True, as_text=True)
        data = safe_dict(raw_data)
    if not isinstance(data, dict) or not data:
        raise ValueError(error_message)
    return data


def normalize_analysis_payload(analysis):
    analysis = _coerce_dict(analysis)
    if not analysis:
        return {}

    tumor_payload = _coerce_dict(analysis.get('tumor'))
    if tumor_payload:
        tumor_payload['confidence'] = format_confidence_percentage(tumor_payload.get('confidence'))
        if 'tumor_volume_mm3' in tumor_payload and 'volume_mm3' not in tumor_payload:
            tumor_payload['volume_mm3'] = tumor_payload.get('tumor_volume_mm3')
    analysis['tumor'] = tumor_payload

    alzheimers_payload = _coerce_dict(analysis.get('alzheimers'))
    if alzheimers_payload:
        alzheimers_payload['confidence'] = format_confidence_percentage(alzheimers_payload.get('confidence'))
    analysis['alzheimers'] = alzheimers_payload

    primary_result = _coerce_dict(analysis.get('primary_result'))
    if primary_result and 'confidence' in primary_result:
        primary_result['confidence'] = format_confidence_percentage(primary_result.get('confidence'))
    analysis['primary_result'] = primary_result

    analysis['tumor_confidence'] = format_confidence_percentage(analysis.get('tumor_confidence'))
    analysis['alzheimer_confidence'] = format_confidence_percentage(analysis.get('alzheimer_confidence'))
    analysis['confidence'] = format_confidence_percentage(analysis.get('confidence'))
    return analysis


def build_report_image_bundle(report):
    analysis = _coerce_dict(report.get('analysis'))
    analysis = normalize_analysis_payload(analysis) if analysis else {}
    analysis_images = _coerce_dict(analysis.get('images'))
    segmentation = _coerce_dict(analysis.get('segmentation'))
    enhancement = _coerce_dict(analysis.get('enhancement'))

    existing_images = _coerce_dict(report.get('report_images'))
    normalized_images = {
        'input_image': _first_non_empty(
            existing_images.get('input_image'),
            report.get('input_image'),
            report.get('image'),
        ),
        'original_mri': _first_non_empty(
            existing_images.get('original_mri'),
            report.get('original_mri'),
            report.get('original_image'),
            _extract_image_payload(analysis_images.get('original')),
            analysis.get('original_image_base64'),
            enhancement.get('original_image_base64'),
        ),
        'enhanced_mri': _first_non_empty(
            existing_images.get('enhanced_mri'),
            report.get('enhanced_mri'),
            report.get('enhanced_image'),
            _extract_image_payload(analysis_images.get('enhanced')),
            analysis.get('enhanced_image_base64'),
            enhancement.get('enhanced_image_base64'),
        ),
        'segmentation_overlay': _first_non_empty(
            existing_images.get('segmentation_overlay'),
            report.get('segmentation_overlay'),
            report.get('segmentation_image'),
            report.get('segmented_image'),
            _extract_image_payload(analysis_images.get('overlay')),
            _extract_image_payload(segmentation.get('overlay_image')),
            analysis.get('segmentation_image'),
        ),
        'segmentation_mask': _first_non_empty(
            existing_images.get('segmentation_mask'),
            report.get('segmentation_mask'),
            _extract_image_payload(analysis_images.get('mask')),
            _extract_image_payload(segmentation.get('mask_image')),
        ),
    }
    return normalized_images


def normalize_report_record(report):
    if not isinstance(report, dict):
        return report

    report['analysis'] = normalize_analysis_payload(report.get('analysis'))
    report_images = build_report_image_bundle(report)
    report['report_images'] = report_images

    report['image'] = _first_non_empty(report.get('image'), report_images.get('input_image'))
    report['input_image'] = _first_non_empty(report.get('input_image'), report_images.get('input_image'), report.get('image'))
    report['original_mri'] = _first_non_empty(report.get('original_mri'), report_images.get('original_mri'))
    report['enhanced_mri'] = _first_non_empty(report.get('enhanced_mri'), report_images.get('enhanced_mri'))
    report['segmentation_overlay'] = _first_non_empty(report.get('segmentation_overlay'), report_images.get('segmentation_overlay'))
    report['segmentation_mask'] = _first_non_empty(report.get('segmentation_mask'), report_images.get('segmentation_mask'))
    report['original_image'] = _first_non_empty(report.get('original_image'), report['original_mri'])
    report['enhanced_image'] = _first_non_empty(report.get('enhanced_image'), report['enhanced_mri'])
    report['segmentation_image'] = _first_non_empty(report.get('segmentation_image'), report['segmentation_overlay'])
    report['segmented_image'] = _first_non_empty(report.get('segmented_image'), report['segmentation_overlay'])

    tumor_confidence = format_confidence_percentage(
        _first_non_empty(
            report.get('tumor_confidence'),
            (report['analysis'].get('tumor') or {}).get('confidence'),
            _coerce_dict(report.get('model_confidences')).get('tumor'),
        )
    )
    alzheimer_confidence = format_confidence_percentage(
        _first_non_empty(
            report.get('alzheimer_confidence'),
            (report['analysis'].get('alzheimers') or {}).get('confidence'),
            _coerce_dict(report.get('model_confidences')).get('alzheimers'),
        )
    )
    ai_confidence = format_confidence_percentage(
        _first_non_empty(
            report.get('ai_confidence'),
            report.get('confidence'),
            (report['analysis'].get('primary_result') or {}).get('confidence'),
            report['analysis'].get('confidence'),
            tumor_confidence,
            alzheimer_confidence,
        )
    )

    report['tumor_confidence'] = tumor_confidence
    report['alzheimer_confidence'] = alzheimer_confidence
    report['ai_confidence'] = ai_confidence
    report['confidence'] = ai_confidence

    model_confidences = _coerce_dict(report.get('model_confidences'))
    if model_confidences:
        model_confidences['tumor'] = tumor_confidence
        model_confidences['alzheimers'] = alzheimer_confidence
        report['model_confidences'] = model_confidences

    analysis_tumor = report['analysis'].get('tumor')
    if isinstance(analysis_tumor, dict):
        analysis_tumor['confidence'] = tumor_confidence
    analysis_alzheimers = report['analysis'].get('alzheimers')
    if isinstance(analysis_alzheimers, dict):
        analysis_alzheimers['confidence'] = alzheimer_confidence
    if isinstance(report['analysis'].get('primary_result'), dict) and report['analysis']['primary_result'].get('confidence') is None:
        report['analysis']['primary_result']['confidence'] = ai_confidence
    report['analysis']['confidence'] = ai_confidence

    return report

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            loaded = safe_dict(json.load(f))
            return {
                "doctors": _coerce_dict_list(loaded.get("doctors")),
                "patients": _coerce_dict_list(loaded.get("patients")),
            }
    return {"doctors": [], "patients": []}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_reports():
    if os.path.exists(REPORTS_FILE):
        with open(REPORTS_FILE, 'r') as f:
            return [normalize_report_record(report) for report in _coerce_dict_list(json.load(f))]
    return []

def save_reports(reports):
    with open(REPORTS_FILE, 'w') as f:
        json.dump([normalize_report_record(report) for report in reports], f, indent=2)

def load_patient_details():
    if os.path.exists(PATIENT_DETAILS_FILE):
        with open(PATIENT_DETAILS_FILE, 'r') as f:
            return _coerce_dict(json.load(f))
    return {}

def save_patient_details(details):
    with open(PATIENT_DETAILS_FILE, 'w') as f:
        json.dump(details, f, indent=2)

# Authentication decorator
def login_required(user_type=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': 'Please login first'}), 401
            if user_type and session.get('user_type') != user_type:
                return jsonify({'error': 'Unauthorized access'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def load_brain_model():
    global brain_model
    if brain_model is None:
        brain_model = safe_load_keras_model(os.path.join(BASE_DIR, 'brain_model_new.h5'))
        if brain_model is None:
            print("Brain model not found. Using fallback mode.")
    return brain_model

def load_alz_model():
    global alz_model
    if alz_model is None:
        alz_model = safe_load_keras_model(os.path.join(BASE_DIR, 'alz_model_new.h5'))
        if alz_model is None:
            print("Alzheimer model not found. Using fallback mode.")
    return alz_model

def get_detailed_data(cls):
    data = {
        "glioma tumor": {
            "symptoms": "Severe headaches, nausea, vomiting, seizures, cognitive changes, vision problems, balance issues",
            "treatment": "Surgery, radiation therapy, chemotherapy, targeted therapy, regular MRI monitoring",
            "description": "Glioma tumors originate in the glial cells of the brain. They can be aggressive and require immediate medical attention.",
            "risk_factors": "Family history, age (45-65), radiation exposure",
            "prevention": "Regular check-ups, healthy lifestyle, avoid radiation exposure",
            "severity": "High",
            "urgency": "Immediate medical attention required"
        },
        "meningioma tumor": {
            "symptoms": "Headaches, hearing loss, vision problems, memory loss, seizures, weakness in limbs",
            "treatment": "Surgical removal, radiation therapy, observation for small tumors, regular follow-ups",
            "description": "Meningiomas are typically benign tumors that grow from the meninges, the protective layers around the brain.",
            "risk_factors": "Age (60+), female gender, genetic disorders, radiation exposure",
            "prevention": "Regular neurological check-ups, manage risk factors",
            "severity": "Medium",
            "urgency": "Consult neurologist soon"
        },
        "no tumor": {
            "symptoms": "No tumor-related symptoms detected",
            "treatment": "No treatment required. Regular health check-ups recommended",
            "description": "No abnormal growth detected in the brain scan. Continue maintaining a healthy lifestyle.",
            "risk_factors": "N/A",
            "prevention": "Maintain healthy diet, regular exercise, avoid smoking, limit alcohol",
            "severity": "None",
            "urgency": "No immediate action needed"
        },
        "pituitary tumor": {
            "symptoms": "Vision problems, headaches, hormonal imbalances, weight changes, fatigue, infertility",
            "treatment": "Surgery, radiation therapy, medication to control hormone levels, regular monitoring",
            "description": "Pituitary tumors develop in the pituitary gland and can affect hormone production throughout the body.",
            "risk_factors": "Genetic conditions, family history, age (30-40)",
            "prevention": "Regular endocrine check-ups, healthy lifestyle",
            "severity": "Medium-High",
            "urgency": "Schedule appointment with endocrinologist"
        },
        "MildDementia": {
            "symptoms": "Forgetfulness, difficulty finding words, losing things, trouble with planning, mood changes",
            "treatment": "Cognitive therapy, medication (cholinesterase inhibitors), lifestyle modifications, support groups",
            "description": "Early stage of dementia with noticeable cognitive decline that affects daily activities.",
            "risk_factors": "Age (65+), family history, cardiovascular disease, diabetes",
            "prevention": "Mental exercises, physical activity, healthy diet, social engagement",
            "severity": "Mild",
            "urgency": "Early intervention recommended"
        },
        "ModerateDementia": {
            "symptoms": "Significant memory loss, confusion, personality changes, difficulty with daily tasks, sleep disturbances",
            "treatment": "Medications, structured routine, caregiver support, safety modifications, occupational therapy",
            "description": "Moderate stage dementia with clear cognitive impairment requiring assistance with daily activities.",
            "risk_factors": "Age, genetics, previous head injuries, lifestyle factors",
            "prevention": "Early intervention, manage cardiovascular health, brain exercises",
            "severity": "Moderate",
            "urgency": "Medical care required"
        },
        "NonDementia": {
            "symptoms": "Normal cognitive function, no signs of dementia detected",
            "treatment": "No treatment required. Preventive measures recommended",
            "description": "No dementia indicators detected. Brain appears to be functioning normally.",
            "risk_factors": "N/A",
            "prevention": "Brain-healthy lifestyle: exercise, mental stimulation, social connection, Mediterranean diet",
            "severity": "None",
            "urgency": "No immediate action needed"
        },
        "VeryMildDementia": {
            "symptoms": "Very mild cognitive changes, occasional forgetfulness, minimal impact on daily life",
            "treatment": "Monitoring, cognitive exercises, lifestyle changes, regular follow-ups",
            "description": "Very early stage with subtle cognitive changes that may not significantly affect daily functioning.",
            "risk_factors": "Age, family history, cardiovascular health",
            "prevention": "Brain training, physical exercise, stress management, healthy diet",
            "severity": "Very Mild",
            "urgency": "Preventive measures recommended"
        },
        "MCI": {
            "symptoms": "Mild memory changes, subtle lapses in concentration, occasional difficulty with complex tasks",
            "treatment": "Monitoring, lifestyle changes, cognitive stimulation, regular neurological follow-up",
            "description": "Mild Cognitive Impairment is an early clinical stage where cognitive changes are present but daily independence is usually preserved.",
            "risk_factors": "Age, family history, cardiovascular disease, diabetes, prior brain injury",
            "prevention": "Regular exercise, sleep hygiene, cognitive activity, blood pressure and glucose control",
            "severity": "Very Mild",
            "urgency": "Early specialist review recommended"
        },
        "Early": {
            "symptoms": "Persistent forgetfulness, word-finding difficulty, reduced task planning and mild confusion",
            "treatment": "Cognitive therapy, medication review, memory support planning, regular clinical assessment",
            "description": "Early Alzheimer staging suggests clinically noticeable decline that benefits from early intervention and structured care planning.",
            "risk_factors": "Age, genetics, cardiovascular disease, sedentary lifestyle",
            "prevention": "Cognitive exercise, Mediterranean diet, physical activity, vascular risk control",
            "severity": "Mild",
            "urgency": "Neurology consultation advised"
        },
        "Moderate": {
            "symptoms": "Increasing confusion, functional decline, impaired daily activities, mood and behavior changes",
            "treatment": "Medication optimization, caregiver planning, occupational therapy, safety modifications",
            "description": "Moderate Alzheimer staging indicates meaningful cognitive and functional impairment requiring structured support.",
            "risk_factors": "Age, genetics, chronic disease burden, prior neurodegeneration",
            "prevention": "Early treatment adherence, structured routines, cardiovascular health management",
            "severity": "Moderate",
            "urgency": "Ongoing medical supervision required"
        },
        "Severe": {
            "symptoms": "Profound memory loss, communication difficulty, dependence for daily living, behavioral and mobility changes",
            "treatment": "Comprehensive dementia care, caregiver support, fall prevention, nutrition and safety management",
            "description": "Severe Alzheimer staging reflects advanced neurocognitive decline and high clinical care needs.",
            "risk_factors": "Progressive neurodegenerative disease",
            "prevention": "Early-stage management may slow progression, but advanced disease needs supportive care planning",
            "severity": "High",
            "urgency": "Specialist-led dementia care required"
        }
    }
    alias_map = {
        "Early Stage": "Early",
        "Moderate Stage": "Moderate",
        "Severe Stage": "Severe",
    }
    normalized_cls = alias_map.get(cls, cls)
    return data.get(normalized_cls, {
        "symptoms": "Unknown",
        "treatment": "Consult a specialist",
        "description": "Please consult with a healthcare professional for accurate diagnosis",
        "risk_factors": "Please consult a doctor",
        "prevention": "Regular health check-ups recommended",
        "severity": "Unknown",
        "urgency": "Consult doctor"
    })

app.register_blueprint(analysis_bp)
app.register_blueprint(
    create_chat_blueprint(
        load_users,
        login_required,
        app.config['CHAT_DB_PATH'],
        save_users,
        app.config['UPLOAD_FOLDER'],
    )
)
app.register_blueprint(
    create_appointment_blueprint(
        login_required,
        load_users,
        save_users,
        app.config['APPOINTMENT_DB_PATH']
    )
)
register_chat_socketio(socketio, load_users, save_users, app.config['CHAT_DB_PATH'])
register_appointment_socketio(socketio, app.config['APPOINTMENT_DB_PATH'])

# Routes
@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/healthz')
def healthcheck():
    return jsonify({'status': 'ok'}), 200

@app.route('/patient')
def patient_portal():
    return render_template('patient.html')

@app.route('/doctor')
def doctor_portal():
    return render_template('doctor.html')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/patient/register', methods=['POST'])
def patient_register():
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    username = normalize_text(data.get('username'))
    email = normalize_text(data.get('email'))
    password = data.get('password')
    full_name = normalize_text(data.get('full_name'))
    age = normalize_text(data.get('age'))
    gender = normalize_text(data.get('gender'))
    phone = normalize_text(data.get('phone'))
    address = normalize_text(data.get('address'))
    medical_history = normalize_text(data.get('medical_history'))
    
    if not all([username, email, password, full_name]):
        return jsonify({'error': 'All fields are required'}), 400
    
    users = load_users()
    
    if any(usernames_match(u.get('username'), username) for u in users['patients']):
        return jsonify({'error': 'Username already exists'}), 400
    
    new_patient = {
        'id': len(users['patients']) + 1,
        'username': username,
        'email': email,
        'password': hash_password(password),
        'full_name': full_name,
        'age': age,
        'gender': gender,
        'phone': phone,
        'address': address,
        'medical_history': medical_history,
        'created_at': datetime.now().isoformat(),
        'assigned_doctor': None
    }
    
    users['patients'].append(new_patient)
    save_users(users)
    
    patient_details = load_patient_details()
    patient_details[str(new_patient['id'])] = {
        'full_name': full_name,
        'age': age,
        'gender': gender,
        'phone': phone,
        'address': address,
        'medical_history': medical_history,
        'reports': []
    }
    save_patient_details(patient_details)
    
    return jsonify({'success': True, 'message': 'Registration successful'})

@app.route('/api/doctor/register', methods=['POST'])
def doctor_register():
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    username = normalize_text(data.get('username'))
    email = normalize_text(data.get('email'))
    password = data.get('password')
    full_name = normalize_text(data.get('full_name'))
    specialization = normalize_text(data.get('specialization'))
    license_number = normalize_text(data.get('license_number'))
    hospital = normalize_text(data.get('hospital'))
    experience = normalize_text(data.get('experience'))
    
    if not all([username, email, password, full_name, specialization, license_number]):
        return jsonify({'error': 'All fields are required'}), 400
    
    users = load_users()
    
    if any(usernames_match(u.get('username'), username) for u in users['doctors']):
        return jsonify({'error': 'Username already exists'}), 400
    
    new_doctor = {
        'id': len(users['doctors']) + 1,
        'username': username,
        'email': email,
        'password': hash_password(password),
        'full_name': full_name,
        'specialization': specialization,
        'license_number': license_number,
        'hospital': hospital,
        'experience': experience,
        'created_at': datetime.now().isoformat(),
        'patients': []
    }
    
    users['doctors'].append(new_doctor)
    save_users(users)
    
    return jsonify({'success': True, 'message': 'Registration successful'})

@app.route('/api/patient/login', methods=['POST'])
def patient_login():
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    username = normalize_text(data.get('username'))
    password = data.get('password')
    
    users = load_users()
    patient = next((p for p in users['patients'] if usernames_match(p.get('username'), username)), None)
    
    if patient and patient['password'] == hash_password(password):
        session['user_id'] = patient['id']
        session['username'] = normalize_text(patient['username'])
        session['user_type'] = 'patient'
        session['full_name'] = patient['full_name']
        
        return jsonify({
            'success': True,
            'user_type': 'patient',
            'username': normalize_text(patient['username']),
            'full_name': patient['full_name'],
            'patient_id': patient['id']
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/doctor/login', methods=['POST'])
def doctor_login():
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    username = normalize_text(data.get('username'))
    password = data.get('password')
    
    users = load_users()
    doctor = next((d for d in users['doctors'] if usernames_match(d.get('username'), username)), None)
    
    if doctor and doctor['password'] == hash_password(password):
        session['user_id'] = doctor['id']
        session['username'] = normalize_text(doctor['username'])
        session['user_type'] = 'doctor'
        session['full_name'] = doctor['full_name']
        
        return jsonify({
            'success': True,
            'user_type': 'doctor',
            'username': normalize_text(doctor['username']),
            'full_name': doctor['full_name'],
            'doctor_id': doctor['id']
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/check-auth')
def check_auth():
    if 'user_id' in session:
        payload = {
            'authenticated': True,
            'user_id': session.get('user_id'),
            'user_type': session.get('user_type'),
            'username': session.get('username'),
            'full_name': session.get('full_name')
        }
        if session.get('user_type') == 'patient':
            payload['patient_id'] = session.get('user_id')
        elif session.get('user_type') == 'doctor':
            payload['doctor_id'] = session.get('user_id')
        return jsonify(payload)
    return jsonify({'authenticated': False})

@app.route('/api/patient/upload', methods=['POST'])
@login_required('patient')
def patient_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    detection_type = request.form.get('type', 'brain')
    if detection_type not in ['brain', 'alz']:
        detection_type = 'brain'
    symptoms = request.form.get('symptoms', '')
    notes = request.form.get('notes', '')
    patient_name = request.form.get('patient_name', '')
    patient_age = request.form.get('patient_age', '')
    patient_gender = request.form.get('patient_gender', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            image_bytes = file.read()
            if not image_bytes:
                return jsonify({'error': 'Uploaded image is empty'}), 400

            analysis = analyze_medical_image(
                image_bytes=image_bytes,
                detection_type=detection_type,
                voxel_metadata={
                    'pixel_spacing_x': request.form.get('pixel_spacing_x'),
                    'pixel_spacing_y': request.form.get('pixel_spacing_y'),
                    'slice_thickness': request.form.get('slice_thickness')
                }
            )
            analysis = safe_dict(analysis)
            if not analysis:
                return jsonify({'error': 'Invalid analysis result'}), 500

            tumor_result = safe_dict(analysis.get('tumor'))
            alzheimer_result = safe_dict(analysis.get('alzheimers'))
            if not tumor_result or not alzheimer_result:
                return jsonify({'error': 'Invalid model output'}), 500

            if detection_type == 'brain':
                result = tumor_result.get('classification') or ('Tumor Detected' if tumor_result.get('detected') else 'No Tumor')
                confidence = format_confidence_percentage(tumor_result.get('confidence'))
            else:
                result = alzheimer_result.get('stage') if alzheimer_result.get('detected') else 'NonDementia'
                confidence = format_confidence_percentage(alzheimer_result.get('confidence'))

            detailed_info = get_detailed_data(result)
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            report_images = {
                'input_image': img_base64,
                'original_mri': analysis.get('original_image_base64') or img_base64,
                'enhanced_mri': analysis.get('enhanced_image_base64'),
                'segmentation_overlay': analysis.get('segmentation_image'),
                'segmentation_mask': analysis.get('segmentation_mask'),
            }
            
            reports = load_reports()
            report = normalize_report_record({
                'id': len(reports) + 1,
                'patient_id': session['user_id'],
                'patient_name': patient_name or session.get('full_name'),
                'patient_age': patient_age,
                'patient_gender': patient_gender,
                'type': detection_type,
                'result': result,
                'symptoms': symptoms,
                'notes': notes,
                'detailed_info': detailed_info,
                'date': datetime.now().isoformat(),
                'image': img_base64,
                'input_image': img_base64,
                'original_image': analysis.get('original_image_base64') or img_base64,
                'original_mri': analysis.get('original_image_base64') or img_base64,
                'original_image_path': analysis.get('original_image'),
                'enhanced_image': analysis.get('enhanced_image_base64'),
                'enhanced_mri': analysis.get('enhanced_image_base64'),
                'enhanced_image_path': analysis.get('enhanced_image'),
                'filename': filename,
                'status': 'pending',
                'doctor_notes': '',
                'prescription': '',
                'follow_up': '',
                'ai_result': result,
                'ai_confidence': confidence,
                'tumor_confidence': tumor_result.get('confidence'),
                'alzheimer_confidence': alzheimer_result.get('confidence'),
                'analysis': analysis,
                'tumor_detected': tumor_result.get('detected', False),
                'tumor_grade': tumor_result.get('grade'),
                'tumor_volume_mm3': tumor_result.get('volume_mm3'),
                'alzheimer_detected': alzheimer_result.get('detected', False),
                'alzheimer_stage': alzheimer_result.get('stage'),
                'segmentation_image': analysis.get('segmentation_image'),
                'segmentation_overlay': analysis.get('segmentation_image'),
                'segmented_image': analysis.get('segmentation_image'),
                'segmentation_mask': analysis.get('segmentation_mask'),
                'overlay_image_path': analysis.get('overlay_image'),
                'mask_image_path': analysis.get('mask_image'),
                'study_id': analysis.get('study_id'),
                'report_images': report_images,
                'asset_paths': {
                    'input_image': None,
                    'original_image': analysis.get('original_image'),
                    'enhanced_image': analysis.get('enhanced_image'),
                    'overlay_image': analysis.get('overlay_image'),
                    'mask_image': analysis.get('mask_image'),
                },
                'model_confidences': {
                    'tumor': tumor_result.get('confidence'),
                    'alzheimers': alzheimer_result.get('confidence'),
                }
            })
            
            reports.append(report)
            save_reports(reports)
            
            patient_details = load_patient_details()
            if str(session['user_id']) in patient_details:
                patient_details[str(session['user_id'])]['reports'].append(report['id'])
                save_patient_details(patient_details)
            
            return jsonify({
                'success': True,
                'result': result,
                'confidence': confidence,
                'detailed_info': detailed_info,
                'report_id': report['id'],
                'analysis': analysis,
                'segmentation_image': analysis.get('segmentation_image')
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/patient/reports')
@login_required('patient')
def get_patient_reports():
    reports = load_reports()
    patient_reports = [r for r in reports if r['patient_id'] == session['user_id']]
    patient_reports.sort(key=lambda x: x['date'], reverse=True)
    return jsonify({'reports': patient_reports})

@app.route('/api/patient/report/<int:report_id>')
@login_required('patient')
def get_patient_report(report_id):
    reports = load_reports()
    report = next((r for r in reports if r['id'] == report_id and r['patient_id'] == session['user_id']), None)
    if report:
        return jsonify({'report': report})
    return jsonify({'error': 'Report not found'}), 404

@app.route('/api/patient/report/<int:report_id>/download')
@login_required('patient')
def download_report_pdf(report_id):
    reports = load_reports()
    report = next((r for r in reports if r['id'] == report_id and r['patient_id'] == session['user_id']), None)
    
    if not report:
        return jsonify({'error': 'Report not found'}), 404

    buffer = build_medical_report_pdf(report)
    
    return send_file(buffer, as_attachment=True, download_name="medical_report_" + str(report_id) + ".pdf", mimetype='application/pdf')

@app.route('/api/doctor/patients')
@login_required('doctor')
def get_doctor_patients():
    reports = load_reports()
    users = load_users()
    
    patient_ids = list(set(r['patient_id'] for r in reports))
    patients = []
    
    for pid in patient_ids:
        patient = next((p for p in users['patients'] if p['id'] == pid), None)
        if patient:
            patient_reports = [r for r in reports if r['patient_id'] == pid]
            pending_reports = [r for r in patient_reports if r.get('status') == 'pending']
            approved_reports = [r for r in patient_reports if r.get('status') in ['approved', 'sent']]
            patients.append({
                'id': patient['id'],
                'name': patient['full_name'],
                'age': patient.get('age', 'N/A'),
                'gender': patient.get('gender', 'N/A'),
                'email': patient.get('email', ''),
                'phone': patient.get('phone', ''),
                'reports_count': len(patient_reports),
                'pending_reports': len(pending_reports),
                'approved_reports': len(approved_reports),
                'last_report': patient_reports[-1]['date'] if patient_reports else None
            })
    
    return jsonify({'patients': patients})

@app.route('/api/doctor/patient/<int:patient_id>')
@login_required('doctor')
def get_patient_details(patient_id):
    users = load_users()
    reports = load_reports()
    patient_details = load_patient_details()
    
    patient = next((p for p in users['patients'] if p['id'] == patient_id), None)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    patient_reports = [r for r in reports if r['patient_id'] == patient_id]
    patient_reports.sort(key=lambda x: x['date'], reverse=True)
    
    details = patient_details.get(str(patient_id), {})
    
    return jsonify({
        'patient': {
            'id': patient['id'],
            'full_name': patient['full_name'],
            'email': patient['email'],
            'age': patient.get('age', 'N/A'),
            'gender': patient.get('gender', 'N/A'),
            'phone': patient.get('phone', ''),
            'address': patient.get('address', ''),
            'medical_history': patient.get('medical_history', ''),
            'created_at': patient.get('created_at', '')
        },
        'reports': patient_reports,
        'additional_info': details
    })

@app.route('/api/doctor/report/<int:report_id>')
@login_required('doctor')
def get_report(report_id):
    reports = load_reports()
    report = next((r for r in reports if r['id'] == report_id), None)
    if report:
        return jsonify({'report': report})
    return jsonify({'error': 'Report not found'}), 404

@app.route('/api/doctor/report/<int:report_id>/update', methods=['POST'])
@login_required('doctor')
def update_report(report_id):
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    reports = load_reports()
    
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    report['doctor_notes'] = data.get('doctor_notes', '')
    report['prescription'] = data.get('prescription', '')
    report['follow_up'] = data.get('follow_up', '')
    report['status'] = 'reviewed'
    report['reviewed_by'] = session['full_name']
    report['reviewed_date'] = datetime.now().isoformat()
    
    save_reports(reports)
    
    return jsonify({'success': True, 'message': 'Report saved as draft'})

@app.route('/api/doctor/report/<int:report_id>/approve', methods=['POST'])
@login_required('doctor')
def approve_report(report_id):
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    reports = load_reports()
    
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    report['doctor_notes'] = data.get('doctor_notes', '')
    report['prescription'] = data.get('prescription', '')
    report['follow_up'] = data.get('follow_up', '')
    report['status'] = 'approved'
    report['approved_by'] = session['full_name']
    report['approved_date'] = datetime.now().isoformat()
    
    save_reports(reports)
    
    return jsonify({'success': True, 'message': 'Report approved'})

@app.route('/api/doctor/report/<int:report_id>/reject', methods=['POST'])
@login_required('doctor')
def reject_report(report_id):
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    reports = load_reports()
    
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    report['doctor_notes'] = data.get('doctor_notes', '')
    report['status'] = 'rejected'
    report['rejected_by'] = session['full_name']
    report['rejected_date'] = datetime.now().isoformat()
    
    save_reports(reports)
    
    return jsonify({'success': True, 'message': 'Report rejected'})

@app.route('/api/doctor/report/<int:report_id>/send', methods=['POST'])
@login_required('doctor')
def send_report_to_patient(report_id):
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    reports = load_reports()
    
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    report['doctor_notes'] = data.get('doctor_notes', '')
    report['prescription'] = data.get('prescription', '')
    report['follow_up'] = data.get('follow_up', '')
    report['status'] = 'sent'
    report['reviewed_by'] = session['full_name']
    report['sent_date'] = datetime.now().isoformat()
    
    save_reports(reports)
    
    return jsonify({'success': True, 'message': 'Report sent to patient'})

@app.route('/api/doctor/stats')
@login_required('doctor')
def get_doctor_stats():
    reports = load_reports()
    
    total_patients = len(set(r['patient_id'] for r in reports))
    pending_reports = len([r for r in reports if r.get('status') == 'pending'])
    approved_reports = len([r for r in reports if r.get('status') == 'approved'])
    sent_reports = len([r for r in reports if r.get('status') == 'sent'])
    
    return jsonify({
        'total_patients': total_patients,
        'pending_reports': pending_reports,
        'approved_reports': approved_reports,
        'sent_reports': sent_reports
    })


@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({
        "success": False,
        "error": str(e)
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    debug_mode = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'}
    socketio.run(app, debug=debug_mode, use_reloader=False, host='0.0.0.0', port=port)
