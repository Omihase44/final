import copy
import tempfile
from pathlib import Path

import pytest
from flask import Flask

from routes.appointment_routes import create_appointment_blueprint
from routes.chat_routes import create_chat_blueprint, register_chat_socketio


@pytest.fixture
def sample_users():
    return {
        "patients": [
            {
                "id": 101,
                "username": "patient_one",
                "full_name": "Patient One",
                "email": "patient@example.com",
                "phone": "9999999999",
                "age": "42",
                "assigned_doctor": 201,
                "created_at": "2026-04-05T10:00:00",
            }
        ],
        "doctors": [
            {
                "id": 201,
                "username": "doctor_one",
                "full_name": "Doctor One",
                "email": "doctor@example.com",
                "specialization": "Neurology",
                "hospital": "Neuro Hospital",
                "experience": "8",
                "patients": [101],
                "created_at": "2026-04-05T10:00:00",
            }
        ],
    }


@pytest.fixture
def chat_app(sample_users):
    users_state = copy.deepcopy(sample_users)
    upload_root = tempfile.mkdtemp(prefix="chat_uploads_")
    db_path = str(Path(tempfile.mkdtemp(prefix="chat_db_")) / "chat.sqlite3")

    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.testing = True

    def login_required(user_type=None):
        def decorator(func):
            from functools import wraps
            from flask import jsonify, session

            @wraps(func)
            def wrapped(*args, **kwargs):
                if "user_id" not in session:
                    return jsonify({"error": "Please login first"}), 401
                if user_type and session.get("user_type") != user_type:
                    return jsonify({"error": "Unauthorized access"}), 403
                return func(*args, **kwargs)

            return wrapped

        return decorator

    def load_users():
        return copy.deepcopy(users_state)

    def save_users(updated_users):
        users_state.clear()
        users_state.update(copy.deepcopy(updated_users))

    app.register_blueprint(
        create_chat_blueprint(
            load_users,
            login_required,
            db_path=db_path,
            save_users_func=save_users,
            upload_root=upload_root,
        )
    )

    socketio = pytest.importorskip("flask_socketio").SocketIO(app, async_mode="threading", cors_allowed_origins="*")
    register_chat_socketio(socketio, load_users, save_users_func=save_users, db_path=db_path)

    return app, socketio, db_path


@pytest.fixture
def appointment_app(sample_users):
    users_state = copy.deepcopy(sample_users)
    db_path = str(Path(tempfile.mkdtemp(prefix="appointment_db_")) / "appointments.sqlite3")

    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.testing = True

    def login_required(user_type=None):
        def decorator(func):
            from functools import wraps
            from flask import jsonify, session

            @wraps(func)
            def wrapped(*args, **kwargs):
                if "user_id" not in session:
                    return jsonify({"error": "Please login first"}), 401
                if user_type and session.get("user_type") != user_type:
                    return jsonify({"error": "Unauthorized access"}), 403
                return func(*args, **kwargs)

            return wrapped

        return decorator

    def load_users():
        return copy.deepcopy(users_state)

    def save_users(updated_users):
        users_state.clear()
        users_state.update(copy.deepcopy(updated_users))

    app.register_blueprint(
        create_appointment_blueprint(
            login_required,
            load_users,
            save_users_func=save_users,
            db_path=db_path,
        )
    )
    return app, db_path
