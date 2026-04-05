import os
import json
import sqlite3
from datetime import date, datetime, time, timedelta
from hashlib import sha1
from typing import Optional, Tuple

from flask import Blueprint, jsonify, request, session

from services.chat_service import get_doctor_record, get_patient_record, get_socket_room


_BROKEN_SQLITE_PATHS = set()
_MEMORY_KEEPALIVE_CONNECTIONS = {}
_REGISTERED_SOCKETIOS = {}
VALID_APPOINTMENT_STATUSES = {"pending", "confirmed", "completed", "cancelled"}
ACTIVE_APPOINTMENT_STATUSES = {"pending", "confirmed"}
DEFAULT_APPOINTMENT_TIME_SLOTS = [
    "9:00 AM",
    "9:30 AM",
    "10:00 AM",
    "10:30 AM",
    "11:00 AM",
    "11:30 AM",
    "2:00 PM",
    "2:30 PM",
    "3:00 PM",
    "3:30 PM",
    "4:00 PM",
    "5:00 PM",
]


def _coerce_payload_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def _configure_connection(connection: sqlite3.Connection) -> None:
    for pragma in ("PRAGMA foreign_keys=ON", "PRAGMA journal_mode=WAL", "PRAGMA synchronous=NORMAL", "PRAGMA busy_timeout=30000"):
        try:
            connection.execute(pragma)
        except sqlite3.OperationalError:
            continue


def _shared_memory_uri(db_path: str) -> str:
    database_hash = sha1(os.path.abspath(db_path).encode("utf-8")).hexdigest()
    return f"file:neuro_appointments_{database_hash}?mode=memory&cache=shared"


def _open_memory_connection(db_path: str) -> sqlite3.Connection:
    resolved_path = os.path.abspath(db_path)
    uri = _shared_memory_uri(resolved_path)

    if resolved_path not in _MEMORY_KEEPALIVE_CONNECTIONS:
        keepalive_connection = sqlite3.connect(uri, timeout=30, uri=True)
        _configure_connection(keepalive_connection)
        _MEMORY_KEEPALIVE_CONNECTIONS[resolved_path] = keepalive_connection

    connection = sqlite3.connect(uri, timeout=30, uri=True)
    _configure_connection(connection)
    return connection


def _open_connection(db_path: str) -> sqlite3.Connection:
    resolved_path = os.path.abspath(db_path)
    if resolved_path in _BROKEN_SQLITE_PATHS:
        return _open_memory_connection(resolved_path)

    connection = sqlite3.connect(resolved_path, timeout=30)
    _configure_connection(connection)
    return connection


def _parse_int_field(value, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}.") from exc


def _normalize_specialization(value) -> str:
    normalized = str(value or "").strip()
    return normalized or "General Consultation"


def _slugify_label(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in str(value or ""))
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "general-consultation"


def _normalize_date_string(value) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("Date is required.")
    try:
        return datetime.strptime(normalized, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError("Invalid appointment date.") from exc


def _normalize_time_string(value) -> str:
    normalized = str(value or "").strip().upper().replace(".", "")
    if not normalized:
        raise ValueError("Time is required.")

    for time_format in ("%H:%M", "%I:%M %p", "%I:%M%p"):
        try:
            parsed_value = datetime.strptime(normalized, time_format)
            return parsed_value.strftime("%I:%M %p").lstrip("0")
        except ValueError:
            continue
    raise ValueError("Invalid appointment time.")


def _safe_date_sort_key(value) -> date:
    try:
        return datetime.strptime(str(value or ""), "%Y-%m-%d").date()
    except ValueError:
        return date.max


def _safe_time_sort_key(value) -> time:
    try:
        normalized_value = _normalize_time_string(value)
        return datetime.strptime(normalized_value, "%I:%M %p").time()
    except ValueError:
        return time.max


def _assign_patient_to_doctor(users_payload: dict, patient_id: int, doctor_id: int) -> Tuple[Optional[dict], Optional[dict]]:
    patient_record = get_patient_record(users_payload, patient_id)
    doctor_record = get_doctor_record(users_payload, doctor_id)
    if patient_record is None or doctor_record is None:
        return patient_record, doctor_record

    patient_record["assigned_doctor"] = int(doctor_id)

    for doctor in users_payload.get("doctors", []):
        if not isinstance(doctor, dict):
            continue

        patient_ids = []
        for value in doctor.get("patients", []):
            try:
                patient_ids.append(int(value))
            except (TypeError, ValueError):
                continue

        if int(doctor.get("id", -1)) == int(doctor_id):
            if int(patient_id) not in patient_ids:
                patient_ids.append(int(patient_id))
        else:
            patient_ids = [existing_id for existing_id in patient_ids if existing_id != int(patient_id)]

        doctor["patients"] = patient_ids

    return patient_record, doctor_record


def _emit_appointment_update(db_path: str, payload: dict) -> None:
    socketio = _REGISTERED_SOCKETIOS.get(os.path.abspath(db_path))
    if socketio is None:
        return

    socketio.emit("appointment_updated", payload, to=get_socket_room(payload["patient_id"]))
    socketio.emit("appointment_updated", payload, to=get_socket_room(payload["doctor_id"]))


def _serialize_appointment_row(row: sqlite3.Row, patient_lookup: dict, doctor_lookup: dict) -> dict:
    patient_record = patient_lookup.get(row["patient_id"], {})
    doctor_record = doctor_lookup.get(row["doctor_id"], {})
    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "patient_name": patient_record.get("full_name") or patient_record.get("username") or f"Patient {row['patient_id']}",
        "doctor_id": row["doctor_id"],
        "doctor_name": doctor_record.get("full_name") or doctor_record.get("username") or f"Doctor {row['doctor_id']}",
        "doctor_specialization": _normalize_specialization(doctor_record.get("specialization")),
        "doctor_hospital": str(doctor_record.get("hospital") or "").strip(),
        "doctor_experience": str(doctor_record.get("experience") or "").strip(),
        "date": row["date"],
        "time": row["time"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


def _create_appointment_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_appointments_patient_doctor_date
        ON appointments(patient_id, doctor_id, date, time)
        """
    )


def init_appointment_database(db_path: str) -> None:
    resolved_path = os.path.abspath(db_path)
    connection = None
    try:
        connection = _open_connection(resolved_path)
        _create_appointment_schema(connection)
        connection.commit()
    except sqlite3.OperationalError:
        _BROKEN_SQLITE_PATHS.add(resolved_path)
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass
        connection = _open_memory_connection(resolved_path)
        _create_appointment_schema(connection)
        connection.commit()
    finally:
        if connection is not None:
            connection.close()


def register_appointment_socketio(socketio, db_path: str = "appointments.sqlite3") -> None:
    resolved_db_path = os.path.abspath(db_path)
    init_appointment_database(resolved_db_path)
    _REGISTERED_SOCKETIOS[resolved_db_path] = socketio


def create_appointment_blueprint(
    login_required_factory,
    load_users_func,
    save_users_func=None,
    db_path: str = "appointments.sqlite3",
):
    resolved_db_path = os.path.abspath(db_path)
    init_appointment_database(resolved_db_path)

    appointment_bp = Blueprint("appointments", __name__)

    def get_connection():
        connection = _open_connection(resolved_db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def build_directory_maps():
        users_payload = load_users_func()
        patient_lookup = {
            int(patient.get("id")): patient
            for patient in users_payload.get("patients", [])
        }
        doctor_lookup = {
            int(doctor.get("id")): doctor
            for doctor in users_payload.get("doctors", [])
        }
        return users_payload, patient_lookup, doctor_lookup

    @appointment_bp.route("/appointment_options", methods=["GET"])
    @appointment_bp.route("/api/appointment_options", methods=["GET"])
    @login_required_factory("patient")
    def get_appointment_options():
        try:
            users_payload = load_users_func()
            doctor_payloads = []
            specialty_index = {}

            for doctor in users_payload.get("doctors", []):
                if not isinstance(doctor, dict):
                    continue

                try:
                    doctor_id = int(doctor.get("id"))
                except (TypeError, ValueError):
                    continue

                specialization = _normalize_specialization(doctor.get("specialization"))
                doctor_payload = {
                    "id": doctor_id,
                    "full_name": doctor.get("full_name") or doctor.get("username") or f"Doctor {doctor_id}",
                    "specialization": specialization,
                    "experience": str(doctor.get("experience") or "").strip(),
                    "hospital": str(doctor.get("hospital") or "").strip(),
                    "email": str(doctor.get("email") or "").strip(),
                    "license_number": str(doctor.get("license_number") or "").strip(),
                }
                doctor_payloads.append(doctor_payload)

                specialty_key = _slugify_label(specialization)
                specialty_entry = specialty_index.setdefault(
                    specialty_key,
                    {
                        "id": specialty_key,
                        "label": specialization,
                        "doctor_count": 0,
                    },
                )
                specialty_entry["doctor_count"] += 1

            doctor_payloads.sort(key=lambda item: (item["specialization"].lower(), item["full_name"].lower()))
            specialties = sorted(specialty_index.values(), key=lambda item: item["label"].lower())

            return jsonify(
                {
                    "success": True,
                    "specialties": specialties,
                    "doctors": doctor_payloads,
                }
            )
        except Exception as exc:
            return jsonify({"error": f"Failed to load appointment options: {exc}"}), 500

    @appointment_bp.route("/appointment_slots/<int:doctor_id>", methods=["GET"])
    @appointment_bp.route("/api/appointment_slots/<int:doctor_id>", methods=["GET"])
    @login_required_factory("patient")
    def get_appointment_slots(doctor_id: int):
        try:
            users_payload = load_users_func()
            doctor_record = get_doctor_record(users_payload, doctor_id)
            if doctor_record is None:
                return jsonify({"error": "Doctor not found."}), 404

            earliest_date = (date.today() + timedelta(days=1)).isoformat()
            booked_slots_by_date = {}
            with get_connection() as connection:
                rows = connection.execute(
                    """
                    SELECT date, time, status
                    FROM appointments
                    WHERE doctor_id = ? AND date >= ?
                    """,
                    (doctor_id, earliest_date),
                ).fetchall()

            for row in rows:
                if str(row["status"] or "").lower() not in ACTIVE_APPOINTMENT_STATUSES:
                    continue
                try:
                    normalized_time = _normalize_time_string(row["time"])
                except ValueError:
                    continue
                booked_slots_by_date.setdefault(row["date"], set()).add(normalized_time)

            slots = []
            for offset in range(1, 8):
                slot_date = date.today() + timedelta(days=offset)
                slot_date_value = slot_date.isoformat()
                booked_slots = booked_slots_by_date.get(slot_date_value, set())
                available_slots = [slot for slot in DEFAULT_APPOINTMENT_TIME_SLOTS if slot not in booked_slots]
                slots.append(
                    {
                        "date": slot_date_value,
                        "label": slot_date.strftime("%A, %B %d, %Y"),
                        "day": slot_date.strftime("%a"),
                        "day_number": slot_date.day,
                        "month": slot_date.strftime("%b"),
                        "available_slots": available_slots,
                        "booked_slots": sorted(booked_slots, key=_safe_time_sort_key),
                    }
                )

            return jsonify(
                {
                    "success": True,
                    "doctor": {
                        "id": doctor_id,
                        "full_name": doctor_record.get("full_name") or doctor_record.get("username") or f"Doctor {doctor_id}",
                        "specialization": _normalize_specialization(doctor_record.get("specialization")),
                    },
                    "slots": slots,
                }
            )
        except Exception as exc:
            return jsonify({"error": f"Failed to load appointment slots: {exc}"}), 500

    @appointment_bp.route("/book_appointment", methods=["POST"])
    @appointment_bp.route("/api/book_appointment", methods=["POST"])
    @login_required_factory()
    def book_appointment():
        try:
            payload = _coerce_payload_dict(request.get_json(silent=True))
            if payload is None:
                payload = request.form.to_dict() if request.form else None
            if not isinstance(payload, dict):
                return jsonify({"error": "Invalid request"}), 400
            current_user_id = int(session.get("user_id"))
            current_role = session.get("user_type")

            patient_id = _parse_int_field(
                payload.get("patient_id", current_user_id if current_role == "patient" else None),
                "patient_id",
            )
            doctor_id = _parse_int_field(
                payload.get("doctor_id", current_user_id if current_role == "doctor" else None),
                "doctor_id",
            )
            appointment_date = _normalize_date_string(payload.get("date"))
            appointment_time = _normalize_time_string(payload.get("time"))

            if current_role == "patient" and patient_id != current_user_id:
                return jsonify({"error": "Patient mismatch for the authenticated user."}), 403
            if current_role == "doctor" and doctor_id != current_user_id:
                return jsonify({"error": "Doctor mismatch for the authenticated user."}), 403

            users_payload = load_users_func()
            patient_record = get_patient_record(users_payload, patient_id)
            doctor_record = get_doctor_record(users_payload, doctor_id)
            if patient_record is None:
                return jsonify({"error": "Patient not found."}), 404
            if doctor_record is None:
                return jsonify({"error": "Doctor not found."}), 404

            with get_connection() as connection:
                conflicting_appointment = connection.execute(
                    """
                    SELECT id
                    FROM appointments
                    WHERE doctor_id = ? AND date = ? AND time = ? AND status IN (?, ?)
                    LIMIT 1
                    """,
                    (doctor_id, appointment_date, appointment_time, "pending", "confirmed"),
                ).fetchone()
                if conflicting_appointment is not None:
                    return jsonify({"error": "The selected time slot is no longer available."}), 409

                patient_record, doctor_record = _assign_patient_to_doctor(users_payload, patient_id, doctor_id)
                if save_users_func is not None:
                    save_users_func(users_payload)

                cursor = connection.execute(
                    """
                    INSERT INTO appointments (patient_id, doctor_id, date, time, status)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (patient_id, doctor_id, appointment_date, appointment_time, "pending"),
                )
                connection.commit()

            appointment_payload = {
                "id": cursor.lastrowid,
                "patient_id": patient_id,
                "patient_name": patient_record.get("full_name") or patient_record.get("username") or f"Patient {patient_id}",
                "doctor_id": doctor_id,
                "doctor_name": doctor_record.get("full_name") or doctor_record.get("username") or f"Doctor {doctor_id}",
                "doctor_specialization": _normalize_specialization(doctor_record.get("specialization")),
                "doctor_hospital": str(doctor_record.get("hospital") or "").strip(),
                "doctor_experience": str(doctor_record.get("experience") or "").strip(),
                "date": appointment_date,
                "time": appointment_time,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
            }
            _emit_appointment_update(resolved_db_path, appointment_payload)

            return jsonify(
                {
                    "success": True,
                    "appointment": appointment_payload,
                }
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to book appointment: {exc}"}), 500

    @appointment_bp.route("/get_appointments/<int:user_id>", methods=["GET"])
    @appointment_bp.route("/api/get_appointments/<int:user_id>", methods=["GET"])
    @login_required_factory()
    def get_appointments(user_id: int):
        try:
            current_user_id = int(session.get("user_id"))
            current_role = session.get("user_type")
            if user_id != current_user_id:
                return jsonify({"error": "Unauthorized appointment access."}), 403

            _, patient_lookup, doctor_lookup = build_directory_maps()
            query_field = "patient_id" if current_role == "patient" else "doctor_id"
            with get_connection() as connection:
                rows = connection.execute(
                    f"""
                    SELECT id, patient_id, doctor_id, date, time, status, created_at
                    FROM appointments
                    WHERE {query_field} = ?
                    ORDER BY date ASC, time ASC, id DESC
                    """,
                    (user_id,),
                ).fetchall()

            appointments = [_serialize_appointment_row(row, patient_lookup, doctor_lookup) for row in rows]
            appointments.sort(key=lambda item: (_safe_date_sort_key(item["date"]), _safe_time_sort_key(item["time"]), item["id"]))

            return jsonify({"success": True, "appointments": appointments, "count": len(appointments)})
        except Exception as exc:
            return jsonify({"error": f"Failed to fetch appointments: {exc}"}), 500

    @appointment_bp.route("/appointments/<int:appointment_id>/status", methods=["POST"])
    @appointment_bp.route("/api/appointments/<int:appointment_id>/status", methods=["POST"])
    @login_required_factory()
    def update_appointment_status(appointment_id: int):
        try:
            payload = _coerce_payload_dict(request.get_json(silent=True))
            if payload is None:
                payload = request.form.to_dict() if request.form else None
            if not isinstance(payload, dict):
                return jsonify({"error": "Invalid request"}), 400
            requested_status = str(payload.get("status") or "").strip().lower()
            if requested_status not in VALID_APPOINTMENT_STATUSES:
                return jsonify({"error": "Invalid appointment status."}), 400

            current_user_id = int(session.get("user_id"))
            current_role = session.get("user_type")

            with get_connection() as connection:
                appointment_row = connection.execute(
                    """
                    SELECT id, patient_id, doctor_id, date, time, status, created_at
                    FROM appointments
                    WHERE id = ?
                    """,
                    (appointment_id,),
                ).fetchone()

                if appointment_row is None:
                    return jsonify({"error": "Appointment not found."}), 404

                if current_role == "doctor":
                    if int(appointment_row["doctor_id"]) != current_user_id:
                        return jsonify({"error": "Unauthorized appointment update."}), 403
                elif current_role == "patient":
                    if int(appointment_row["patient_id"]) != current_user_id:
                        return jsonify({"error": "Unauthorized appointment update."}), 403
                    if requested_status != "cancelled":
                        return jsonify({"error": "Patients can only cancel appointments."}), 403
                else:
                    return jsonify({"error": "Unsupported appointment update role."}), 403

                connection.execute(
                    "UPDATE appointments SET status = ? WHERE id = ?",
                    (requested_status, appointment_id),
                )
                connection.commit()

                updated_row = connection.execute(
                    """
                    SELECT id, patient_id, doctor_id, date, time, status, created_at
                    FROM appointments
                    WHERE id = ?
                    """,
                    (appointment_id,),
                ).fetchone()

            _, patient_lookup, doctor_lookup = build_directory_maps()
            appointment_payload = _serialize_appointment_row(updated_row, patient_lookup, doctor_lookup)
            _emit_appointment_update(resolved_db_path, appointment_payload)

            return jsonify({"success": True, "appointment": appointment_payload})
        except Exception as exc:
            return jsonify({"error": f"Failed to update appointment: {exc}"}), 500

    return appointment_bp
