from typing import Dict, Optional


CHAT_MESSAGE_TYPES = {"text", "image", "video", "audio"}


def resolve_counterpart_role(current_role: Optional[str]) -> str:
    return "doctor" if current_role == "patient" else "patient"


def normalize_message_text(message: object) -> str:
    return str(message or "").strip()


def normalize_message_type(message_type: object) -> str:
    normalized = str(message_type or "text").strip().lower()
    if normalized not in CHAT_MESSAGE_TYPES:
        return "text"
    return normalized


def normalize_file_url(file_url: object) -> Optional[str]:
    normalized = str(file_url or "").strip()
    return normalized or None


def get_patient_record(users: Dict[str, list], patient_source_id: int) -> Optional[dict]:
    return next(
        (patient for patient in users.get("patients", []) if int(patient.get("id", -1)) == int(patient_source_id)),
        None,
    )


def get_doctor_record(users: Dict[str, list], doctor_source_id: int) -> Optional[dict]:
    return next(
        (doctor for doctor in users.get("doctors", []) if int(doctor.get("id", -1)) == int(doctor_source_id)),
        None,
    )


def detect_user_role(users: Dict[str, list], source_id: int) -> Optional[str]:
    if get_patient_record(users, source_id) is not None:
        return "patient"
    if get_doctor_record(users, source_id) is not None:
        return "doctor"
    return None


def get_user_record(users: Dict[str, list], source_id: int, role: Optional[str] = None) -> Optional[dict]:
    detected_role = role or detect_user_role(users, source_id)
    if detected_role == "patient":
        return get_patient_record(users, source_id)
    if detected_role == "doctor":
        return get_doctor_record(users, source_id)
    return None


def get_display_name(record: Optional[dict], fallback_id: int) -> str:
    if record is None:
        return f"User {int(fallback_id)}"
    return str(record.get("full_name") or record.get("username") or f"User {int(fallback_id)}")


def ensure_patient_doctor_roles(sender_role: str, receiver_role: str) -> None:
    if sender_role == receiver_role:
        raise ValueError("Messages are only allowed between patients and doctors.")
    if {sender_role, receiver_role} != {"patient", "doctor"}:
        raise ValueError("Chat is restricted to patient-doctor conversations.")


def ensure_assigned_doctor(
    users: Dict[str, list],
    patient_source_id: int,
    doctor_source_id: int,
) -> bool:
    patient_record = get_patient_record(users, patient_source_id)
    doctor_record = get_doctor_record(users, doctor_source_id)
    if patient_record is None or doctor_record is None:
        return False

    assigned_doctor = patient_record.get("assigned_doctor")
    if assigned_doctor in (None, "", 0):
        patient_record["assigned_doctor"] = int(doctor_source_id)
    elif int(assigned_doctor) != int(doctor_source_id):
        return False

    doctor_patients = doctor_record.setdefault("patients", [])
    patient_ids = [int(value) for value in doctor_patients if value is not None]
    if int(patient_source_id) not in patient_ids:
        doctor_patients.append(int(patient_source_id))
    return True


def get_allowed_chat_user_ids(users: Dict[str, list], current_role: str, current_source_id: int) -> Optional[set[int]]:
    if current_role == "patient":
        patient_record = get_patient_record(users, current_source_id)
        if patient_record is None:
            return set()
        assigned_doctor = patient_record.get("assigned_doctor")
        if assigned_doctor in (None, "", 0):
            return None
        return {int(assigned_doctor)}

    doctor_record = get_doctor_record(users, current_source_id)
    if doctor_record is None:
        return set()

    patient_ids = set(int(patient_id) for patient_id in doctor_record.get("patients", []) if patient_id is not None)
    for patient in users.get("patients", []):
        assigned_doctor = patient.get("assigned_doctor")
        if assigned_doctor not in (None, "", 0) and int(assigned_doctor) == int(current_source_id):
            patient_ids.add(int(patient["id"]))
    return patient_ids or None


def get_socket_room(user_id: int) -> str:
    return f"user_{int(user_id)}"


def build_chat_payload(
    message_id: int,
    sender_id: int,
    sender_role: str,
    receiver_id: int,
    receiver_role: str,
    message: str,
    timestamp: str,
    sender_name: Optional[str] = None,
    receiver_name: Optional[str] = None,
    file_url: Optional[str] = None,
    message_type: str = "text",
) -> dict:
    return {
        "id": int(message_id),
        "sender_id": int(sender_id),
        "sender_role": sender_role,
        "sender_name": sender_name,
        "receiver_id": int(receiver_id),
        "receiver_role": receiver_role,
        "receiver_name": receiver_name,
        "message": message,
        "file_url": file_url,
        "type": normalize_message_type(message_type),
        "timestamp": timestamp,
    }
