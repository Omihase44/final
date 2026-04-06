from typing import Optional

from services.chat_service import get_socket_room


DOCTOR_ROOM = "role_doctors"
PATIENT_ROOM = "role_patients"


def get_role_room(role: Optional[str]) -> Optional[str]:
    normalized_role = str(role or "").strip().lower()
    if normalized_role == "doctor":
        return DOCTOR_ROOM
    if normalized_role == "patient":
        return PATIENT_ROOM
    return None


def join_default_socket_rooms(join_room, user_id: int, user_role: Optional[str] = None) -> None:
    join_room(get_socket_room(int(user_id), user_role))
    role_room = get_role_room(user_role)
    if role_room:
        join_room(role_room)


def _emit(socketio, event_name: str, payload: dict, room: Optional[str]) -> None:
    if socketio is None or not room:
        return
    socketio.emit(event_name, payload, to=room)


def _report_payload(report: dict, doctor_id: Optional[int] = None) -> dict:
    return {
        "report_id": report.get("id"),
        "patient_id": report.get("patient_id"),
        "patient_name": report.get("patient_name"),
        "doctor_id": doctor_id,
        "status": report.get("status"),
        "result": report.get("result"),
        "confidence": report.get("confidence") or report.get("ai_confidence"),
        "detection_type": report.get("type"),
        "date": report.get("date"),
        "sent_date": report.get("sent_date"),
        "reviewed_date": report.get("reviewed_date"),
        "approved_date": report.get("approved_date"),
        "rejected_date": report.get("rejected_date"),
    }


def emit_scan_uploaded(socketio, report: dict, doctor_id: Optional[int] = None) -> None:
    payload = _report_payload(report, doctor_id=doctor_id)
    if doctor_id is not None:
        _emit(socketio, "scan_uploaded", payload, get_socket_room(int(doctor_id), "doctor"))
    else:
        _emit(socketio, "scan_uploaded", payload, DOCTOR_ROOM)


def emit_report_status_updated(socketio, report: dict, doctor_id: Optional[int] = None) -> None:
    payload = _report_payload(report, doctor_id=doctor_id)
    _emit(socketio, "report_status_updated", payload, get_socket_room(int(report.get("patient_id", 0)), "patient"))
    if doctor_id is not None:
        _emit(socketio, "report_status_updated", payload, get_socket_room(int(doctor_id), "doctor"))
