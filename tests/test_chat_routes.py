from services.chat_service import get_socket_room
from tests.helpers import login_session


def test_socket_rooms_are_role_aware_for_overlapping_ids():
    assert get_socket_room(1, "patient") == "user_patient_1"
    assert get_socket_room(1, "doctor") == "user_doctor_1"
    assert get_socket_room(1, "patient") != get_socket_room(1, "doctor")


def test_chat_message_status_progression(chat_app):
    app, socketio, _ = chat_app
    patient_client = app.test_client()
    doctor_client = app.test_client()

    login_session(patient_client, user_id=101, user_type="patient", full_name="Patient One")
    response = patient_client.post(
        "/chat/send",
        json={
            "sender_id": 101,
            "receiver_id": 201,
            "message": "Hello doctor",
        },
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "sent"

    login_session(doctor_client, user_id=201, user_type="doctor", full_name="Doctor One")
    socket_client = socketio.test_client(app, flask_test_client=doctor_client)
    assert socket_client.is_connected()
    socket_client.emit("register_user", {"user_id": 201})

    delivered_history = patient_client.get("/chat/history/201").get_json()
    assert delivered_history[0]["status"] == "delivered"

    seen_history = doctor_client.get("/chat/history/101").get_json()
    assert seen_history[0]["status"] == "seen"

    patient_history = patient_client.get("/chat/history/201").get_json()
    assert patient_history[0]["status"] == "seen"
    assert patient_history[0]["conversation_key"] == "101:201"

    socket_client.disconnect()


def test_chat_users_includes_unread_count(chat_app):
    app, _, _ = chat_app
    patient_client = app.test_client()
    doctor_client = app.test_client()
    login_session(patient_client, user_id=101, user_type="patient", full_name="Patient One")
    patient_client.post(
        "/chat/send",
        json={
            "sender_id": 101,
            "receiver_id": 201,
            "message": "Need guidance",
        },
    )

    login_session(doctor_client, user_id=201, user_type="doctor", full_name="Doctor One")
    response = doctor_client.get("/api/chat_users")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["users"][0]["unread_count"] >= 1
