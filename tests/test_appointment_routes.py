from tests.helpers import login_session


def test_book_appointment_alias_persists_booking_fields(appointment_app):
    app, _ = appointment_app
    client = app.test_client()
    login_session(client, user_id=101, user_type="patient", full_name="Patient One")

    response = client.post(
        "/book-appointment",
        json={
            "patient_id": 101,
            "doctor_id": 201,
            "date": "2026-04-10",
            "time": "10:00 AM",
            "full_name": "Patient One",
            "phone": "9999999999",
            "age": "42",
            "reason": "Follow-up consultation",
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["appointment"]["patient_name"] == "Patient One"
    assert payload["appointment"]["phone"] == "9999999999"
    assert payload["appointment"]["age"] == "42"
    assert payload["appointment"]["reason"] == "Follow-up consultation"


def test_doctor_appointments_endpoint_returns_calendar(appointment_app):
    app, _ = appointment_app
    patient_client = app.test_client()
    login_session(patient_client, user_id=101, user_type="patient", full_name="Patient One")
    patient_client.post(
        "/book-appointment",
        json={
            "patient_id": 101,
            "doctor_id": 201,
            "date": "2026-04-10",
            "time": "10:00 AM",
            "full_name": "Patient One",
            "phone": "9999999999",
            "age": "42",
            "reason": "Consultation",
        },
    )

    doctor_client = app.test_client()
    login_session(doctor_client, user_id=201, user_type="doctor", full_name="Doctor One")
    response = doctor_client.get("/api/doctor-appointments?view=week&start=2026-04-06")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["view"] == "week"
    assert len(payload["calendar"]) == 7
    assert payload["appointments"][0]["patient_name"] == "Patient One"


def test_patient_cannot_confirm_appointment(appointment_app):
    app, _ = appointment_app
    patient_client = app.test_client()
    login_session(patient_client, user_id=101, user_type="patient", full_name="Patient One")
    booking = patient_client.post(
        "/book-appointment",
        json={
            "patient_id": 101,
            "doctor_id": 201,
            "date": "2026-04-10",
            "time": "11:00 AM",
        },
    ).get_json()

    response = patient_client.post(
        f"/api/appointments/{booking['appointment']['id']}/status",
        json={"status": "confirmed"},
    )

    assert response.status_code == 403
