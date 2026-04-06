import json
import sqlite3

from services.data_store import bootstrap_platform_data, sync_reports_to_db, sync_users_to_db


def test_sync_users_and_reports_to_db(tmp_path):
    db_path = tmp_path / "platform.sqlite3"
    users_payload = {
        "patients": [
            {"id": 11, "username": "patient", "full_name": "Patient Example", "assigned_doctor": 21}
        ],
        "doctors": [
            {"id": 21, "username": "doctor", "full_name": "Doctor Example", "patients": [11]}
        ],
    }
    reports_payload = [
        {"id": 7, "patient_id": 11, "type": "brain", "result": "no tumor", "status": "pending", "date": "2026-04-05T10:00:00"}
    ]

    bootstrap_platform_data(str(db_path), users_payload, reports_payload)

    with sqlite3.connect(db_path) as connection:
        user_rows = connection.execute("SELECT source_id, role, payload_json FROM users ORDER BY source_id").fetchall()
        report_rows = connection.execute("SELECT id, patient_id, payload_json FROM reports").fetchall()

    assert len(user_rows) == 2
    assert user_rows[0][0] == 11
    assert json.loads(user_rows[0][2])["full_name"] == "Patient Example"
    assert len(report_rows) == 1
    assert report_rows[0][0] == 7
    assert json.loads(report_rows[0][2])["result"] == "no tumor"


def test_sync_prunes_removed_rows(tmp_path):
    db_path = tmp_path / "platform.sqlite3"
    sync_users_to_db(
        str(db_path),
        {
            "patients": [{"id": 1, "username": "patient-a"}],
            "doctors": [{"id": 2, "username": "doctor-a"}],
        },
    )
    sync_reports_to_db(str(db_path), [{"id": 10, "patient_id": 1, "status": "pending"}])

    sync_users_to_db(str(db_path), {"patients": [], "doctors": []})
    sync_reports_to_db(str(db_path), [])

    with sqlite3.connect(db_path) as connection:
        user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        report_count = connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0]

    assert user_count == 0
    assert report_count == 0
