import json
import os
import sqlite3
from contextlib import closing
from typing import Dict, Iterable, List


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


def _configure_connection(connection: sqlite3.Connection) -> None:
    for pragma in (
        "PRAGMA foreign_keys=ON",
        "PRAGMA journal_mode=MEMORY",
        "PRAGMA synchronous=OFF",
        "PRAGMA busy_timeout=30000",
    ):
        try:
            connection.execute(pragma)
        except sqlite3.OperationalError:
            continue


def _open_connection(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(os.path.abspath(db_path), timeout=30)
    _configure_connection(connection)
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('patient', 'doctor')),
            username TEXT,
            email TEXT,
            full_name TEXT,
            password_hash TEXT,
            age TEXT,
            gender TEXT,
            phone TEXT,
            address TEXT,
            medical_history TEXT,
            specialization TEXT,
            license_number TEXT,
            hospital TEXT,
            experience TEXT,
            assigned_doctor INTEGER,
            patients_json TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_id, role)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            detection_type TEXT,
            result TEXT,
            status TEXT,
            ai_confidence TEXT,
            report_date TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_users_source_role
        ON users(source_id, role)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reports_patient_status_date
        ON reports(patient_id, status, report_date)
        """
    )


def init_platform_database(db_path: str) -> None:
    with closing(_open_connection(db_path)) as connection:
        _create_schema(connection)
        connection.commit()


def _prune_rows_for_user_pairs(connection: sqlite3.Connection, valid_pairs: List[tuple[int, str]]) -> None:
    valid_ids_by_role = {"patient": set(), "doctor": set()}
    for source_id, role in valid_pairs:
        normalized_role = str(role or "").strip().lower()
        if normalized_role in valid_ids_by_role:
            valid_ids_by_role[normalized_role].add(int(source_id))

    for role, source_ids in valid_ids_by_role.items():
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            connection.execute(
                f"DELETE FROM users WHERE role = ? AND source_id NOT IN ({placeholders})",
                [role, *sorted(source_ids)],
            )
        else:
            connection.execute("DELETE FROM users WHERE role = ?", (role,))

    connection.execute(
        "DELETE FROM users WHERE role NOT IN (?, ?)",
        ("patient", "doctor"),
    )


def _prune_missing_rows(
    connection: sqlite3.Connection,
    table_name: str,
    key_column: str,
    current_ids: Iterable[int],
) -> None:
    current_ids = [int(value) for value in current_ids]
    if current_ids:
        placeholders = ",".join("?" for _ in current_ids)
        connection.execute(
            f"DELETE FROM {table_name} WHERE {key_column} NOT IN ({placeholders})",
            current_ids,
        )
    else:
        connection.execute(f"DELETE FROM {table_name}")


def sync_users_to_db(db_path: str, users_payload: Dict[str, List[dict]]) -> None:
    init_platform_database(db_path)
    with closing(_open_connection(db_path)) as connection:
        valid_pairs = []
        for role, collection_name in (("patient", "patients"), ("doctor", "doctors")):
            for user in users_payload.get(collection_name, []):
                try:
                    source_id = int(user.get("id"))
                except (TypeError, ValueError):
                    continue
                valid_pairs.append((source_id, role))
                connection.execute(
                    """
                    INSERT INTO users (
                        source_id, role, username, email, full_name, password_hash, age, gender,
                        phone, address, medical_history, specialization, license_number, hospital,
                        experience, assigned_doctor, patients_json, payload_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(source_id, role) DO UPDATE SET
                        username = excluded.username,
                        email = excluded.email,
                        full_name = excluded.full_name,
                        password_hash = excluded.password_hash,
                        age = excluded.age,
                        gender = excluded.gender,
                        phone = excluded.phone,
                        address = excluded.address,
                        medical_history = excluded.medical_history,
                        specialization = excluded.specialization,
                        license_number = excluded.license_number,
                        hospital = excluded.hospital,
                        experience = excluded.experience,
                        assigned_doctor = excluded.assigned_doctor,
                        patients_json = excluded.patients_json,
                        payload_json = excluded.payload_json,
                        created_at = excluded.created_at,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        source_id,
                        role,
                        user.get("username"),
                        user.get("email"),
                        user.get("full_name"),
                        user.get("password"),
                        user.get("age"),
                        user.get("gender"),
                        user.get("phone"),
                        user.get("address"),
                        user.get("medical_history"),
                        user.get("specialization"),
                        user.get("license_number"),
                        user.get("hospital"),
                        user.get("experience"),
                        user.get("assigned_doctor"),
                        _json_dumps(user.get("patients", [])),
                        _json_dumps(user),
                        user.get("created_at"),
                    ),
                )

        _prune_rows_for_user_pairs(connection, valid_pairs)
        connection.commit()


def sync_reports_to_db(db_path: str, reports_payload: List[dict]) -> None:
    init_platform_database(db_path)
    with closing(_open_connection(db_path)) as connection:
        report_ids = []
        for report in reports_payload:
            try:
                report_id = int(report.get("id"))
            except (TypeError, ValueError):
                continue
            report_ids.append(report_id)
            connection.execute(
                """
                INSERT INTO reports (
                    id, patient_id, detection_type, result, status, ai_confidence, report_date,
                    payload_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    patient_id = excluded.patient_id,
                    detection_type = excluded.detection_type,
                    result = excluded.result,
                    status = excluded.status,
                    ai_confidence = excluded.ai_confidence,
                    report_date = excluded.report_date,
                    payload_json = excluded.payload_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    report_id,
                    int(report.get("patient_id") or 0),
                    report.get("type"),
                    report.get("result"),
                    report.get("status"),
                    report.get("ai_confidence") or report.get("confidence"),
                    report.get("date"),
                    _json_dumps(report),
                ),
            )

        _prune_missing_rows(connection, "reports", "id", report_ids)
        connection.commit()


def bootstrap_platform_data(db_path: str, users_payload: Dict[str, List[dict]], reports_payload: List[dict]) -> None:
    init_platform_database(db_path)
    sync_users_to_db(db_path, users_payload)
    sync_reports_to_db(db_path, reports_payload)
