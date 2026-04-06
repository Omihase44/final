import os
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from hashlib import sha1
from typing import Dict, Optional
from threading import Lock

from flask import Blueprint, jsonify, request, session
from werkzeug.utils import secure_filename

from socket_handler import join_default_socket_rooms

try:
    from flask_socketio import join_room
except ImportError:  # pragma: no cover - optional until dependency is installed
    join_room = None

from services.chat_service import (
    build_conversation_key,
    build_chat_payload,
    ensure_assigned_doctor,
    ensure_patient_doctor_roles,
    get_allowed_chat_user_ids,
    get_display_name,
    get_socket_room,
    get_user_record,
    normalize_file_url,
    normalize_message_text,
    normalize_message_type,
    resolve_counterpart_role,
)


MAX_MESSAGE_LENGTH = 2000
CHAT_MEDIA_TYPES = {
    "image": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
    "audio": {".mp3", ".wav", ".ogg", ".aac", ".m4a"},
}
CHAT_MEDIA_ACCEPTED_EXTENSIONS = {
    extension for extensions in CHAT_MEDIA_TYPES.values() for extension in extensions
}
_BROKEN_SQLITE_PATHS = set()
_MEMORY_KEEPALIVE_CONNECTIONS = {}
_REGISTERED_SOCKETIOS = {}
_ACTIVE_SOCKET_USERS = {}
_ACTIVE_SOCKET_USERS_LOCK = Lock()


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
    try:
        connection.execute("PRAGMA foreign_keys=ON")
    except sqlite3.OperationalError:
        pass

    try:
        connection.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        try:
            connection.execute("PRAGMA journal_mode=MEMORY")
        except sqlite3.OperationalError:
            pass

    for pragma in ("PRAGMA synchronous=NORMAL", "PRAGMA busy_timeout=30000"):
        try:
            connection.execute(pragma)
        except sqlite3.OperationalError:
            continue


def _shared_memory_uri(db_path: str) -> str:
    database_hash = sha1(os.path.abspath(db_path).encode("utf-8")).hexdigest()
    return f"file:neuro_chat_{database_hash}?mode=memory&cache=shared"


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


def _ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_definition: str) -> None:
    existing_columns = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in existing_columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def _create_chat_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('patient', 'doctor')),
            username TEXT,
            full_name TEXT,
            email TEXT,
            created_at TEXT,
            UNIQUE(source_id, role)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            message TEXT NOT NULL DEFAULT '',
            file_url TEXT,
            type TEXT NOT NULL DEFAULT 'text',
            timestamp TEXT NOT NULL,
            FOREIGN KEY(sender_id) REFERENCES users(id),
            FOREIGN KEY(receiver_id) REFERENCES users(id)
        )
        """
    )
    _ensure_column(connection, "messages", "file_url", "TEXT")
    _ensure_column(connection, "messages", "type", "TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(connection, "messages", "status", "TEXT NOT NULL DEFAULT 'sent'")
    _ensure_column(connection, "messages", "delivered_at", "TEXT")
    _ensure_column(connection, "messages", "seen_at", "TEXT")
    _ensure_column(connection, "messages", "conversation_key", "TEXT")
    connection.execute(
        """
        UPDATE messages
        SET status = 'sent'
        WHERE status IS NULL OR TRIM(status) = ''
        """
    )
    connection.execute(
        """
        UPDATE messages
        SET conversation_key = (
            SELECT
                CASE
                    WHEN sender.source_id <= receiver.source_id
                    THEN printf('%d:%d', sender.source_id, receiver.source_id)
                    ELSE printf('%d:%d', receiver.source_id, sender.source_id)
                END
            FROM users AS sender, users AS receiver
            WHERE sender.id = messages.sender_id AND receiver.id = messages.receiver_id
        )
        WHERE conversation_key IS NULL OR TRIM(conversation_key) = ''
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_sender_receiver_timestamp
        ON messages(sender_id, receiver_id, timestamp)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp
        ON messages(timestamp)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_timestamp
        ON messages(conversation_key, timestamp)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_status
        ON messages(status)
        """
    )


def init_chat_database(db_path: str) -> None:
    resolved_path = os.path.abspath(db_path)
    connection = None
    try:
        connection = _open_connection(resolved_path)
        _create_chat_schema(connection)
        connection.commit()
    except sqlite3.OperationalError:
        _BROKEN_SQLITE_PATHS.add(resolved_path)
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass
        connection = _open_memory_connection(resolved_path)
        _create_chat_schema(connection)
        connection.commit()
    finally:
        if connection is not None:
            connection.close()


def sync_users_to_chat_db(db_path: str, users: Dict[str, list]) -> None:
    init_chat_database(db_path)
    connection = _open_connection(db_path)
    try:
        for role, collection_name in (("patient", "patients"), ("doctor", "doctors")):
            for user in users.get(collection_name, []):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO users (source_id, role, username, full_name, email, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user.get("id"),
                        role,
                        user.get("username"),
                        user.get("full_name"),
                        user.get("email"),
                        user.get("created_at"),
                    ),
                )
                connection.execute(
                    """
                    UPDATE users
                    SET username = ?, full_name = ?, email = ?, created_at = ?
                    WHERE source_id = ? AND role = ?
                    """,
                    (
                        user.get("username"),
                        user.get("full_name"),
                        user.get("email"),
                        user.get("created_at"),
                        user.get("id"),
                        role,
                    ),
                )
        connection.commit()
    finally:
        connection.close()


def _parse_int_field(value, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}.") from exc


def _parse_optional_int_field(value, field_name: str):
    if value in (None, ""):
        return None
    return _parse_int_field(value, field_name)


def _get_connection_with_rows(db_path: str) -> sqlite3.Connection:
    connection = _open_connection(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _get_chat_user(connection: sqlite3.Connection, source_id: int, role: str):
    return connection.execute(
        "SELECT * FROM users WHERE source_id = ? AND role = ?",
        (source_id, role),
    ).fetchone()


def _socket_user_key(user_id: int, user_role: Optional[str] = None) -> str:
    normalized_role = str(user_role or "").strip().lower() or "unknown"
    return f"{normalized_role}:{int(user_id)}"


def _track_active_socket(db_path: str, user_id: int, user_role: Optional[str], socket_id: str) -> None:
    resolved_path = os.path.abspath(db_path)
    with _ACTIVE_SOCKET_USERS_LOCK:
        user_sockets = _ACTIVE_SOCKET_USERS.setdefault(resolved_path, {})
        user_sockets.setdefault(_socket_user_key(user_id, user_role), set()).add(socket_id)


def _remove_active_socket(db_path: str, socket_id: str) -> None:
    resolved_path = os.path.abspath(db_path)
    with _ACTIVE_SOCKET_USERS_LOCK:
        user_sockets = _ACTIVE_SOCKET_USERS.get(resolved_path, {})
        empty_user_ids = []
        for user_id, socket_ids in user_sockets.items():
            if socket_id in socket_ids:
                socket_ids.discard(socket_id)
            if not socket_ids:
                empty_user_ids.append(user_id)
        for user_id in empty_user_ids:
            user_sockets.pop(user_id, None)
        if not user_sockets:
            _ACTIVE_SOCKET_USERS.pop(resolved_path, None)


def _is_user_online(db_path: str, user_id: int, user_role: Optional[str] = None) -> bool:
    resolved_path = os.path.abspath(db_path)
    with _ACTIVE_SOCKET_USERS_LOCK:
        socket_ids = (_ACTIVE_SOCKET_USERS.get(resolved_path) or {}).get(_socket_user_key(user_id, user_role), set())
        return bool(socket_ids)


def _build_status_payload(
    message_id: int,
    sender_id: int,
    sender_role: str,
    receiver_id: int,
    receiver_role: str,
    status: str,
    conversation_key: str,
    delivered_at: Optional[str] = None,
    seen_at: Optional[str] = None,
) -> dict:
    return {
        "message_id": int(message_id),
        "sender_id": int(sender_id),
        "sender_role": sender_role,
        "receiver_id": int(receiver_id),
        "receiver_role": receiver_role,
        "status": str(status or "sent").strip().lower(),
        "delivered_at": delivered_at,
        "seen_at": seen_at,
        "conversation_key": conversation_key,
    }


def _emit_message_status(db_path: str, payload: dict) -> None:
    socketio = _REGISTERED_SOCKETIOS.get(os.path.abspath(db_path))
    if socketio is None:
        return

    sender_room = get_socket_room(payload["sender_id"], payload.get("sender_role"))
    receiver_room = get_socket_room(payload["receiver_id"], payload.get("receiver_role"))
    socketio.emit("message_status", payload, to=sender_room)
    if receiver_room != sender_room:
        socketio.emit("message_status", payload, to=receiver_room)


def _broadcast_message(db_path: str, payload: dict) -> None:
    socketio = _REGISTERED_SOCKETIOS.get(os.path.abspath(db_path))
    if socketio is None:
        return

    sender_room = get_socket_room(payload["sender_id"], payload.get("sender_role"))
    receiver_room = get_socket_room(payload["receiver_id"], payload.get("receiver_role"))
    socketio.emit("receive_message", payload, to=sender_room)
    if receiver_room != sender_room:
        socketio.emit("receive_message", payload, to=receiver_room)


def _infer_media_type(filename: str, mimetype: Optional[str] = None) -> str:
    normalized_name = str(filename or "").lower()
    extension = os.path.splitext(normalized_name)[1]
    for media_type, extensions in CHAT_MEDIA_TYPES.items():
        if extension in extensions:
            return media_type
    if mimetype:
        major_type = mimetype.split("/", 1)[0].lower()
        if major_type in CHAT_MEDIA_TYPES:
            return major_type
    raise ValueError("Unsupported media type.")


def _serialize_message_row(row) -> dict:
    return {
        "id": row["id"],
        "sender_id": row["sender_source_id"],
        "sender_role": row["sender_role"],
        "sender_name": row["sender_name"],
        "receiver_id": row["receiver_source_id"],
        "receiver_role": row["receiver_role"],
        "receiver_name": row["receiver_name"],
        "message": row["message"],
        "file_url": row["file_url"],
        "type": row["type"],
        "timestamp": row["timestamp"],
        "status": row["status"],
        "delivered_at": row["delivered_at"],
        "seen_at": row["seen_at"],
        "conversation_key": row["conversation_key"],
    }


def _mark_messages_delivered(db_path: str, receiver_source_id: int, receiver_role: Optional[str]) -> list[dict]:
    if receiver_role not in {"patient", "doctor"}:
        return []

    with closing(_get_connection_with_rows(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT
                messages.id,
                sender.source_id AS sender_source_id,
                sender.role AS sender_role,
                receiver.source_id AS receiver_source_id,
                receiver.role AS receiver_role,
                messages.conversation_key
            FROM messages
            JOIN users AS sender ON sender.id = messages.sender_id
            JOIN users AS receiver ON receiver.id = messages.receiver_id
            WHERE receiver.source_id = ?
              AND receiver.role = ?
              AND (messages.status IS NULL OR messages.status = 'sent')
            """,
            (receiver_source_id, receiver_role),
        ).fetchall()
        if not rows:
            return []

        delivered_at = datetime.now().isoformat()
        message_ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in message_ids)
        connection.execute(
            f"""
            UPDATE messages
            SET status = 'delivered',
                delivered_at = COALESCE(delivered_at, ?)
            WHERE id IN ({placeholders})
            """,
            [delivered_at, *message_ids],
        )
        connection.commit()

    return [
        _build_status_payload(
            message_id=row["id"],
            sender_id=row["sender_source_id"],
            sender_role=row["sender_role"],
            receiver_id=row["receiver_source_id"],
            receiver_role=row["receiver_role"],
            status="delivered",
            delivered_at=delivered_at,
            seen_at=None,
            conversation_key=row["conversation_key"] or build_conversation_key(row["sender_source_id"], row["receiver_source_id"]),
        )
        for row in rows
    ]


def _mark_conversation_seen(
    db_path: str,
    current_source_id: int,
    current_role: str,
    counterpart_source_id: int,
    counterpart_role: str,
) -> list[dict]:
    with closing(_get_connection_with_rows(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT
                messages.id,
                sender.source_id AS sender_source_id,
                sender.role AS sender_role,
                receiver.source_id AS receiver_source_id,
                receiver.role AS receiver_role,
                messages.delivered_at,
                messages.conversation_key
            FROM messages
            JOIN users AS sender ON sender.id = messages.sender_id
            JOIN users AS receiver ON receiver.id = messages.receiver_id
            WHERE sender.source_id = ?
              AND sender.role = ?
              AND receiver.source_id = ?
              AND receiver.role = ?
              AND COALESCE(messages.status, 'sent') != 'seen'
            """,
            (counterpart_source_id, counterpart_role, current_source_id, current_role),
        ).fetchall()
        if not rows:
            return []

        event_time = datetime.now().isoformat()
        message_ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in message_ids)
        connection.execute(
            f"""
            UPDATE messages
            SET status = 'seen',
                delivered_at = COALESCE(delivered_at, ?),
                seen_at = ?
            WHERE id IN ({placeholders})
            """,
            [event_time, event_time, *message_ids],
        )
        connection.commit()

    return [
        _build_status_payload(
            message_id=row["id"],
            sender_id=row["sender_source_id"],
            sender_role=row["sender_role"],
            receiver_id=row["receiver_source_id"],
            receiver_role=row["receiver_role"],
            status="seen",
            delivered_at=row["delivered_at"] or event_time,
            seen_at=event_time,
            conversation_key=row["conversation_key"] or build_conversation_key(row["sender_source_id"], row["receiver_source_id"]),
        )
        for row in rows
    ]


def _create_message_payload(
    db_path: str,
    load_users_func,
    save_users_func,
    sender_source_id: int,
    receiver_source_id: int,
    message: str,
    authenticated_user_id: Optional[int],
    authenticated_role: Optional[str],
    message_type: str = "text",
    file_url: Optional[str] = None,
) -> dict:
    if authenticated_user_id is None or authenticated_role not in {"patient", "doctor"}:
        raise PermissionError("Please login first.")
    if sender_source_id != int(authenticated_user_id):
        raise PermissionError("Sender mismatch for the authenticated user.")

    message = normalize_message_text(message)
    file_url = normalize_file_url(file_url)
    message_type = normalize_message_type(message_type)
    if not message and not file_url:
        raise ValueError("Message or media is required.")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError("Message is too long.")
    if file_url and message_type == "text":
        message_type = _infer_media_type(file_url)
    if message_type != "text" and not file_url:
        raise ValueError("Media messages require a file URL.")

    users_payload = load_users_func()
    sender_record = get_user_record(users_payload, sender_source_id, authenticated_role)
    receiver_role = resolve_counterpart_role(authenticated_role)
    receiver_record = get_user_record(users_payload, receiver_source_id, receiver_role)
    if sender_record is None:
        raise ValueError("Authenticated sender not found.")
    if receiver_record is None:
        raise ValueError("Receiver must be a valid patient or doctor.")

    ensure_patient_doctor_roles(authenticated_role, receiver_role)

    if authenticated_role == "patient":
        if not ensure_assigned_doctor(users_payload, sender_source_id, receiver_source_id):
            raise PermissionError("Patient messages must target the assigned doctor.")
        if save_users_func is not None:
            save_users_func(users_payload)
    else:
        allowed_patient_ids = get_allowed_chat_user_ids(users_payload, authenticated_role, sender_source_id)
        if allowed_patient_ids is not None and receiver_source_id not in allowed_patient_ids:
            raise PermissionError("Doctor can only message linked patients.")

    sync_users_to_chat_db(db_path, users_payload)
    timestamp = datetime.now().isoformat()
    with closing(_get_connection_with_rows(db_path)) as connection:
        sender = _get_chat_user(connection, sender_source_id, authenticated_role)
        receiver = _get_chat_user(connection, receiver_source_id, receiver_role)
        if sender is None:
            raise ValueError("Authenticated user not found in chat directory.")
        if receiver is None:
            raise ValueError("Receiver must be a valid patient or doctor.")
        if sender["id"] == receiver["id"]:
            raise ValueError("Sender and receiver must be different users.")
        ensure_patient_doctor_roles(sender["role"], receiver["role"])
        conversation_key = build_conversation_key(sender_source_id, receiver_source_id)
        status = "delivered" if _is_user_online(db_path, receiver_source_id, receiver_role) else "sent"
        delivered_at = timestamp if status == "delivered" else None

        cursor = connection.execute(
            """
            INSERT INTO messages (
                sender_id, receiver_id, message, file_url, type, timestamp,
                status, delivered_at, seen_at, conversation_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sender["id"],
                receiver["id"],
                message,
                file_url,
                message_type,
                timestamp,
                status,
                delivered_at,
                None,
                conversation_key,
            ),
        )
        connection.commit()

    payload = build_chat_payload(
        message_id=cursor.lastrowid,
        sender_id=sender_source_id,
        sender_role=authenticated_role,
        sender_name=get_display_name(sender_record, sender_source_id),
        receiver_id=receiver_source_id,
        receiver_role=receiver_role,
        receiver_name=get_display_name(receiver_record, receiver_source_id),
        message=message,
        file_url=file_url,
        message_type=message_type,
        timestamp=timestamp,
        status=status,
        delivered_at=delivered_at,
        seen_at=None,
        conversation_key=conversation_key,
    )
    _broadcast_message(db_path, payload)
    if status != "sent":
        _emit_message_status(
            db_path,
            _build_status_payload(
                message_id=cursor.lastrowid,
                sender_id=sender_source_id,
                sender_role=authenticated_role,
                receiver_id=receiver_source_id,
                receiver_role=receiver_role,
                status=status,
                delivered_at=delivered_at,
                seen_at=None,
                conversation_key=conversation_key,
            ),
        )
    return payload


def register_chat_socketio(socketio, load_users_func, save_users_func=None, db_path: str = "db.sqlite3") -> None:
    resolved_db_path = os.path.abspath(db_path)
    init_chat_database(resolved_db_path)
    _REGISTERED_SOCKETIOS[resolved_db_path] = socketio

    if join_room is None:
        return

    @socketio.on("connect")
    def handle_connect():
        user_id = session.get("user_id")
        if user_id:
            join_default_socket_rooms(join_room, int(user_id), session.get("user_type"))
            _track_active_socket(resolved_db_path, int(user_id), session.get("user_type"), request.sid)
            for status_payload in _mark_messages_delivered(resolved_db_path, int(user_id), session.get("user_type")):
                _emit_message_status(resolved_db_path, status_payload)

    @socketio.on("register_user")
    def handle_register_user(data):
        user_id = session.get("user_id")
        if user_id is None:
            return {"error": "Please login first."}

        requested_user_id = None
        if isinstance(data, dict) and data.get("user_id") not in (None, ""):
            requested_user_id = _parse_int_field(data.get("user_id"), "user_id")
        if requested_user_id is not None and requested_user_id != int(user_id):
            return {"error": "User registration mismatch."}

        join_default_socket_rooms(join_room, int(user_id), session.get("user_type"))
        _track_active_socket(resolved_db_path, int(user_id), session.get("user_type"), request.sid)
        delivered_payloads = _mark_messages_delivered(resolved_db_path, int(user_id), session.get("user_type"))
        for status_payload in delivered_payloads:
            _emit_message_status(resolved_db_path, status_payload)
        return {"success": True, "user_id": int(user_id)}

    @socketio.on("disconnect")
    def handle_disconnect():
        _remove_active_socket(resolved_db_path, request.sid)

    @socketio.on("send_message")
    def handle_socket_message(data):
        try:
            payload = _coerce_payload_dict(data)
            if payload is None:
                return {"error": "Invalid JSON"}
            message_payload = _create_message_payload(
                db_path=resolved_db_path,
                load_users_func=load_users_func,
                save_users_func=save_users_func,
                sender_source_id=_parse_int_field(payload.get("sender_id"), "sender_id"),
                receiver_source_id=_parse_int_field(payload.get("receiver_id"), "receiver_id"),
                message=payload.get("message", ""),
                file_url=payload.get("file_url"),
                message_type=payload.get("type", "text"),
                authenticated_user_id=session.get("user_id"),
                authenticated_role=session.get("user_type"),
            )
            return {"success": True, "message": message_payload}
        except PermissionError as exc:
            return {"error": str(exc)}
        except (ValueError, TypeError) as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Failed to send message: {exc}"}


def create_chat_blueprint(
    load_users_func,
    login_required_factory,
    db_path: str = "db.sqlite3",
    save_users_func=None,
    upload_root: str = "uploads",
):
    resolved_db_path = os.path.abspath(db_path)
    resolved_upload_root = os.path.abspath(upload_root)
    chat_media_root = os.path.join(resolved_upload_root, "chat_media")
    os.makedirs(chat_media_root, exist_ok=True)
    init_chat_database(resolved_db_path)

    chat_bp = Blueprint("chat", __name__)

    def sync_chat_users():
        sync_users_to_chat_db(resolved_db_path, load_users_func())

    @chat_bp.route("/chat_users", methods=["GET"])
    @chat_bp.route("/api/chat_users", methods=["GET"])
    @login_required_factory()
    def get_chat_users():
        try:
            sync_chat_users()
            current_role = session.get("user_type")
            counterpart_role = resolve_counterpart_role(current_role)
            current_source_id = int(session.get("user_id"))
            users_payload = load_users_func()
            allowed_counterpart_ids = get_allowed_chat_user_ids(users_payload, current_role, current_source_id)

            with closing(_get_connection_with_rows(resolved_db_path)) as connection:
                current_user = _get_chat_user(connection, current_source_id, current_role)
                if current_user is None:
                    return jsonify({"error": "Authenticated user not found in chat directory."}), 404

                rows = connection.execute(
                    """
                    SELECT
                        users.source_id,
                        users.role,
                        users.username,
                        users.full_name,
                        users.email,
                        users.created_at,
                        MAX(messages.timestamp) AS last_message_at,
                        COUNT(messages.id) AS message_count,
                        SUM(
                            CASE
                                WHEN messages.receiver_id = ? AND COALESCE(messages.status, 'sent') != 'seen'
                                THEN 1
                                ELSE 0
                            END
                        ) AS unread_count
                    FROM users
                    LEFT JOIN messages
                        ON (
                            (messages.sender_id = ? AND messages.receiver_id = users.id)
                            OR (messages.sender_id = users.id AND messages.receiver_id = ?)
                        )
                    WHERE users.role = ?
                    GROUP BY users.id, users.source_id, users.role, users.username, users.full_name, users.email, users.created_at
                    ORDER BY COALESCE(MAX(messages.timestamp), users.created_at) DESC, users.full_name ASC
                    """,
                    (current_user["id"], current_user["id"], current_user["id"], counterpart_role),
                ).fetchall()

            chat_users = [
                {
                    "id": row["source_id"],
                    "role": row["role"],
                    "username": row["username"],
                    "full_name": row["full_name"],
                    "email": row["email"],
                    "created_at": row["created_at"],
                    "last_message_at": row["last_message_at"],
                    "message_count": row["message_count"],
                    "unread_count": row["unread_count"] or 0,
                }
                for row in rows
                if allowed_counterpart_ids is None or row["source_id"] in allowed_counterpart_ids
            ]

            return jsonify(
                {
                    "success": True,
                    "users": chat_users,
                    "count": len(chat_users),
                }
            )
        except Exception as exc:
            return jsonify({"error": f"Failed to fetch chat users: {exc}"}), 500

    @chat_bp.route("/upload_media", methods=["POST"])
    @chat_bp.route("/api/upload_media", methods=["POST"])
    @login_required_factory()
    def upload_media():
        try:
            uploaded_file = (
                request.files.get("file")
                or request.files.get("media")
                or request.files.get("attachment")
            )
            if uploaded_file is None or uploaded_file.filename == "":
                return jsonify({"error": "No media file provided."}), 400

            safe_name = secure_filename(uploaded_file.filename)
            if not safe_name:
                return jsonify({"error": "Invalid media filename."}), 400

            media_type = request.form.get("type")
            if media_type:
                media_type = normalize_message_type(media_type)
            else:
                media_type = _infer_media_type(safe_name, uploaded_file.mimetype)

            extension = os.path.splitext(safe_name)[1].lower()
            if extension not in CHAT_MEDIA_ACCEPTED_EXTENSIONS:
                return jsonify({"error": "Unsupported media file extension."}), 400

            unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{safe_name}"
            saved_path = os.path.join(chat_media_root, unique_name)
            uploaded_file.save(saved_path)
            file_url = f"/uploads/chat_media/{unique_name}"

            return jsonify(
                {
                    "success": True,
                    "file_url": file_url,
                    "url": file_url,
                    "type": media_type,
                    "filename": unique_name,
                }
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to upload media: {exc}"}), 500

    @chat_bp.route("/send_message", methods=["POST"])
    @chat_bp.route("/api/send_message", methods=["POST"])
    @chat_bp.route("/chat/send", methods=["POST"])
    @chat_bp.route("/api/chat/send", methods=["POST"])
    @login_required_factory()
    def send_message():
        try:
            sync_chat_users()
            payload = _coerce_payload_dict(request.get_json(silent=True))
            if payload is None:
                payload = request.form.to_dict() if request.form else None
            if not isinstance(payload, dict):
                return jsonify({"error": "Invalid JSON"}), 400

            message_payload = _create_message_payload(
                db_path=resolved_db_path,
                load_users_func=load_users_func,
                save_users_func=save_users_func,
                sender_source_id=_parse_int_field(payload.get("sender_id", session.get("user_id")), "sender_id"),
                receiver_source_id=_parse_int_field(payload.get("receiver_id"), "receiver_id"),
                message=payload.get("message", ""),
                file_url=payload.get("file_url"),
                message_type=payload.get("type", "text"),
                authenticated_user_id=session.get("user_id"),
                authenticated_role=session.get("user_type"),
            )

            return jsonify({"success": True, **message_payload})
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to send message: {exc}"}), 500

    @chat_bp.route("/get_messages", methods=["GET"])
    @chat_bp.route("/api/get_messages", methods=["GET"])
    @chat_bp.route("/chat/history", methods=["GET"])
    @chat_bp.route("/chat/history/<int:path_user_id>", methods=["GET"])
    @chat_bp.route("/api/chat/history", methods=["GET"])
    @chat_bp.route("/api/chat/history/<int:path_user_id>", methods=["GET"])
    @login_required_factory()
    def get_messages(path_user_id=None):
        try:
            sync_chat_users()
            current_role = session.get("user_type")
            counterpart_role = resolve_counterpart_role(current_role)
            current_source_id = int(session.get("user_id"))
            counterpart_source_id = (
                path_user_id
                or request.args.get("chat_user_id")
                or request.args.get("history_user_id")
                or request.args.get("user_id")
                or request.args.get("counterpart_id")
                or request.args.get("partner_id")
            )
            after_id = _parse_optional_int_field(request.args.get("after_id"), "after_id")
            limit = _parse_optional_int_field(request.args.get("limit"), "limit")
            if limit is not None:
                limit = max(1, min(limit, 500))

            with closing(_get_connection_with_rows(resolved_db_path)) as connection:
                current_user = _get_chat_user(connection, current_source_id, current_role)
                if current_user is None:
                    return jsonify({"error": "Authenticated user not found in chat directory."}), 404

                query = """
                    SELECT
                        messages.id,
                        messages.message,
                        messages.file_url,
                        messages.type,
                        messages.timestamp,
                        messages.status,
                        messages.delivered_at,
                        messages.seen_at,
                        messages.conversation_key,
                        sender.source_id AS sender_source_id,
                        sender.role AS sender_role,
                        sender.full_name AS sender_name,
                        receiver.source_id AS receiver_source_id,
                        receiver.role AS receiver_role,
                        receiver.full_name AS receiver_name
                    FROM messages
                    JOIN users AS sender ON sender.id = messages.sender_id
                    JOIN users AS receiver ON receiver.id = messages.receiver_id
                    WHERE (messages.sender_id = ? OR messages.receiver_id = ?)
                    AND sender.role != receiver.role
                """
                parameters = [current_user["id"], current_user["id"]]

                if after_id is not None:
                    query += " AND messages.id > ?"
                    parameters.append(after_id)

                if counterpart_source_id is not None:
                    counterpart_source_id = _parse_int_field(counterpart_source_id, "user_id")
                    allowed_counterpart_ids = get_allowed_chat_user_ids(load_users_func(), current_role, current_source_id)
                    if allowed_counterpart_ids is not None and counterpart_source_id not in allowed_counterpart_ids:
                        return jsonify({"error": "Conversation user is not linked to the authenticated account."}), 403
                    counterpart_user = _get_chat_user(connection, counterpart_source_id, counterpart_role)
                    if counterpart_user is None:
                        return jsonify({"error": "Conversation user not found."}), 404
                    ensure_patient_doctor_roles(current_user["role"], counterpart_user["role"])
                    for status_payload in _mark_conversation_seen(
                        resolved_db_path,
                        current_source_id=current_source_id,
                        current_role=current_role,
                        counterpart_source_id=counterpart_source_id,
                        counterpart_role=counterpart_role,
                    ):
                        _emit_message_status(resolved_db_path, status_payload)

                    query += """
                        AND (
                            (messages.sender_id = ? AND messages.receiver_id = ?)
                            OR (messages.sender_id = ? AND messages.receiver_id = ?)
                        )
                    """
                    parameters.extend(
                        [
                            current_user["id"],
                            counterpart_user["id"],
                            counterpart_user["id"],
                            current_user["id"],
                        ]
                    )

                query += " ORDER BY messages.timestamp ASC"
                if limit is not None:
                    query += " LIMIT ?"
                    parameters.append(limit)
                rows = connection.execute(query, parameters).fetchall()

            history = [_serialize_message_row(row) for row in rows]

            if request.path.startswith("/chat/history") or request.path.startswith("/api/chat/history"):
                return jsonify(history)

            return jsonify(
                {
                    "success": True,
                    "messages": history,
                    "count": len(history),
                }
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to fetch messages: {exc}"}), 500

    return chat_bp
