def login_session(client, user_id: int, user_type: str, full_name: str) -> None:
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["user_type"] = user_type
        session["full_name"] = full_name
