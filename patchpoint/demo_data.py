"""Tiny synthetic corpus + issue used by `patchpoint demo` — no network access needed."""

from __future__ import annotations

DEMO_FILES: dict[str, str] = {
    "auth/login.py": """
def login(username, password):
    user = find_user(username)
    if user is None or not user.check_password(password):
        raise AuthError("invalid credentials")
    return create_session(user)
""",
    "auth/session.py": """
def create_session(user):
    token = generate_token()
    sessions[token] = user.id
    return token
""",
    "billing/invoice.py": """
def generate_invoice(order):
    total = sum(item.price for item in order.items)
    return Invoice(order.id, total)
""",
    "docs/readme.md": "# Demo project\nA tiny synthetic repo for smoke-testing patchpoint.",
}

DEMO_ISSUE = (
    "Login raises AuthError even with the right password when the session token "
    "already exists for that user."
)
