"""Authentication — single admin password + session cookies (stdlib only).

- Password stored as PBKDF2-HMAC-SHA256(salt) in SQLite settings.
- Sessions: random tokens kept in memory with TTL; delivered as HttpOnly cookie.
- Basic global lockout after repeated failures (home single-admin device).
- First run sets a default password "admin" (flagged) → dashboard nags to change.
"""

import hashlib
import hmac
import secrets
import threading
import time

import store

_SESS_TTL = 7 * 24 * 3600
_LOCK_FAILS = 5
_LOCK_SECS = 30

_sessions = {}                 # token -> expiry epoch
_lock = threading.Lock()
_fail = {"n": 0, "until": 0.0}


def _hash(pw, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(),
                               bytes.fromhex(salt_hex), 120_000).hex()


def init():
    if not store.get_setting("auth_hash"):
        set_password("admin", default=True)


def set_password(pw, default=False):
    if not pw:
        return False
    salt = secrets.token_hex(16)
    store.set_setting("auth_salt", salt)
    store.set_setting("auth_hash", _hash(pw, salt))
    store.set_setting("auth_default", "1" if default else "0")
    return True


def is_default():
    return store.get_setting("auth_default", "0") == "1"


def verify(pw):
    salt = store.get_setting("auth_salt")
    h = store.get_setting("auth_hash")
    if not salt or not h:
        return False
    return hmac.compare_digest(_hash(pw, salt), h)


def locked():
    return time.time() < _fail["until"]


def login(pw):
    """Return (token, error). token is None on failure."""
    if locked():
        return None, "잠시 후 다시 시도하세요"
    if not verify(pw):
        _fail["n"] += 1
        if _fail["n"] >= _LOCK_FAILS:
            _fail["until"] = time.time() + _LOCK_SECS
            _fail["n"] = 0
        return None, "비밀번호가 올바르지 않습니다"
    _fail["n"] = 0
    tok = secrets.token_urlsafe(24)
    with _lock:
        _sessions[tok] = time.time() + _SESS_TTL
    return tok, None


def valid(tok):
    if not tok:
        return False
    with _lock:
        exp = _sessions.get(tok)
        if exp and exp > time.time():
            return True
        _sessions.pop(tok, None)
    return False


def logout(tok):
    with _lock:
        _sessions.pop(tok, None)


def change_password(old, new):
    if not verify(old):
        return False, "현재 비밀번호가 올바르지 않습니다"
    if not new or len(new) < 4:
        return False, "새 비밀번호는 4자 이상이어야 합니다"
    set_password(new, default=False)
    with _lock:
        _sessions.clear()          # force re-login everywhere
    return True, None
