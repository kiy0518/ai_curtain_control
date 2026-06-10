"""SQLite persistence: settings (key/value) + curtain schedules.

Pure stdlib (sqlite3). Thread-safe via a single shared connection + lock
(check_same_thread=False). DB lives at the repo root as ``curtain.db``.
"""

import os
import sqlite3
import threading

_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "curtain.db")
_lock = threading.Lock()
_conn = sqlite3.connect(_DB, check_same_thread=False)
_conn.row_factory = sqlite3.Row


def init():
    with _lock:
        _conn.executescript("""
        CREATE TABLE IF NOT EXISTS settings(
            key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS schedules(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, action TEXT,            -- OPEN/CLOSE/STOP
            kind TEXT,                          -- 'time' | 'sun'
            hh INTEGER, mm INTEGER,             -- for kind='time'
            sun_event TEXT, sun_offset INTEGER, -- 'sunrise'/'sunset', minutes (+/-)
            days TEXT,                          -- "" = daily, else csv weekdays 0=Mon
            enabled INTEGER DEFAULT 1);
        """)
        _conn.commit()
    # sensible defaults (Seoul) so sun schedules work out of the box
    if get_setting("lat") is None:
        set_setting("lat", "37.5665")
        set_setting("lon", "126.9780")


# --- settings --------------------------------------------------------------
def get_setting(key, default=None):
    with _lock:
        r = _conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_setting(key, value):
    with _lock:
        _conn.execute("INSERT INTO settings(key,value) VALUES(?,?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, str(value)))
        _conn.commit()


# --- schedules -------------------------------------------------------------
def list_schedules():
    with _lock:
        rows = _conn.execute("SELECT * FROM schedules ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def add_schedule(s):
    with _lock:
        cur = _conn.execute(
            "INSERT INTO schedules(name,action,kind,hh,mm,sun_event,sun_offset,days,enabled)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (s.get("name", ""), s.get("action", "OPEN"), s.get("kind", "time"),
             s.get("hh"), s.get("mm"), s.get("sun_event"),
             int(s.get("sun_offset", 0) or 0), s.get("days", ""),
             1 if s.get("enabled", True) else 0))
        _conn.commit()
        return cur.lastrowid


def delete_schedule(sid):
    with _lock:
        _conn.execute("DELETE FROM schedules WHERE id=?", (sid,))
        _conn.commit()


def set_enabled(sid, enabled):
    with _lock:
        _conn.execute("UPDATE schedules SET enabled=? WHERE id=?",
                      (1 if enabled else 0, sid))
        _conn.commit()
