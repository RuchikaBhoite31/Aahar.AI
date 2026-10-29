"""SQLite persistence (replaces Firebase Auth/Firestore from the Flutter app)."""
import hashlib, hmac, json, secrets, sqlite3, time
from contextlib import contextmanager
from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, email TEXT UNIQUE, name TEXT, pw TEXT, salt TEXT,
  allergies TEXT DEFAULT '[]', medical TEXT DEFAULT '{}', created REAL);
CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS logs(id INTEGER PRIMARY KEY, user_id INTEGER, date TEXT, data TEXT, created REAL);
CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, user_id INTEGER, barcode TEXT, data TEXT, created REAL);
"""


@contextmanager
def conn():
    c = sqlite3.connect(settings.db_path)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def _hash(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()


def create_user(email, name, pw):
    salt = secrets.token_hex(16)
    with conn() as c:
        cur = c.execute("INSERT INTO users(email,name,pw,salt,created) VALUES(?,?,?,?,?)",
                        (email.lower(), name, _hash(pw, salt), salt, time.time()))
        return cur.lastrowid


def check_login(email, pw):
    with conn() as c:
        u = c.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
    if u and hmac.compare_digest(u["pw"], _hash(pw, u["salt"])):
        return u["id"]
    return None


def new_session(uid) -> str:
    t = secrets.token_urlsafe(32)
    with conn() as c:
        c.execute("INSERT INTO sessions VALUES(?,?,?)", (t, uid, time.time()))
    return t


def public_user(r):
    return {"id": r["id"], "email": r["email"], "name": r["name"],
            "allergies": json.loads(r["allergies"]), "medical": json.loads(r["medical"])}


def user_by_token(t):
    if not t:
        return None
    with conn() as c:
        r = c.execute("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?", (t,)).fetchone()
    return public_user(r) if r else None


def drop_session(t):
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (t,))


def update_user(uid, **f):
    with conn() as c:
        for k in ("name", "allergies", "medical"):
            if f.get(k) is not None:
                v = f[k] if k == "name" else json.dumps(f[k])
                c.execute(f"UPDATE users SET {k}=? WHERE id=?", (v, uid))
        return public_user(c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())


def add_row(table, uid, **cols):
    assert table in ("logs", "scans")
    with conn() as c:
        keys = ",".join(cols)
        cur = c.execute(f"INSERT INTO {table}(user_id,{keys},created) VALUES(?,{','.join('?' * len(cols))},?)",
                        (uid, *cols.values(), time.time()))
        return cur.lastrowid


def list_rows(table, uid, limit=100):
    assert table in ("logs", "scans")
    with conn() as c:
        return [dict(r) for r in c.execute(
            f"SELECT * FROM {table} WHERE user_id=? ORDER BY created DESC LIMIT ?", (uid, limit))]


def delete_log(uid, lid):
    with conn() as c:
        c.execute("DELETE FROM logs WHERE id=? AND user_id=?", (lid, uid))
