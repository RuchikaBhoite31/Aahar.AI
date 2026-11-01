"""FastAPI app: JSON API + static single-page frontend."""
import json, re
from datetime import date
from typing import Optional
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token

from . import db, scan, scenarios
from .allergens import MAJOR_ALLERGENS
from .config import DATA, WEB, settings
from .pipeline import ask, retriever

app = FastAPI(title="Aahar.AI", version="1.0.0")


def _tok(authorization: str) -> str:
    return authorization.removeprefix("Bearer ").strip()


def me(authorization: str = Header(default="")):
    u = db.user_by_token(_tok(authorization))
    if not u:
        raise HTTPException(401, "Please log in again.")
    return u


class Signup(BaseModel):
    name: str = Field(min_length=2, max_length=50, pattern=r"^[a-zA-Z\s\-']+$")
    email: str = Field(pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(min_length=6, max_length=128)


class Login(BaseModel):
    email: str
    password: str


class GoogleLogin(BaseModel):
    credential: str = Field(min_length=100, max_length=10000)


class Profile(BaseModel):
    name: Optional[str] = None
    allergies: Optional[list[str]] = None
    medical: Optional[dict] = None


class Ask(BaseModel):
    question: str = Field(min_length=2, max_length=600)
    history: list[tuple[str, str]] = []


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=400)
    turn: int = 0
    history: list[str] = Field(default_factory=list, max_length=10)


@app.get("/api/health")
def health():
    return {"ok": True, "llm": settings.provider, "google_auth": bool(settings.google_client_id),
            "chunks": len(retriever().chunks)}


@app.get("/api/auth/google/config")
def google_auth_config():
    # OAuth client IDs are public identifiers; the signing secret must never be sent to the browser.
    return {"enabled": bool(settings.google_client_id), "client_id": settings.google_client_id or None}


@app.post("/api/auth/google")
def google_login(b: GoogleLogin):
    if not settings.google_client_id:
        raise HTTPException(503, "Google sign-in is not configured for this deployment.")
    try:
        claims = id_token.verify_oauth2_token(
            b.credential, GoogleAuthRequest(), settings.google_client_id
        )
    except ValueError:
        raise HTTPException(401, "Google sign-in could not be verified. Please try again.")
    except GoogleAuthError:
        raise HTTPException(503, "Google verification is temporarily unavailable. Please try again.")
    email = str(claims.get("email", "")).strip().lower()
    name = str(claims.get("name", "")).strip()
    if claims.get("email_verified") is not True or not re.fullmatch(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", email
    ):
        raise HTTPException(401, "Google must provide a verified email address.")
    uid = db.get_or_create_google_user(email, name)
    token = db.new_session(uid)
    return {"token": token, "user": db.user_by_token(token)}


@app.post("/api/auth/signup")
def signup(b: Signup):
    if not re.search(r"[a-zA-Z]", b.password) or not re.search(r"\d", b.password):
        raise HTTPException(422, "Password must contain at least one letter and one number")
    try:
        uid = db.create_user(b.email, b.name.strip(), b.password)
    except Exception:
        raise HTTPException(409, "An account with this email already exists.")
    t = db.new_session(uid)
    return {"token": t, "user": db.user_by_token(t)}


@app.post("/api/auth/login")
def login(b: Login):
    uid = db.check_login(b.email, b.password)
    if not uid:
        raise HTTPException(401, "Invalid email or password.")
    t = db.new_session(uid)
    return {"token": t, "user": db.user_by_token(t)}


@app.post("/api/auth/logout")
def logout(authorization: str = Header(default="")):
    db.drop_session(_tok(authorization))
    return {"ok": True}


@app.get("/api/me")
def get_me(u=Depends(me)):
    return u


@app.put("/api/me")
def put_me(b: Profile, u=Depends(me)):
    if b.allergies is not None:
        b.allergies = [a for a in b.allergies if a in MAJOR_ALLERGENS]
    return db.update_user(u["id"], **b.model_dump())


@app.get("/api/allergens")
def allergens():
    return MAJOR_ALLERGENS


@app.post("/api/ask")
def api_ask(b: Ask, u=Depends(me)):
    return ask(b.question, u["allergies"], b.history)


@app.get("/api/learn/{module}")
def learn(module: str):
    f = {"educational": "educational_module.json", "behavioural": "behavioural_module.json"}.get(module)
    if not f:
        raise HTTPException(404)
    return json.loads((DATA / "knowledge" / f).read_text(encoding="utf-8"))


@app.get("/api/logs")
def logs(u=Depends(me)):
    return [{**json.loads(r["data"]), "id": r["id"], "date": r["date"]} for r in db.list_rows("logs", u["id"])]


@app.post("/api/logs")
def add_log(body: dict, u=Depends(me)):
    d = str(body.get("date") or date.today().isoformat())[:10]
    return {"id": db.add_row("logs", u["id"], date=d, data=json.dumps(body))}


@app.delete("/api/logs/{lid}")
def del_log(lid: int, u=Depends(me)):
    db.delete_log(u["id"], lid)
    return {"ok": True}


@app.get("/api/scan/{barcode}")
def api_scan(barcode: str, u=Depends(me)):
    res = scan.lookup(barcode, u["allergies"])
    if res.get("name"):
        db.add_row("scans", u["id"], barcode=res["barcode"], data=json.dumps(res))
    return res


@app.get("/api/scans")
def scans(u=Depends(me)):
    return [json.loads(r["data"]) for r in db.list_rows("scans", u["id"], 10)]


@app.get("/api/scenarios")
def scen():
    return scenarios.cards()


@app.get("/api/scenarios/{sid}/start")
def scen_start(sid: str, u=Depends(me)):
    return scenarios.start(sid)


@app.post("/api/scenarios/{sid}/chat")
def scen_chat(sid: str, b: Chat, u=Depends(me)):
    return scenarios.reply(sid, b.message, u["allergies"], b.turn, b.history)


app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
