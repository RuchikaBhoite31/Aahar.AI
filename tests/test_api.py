from fastapi.testclient import TestClient
from aahar.api import app

c = TestClient(app)


def auth(email="a@b.ie"):
    r = c.post("/api/auth/signup", json={"name": "Alex Doe", "email": email, "password": "abc12345"})
    assert r.status_code == 200
    return {"Authorization": "Bearer " + r.json()["token"]}


def test_auth_flow_and_validation():
    h = auth()
    assert c.get("/api/me", headers=h).json()["email"] == "a@b.ie"
    assert c.post("/api/auth/signup", json={"name": "Alex Doe", "email": "a@b.ie", "password": "abc12345"}).status_code == 409
    assert c.post("/api/auth/signup", json={"name": "Al", "email": "bad", "password": "abc12345"}).status_code == 422
    assert c.post("/api/auth/signup", json={"name": "Alex Doe", "email": "x@y.ie", "password": "abcdefgh"}).status_code == 422  # needs a digit
    assert c.post("/api/auth/login", json={"email": "a@b.ie", "password": "wrong"}).status_code == 401
    assert c.post("/api/auth/login", json={"email": "A@B.ie", "password": "abc12345"}).status_code == 200
    assert c.get("/api/me").status_code == 401
    c.post("/api/auth/logout", headers=h)
    assert c.get("/api/me", headers=h).status_code == 401


def test_google_sign_in_config_and_verified_identity(monkeypatch):
    from aahar.api import id_token
    from aahar.config import settings

    assert c.get("/api/auth/google/config").json() == {"enabled": False, "client_id": None}
    assert c.post("/api/auth/google", json={"credential": "x" * 120}).status_code == 503

    monkeypatch.setattr(settings, "google_client_id", "web-client-id.apps.googleusercontent.com")
    assert c.get("/api/auth/google/config").json() == {
        "enabled": True,
        "client_id": "web-client-id.apps.googleusercontent.com",
    }
    monkeypatch.setattr(id_token, "verify_oauth2_token", lambda token, request, audience: {
        "email": "Google.User@example.com", "email_verified": True, "name": "Google User",
    })
    first = c.post("/api/auth/google", json={"credential": "g" * 120})
    assert first.status_code == 200
    assert first.json()["user"]["email"] == "google.user@example.com"
    assert c.get("/api/me", headers={"Authorization": "Bearer " + first.json()["token"]}).status_code == 200

    second = c.post("/api/auth/google", json={"credential": "g" * 120}).json()
    assert second["user"]["id"] == first.json()["user"]["id"]

    monkeypatch.setattr(id_token, "verify_oauth2_token", lambda token, request, audience: {
        "email": "unverified@example.com", "email_verified": False, "name": "Unverified",
    })
    assert c.post("/api/auth/google", json={"credential": "g" * 120}).status_code == 401


def test_profile_ask_logs_scenarios():
    h = auth("p@q.ie")
    me = c.put("/api/me", json={"allergies": ["Peanuts", "Bogus"], "medical": {"emergencyContact": {"name": "Mum", "phone": "087"}}}, headers=h).json()
    assert me["allergies"] == ["Peanuts"] and me["medical"]["emergencyContact"]["name"] == "Mum"
    a = c.post("/api/ask", json={"question": "What is anaphylaxis?"}, headers=h).json()
    assert a["citations"] and a["mode"] in ("extractive", "llm")
    lid = c.post("/api/logs", json={"severity": "mild", "foodEaten": "cake", "date": "2026-10-29"}, headers=h).json()["id"]
    assert c.get("/api/logs", headers=h).json()[0]["foodEaten"] == "cake"
    other = auth("o@q.ie")
    c.delete(f"/api/logs/{lid}", headers=other)                      # another user cannot delete it
    assert len(c.get("/api/logs", headers=h).json()) == 1
    c.delete(f"/api/logs/{lid}", headers=h)
    assert c.get("/api/logs", headers=h).json() == []
    ids = [s["id"] for s in c.get("/api/scenarios").json()]
    assert {"restaurant_beginner", "party_intermediate", "dinner_with_friends"} <= set(ids)
    s = c.post("/api/scenarios/restaurant_intermediate/chat", json={"message": "does the prawn cocktail contain egg?"}, headers=h).json()
    assert "Declared allergens" in s["text"] and s["points"] == 0
    followup = c.post("/api/scenarios/restaurant_intermediate/chat", json={
        "message": "Does it contain peanuts?", "turn": 1,
        "history": ["Can you check the Chicken Satay Skewers ingredients?"],
    }, headers=h).json()
    assert "Chicken Satay Skewers" in followup["text"]
    assert "Declared allergens: peanuts, soybeans" in followup["text"]
    assert c.post("/api/scenarios/party_intermediate/chat", json={"message": "sure"}, headers=h).json()["points"] == 0


def test_frontend_and_assets_served():
    assert "Aahar.AI" in c.get("/").text
    assert c.get("/assets/icons/chat-bot.png").status_code == 200
