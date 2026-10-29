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
    assert "Declared allergens" in s["text"] and s["points"] == 10
    assert c.post("/api/scenarios/party_intermediate/chat", json={"message": "sure"}, headers=h).json()["points"] == 0


def test_frontend_and_assets_served():
    assert "Aahar.AI" in c.get("/").text
    assert c.get("/assets/icons/chat-bot.png").status_code == 200
