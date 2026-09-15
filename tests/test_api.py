import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "index_loaded" in data
    assert "total_chunks" in data

def test_ask_without_index():
    # If no papers are loaded, ask should return 400
    from app.api import _state
    _state["index"] = None
    response = client.post("/ask", json={"question": "What is LoRA?"})
    assert response.status_code == 400
    assert "No papers indexed yet" in response.json()["detail"]
