from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_valid_landmarks():
    response = client.post("/predict", json={
        "landmarks": [[0.1, 0.2, 0.0]] * 21
    })
    assert response.status_code == 200
    assert "gesture" in response.json()

def test_invalid_landmarks():
    response = client.post("/predict", json={
        "landmarks": [[0.1, 0.2]] * 21  # Missing Z
    })
    assert response.status_code == 200
    assert "error" in response.json()
