
from fastapi.testclient import TestClient
from app import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "I was charged twice"},
        )
        assert response.status_code == 200

        result = response.json()
        assert "category" in result
        assert "priority" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1


def test_empty_text():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": ""},
        )
        assert response.status_code == 422


def test_whitespace_text():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "   "},
        )
        assert response.status_code == 422


def test_batch_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict-batch",
            json={
                "tickets": [
                    {"text": "I was charged twice"},
                    {"text": "I cannot log in"},
                ]
            },
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2
