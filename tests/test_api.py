import pytest
from fastapi.testclient import TestClient
from app.main import app
from pathlib import Path


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "message" in data


def test_readiness_endpoint(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_validate_access_endpoint(client):
    request_data = {
        "user_role": "Admin",
        "document_type": "INTERNAL",
    }
    response = client.post("/validate-access", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "allowed" in data


def test_validate_access_denied(client):
    request_data = {
        "user_role": "Contractor",
        "document_type": "CONFIDENTIAL",
    }
    response = client.post("/validate-access", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["allowed"] is False


def test_ask_without_index(client):
    request_data = {
        "query": "What is ProductA?",
        "user_role": "Admin",
    }
    response = client.post("/ask", json=request_data)
    assert response.status_code in [503, 500]


def test_trace_endpoint(client):
    response = client.get("/debug/trace")
    assert response.status_code == 404
