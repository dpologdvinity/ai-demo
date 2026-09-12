import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint(client: TestClient):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_algorithm_info_endpoint(client: TestClient):
    """Test algorithm info endpoint structure"""
    # This will be expanded as algorithms are added
    pass


def test_training_endpoint_validation(client: TestClient):
    """Test training endpoint parameter validation"""
    # This will be expanded as algorithms are added
    pass
