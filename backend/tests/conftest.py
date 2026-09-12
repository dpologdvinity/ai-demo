import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    """Test client for FastAPI"""
    return TestClient(app)


@pytest.fixture
def sample_training_params():
    """Sample training parameters"""
    return {
        "learning_rate": 0.001,
        "n_iterations": 100,
        "random_state": 42
    }
