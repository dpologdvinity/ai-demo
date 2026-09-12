"""Test script for GAN route directly."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routes.deep_learning import router

# Create a minimal app just for testing the deep learning routes
app = FastAPI()
app.include_router(router, prefix="/api")

client = TestClient(app)


def test_gan_info():
    """Test GAN info endpoint."""
    print("\n1. Testing GET /api/deep-learning/gan/info...")
    response = client.get("/api/deep-learning/gan/info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    print(f"   ✓ Got metadata for: {data['metadata']['name']}")
    print(f"   ✓ Dataset: {data['dataset']['name']}")
    return data


def test_gan_train():
    """Test GAN training endpoint."""
    print("\n2. Testing POST /api/deep-learning/gan/train...")

    payload = {
        "latent_dim": 100,
        "g_hidden": 256,
        "d_hidden": 256,
        "learning_rate": 0.0002,
        "epochs": 10,  # Minimum is 10
        "batch_size": 64,
        "random_state": 42
    }

    response = client.post("/api/deep-learning/gan/train", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    print(f"   ✓ Training completed in {data['execution_time_ms']:.2f}ms")
    print(f"   ✓ Loss history: {len(data['loss_history'])} epochs")
    print(f"   ✓ Final samples: {len(data['final_samples'])} images")
    print(f"   ✓ Interpolated samples: {len(data['interpolated_samples'])} images")
    print(f"   ✓ Decision boundary points: {len(data['decision_boundary']['scores'])}")

    final = data['loss_history'][-1]
    print(f"   ✓ Final G loss: {final['g_loss']:.4f}")
    print(f"   ✓ Final D loss: {final['d_loss']:.4f}")
    print(f"   ✓ D real acc: {final['d_real_accuracy']:.4f}")
    print(f"   ✓ D fake acc: {final['d_fake_accuracy']:.4f}")

    return data


if __name__ == "__main__":
    try:
        print("Testing GAN routes...")
        test_gan_info()
        test_gan_train()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
