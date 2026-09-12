"""Test script for GAN API endpoint."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_gan_info_endpoint():
    """Test GAN info endpoint."""
    print("\n1. Testing GET /api/deep-learning/gan/info...")
    response = client.get("/api/deep-learning/gan/info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    assert "metadata" in data
    assert "dataset" in data

    metadata = data["metadata"]
    assert metadata["id"] == "gan"
    assert metadata["name"] == "GAN (Generative Adversarial Network)"
    assert metadata["slug"] == "gan"
    assert metadata["category"] == "deep_learning"
    assert metadata["difficulty"] == "advanced"

    print(f"   ✓ Metadata: {metadata['name']}")
    print(f"   ✓ Parameters: {len(metadata['parameters'])}")
    print(f"   ✓ Use cases: {len(metadata['use_cases'])}")
    print(f"   ✓ Dataset: {data['dataset']['name']}")


def test_gan_train_endpoint():
    """Test GAN training endpoint."""
    print("\n2. Testing POST /api/deep-learning/gan/train...")

    # Test with default parameters (but fewer epochs)
    payload = {
        "latent_dim": 100,
        "g_hidden": 256,
        "d_hidden": 256,
        "learning_rate": 0.0002,
        "epochs": 3,  # Small for testing
        "batch_size": 64,
        "random_state": 42
    }

    response = client.post("/api/deep-learning/gan/train", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()

    # Check response structure
    assert "loss_history" in data
    assert "generated_samples" in data
    assert "final_samples" in data
    assert "interpolated_samples" in data
    assert "decision_boundary" in data
    assert "visualization_data" in data
    assert "execution_time_ms" in data
    assert "model_info" in data

    # Check loss history
    assert len(data["loss_history"]) == 3  # 3 epochs
    first_loss = data["loss_history"][0]
    assert "epoch" in first_loss
    assert "g_loss" in first_loss
    assert "d_loss" in first_loss
    assert "d_real_loss" in first_loss
    assert "d_fake_loss" in first_loss
    assert "d_real_accuracy" in first_loss
    assert "d_fake_accuracy" in first_loss

    # Check generated samples
    assert len(data["generated_samples"]) > 0
    assert "epoch" in data["generated_samples"][0]
    assert "samples" in data["generated_samples"][0]

    # Check final samples
    assert len(data["final_samples"]) == 16
    assert len(data["final_samples"][0]) == 64  # 8x8 = 64 pixels

    # Check interpolated samples
    assert data["interpolated_samples"] is not None
    assert len(data["interpolated_samples"]) > 0

    # Check decision boundary
    assert data["decision_boundary"] is not None
    assert "latent_vectors" in data["decision_boundary"]
    assert "scores" in data["decision_boundary"]
    assert "images" in data["decision_boundary"]

    # Check visualization data
    viz_data = data["visualization_data"]
    assert "sample_epochs" in viz_data
    assert "loss_epochs" in viz_data
    assert "n_samples" in viz_data
    assert "image_shape" in viz_data
    assert viz_data["image_shape"] == [8, 8]

    # Check model info
    model_info = data["model_info"]
    assert "latent_dim" in model_info
    assert "generator_params" in model_info
    assert "discriminator_params" in model_info
    assert "total_params" in model_info
    assert model_info["latent_dim"] == 100

    print(f"   ✓ Loss history: {len(data['loss_history'])} epochs")
    print(f"   ✓ Generated samples: {len(data['generated_samples'])} checkpoints")
    print(f"   ✓ Final samples: {len(data['final_samples'])} images")
    print(f"   ✓ Interpolated samples: {len(data['interpolated_samples'])} images")
    print(f"   ✓ Decision boundary: {len(data['decision_boundary']['scores'])} points")
    print(f"   ✓ Execution time: {data['execution_time_ms']:.2f}ms")
    print(f"   ✓ Generator params: {model_info['generator_params']:,}")
    print(f"   ✓ Discriminator params: {model_info['discriminator_params']:,}")

    # Check final metrics
    final_loss = data["loss_history"][-1]
    print(f"   ✓ Final G loss: {final_loss['g_loss']:.4f}")
    print(f"   ✓ Final D loss: {final_loss['d_loss']:.4f}")
    print(f"   ✓ Final D real acc: {final_loss['d_real_accuracy']:.4f}")
    print(f"   ✓ Final D fake acc: {final_loss['d_fake_accuracy']:.4f}")


def test_gan_parameter_validation():
    """Test GAN parameter validation."""
    print("\n3. Testing parameter validation...")

    # Test invalid latent_dim (too small)
    payload = {
        "latent_dim": 10,  # Below minimum of 32
        "epochs": 3
    }
    response = client.post("/api/deep-learning/gan/train", json=payload)
    assert response.status_code == 422, f"Expected 422 for invalid latent_dim, got {response.status_code}"
    print("   ✓ Validated latent_dim minimum")

    # Test invalid epochs (too small)
    payload = {
        "latent_dim": 100,
        "epochs": 5  # Below minimum of 10
    }
    response = client.post("/api/deep-learning/gan/train", json=payload)
    assert response.status_code == 422, f"Expected 422 for invalid epochs, got {response.status_code}"
    print("   ✓ Validated epochs minimum")

    # Test invalid learning_rate (too large)
    payload = {
        "latent_dim": 100,
        "epochs": 10,
        "learning_rate": 0.01  # Above maximum of 0.001
    }
    response = client.post("/api/deep-learning/gan/train", json=payload)
    assert response.status_code == 422, f"Expected 422 for invalid learning_rate, got {response.status_code}"
    print("   ✓ Validated learning_rate maximum")


def test_list_algorithms():
    """Test that GAN appears in algorithms list."""
    print("\n4. Testing GET /api/deep-learning/algorithms...")
    response = client.get("/api/deep-learning/algorithms")
    assert response.status_code == 200

    algorithms = response.json()
    gan_algo = next((algo for algo in algorithms if algo["id"] == "gan"), None)
    assert gan_algo is not None, "GAN not found in algorithms list"

    print(f"   ✓ Found GAN in list of {len(algorithms)} algorithms")
    print(f"   ✓ GAN name: {gan_algo['name']}")
    print(f"   ✓ GAN description: {gan_algo['description']}")


if __name__ == "__main__":
    try:
        print("Testing GAN API endpoints...")
        test_gan_info_endpoint()
        test_gan_train_endpoint()
        test_gan_parameter_validation()
        test_list_algorithms()
        print("\n✓ All API tests passed!")
    except AssertionError as e:
        print(f"\n✗ Test failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
