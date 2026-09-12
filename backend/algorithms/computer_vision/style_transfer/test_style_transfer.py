"""Test script for Neural Style Transfer implementation."""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

from algorithms.computer_vision.style_transfer import (
    StyleTransferModel,
    StyleTransferRequest,
    get_available_content_images,
    get_available_style_images,
    get_dataset_info
)


def test_dataset_info():
    """Test dataset information retrieval."""
    print("=" * 60)
    print("Testing Dataset Info")
    print("=" * 60)

    dataset = get_dataset_info()
    print(f"\nDataset: {dataset['name']}")
    print(f"Description: {dataset['description']}")
    print(f"Content Images: {dataset['num_content_images']}")
    print(f"Style Images: {dataset['num_style_images']}")
    print(f"Categories: {', '.join(dataset['content_categories'])}")

    print("\n--- Available Content Images ---")
    for img in get_available_content_images():
        print(f"  [{img['index']}] {img['description']}")

    print("\n--- Available Style Images ---")
    for img in get_available_style_images():
        print(f"  [{img['index']}] {img['description']}")


def test_model_initialization():
    """Test model initialization."""
    print("\n" + "=" * 60)
    print("Testing Model Initialization")
    print("=" * 60)

    model = StyleTransferModel()
    print(f"\nModel initialized successfully")
    print(f"Device: {model.device}")
    print(f"VGG19 loaded: {model.vgg is not None}")


def test_request_creation():
    """Test request creation with different parameters."""
    print("\n" + "=" * 60)
    print("Testing Request Creation")
    print("=" * 60)

    # Basic request
    request1 = StyleTransferRequest(
        content_image_index=0,
        style_image_index=0
    )
    print("\n1. Basic Request (defaults):")
    print(f"   Content: {request1.content_image_index}")
    print(f"   Style: {request1.style_image_index}")
    print(f"   Iterations: {request1.iterations}")
    print(f"   Content weight: {request1.content_weight}")
    print(f"   Style weight: {request1.style_weight}")

    # Strong stylization
    request2 = StyleTransferRequest(
        content_image_index=2,
        style_image_index=1,
        style_weight=5000000.0,
        content_weight=0.5,
        iterations=500
    )
    print("\n2. Strong Stylization:")
    print(f"   Content weight: {request2.content_weight} (low = less content)")
    print(f"   Style weight: {request2.style_weight} (high = strong style)")
    print(f"   Iterations: {request2.iterations}")

    # Fast preview
    request3 = StyleTransferRequest(
        content_image_index=1,
        style_image_index=2,
        image_size=256,
        iterations=100,
        learning_rate=0.005
    )
    print("\n3. Fast Preview:")
    print(f"   Image size: {request3.image_size}px")
    print(f"   Iterations: {request3.iterations}")
    print(f"   Learning rate: {request3.learning_rate}")


def test_algorithm_info():
    """Test algorithm information retrieval."""
    print("\n" + "=" * 60)
    print("Testing Algorithm Info")
    print("=" * 60)

    info = StyleTransferModel.get_algorithm_info()

    print("\n--- Dataset Info ---")
    print(f"Name: {info['dataset']['name']}")
    print(f"Source: {info['dataset']['source']}")

    print("\n--- Algorithm Details ---")
    details = info['algorithm_details']
    print(f"Method: {details['method']}")
    print(f"Backbone: {details['backbone']}")
    print(f"Content Rep: {details['content_representation']}")
    print(f"Style Rep: {details['style_representation']}")
    print(f"Optimization: {details['optimization']}")


def test_style_transfer_dry_run():
    """Test style transfer without actual execution (dry run)."""
    print("\n" + "=" * 60)
    print("Style Transfer Dry Run")
    print("=" * 60)

    print("\nNote: Full execution requires downloading images and running")
    print("optimization for several minutes. This is a dry run showing the")
    print("workflow without actual execution.")

    # Create model
    model = StyleTransferModel()

    # Create request
    request = StyleTransferRequest(
        content_image_index=0,
        style_image_index=0,
        iterations=300,
        content_weight=1.0,
        style_weight=1000000.0,
        learning_rate=0.003,
        image_size=512
    )

    print("\n--- Request Parameters ---")
    print(f"Content Image: Mountain landscape (index {request.content_image_index})")
    print(f"Style Image: Van Gogh Starry Night (index {request.style_image_index})")
    print(f"Optimization:")
    print(f"  - Iterations: {request.iterations}")
    print(f"  - Learning rate: {request.learning_rate}")
    print(f"  - Content weight: {request.content_weight}")
    print(f"  - Style weight: {request.style_weight}")
    print(f"Output size: {request.image_size}x{request.image_size}")

    print("\n--- Expected Workflow ---")
    print("1. Download content image (if not cached)")
    print("2. Download style image (if not cached)")
    print("3. Load VGG19 model")
    print("4. Preprocess images (resize, normalize)")
    print("5. Initialize generated image = content image")
    print("6. Extract target features:")
    print("   - Content: conv4_2 features")
    print("   - Style: Gram matrices from conv1_1, conv2_1, conv3_1, conv4_1, conv5_1")
    print("7. Optimization loop (300 iterations):")
    print("   - Forward pass through VGG19")
    print("   - Compute content loss (MSE)")
    print("   - Compute style loss (Gram matrix MSE)")
    print("   - Total loss = 1.0 * content + 1000000.0 * style")
    print("   - Backward pass (gradients)")
    print("   - Update generated image")
    print("   - Clamp pixel values")
    print("8. Convert result to displayable format")
    print("9. Prepare visualization data")

    print("\n--- Expected Response ---")
    print("Response includes:")
    print("  - Generated stylized image (base64)")
    print("  - Content image (base64)")
    print("  - Style image (base64)")
    print("  - Loss history (every 10 iterations)")
    print("  - Statistics (loss reduction, convergence rate)")
    print("  - Execution time")
    print("  - Model info")

    print("\n--- Execution Time Estimate ---")
    print("CPU:")
    print(f"  - 256x256: ~5-10 seconds")
    print(f"  - 512x512: ~20-30 seconds")
    print(f"  - 1024x1024: ~60-90 seconds")
    print("\nGPU:")
    print(f"  - 256x256: ~2-3 seconds")
    print(f"  - 512x512: ~5-8 seconds")
    print(f"  - 1024x1024: ~15-20 seconds")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NEURAL STYLE TRANSFER - TEST SUITE")
    print("=" * 60)

    try:
        test_dataset_info()
        test_model_initialization()
        test_request_creation()
        test_algorithm_info()
        test_style_transfer_dry_run()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\n--- Next Steps ---")
        print("1. Start the FastAPI server:")
        print("   cd backend && python -m uvicorn main:app --reload")
        print("\n2. Test the endpoint:")
        print("   POST http://localhost:8000/api/computer-vision/style-transfer/stylize")
        print("\n3. Get algorithm info:")
        print("   GET http://localhost:8000/api/computer-vision/style-transfer/info")
        print("\n4. List all CV algorithms:")
        print("   GET http://localhost:8000/api/computer-vision/algorithms")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
