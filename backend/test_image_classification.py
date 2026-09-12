"""Test script for Image Classification implementation."""

import sys
import asyncio
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from algorithms.computer_vision.image_classification import (
    ImageClassificationModel,
    ImageClassificationRequest
)


async def test_image_classification():
    """Test the image classification implementation."""
    print("=" * 80)
    print("Testing Image Classification (CNN)")
    print("=" * 80)

    # Test with ResNet18
    print("\n1. Testing ResNet18...")
    print("-" * 80)

    request = ImageClassificationRequest(
        model_name="resnet18",
        top_k=5,
        confidence_threshold=0.01,
        image_index=0  # Cat image
    )

    try:
        model = ImageClassificationModel(model_name=request.model_name)
        response = model.process_request(request)

        print(f"✓ Model: {response.model_info['model_name']}")
        print(f"✓ Device: {response.model_info['device']}")
        print(f"✓ Execution time: {response.execution_time_ms:.2f}ms")
        print(f"✓ Total parameters: {response.model_info['total_parameters']:,}")
        print(f"\nImage Info:")
        print(f"  - Size: {response.image_info['width']}x{response.image_info['height']}")
        print(f"  - Channels: {response.image_info['channels']}")

        print(f"\nTop {len(response.predictions)} Predictions:")
        for i, pred in enumerate(response.predictions, 1):
            print(f"  {i}. {pred.class_name:30s} - {pred.probability:6.2f}% (confidence: {pred.confidence:.4f})")

        print(f"\nStatistics:")
        print(f"  - Total predictions: {response.statistics.total_predictions}")
        print(f"  - Top confidence: {response.statistics.top_confidence:.4f}")
        print(f"  - Confidence spread: {response.statistics.confidence_spread:.4f}")
        print(f"  - Entropy: {response.statistics.entropy:.2f}")

        print("\n✓ ResNet18 test passed!")

    except Exception as e:
        print(f"✗ ResNet18 test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Test with MobileNetV2
    print("\n2. Testing MobileNetV2...")
    print("-" * 80)

    request2 = ImageClassificationRequest(
        model_name="mobilenet_v2",
        top_k=3,
        confidence_threshold=0.1,
        image_index=1  # Dog image
    )

    try:
        model2 = ImageClassificationModel(model_name=request2.model_name)
        response2 = model2.process_request(request2)

        print(f"✓ Model: {response2.model_info['model_name']}")
        print(f"✓ Execution time: {response2.execution_time_ms:.2f}ms")
        print(f"\nTop {len(response2.predictions)} Predictions:")
        for i, pred in enumerate(response2.predictions, 1):
            print(f"  {i}. {pred.class_name:30s} - {pred.probability:6.2f}%")

        print("\n✓ MobileNetV2 test passed!")

    except Exception as e:
        print(f"✗ MobileNetV2 test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Test with different image
    print("\n3. Testing with airplane image...")
    print("-" * 80)

    request3 = ImageClassificationRequest(
        model_name="resnet18",
        top_k=5,
        confidence_threshold=0.05,
        image_index=2  # Airplane image
    )

    try:
        response3 = model.process_request(request3)

        print(f"✓ Execution time: {response3.execution_time_ms:.2f}ms")
        print(f"\nTop {len(response3.predictions)} Predictions:")
        for i, pred in enumerate(response3.predictions, 1):
            print(f"  {i}. {pred.class_name:30s} - {pred.probability:6.2f}%")

        print("\n✓ Airplane test passed!")

    except Exception as e:
        print(f"✗ Airplane test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Test metadata
    print("\n4. Testing metadata...")
    print("-" * 80)

    try:
        algo_info = ImageClassificationModel.get_algorithm_info()
        print(f"✓ Dataset: {algo_info['dataset']['name']}")
        print(f"✓ Number of classes: {algo_info['dataset']['num_classes']}")
        print(f"✓ Available images: {len(algo_info['dataset']['available_images'])}")
        print(f"✓ Model variants: {len(algo_info['model_variants'])}")

        print("\n✓ Metadata test passed!")

    except Exception as e:
        print(f"✗ Metadata test failed: {str(e)}")
        return False

    print("\n" + "=" * 80)
    print("All tests passed successfully!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    asyncio.run(test_image_classification())
