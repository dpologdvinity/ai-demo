"""Quick test script for YOLO implementation."""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from algorithms.computer_vision.yolo import YOLOModel, YOLORequest


def test_yolo_basic():
    """Test basic YOLO functionality."""
    print("Testing YOLO Object Detection Implementation...")
    print("-" * 50)

    # Create a request
    request = YOLORequest(
        confidence_threshold=0.25,
        iou_threshold=0.45,
        model_version='yolov8n',
        max_detections=100,
        image_index=0,
        class_filter='all'
    )

    print(f"Request parameters: {request.model_dump()}")
    print()

    # Initialize model
    print("Initializing YOLO model...")
    model = YOLOModel(model_version=request.model_version)

    # Process request
    print("Running object detection...")
    try:
        response = model.process_request(request)

        print(f"\nDetection completed successfully!")
        print(f"Execution time: {response.execution_time_ms:.2f}ms")
        print(f"Total detections: {response.statistics.total_detections}")
        print(f"Average confidence: {response.statistics.avg_confidence:.3f}")
        print(f"\nDetected objects:")
        for det in response.detections[:10]:  # Show first 10
            print(f"  - {det.class_name}: {det.confidence:.3f}")

        print(f"\nClass counts:")
        for class_name, count in response.statistics.class_counts.items():
            print(f"  - {class_name}: {count}")

        print(f"\nConfidence distribution:")
        for range_name, count in response.statistics.confidence_distribution.items():
            print(f"  - {range_name}: {count}")

        print("\n" + "=" * 50)
        print("Test PASSED!")
        return True

    except Exception as e:
        print(f"\nTest FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_yolo_basic()
    sys.exit(0 if success else 1)
