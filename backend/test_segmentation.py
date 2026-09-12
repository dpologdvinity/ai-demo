#!/usr/bin/env python3
"""Test script for semantic segmentation implementation."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.computer_vision.semantic_segmentation import (
    SegmentationModel,
    SegmentationRequest
)


def test_segmentation():
    """Test semantic segmentation with default parameters."""
    print("=" * 60)
    print("Testing Semantic Segmentation Implementation")
    print("=" * 60)

    # Create request with default parameters
    request = SegmentationRequest(
        num_classes=21,
        confidence_threshold=0.5,
        model_backbone='resnet50',
        image_size=512,
        image_index=0
    )

    print("\nRequest Parameters:")
    print(f"  - Number of classes: {request.num_classes}")
    print(f"  - Confidence threshold: {request.confidence_threshold}")
    print(f"  - Model backbone: {request.model_backbone}")
    print(f"  - Image size: {request.image_size}")
    print(f"  - Image index: {request.image_index}")

    # Initialize model
    print("\nInitializing model...")
    model = SegmentationModel(
        model_backbone=request.model_backbone,
        num_classes=request.num_classes
    )

    # Process request
    print("\nProcessing segmentation request...")
    try:
        response = model.process_request(request)

        print("\nSegmentation Results:")
        print(f"  - Success: {response.success}")
        print(f"  - Execution time: {response.execution_time_ms:.2f} ms")
        print(f"  - Total classes found: {response.statistics.total_classes}")
        print(f"  - Total pixels: {response.statistics.total_pixels:,}")
        print(f"  - Mean confidence: {response.statistics.mean_confidence:.3f}")

        print("\nClass Distribution:")
        for i, class_info in enumerate(response.statistics.class_info[:5], 1):
            print(f"  {i}. {class_info.class_name}: "
                  f"{class_info.pixel_count:,} pixels ({class_info.percentage:.2f}%)")

        if len(response.statistics.class_info) > 5:
            print(f"  ... and {len(response.statistics.class_info) - 5} more classes")

        print("\nModel Info:")
        for key, value in response.model_info.items():
            print(f"  - {key}: {value}")

        print("\nVisualization Data Keys:")
        for key in response.visualization_data.keys():
            if 'image' in key:
                print(f"  - {key}: [base64 image data]")
            else:
                print(f"  - {key}: {type(response.visualization_data[key]).__name__}")

        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_segmentation()
    sys.exit(0 if success else 1)
