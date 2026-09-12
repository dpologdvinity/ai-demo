"""Test script for Instance Segmentation implementation."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.computer_vision.instance_segmentation import (
    InstanceSegmentationModel,
    InstanceSegmentationRequest
)


def test_instance_segmentation():
    """Test instance segmentation with default parameters."""
    print("=" * 80)
    print("Testing Instance Segmentation (Mask R-CNN)")
    print("=" * 80)

    # Create request with default parameters
    request = InstanceSegmentationRequest(
        confidence_threshold=0.5,
        model_backbone='resnet50',
        mask_threshold=0.5,
        max_instances=100,
        image_index=0,
        nms_threshold=0.5
    )

    print(f"\nRequest parameters:")
    print(f"  - Confidence threshold: {request.confidence_threshold}")
    print(f"  - Model backbone: {request.model_backbone}")
    print(f"  - Mask threshold: {request.mask_threshold}")
    print(f"  - Max instances: {request.max_instances}")
    print(f"  - Image index: {request.image_index}")
    print(f"  - NMS threshold: {request.nms_threshold}")

    # Initialize model
    print("\nInitializing Mask R-CNN model...")
    model = InstanceSegmentationModel(model_backbone=request.model_backbone)

    # Process request
    print("Running instance segmentation...")
    try:
        response = model.process_request(request)

        print(f"\n{'=' * 80}")
        print("RESULTS")
        print("=" * 80)

        # Print execution time
        print(f"\nExecution time: {response.execution_time_ms:.2f} ms")

        # Print statistics
        stats = response.statistics
        print(f"\nInstance Statistics:")
        print(f"  - Total instances: {stats.total_instances}")
        print(f"  - Unique classes: {stats.unique_classes}")
        print(f"  - Average confidence: {stats.avg_confidence:.3f}")
        print(f"  - Total mask area: {stats.total_mask_area:,} pixels")
        print(f"  - Coverage: {stats.coverage_percentage:.2f}%")
        print(f"  - Average instance size: {stats.avg_instance_size:.1f} pixels")

        print(f"\nClass Distribution:")
        for class_name, count in sorted(stats.class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {class_name}: {count}")

        print(f"\nConfidence Distribution:")
        for range_name, count in stats.confidence_distribution.items():
            print(f"  - {range_name}: {count}")

        # Print top instances
        print(f"\nTop 10 Detected Instances:")
        print(f"{'ID':<4} {'Class':<20} {'Confidence':<12} {'Area (px)':<12} {'BBox'}")
        print("-" * 80)
        for inst in response.instances[:10]:
            bbox_str = f"[{inst.bbox[0]:.1f}, {inst.bbox[1]:.1f}, {inst.bbox[2]:.1f}, {inst.bbox[3]:.1f}]"
            print(f"{inst.instance_id:<4} {inst.class_name:<20} {inst.confidence:<12.3f} {inst.mask_area:<12,} {bbox_str}")

        # Model info
        print(f"\nModel Information:")
        print(f"  - Name: {response.model_info['name']}")
        print(f"  - Backbone: {response.model_info['backbone']}")
        print(f"  - Framework: {response.model_info['framework']}")
        print(f"  - Pretrained on: {response.model_info['pretrained_on']}")
        print(f"  - Device: {response.model_info['device']}")

        # Image info
        print(f"\nImage Information:")
        print(f"  - Size: {response.image_info['size']}")
        print(f"  - Index: {response.image_info['index']}")

        # Visualization data
        print(f"\nVisualization Data:")
        print(f"  - Original image: {len(response.visualization_data['original_image'])} bytes (base64)")
        print(f"  - Colored masks: {len(response.visualization_data['colored_masks'])} bytes (base64)")
        print(f"  - Overlay image: {len(response.visualization_data['overlay_image'])} bytes (base64)")
        print(f"  - Annotated image: {len(response.visualization_data['annotated_image'])} bytes (base64)")
        print(f"  - Instance colors: {len(response.visualization_data['instance_colors'])} entries")

        print(f"\n{'=' * 80}")
        print("✓ Instance Segmentation test completed successfully!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n✗ Error during instance segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_algorithm_info():
    """Test algorithm info retrieval."""
    print("\n" + "=" * 80)
    print("Testing Algorithm Info")
    print("=" * 80)

    try:
        info = InstanceSegmentationModel.get_algorithm_info()

        print("\nDataset Information:")
        dataset = info['dataset']
        print(f"  - Name: {dataset['name']}")
        print(f"  - Description: {dataset['description']}")
        print(f"  - Number of classes: {dataset['num_classes']}")
        print(f"  - Number of samples: {dataset['num_samples']}")

        print("\nSample Image Descriptions:")
        for idx, desc in enumerate(dataset['sample_descriptions']):
            print(f"  [{idx}] {desc}")

        print("\nAlgorithm Details:")
        algo = info['algorithm_details']
        print(f"  - Algorithm: {algo['algorithm']}")
        print(f"  - Description: {algo['description']}")
        print(f"  - Framework: {algo['framework']}")
        print(f"  - Pretrained dataset: {algo['pretrained_dataset']}")

        print("\nCapabilities:")
        for cap in algo['capabilities']:
            print(f"  - {cap}")

        print("\n✓ Algorithm info test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error retrieving algorithm info: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🚀 Starting Instance Segmentation Tests\n")

    # Run tests
    test1_passed = test_algorithm_info()
    test2_passed = test_instance_segmentation()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Algorithm Info Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Instance Segmentation Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("=" * 80)

    # Exit with appropriate code
    sys.exit(0 if (test1_passed and test2_passed) else 1)
