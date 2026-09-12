"""Test script for Instance Segmentation API endpoints."""

import sys
import os
import asyncio

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.routes.computer_vision import (
    segment_instances,
    get_instance_segmentation_info,
    AlgorithmRegistry
)
from algorithms.computer_vision.instance_segmentation import InstanceSegmentationRequest


async def test_info_endpoint():
    """Test the /info endpoint."""
    print("=" * 80)
    print("Testing /instance-segmentation/info endpoint")
    print("=" * 80)

    try:
        info = await get_instance_segmentation_info()

        print("\nMetadata:")
        print(f"  - ID: {info['metadata']['id']}")
        print(f"  - Name: {info['metadata']['name']}")
        print(f"  - Category: {info['metadata']['category']}")
        print(f"  - Difficulty: {info['metadata']['difficulty']}")
        print(f"  - Description: {info['metadata']['description']}")

        print("\nParameters:")
        for param in info['metadata']['parameters']:
            print(f"  - {param['name']}: {param['label']} (default: {param['default']})")

        print("\nUse Cases:")
        for use_case in info['metadata']['use_cases']:
            print(f"  - {use_case}")

        print("\nDataset:")
        print(f"  - Name: {info['dataset']['name']}")
        print(f"  - Classes: {info['dataset']['num_classes']}")
        print(f"  - Samples: {info['dataset']['num_samples']}")

        print("\n✓ Info endpoint test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Info endpoint test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_segment_endpoint():
    """Test the /segment endpoint."""
    print("\n" + "=" * 80)
    print("Testing /instance-segmentation/segment endpoint")
    print("=" * 80)

    try:
        # Create request
        request = InstanceSegmentationRequest(
            confidence_threshold=0.6,
            model_backbone='resnet50',
            mask_threshold=0.5,
            max_instances=50,
            image_index=0,
            nms_threshold=0.5
        )

        print(f"\nRequest: {request.model_dump()}")

        # Call endpoint
        response = await segment_instances(request)

        print(f"\n✓ Segmentation successful!")
        print(f"  - Execution time: {response.execution_time_ms:.2f} ms")
        print(f"  - Total instances: {response.statistics.total_instances}")
        print(f"  - Unique classes: {response.statistics.unique_classes}")
        print(f"  - Average confidence: {response.statistics.avg_confidence:.3f}")
        print(f"  - Coverage: {response.statistics.coverage_percentage:.2f}%")

        print("\n  Top instances:")
        for i, inst in enumerate(response.instances[:5], 1):
            print(f"    {i}. {inst.class_name}: {inst.confidence:.3f} ({inst.mask_area:,} px)")

        print("\n✓ Segment endpoint test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Segment endpoint test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_algorithm_registry():
    """Test that the algorithm is registered."""
    print("\n" + "=" * 80)
    print("Testing Algorithm Registry")
    print("=" * 80)

    try:
        # Get algorithm from registry
        metadata = AlgorithmRegistry.get("instance-segmentation")

        if metadata:
            print(f"\n✓ Algorithm found in registry:")
            print(f"  - ID: {metadata.id}")
            print(f"  - Name: {metadata.name}")
            print(f"  - Slug: {metadata.slug}")
            print(f"  - Category: {metadata.category}")
            print(f"  - Difficulty: {metadata.difficulty}")
            print(f"  - Tags: {', '.join(metadata.tags)}")

            print("\n✓ Registry test passed!")
            return True
        else:
            print("\n✗ Algorithm not found in registry!")
            return False

    except Exception as e:
        print(f"\n✗ Registry test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n🚀 Starting Instance Segmentation API Tests\n")

    # Run tests
    test1_passed = await test_algorithm_registry()
    test2_passed = await test_info_endpoint()
    test3_passed = await test_segment_endpoint()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Algorithm Registry Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Info Endpoint Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Segment Endpoint Test: {'✓ PASSED' if test3_passed else '✗ FAILED'}")
    print("=" * 80)

    # Exit with appropriate code
    all_passed = test1_passed and test2_passed and test3_passed
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
