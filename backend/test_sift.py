"""Test script for SIFT implementation."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.computer_vision.sift import SIFTModel, SIFTRequest


def test_sift_basic():
    """Test basic SIFT feature detection."""
    print("Testing SIFT feature detection...")

    # Create request
    request = SIFTRequest(
        nfeatures=500,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
        image_index=0,
        match_mode=False
    )

    # Initialize model
    model = SIFTModel(
        nfeatures=request.nfeatures,
        nOctaveLayers=request.nOctaveLayers,
        contrastThreshold=request.contrastThreshold,
        edgeThreshold=request.edgeThreshold,
        sigma=request.sigma
    )

    # Process request
    response = model.process_request(request)

    print(f"✓ SIFT detection completed in {response.execution_time_ms:.2f}ms")
    print(f"✓ Detected {response.statistics.keypoint_count} keypoints")
    print(f"✓ Average scale: {response.statistics.average_scale:.2f}")
    print(f"✓ Average response: {response.statistics.average_response:.4f}")
    print(f"✓ Descriptor dimensions: {response.statistics.descriptor_dimensions}")
    print(f"✓ Number of keypoints returned: {len(response.keypoints)}")

    # Check visualizations
    assert 'original_image' in response.visualization_data
    assert 'keypoints_image' in response.visualization_data
    assert 'descriptor_heatmap' in response.visualization_data
    print("✓ All visualizations generated")

    print("\nBasic SIFT test passed!")


def test_sift_matching():
    """Test SIFT feature matching."""
    print("\nTesting SIFT feature matching...")

    # Create request with matching mode
    request = SIFTRequest(
        nfeatures=500,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
        image_index=0,
        match_mode=True,
        match_image_index=1
    )

    # Initialize model
    model = SIFTModel(
        nfeatures=request.nfeatures,
        nOctaveLayers=request.nOctaveLayers,
        contrastThreshold=request.contrastThreshold,
        edgeThreshold=request.edgeThreshold,
        sigma=request.sigma
    )

    # Process request
    response = model.process_request(request)

    print(f"✓ SIFT matching completed in {response.execution_time_ms:.2f}ms")

    if response.match_statistics:
        print(f"✓ Found {response.match_statistics.match_count} matches")
        print(f"✓ Keypoints in image 1: {response.match_statistics.keypoints_image1}")
        print(f"✓ Keypoints in image 2: {response.match_statistics.keypoints_image2}")
        print(f"✓ Match ratio: {response.match_statistics.match_ratio:.3f}")
        print(f"✓ Average match distance: {response.match_statistics.average_match_distance:.2f}")

        # Check matching visualizations
        assert 'match_image' in response.visualization_data
        assert 'image2_with_keypoints' in response.visualization_data
        print("✓ Matching visualizations generated")
    else:
        print("⚠ Warning: No matches found")

    print("\nSIFT matching test passed!")


def test_sift_algorithm_info():
    """Test SIFT algorithm info retrieval."""
    print("\nTesting SIFT algorithm info...")

    info = SIFTModel.get_algorithm_info()

    assert 'dataset' in info
    assert 'algorithm_details' in info

    dataset = info['dataset']
    print(f"✓ Dataset: {dataset['name']}")
    print(f"✓ Number of images: {dataset['num_images']}")
    print(f"✓ Use cases: {', '.join(dataset['use_cases'][:3])}")

    algo_details = info['algorithm_details']
    assert 'optimal_settings' in algo_details
    assert 'tips' in algo_details
    assert 'matching' in algo_details
    print("✓ Algorithm details retrieved")

    print("\nAlgorithm info test passed!")


if __name__ == "__main__":
    try:
        test_sift_basic()
        test_sift_matching()
        test_sift_algorithm_info()
        print("\n" + "="*50)
        print("ALL SIFT TESTS PASSED!")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
