"""Test Face Detection implementation."""

import pytest
from algorithms.computer_vision.face_detection import (
    FaceDetectionModel,
    FaceDetectionRequest,
    FaceDetectionResponse
)


def test_face_detection_request_schema():
    """Test FaceDetectionRequest schema validation."""
    # Valid request
    request = FaceDetectionRequest(
        scale_factor=1.1,
        min_neighbors=5,
        min_size=30,
        max_size=200,
        image_index=0
    )
    assert request.scale_factor == 1.1
    assert request.min_neighbors == 5
    assert request.min_size == 30
    assert request.max_size == 200
    assert request.image_index == 0

    # Test defaults
    default_request = FaceDetectionRequest()
    assert default_request.scale_factor == 1.1
    assert default_request.min_neighbors == 5
    assert default_request.min_size == 30
    assert default_request.max_size is None


def test_face_detection_request_validation():
    """Test FaceDetectionRequest parameter validation."""
    # Test scale_factor bounds
    with pytest.raises(ValueError):
        FaceDetectionRequest(scale_factor=1.0)  # Too low

    with pytest.raises(ValueError):
        FaceDetectionRequest(scale_factor=1.4)  # Too high

    # Test min_neighbors bounds
    with pytest.raises(ValueError):
        FaceDetectionRequest(min_neighbors=0)  # Too low

    with pytest.raises(ValueError):
        FaceDetectionRequest(min_neighbors=11)  # Too high

    # Test min_size bounds
    with pytest.raises(ValueError):
        FaceDetectionRequest(min_size=10)  # Too low


def test_face_detection_model_initialization():
    """Test FaceDetectionModel initialization."""
    model = FaceDetectionModel()
    assert model.face_cascade is not None
    assert not model.face_cascade.empty()


def test_face_detection_process_request():
    """Test FaceDetectionModel.process_request with sample image."""
    model = FaceDetectionModel()
    request = FaceDetectionRequest(
        scale_factor=1.1,
        min_neighbors=5,
        min_size=30,
        image_index=0
    )

    response = model.process_request(request)

    # Check response structure
    assert isinstance(response, FaceDetectionResponse)
    assert response.success is True
    assert isinstance(response.faces, list)
    assert response.statistics is not None
    assert response.visualization_data is not None
    assert response.execution_time_ms > 0
    assert response.algorithm_info is not None
    assert response.parameters_used is not None
    assert response.image_info is not None

    # Check statistics
    stats = response.statistics
    assert stats.face_count >= 0
    assert stats.image_dimensions is not None
    assert 'width' in stats.image_dimensions
    assert 'height' in stats.image_dimensions
    assert stats.detection_quality in ['High', 'Medium', 'Low']

    # Check visualization data
    viz = response.visualization_data
    assert 'original_image' in viz
    assert 'annotated_image' in viz
    assert 'gray_image' in viz
    assert 'centers_image' in viz
    assert viz['original_image'].startswith('data:image/jpeg;base64,')

    # Check algorithm info
    algo = response.algorithm_info
    assert algo['algorithm_name'] == 'Haar Cascade Face Detection'
    assert 'library' in algo
    assert 'complexity' in algo
    assert 'parameters' in algo


def test_face_detection_with_different_parameters():
    """Test face detection with various parameter combinations."""
    model = FaceDetectionModel()

    # High accuracy settings
    request_high = FaceDetectionRequest(
        scale_factor=1.05,
        min_neighbors=6,
        min_size=30,
        image_index=0
    )
    response_high = model.process_request(request_high)
    assert response_high.statistics.detection_quality == 'High'

    # Fast settings
    request_fast = FaceDetectionRequest(
        scale_factor=1.2,
        min_neighbors=3,
        min_size=40,
        image_index=0
    )
    response_fast = model.process_request(request_fast)
    assert response_fast.success is True


def test_face_box_schema():
    """Test FaceBox schema."""
    from algorithms.computer_vision.face_detection.schema import FaceBox

    face = FaceBox(
        x=100,
        y=150,
        width=80,
        height=80,
        confidence=0.85,
        center={'x': 140, 'y': 190},
        area=6400
    )
    assert face.x == 100
    assert face.y == 150
    assert face.width == 80
    assert face.height == 80
    assert face.confidence == 0.85
    assert face.center['x'] == 140
    assert face.area == 6400


def test_face_statistics_schema():
    """Test FaceStatistics schema."""
    from algorithms.computer_vision.face_detection.schema import FaceStatistics

    stats = FaceStatistics(
        face_count=3,
        average_face_size=5000.5,
        largest_face_size=8000,
        smallest_face_size=3000,
        total_face_area=15000,
        face_density=12.5,
        image_dimensions={'width': 640, 'height': 480},
        detection_quality='High'
    )
    assert stats.face_count == 3
    assert stats.average_face_size == 5000.5
    assert stats.face_density == 12.5
    assert stats.detection_quality == 'High'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
