"""Smoke tests for face detection."""

import pytest
from algorithms.computer_vision.face_detection import (
    FaceDetectionModel,
    FaceDetectionRequest
)


class TestFaceDetectionModel:
    """Basic smoke tests for FaceDetectionModel."""

    def test_face_detection(self):
        """Test basic face detection with default parameters."""
        model = FaceDetectionModel()
        request = FaceDetectionRequest(
            scale_factor=1.1,
            min_neighbors=5,
            min_size=30,
            max_size=None,
            image_index=0
        )
        response = model.process_request(request)

        assert response is not None
        assert hasattr(response, 'faces')
        assert hasattr(response, 'statistics')
        assert hasattr(response, 'execution_time_ms')
        assert response.execution_time_ms >= 0
