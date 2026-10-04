"""Smoke tests for edge detection."""

import pytest
from algorithms.computer_vision.edge_detection import (
    EdgeDetectionModel,
    EdgeDetectionRequest
)


class TestEdgeDetectionModel:
    """Basic smoke tests for EdgeDetectionModel."""

    def test_edge_detection(self):
        """Test basic edge detection with default parameters."""
        model = EdgeDetectionModel()
        request = EdgeDetectionRequest(
            threshold1=50,
            threshold2=150,
            aperture_size=3,
            l2gradient=False,
            image_index=0
        )
        response = model.process_request(request)

        assert response is not None
        assert hasattr(response, 'edges')
        assert hasattr(response, 'statistics')
        assert hasattr(response, 'execution_time_ms')
        assert response.execution_time_ms >= 0
