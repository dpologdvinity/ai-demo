"""Smoke tests for SIFT feature detection."""

import pytest
from algorithms.computer_vision.sift import (
    SIFTModel,
    SIFTRequest
)


class TestSIFTModel:
    """Basic smoke tests for SIFTModel."""

    def test_sift_detection(self):
        """Test basic SIFT feature detection with default parameters."""
        model = SIFTModel(
            nfeatures=500,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        request = SIFTRequest(
            nfeatures=500,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6,
            image_index=0,
            match_mode=False,
            match_image_index=1
        )
        response = model.process_request(request)

        assert response is not None
        assert hasattr(response, 'keypoints')
        assert hasattr(response, 'statistics')
        assert hasattr(response, 'execution_time_ms')
        assert response.execution_time_ms >= 0
