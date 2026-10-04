"""Smoke tests for optical flow."""

import pytest
from algorithms.computer_vision.optical_flow import (
    OpticalFlowModel,
    OpticalFlowRequest
)


class TestOpticalFlowModel:
    """Basic smoke tests for OpticalFlowModel."""

    def test_optical_flow(self):
        """Test basic optical flow computation with default parameters."""
        model = OpticalFlowModel()
        request = OpticalFlowRequest(
            method="farneback",
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            image_pair_index=0
        )
        response = model.process_request(request)

        assert response is not None
        assert hasattr(response, 'flow_magnitude')
        assert hasattr(response, 'statistics')
        assert hasattr(response, 'execution_time_ms')
        assert response.execution_time_ms >= 0
