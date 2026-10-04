"""Smoke tests for optical flow."""

import pytest
import numpy as np
from algorithms.computer_vision.optical_flow import OpticalFlowModel


class TestOpticalFlowModel:
    """Basic smoke tests for OpticalFlowModel."""

    def test_optical_flow(self):
        """Test basic optical flow computation with synthetic frames."""
        model = OpticalFlowModel()

        # Create synthetic test frames (128x128 BGR)
        frame1 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        # Apply horizontal translation to frame2
        M = np.float32([[1, 0, 5], [0, 1, 0]])
        import cv2
        frame2 = cv2.warpAffine(frame2, M, (128, 128))

        # Compute optical flow
        result = model.compute_optical_flow(
            frame1,
            frame2,
            method="farneback",
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3
        )

        assert result is not None
        assert 'magnitude' in result
        assert 'statistics' in result
        assert 'computation_time_ms' in result
        assert result['computation_time_ms'] >= 0
        assert 'flow' in result
        assert result['flow'].shape == (128, 128, 2)
