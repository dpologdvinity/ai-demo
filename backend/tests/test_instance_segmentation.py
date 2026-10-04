"""Smoke tests for instance segmentation."""

import pytest
from algorithms.computer_vision.instance_segmentation import (
    InstanceSegmentationModel,
    InstanceSegmentationRequest
)


class TestInstanceSegmentationModel:
    """Basic smoke tests for InstanceSegmentationModel."""

    def test_instance_segmentation(self):
        """Test basic instance segmentation with default parameters."""
        model = InstanceSegmentationModel(model_backbone="resnet50")
        request = InstanceSegmentationRequest(
            model_backbone="resnet50",
            confidence_threshold=0.5,
            mask_threshold=0.5,
            max_instances=100,
            nms_threshold=0.5,
            image_index=0
        )
        response = model.process_request(request)

        assert response is not None
        assert hasattr(response, 'instances')
        assert hasattr(response, 'statistics')
        assert hasattr(response, 'execution_time_ms')
        assert response.execution_time_ms >= 0
