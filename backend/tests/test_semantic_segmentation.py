"""Smoke tests for semantic segmentation."""

import pytest
from algorithms.computer_vision.semantic_segmentation import (
    SegmentationModel,
    SegmentationRequest
)


class TestSegmentationModel:
    """Basic smoke tests for SegmentationModel."""

    def test_semantic_segmentation(self):
        """Test basic semantic segmentation with default parameters."""
        model = SegmentationModel(model_backbone="resnet50", num_classes=21)
        request = SegmentationRequest(
            model_backbone="resnet50",
            num_classes=21,
            confidence_threshold=0.5,
            image_size=512,
            image_index=0
        )
        response = model.process_request(request)

        assert response is not None
        assert hasattr(response, 'segmentation_mask')
        assert hasattr(response, 'statistics')
        assert hasattr(response, 'execution_time_ms')
        assert response.execution_time_ms >= 0
