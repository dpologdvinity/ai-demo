"""Smoke tests for image classification."""

import pytest
from algorithms.computer_vision.image_classification import (
    ImageClassificationModel,
    ImageClassificationRequest
)


class TestImageClassificationModel:
    """Basic smoke tests for ImageClassificationModel."""

    def test_image_classification(self):
        """Test basic image classification with default parameters."""
        model = ImageClassificationModel(model_name="resnet18")
        request = ImageClassificationRequest(
            model_name="resnet18",
            top_k=5,
            confidence_threshold=0.1,
            image_index=0
        )
        response = model.process_request(request)

        assert response is not None
        assert hasattr(response, 'predictions')
        assert hasattr(response, 'statistics')
        assert hasattr(response, 'execution_time_ms')
        assert response.execution_time_ms >= 0
        assert len(response.predictions) > 0
