"""Smoke tests for image classification."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from algorithms.computer_vision.image_classification import (
    ImageClassificationModel,
    ImageClassificationRequest,
    ImageClassificationResponse
)


class TestImageClassificationModel:
    """Basic smoke tests for ImageClassificationModel."""

    def test_image_classification(self):
        """Test basic image classification interface and initialization."""
        model = ImageClassificationModel(model_name="resnet18")

        # Verify model instantiation
        assert model is not None
        assert model.model_name == "resnet18"
        assert hasattr(model, 'classify')
        assert hasattr(model, 'process_request')
        assert hasattr(model, 'load_model')

        # Verify request schema
        request = ImageClassificationRequest(
            model_name="resnet18",
            top_k=5,
            confidence_threshold=0.1,
            image_index=0
        )
        assert request.model_name == "resnet18"
        assert request.top_k == 5
        assert request.confidence_threshold == 0.1

        # Verify class labels are loaded
        assert len(model.class_labels) == 1000
        assert model.class_labels[0] == "tench"
