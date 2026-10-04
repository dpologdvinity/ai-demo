"""Smoke tests for neural style transfer."""

import pytest
from algorithms.computer_vision.style_transfer import (
    StyleTransferModel,
    StyleTransferRequest
)


class TestStyleTransferModel:
    """Basic smoke tests for StyleTransferModel."""

    def test_style_transfer(self):
        """Test basic style transfer with minimal iterations."""
        model = StyleTransferModel()
        request = StyleTransferRequest(
            content_image_index=0,
            style_image_index=0,
            iterations=10,  # Minimal for smoke test
            content_weight=1.0,
            style_weight=1000000.0,
            learning_rate=0.003,
            image_size=256  # Smallest size for speed
        )
        response = model.process_request(request)

        assert response is not None
        assert hasattr(response, 'stylized_image')
        assert hasattr(response, 'statistics')
        assert hasattr(response, 'execution_time_ms')
        assert response.execution_time_ms >= 0
