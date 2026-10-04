"""Smoke tests for neural style transfer."""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path

from algorithms.computer_vision.style_transfer import (
    StyleTransferModel,
    StyleTransferRequest
)


def create_test_image(size=256):
    """Create a test RGB image as numpy array."""
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)


def save_test_image(path):
    """Save a test image to the given path."""
    img_array = create_test_image(size=256)
    img = Image.fromarray(img_array)
    img.save(path)
    return str(path)


class TestStyleTransferModel:
    """Basic smoke tests for StyleTransferModel."""

    def test_style_transfer(self):
        """Test basic style transfer with minimal iterations."""
        model = StyleTransferModel()

        # Create temporary test images
        with tempfile.TemporaryDirectory() as tmpdir:
            content_path = Path(tmpdir) / "content.jpg"
            style_path = Path(tmpdir) / "style.jpg"

            save_test_image(str(content_path))
            save_test_image(str(style_path))

            # Mock the download functions to return our test images
            with patch('algorithms.computer_vision.style_transfer.model.download_content_image',
                      return_value=str(content_path)), \
                 patch('algorithms.computer_vision.style_transfer.model.download_style_image',
                      return_value=str(style_path)):

                request = StyleTransferRequest(
                    content_image_index=0,
                    style_image_index=0,
                    iterations=50,  # Minimal for smoke test
                    content_weight=1.0,
                    style_weight=1000000.0,
                    learning_rate=0.003,
                    image_size=256  # Smallest size for speed
                )
                response = model.process_request(request)

                assert response is not None
                assert response.success is True
                assert hasattr(response, 'visualization_data')
                assert hasattr(response, 'statistics')
                assert hasattr(response, 'execution_time_ms')
                assert response.execution_time_ms >= 0
                assert response.statistics is not None
                assert response.loss_history is not None
                assert len(response.loss_history) > 0
