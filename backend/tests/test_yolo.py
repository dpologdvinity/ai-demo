"""Smoke tests for YOLO object detection."""

import pytest
from algorithms.computer_vision.yolo import YOLOModel, YOLORequest


class TestYOLOModel:
    """Basic smoke tests for YOLOModel."""

    def test_yolo_detection(self):
        """Test basic YOLO detection with default parameters."""
        model = YOLOModel(model_version="yolov8n")
        request = YOLORequest(
            model_version="yolov8n",
            confidence_threshold=0.25,
            iou_threshold=0.45,
            max_detections=100,
            image_index=0,
            class_filter="all"
        )
        response = model.process_request(request)

        assert response is not None
        assert hasattr(response, 'detections')
        assert hasattr(response, 'statistics')
        assert hasattr(response, 'execution_time_ms')
        assert response.execution_time_ms >= 0
