"""YOLO Object Detection model implementation."""

import time
import base64
from io import BytesIO
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import cv2

from .schema import YOLORequest, YOLOResponse, Detection, DetectionStatistics
from .data import download_sample_image, get_dataset_info


class YOLOModel:
    """YOLO Object Detection model using YOLOv8.

    This class provides a wrapper around the Ultralytics YOLOv8 model
    for object detection tasks.

    Attributes:
        model: The YOLOv8 model instance
        model_size: Size of the model (n, s, or m)
        class_names: COCO dataset class names
    """

    # COCO class filtering groups
    CLASS_FILTERS = {
        'person': ['person'],
        'vehicle': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                   'truck', 'boat'],
        'animal': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                  'bear', 'zebra', 'giraffe']
    }

    def __init__(self, model_version: str = 'yolov8n'):
        """Initialize YOLO model.

        Args:
            model_version: Model version - 'yolov8n', 'yolov8s', 'yolov8m', or 'yolov5s'

        Raises:
            ValueError: If model_version is invalid
        """
        valid_versions = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov5s']
        if model_version not in valid_versions:
            raise ValueError(f"model_version must be one of {valid_versions}")

        self.model_version = model_version
        self.model = None
        self.class_names = None

    def load_model(self):
        """Load the YOLO model.

        Loads the model on first use to avoid unnecessary initialization.
        """
        if self.model is None:
            try:
                from ultralytics import YOLO
                model_name = f'{self.model_version}.pt'
                print(f"Loading YOLO model: {model_name}")
                self.model = YOLO(model_name)
                self.class_names = self.model.names
                print(f"Model loaded successfully. Device: {self.model.device}")
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def detect(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        max_detections: int = 100,
        class_filter: str = 'all'
    ) -> Dict[str, Any]:
        """Run object detection on an image.

        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to return
            class_filter: Filter by class category ('all', 'person', 'vehicle', 'animal')

        Returns:
            Dictionary containing detection results and metadata

        Raises:
            RuntimeError: If detection fails
        """
        # Load model if not already loaded
        self.load_model()

        start_time = time.time()

        try:
            # Run inference
            results = self.model(
                image_path,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                verbose=False
            )

            # Extract detection results
            result = results[0]
            boxes = result.boxes

            # Parse detections
            detections = []
            filter_classes = None
            if class_filter != 'all' and class_filter in self.CLASS_FILTERS:
                filter_classes = set(self.CLASS_FILTERS[class_filter])

            for box in boxes:
                # Get bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy()

                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = self.class_names[class_id]

                # Apply class filter if specified
                if filter_classes and class_name not in filter_classes:
                    continue

                detections.append({
                    'bbox': xyxy.tolist(),
                    'class_name': class_name,
                    'class_id': class_id,
                    'confidence': confidence
                })

            # Load original image for visualization
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Get annotated image
            annotated_image = result.plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            detection_time = (time.time() - start_time) * 1000

            return {
                'detections': detections,
                'original_image': original_image,
                'annotated_image': annotated_image,
                'image_shape': original_image.shape,
                'detection_time_ms': detection_time
            }

        except Exception as e:
            raise RuntimeError(f"Detection failed: {str(e)}")

    def process_request(self, request: YOLORequest) -> YOLOResponse:
        """Process a YOLO detection request.

        Args:
            request: YOLORequest with detection parameters

        Returns:
            YOLOResponse with detection results

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If detection fails
        """
        start_time = time.time()

        # Download sample image
        try:
            image_path = download_sample_image(request.image_index)
        except Exception as e:
            raise ValueError(f"Failed to load sample image: {str(e)}")

        # Update model version if needed
        if self.model_version != request.model_version:
            self.model_version = request.model_version
            self.model = None  # Force reload

        # Run detection
        detection_result = self.detect(
            image_path,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            max_detections=request.max_detections,
            class_filter=request.class_filter
        )

        # Parse detections
        detections = [
            Detection(**det) for det in detection_result['detections']
        ]

        # Calculate statistics
        statistics = self._calculate_statistics(detections)

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            detection_result['original_image'],
            detection_result['annotated_image'],
            detections
        )

        # Get model info
        model_info = self._get_model_info()

        # Get image info
        image_shape = detection_result['image_shape']
        image_info = {
            'width': int(image_shape[1]),
            'height': int(image_shape[0]),
            'channels': int(image_shape[2]),
            'image_index': request.image_index
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return YOLOResponse(
            success=True,
            detections=detections,
            statistics=statistics,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'confidence_threshold': request.confidence_threshold,
                'iou_threshold': request.iou_threshold,
                'model_version': request.model_version,
                'max_detections': request.max_detections,
                'image_index': request.image_index,
                'class_filter': request.class_filter
            },
            image_info=image_info
        )

    def _calculate_statistics(self, detections: List[Detection]) -> DetectionStatistics:
        """Calculate statistics from detections.

        Args:
            detections: List of Detection objects

        Returns:
            DetectionStatistics object
        """
        if not detections:
            return DetectionStatistics(
                total_detections=0,
                class_counts={},
                avg_confidence=0.0,
                confidence_distribution={}
            )

        # Count objects per class
        class_counts = {}
        confidences = []

        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
            confidences.append(det.confidence)

        # Calculate confidence distribution
        confidence_distribution = {
            '0.0-0.3': 0,
            '0.3-0.5': 0,
            '0.5-0.7': 0,
            '0.7-0.9': 0,
            '0.9-1.0': 0
        }

        for conf in confidences:
            if conf < 0.3:
                confidence_distribution['0.0-0.3'] += 1
            elif conf < 0.5:
                confidence_distribution['0.3-0.5'] += 1
            elif conf < 0.7:
                confidence_distribution['0.5-0.7'] += 1
            elif conf < 0.9:
                confidence_distribution['0.7-0.9'] += 1
            else:
                confidence_distribution['0.9-1.0'] += 1

        return DetectionStatistics(
            total_detections=len(detections),
            class_counts=class_counts,
            avg_confidence=float(np.mean(confidences)),
            confidence_distribution=confidence_distribution
        )

    def _prepare_visualization_data(
        self,
        original_image: np.ndarray,
        annotated_image: np.ndarray,
        detections: List[Detection]
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            original_image: Original input image
            annotated_image: Image with bounding boxes drawn
            detections: List of Detection objects

        Returns:
            Dictionary with visualization data
        """
        # Convert images to base64 for frontend
        original_base64 = self._image_to_base64(original_image)
        annotated_base64 = self._image_to_base64(annotated_image)

        # Prepare detection list for display
        detection_list = [
            {
                'bbox': det.bbox,
                'class_name': det.class_name,
                'confidence': round(det.confidence, 3)
            }
            for det in detections
        ]

        # Prepare confidence chart data
        confidence_chart = [
            {
                'class': det.class_name,
                'confidence': round(det.confidence, 3)
            }
            for det in sorted(detections, key=lambda x: x.confidence, reverse=True)
        ]

        return {
            'original_image': original_base64,
            'annotated_image': annotated_base64,
            'detection_list': detection_list,
            'confidence_chart': confidence_chart
        }

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image array to base64 string.

        Args:
            image: Image as numpy array (RGB)

        Returns:
            Base64 encoded image string
        """
        pil_image = Image.fromarray(image.astype('uint8'))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {}

        return {
            'model_name': self.model_version.upper(),
            'model_version': self.model_version,
            'num_classes': len(self.class_names) if self.class_names else 80,
            'framework': 'Ultralytics YOLO',
            'device': str(self.model.device),
            'input_size': '640x640 (default)',
            'architecture': 'YOLO (CSPDarknet + PANet + Detection Head)'
        }

    @staticmethod
    def get_algorithm_info() -> Dict[str, Any]:
        """Get algorithm metadata and dataset information.

        Returns:
            Dictionary with algorithm and dataset information
        """
        dataset_info = get_dataset_info()

        return {
            'dataset': dataset_info,
            'model_variants': {
                'yolov8n': 'YOLOv8 Nano - Fastest, smallest model',
                'yolov8s': 'YOLOv8 Small - Balanced speed and accuracy',
                'yolov8m': 'YOLOv8 Medium - Higher accuracy, slower',
                'yolov5s': 'YOLOv5 Small - Legacy model, good performance'
            }
        }
