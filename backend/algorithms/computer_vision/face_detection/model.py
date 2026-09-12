"""Face Detection (Haar Cascades) model implementation."""

import time
import base64
from io import BytesIO
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from .schema import FaceDetectionRequest, FaceDetectionResponse, FaceStatistics, FaceBox
from .data import download_sample_image, get_dataset_info


class FaceDetectionModel:
    """Face Detection model using Haar Cascade classifiers.

    The Haar Cascade classifier is a machine learning based approach where
    a cascade function is trained from positive and negative images. It uses
    Haar-like features to detect objects (in this case, faces) in images.

    The detector works by sliding a detection window across the image at
    multiple scales and positions, evaluating each window using the cascade
    of classifiers.
    """

    def __init__(self):
        """Initialize Face Detection model with Haar Cascade classifier."""
        # Load pre-trained Haar Cascade for frontal face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")

    def detect_faces(
        self,
        image_path: str,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Tuple[int, int] | None = None
    ) -> Dict[str, Any]:
        """Detect faces in an image using Haar Cascade classifier.

        Args:
            image_path: Path to the input image
            scale_factor: Scale reduction factor (1.05-1.3)
            min_neighbors: Minimum neighbors required for detection (1-10)
            min_size: Minimum face size as (width, height)
            max_size: Maximum face size as (width, height) or None

        Returns:
            Dictionary containing face detection results and metadata

        Raises:
            RuntimeError: If face detection fails
        """
        start_time = time.time()

        try:
            # Load image
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise RuntimeError(f"Failed to load image from {image_path}")

            # Convert to RGB for display
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for face detection (Haar Cascades work on grayscale)
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Enhance contrast for better detection
            gray_image = cv2.equalizeHist(gray_image)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                maxSize=max_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Create annotated image with bounding boxes
            annotated_image = original_image_rgb.copy()
            annotated_pil = Image.fromarray(annotated_image)
            draw = ImageDraw.Draw(annotated_pil)

            # Process detected faces
            face_boxes = []
            for i, (x, y, w, h) in enumerate(faces):
                # Calculate confidence based on neighbors (normalized)
                confidence = min(1.0, min_neighbors / 10.0)

                # Calculate center and area
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                area = w * h

                face_boxes.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': round(confidence, 2),
                    'center': {'x': center_x, 'y': center_y},
                    'area': area
                })

                # Draw bounding box (green with varying thickness based on size)
                thickness = max(2, int(min(w, h) / 50))
                draw.rectangle(
                    [(x, y), (x + w, y + h)],
                    outline=(0, 255, 0),
                    width=thickness
                )

                # Draw face number label
                label = f"#{i+1}"
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                              size=max(12, int(min(w, h) / 10)))
                except:
                    font = ImageFont.load_default()

                # Draw label background
                bbox = draw.textbbox((x, y - 25), label, font=font)
                draw.rectangle(bbox, fill=(0, 255, 0))
                draw.text((x, y - 25), label, fill=(0, 0, 0), font=font)

            annotated_image = np.array(annotated_pil)

            # Create visualization with face centers marked
            centers_image = original_image_rgb.copy()
            centers_pil = Image.fromarray(centers_image)
            draw_centers = ImageDraw.Draw(centers_pil)

            for face in face_boxes:
                cx, cy = face['center']['x'], face['center']['y']
                radius = 5
                draw_centers.ellipse(
                    [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                    fill=(255, 0, 0),
                    outline=(255, 255, 0),
                    width=2
                )

            centers_image = np.array(centers_pil)

            # Calculate statistics
            face_count = len(faces)
            if face_count > 0:
                face_areas = [face['area'] for face in face_boxes]
                average_face_size = np.mean(face_areas)
                largest_face_size = max(face_areas)
                smallest_face_size = min(face_areas)
                total_face_area = sum(face_areas)
            else:
                average_face_size = 0
                largest_face_size = 0
                smallest_face_size = 0
                total_face_area = 0

            total_pixels = original_image_rgb.shape[0] * original_image_rgb.shape[1]
            face_density = (total_face_area / total_pixels) * 100 if total_pixels > 0 else 0

            detection_time = (time.time() - start_time) * 1000

            return {
                'original_image': original_image_rgb,
                'gray_image': gray_image,
                'annotated_image': annotated_image,
                'centers_image': centers_image,
                'face_boxes': face_boxes,
                'face_count': face_count,
                'average_face_size': average_face_size,
                'largest_face_size': largest_face_size,
                'smallest_face_size': smallest_face_size,
                'total_face_area': total_face_area,
                'face_density': face_density,
                'image_shape': original_image_rgb.shape,
                'detection_time_ms': detection_time
            }

        except Exception as e:
            raise RuntimeError(f"Face detection failed: {str(e)}")

    def process_request(self, request: FaceDetectionRequest) -> FaceDetectionResponse:
        """Process a face detection request.

        Args:
            request: FaceDetectionRequest with detection parameters

        Returns:
            FaceDetectionResponse with face detection results

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If face detection fails
        """
        start_time = time.time()

        # Validate parameters
        if request.scale_factor < 1.05 or request.scale_factor > 1.3:
            raise ValueError("scale_factor must be between 1.05 and 1.3")

        if request.min_neighbors < 1 or request.min_neighbors > 10:
            raise ValueError("min_neighbors must be between 1 and 10")

        # Download sample image
        try:
            image_path = download_sample_image(request.image_index)
        except Exception as e:
            raise ValueError(f"Failed to load sample image: {str(e)}")

        # Prepare size parameters
        min_size = (request.min_size, request.min_size)
        max_size = None if request.max_size is None else (request.max_size, request.max_size)

        # Run face detection
        detection_result = self.detect_faces(
            image_path,
            scale_factor=request.scale_factor,
            min_neighbors=request.min_neighbors,
            min_size=min_size,
            max_size=max_size
        )

        # Convert face boxes to schema
        faces = [FaceBox(**face) for face in detection_result['face_boxes']]

        # Determine detection quality based on parameters
        detection_quality = self._assess_detection_quality(
            request.scale_factor,
            request.min_neighbors
        )

        # Calculate statistics
        image_shape = detection_result['image_shape']
        statistics = FaceStatistics(
            face_count=detection_result['face_count'],
            average_face_size=round(detection_result['average_face_size'], 2),
            largest_face_size=detection_result['largest_face_size'],
            smallest_face_size=detection_result['smallest_face_size'],
            total_face_area=detection_result['total_face_area'],
            face_density=round(detection_result['face_density'], 2),
            image_dimensions={
                'width': int(image_shape[1]),
                'height': int(image_shape[0])
            },
            detection_quality=detection_quality
        )

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            detection_result['original_image'],
            detection_result['annotated_image'],
            detection_result['gray_image'],
            detection_result['centers_image']
        )

        # Get algorithm info
        algorithm_info = self._get_algorithm_info()

        # Get image info
        image_info = {
            'width': int(image_shape[1]),
            'height': int(image_shape[0]),
            'channels': int(image_shape[2]),
            'image_index': request.image_index
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return FaceDetectionResponse(
            success=True,
            faces=faces,
            statistics=statistics,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            algorithm_info=algorithm_info,
            parameters_used={
                'scale_factor': request.scale_factor,
                'min_neighbors': request.min_neighbors,
                'min_size': request.min_size,
                'max_size': request.max_size,
                'image_index': request.image_index
            },
            image_info=image_info
        )

    def _assess_detection_quality(self, scale_factor: float, min_neighbors: int) -> str:
        """Assess detection quality based on parameters.

        Args:
            scale_factor: Scale reduction factor
            min_neighbors: Minimum neighbors

        Returns:
            Quality rating: "High", "Medium", or "Low"
        """
        # Lower scale factor + higher min_neighbors = higher quality but slower
        if scale_factor <= 1.1 and min_neighbors >= 5:
            return "High"
        elif scale_factor <= 1.15 and min_neighbors >= 4:
            return "Medium"
        else:
            return "Low"

    def _prepare_visualization_data(
        self,
        original_image: np.ndarray,
        annotated_image: np.ndarray,
        gray_image: np.ndarray,
        centers_image: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            original_image: Original input image (RGB)
            annotated_image: Image with face bounding boxes
            gray_image: Grayscale version used for detection
            centers_image: Image with face centers marked

        Returns:
            Dictionary with visualization data
        """
        # Convert images to base64 for frontend
        original_base64 = self._image_to_base64(original_image)
        annotated_base64 = self._image_to_base64(annotated_image)
        gray_base64 = self._image_to_base64(gray_image)
        centers_base64 = self._image_to_base64(centers_image)

        return {
            'original_image': original_base64,
            'annotated_image': annotated_base64,
            'gray_image': gray_base64,
            'centers_image': centers_base64
        }

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image array to base64 string.

        Args:
            image: Image as numpy array (RGB or grayscale)

        Returns:
            Base64 encoded image string
        """
        # Handle grayscale images
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        pil_image = Image.fromarray(image.astype('uint8'))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def _get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the Haar Cascade face detection algorithm.

        Returns:
            Dictionary with algorithm information
        """
        return {
            'algorithm_name': 'Haar Cascade Face Detection',
            'inventors': 'Paul Viola and Michael Jones (2001)',
            'library': 'OpenCV (cv2.CascadeClassifier)',
            'cascade_type': 'haarcascade_frontalface_default.xml',
            'detection_method': [
                '1. Haar-like feature extraction',
                '2. Integral image computation',
                '3. AdaBoost training (pre-trained)',
                '4. Cascade of classifiers',
                '5. Multi-scale sliding window detection'
            ],
            'complexity': {
                'time': 'O(image_size × scales)',
                'space': 'O(cascade_size)'
            },
            'parameters': {
                'scale_factor': 'How much the image size is reduced at each scale (lower = more accurate but slower)',
                'min_neighbors': 'How many neighbors each rectangle should have to retain it (higher = fewer false positives)',
                'min_size': 'Minimum possible face size (faces smaller than this are ignored)',
                'max_size': 'Maximum possible face size (faces larger than this are ignored)'
            },
            'preprocessing': 'Histogram equalization for contrast enhancement',
            'optimal_settings': {
                'high_accuracy': 'scale_factor=1.05, min_neighbors=6',
                'balanced': 'scale_factor=1.1, min_neighbors=5',
                'fast': 'scale_factor=1.2, min_neighbors=3'
            }
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
            'algorithm_details': {
                'optimal_settings': {
                    'high_accuracy': 'scale_factor=1.05-1.1, min_neighbors=5-6',
                    'balanced': 'scale_factor=1.1, min_neighbors=4-5',
                    'fast': 'scale_factor=1.15-1.2, min_neighbors=3-4'
                },
                'tips': [
                    'Lower scale_factor = more thorough search but slower',
                    'Higher min_neighbors = fewer false positives but may miss faces',
                    'Adjust min_size based on expected face size in image',
                    'Works best with frontal faces and good lighting',
                    'Grayscale conversion and histogram equalization improve detection'
                ],
                'limitations': [
                    'May struggle with profile views or tilted faces',
                    'Performance degrades with poor lighting',
                    'Can produce false positives on face-like patterns',
                    'Not rotation-invariant (faces must be upright)'
                ]
            }
        }
