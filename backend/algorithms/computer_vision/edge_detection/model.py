"""Edge Detection (Canny Algorithm) model implementation."""

import time
import base64
from io import BytesIO
from typing import Dict, Any
import numpy as np
from PIL import Image
import cv2

from .schema import EdgeDetectionRequest, EdgeDetectionResponse, EdgeStatistics
from .data import download_sample_image, get_dataset_info


class EdgeDetectionModel:
    """Edge Detection model using Canny algorithm.

    The Canny edge detector is a multi-stage algorithm that detects edges
    in images by finding local maxima of the image gradient.

    Stages:
        1. Noise reduction (Gaussian blur)
        2. Gradient calculation (Sobel operators)
        3. Non-maximum suppression
        4. Double threshold
        5. Edge tracking by hysteresis
    """

    def __init__(self):
        """Initialize Edge Detection model."""
        self.blur_kernel_size = (5, 5)  # Gaussian blur kernel for noise reduction

    def detect_edges(
        self,
        image_path: str,
        threshold1: int = 50,
        threshold2: int = 150,
        aperture_size: int = 3,
        l2gradient: bool = False
    ) -> Dict[str, Any]:
        """Detect edges in an image using Canny algorithm.

        Args:
            image_path: Path to the input image
            threshold1: Lower threshold for hysteresis (0-255)
            threshold2: Upper threshold for hysteresis (0-255)
            aperture_size: Sobel kernel size (3, 5, or 7)
            l2gradient: Use L2 norm for gradient magnitude

        Returns:
            Dictionary containing edge detection results and metadata

        Raises:
            RuntimeError: If edge detection fails
        """
        start_time = time.time()

        try:
            # Load image
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise RuntimeError(f"Failed to load image from {image_path}")

            # Convert to RGB for display
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for edge detection
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_image, self.blur_kernel_size, 0)

            # Apply Canny edge detection
            edges = cv2.Canny(
                blurred,
                threshold1,
                threshold2,
                apertureSize=aperture_size,
                L2gradient=l2gradient
            )

            # Create colored edge visualization (edges in white on black background)
            edge_image_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            # Create overlay visualization (edges in color on original image)
            overlay_image = original_image_rgb.copy()
            # Make edges stand out - cyan color for visibility
            overlay_image[edges != 0] = [0, 255, 255]  # Cyan edges

            # Calculate edge statistics
            edge_pixel_count = int(np.sum(edges != 0))
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_density = (edge_pixel_count / total_pixels) * 100

            detection_time = (time.time() - start_time) * 1000

            return {
                'original_image': original_image_rgb,
                'gray_image': gray_image,
                'edge_image': edges,
                'edge_image_rgb': edge_image_rgb,
                'overlay_image': overlay_image,
                'edge_pixel_count': edge_pixel_count,
                'total_pixels': total_pixels,
                'edge_density': edge_density,
                'image_shape': original_image_rgb.shape,
                'detection_time_ms': detection_time
            }

        except Exception as e:
            raise RuntimeError(f"Edge detection failed: {str(e)}")

    def process_request(self, request: EdgeDetectionRequest) -> EdgeDetectionResponse:
        """Process an edge detection request.

        Args:
            request: EdgeDetectionRequest with detection parameters

        Returns:
            EdgeDetectionResponse with edge detection results

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If edge detection fails
        """
        start_time = time.time()

        # Validate aperture size
        request.validate_aperture_size()

        # Validate threshold relationship
        if request.threshold1 >= request.threshold2:
            raise ValueError("threshold1 must be less than threshold2")

        # Download sample image
        try:
            image_path = download_sample_image(request.image_index)
        except Exception as e:
            raise ValueError(f"Failed to load sample image: {str(e)}")

        # Run edge detection
        detection_result = self.detect_edges(
            image_path,
            threshold1=request.threshold1,
            threshold2=request.threshold2,
            aperture_size=request.aperture_size,
            l2gradient=request.l2gradient
        )

        # Calculate statistics
        image_shape = detection_result['image_shape']
        threshold_ratio = request.threshold2 / request.threshold1 if request.threshold1 > 0 else 0

        statistics = EdgeStatistics(
            edge_pixel_count=detection_result['edge_pixel_count'],
            total_pixels=detection_result['total_pixels'],
            edge_density=round(detection_result['edge_density'], 2),
            image_dimensions={
                'width': int(image_shape[1]),
                'height': int(image_shape[0])
            },
            threshold_ratio=round(threshold_ratio, 2)
        )

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            detection_result['original_image'],
            detection_result['edge_image_rgb'],
            detection_result['overlay_image'],
            detection_result['gray_image']
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

        return EdgeDetectionResponse(
            success=True,
            statistics=statistics,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            algorithm_info=algorithm_info,
            parameters_used={
                'threshold1': request.threshold1,
                'threshold2': request.threshold2,
                'aperture_size': request.aperture_size,
                'l2gradient': request.l2gradient,
                'image_index': request.image_index
            },
            image_info=image_info
        )

    def _prepare_visualization_data(
        self,
        original_image: np.ndarray,
        edge_image: np.ndarray,
        overlay_image: np.ndarray,
        gray_image: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            original_image: Original input image (RGB)
            edge_image: Binary edge image (RGB)
            overlay_image: Edges overlaid on original image
            gray_image: Grayscale version of original

        Returns:
            Dictionary with visualization data
        """
        # Convert images to base64 for frontend
        original_base64 = self._image_to_base64(original_image)
        edge_base64 = self._image_to_base64(edge_image)
        overlay_base64 = self._image_to_base64(overlay_image)
        gray_base64 = self._image_to_base64(gray_image)

        return {
            'original_image': original_base64,
            'edge_image': edge_base64,
            'overlay_image': overlay_base64,
            'gray_image': gray_base64
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
        """Get information about the Canny edge detection algorithm.

        Returns:
            Dictionary with algorithm information
        """
        return {
            'algorithm_name': 'Canny Edge Detection',
            'inventor': 'John F. Canny (1986)',
            'library': 'OpenCV (cv2.Canny)',
            'stages': [
                '1. Noise Reduction (Gaussian blur)',
                '2. Gradient Calculation (Sobel operators)',
                '3. Non-maximum Suppression',
                '4. Double Threshold',
                '5. Edge Tracking by Hysteresis'
            ],
            'complexity': {
                'time': 'O(width × height)',
                'space': 'O(width × height)'
            },
            'parameters': {
                'threshold1': 'Lower threshold for weak edges',
                'threshold2': 'Upper threshold for strong edges',
                'aperture_size': 'Sobel kernel size for gradient calculation',
                'l2gradient': 'Use L2 norm vs L1 norm for gradient magnitude'
            },
            'optimal_threshold_ratio': '2:1 or 3:1 (threshold2:threshold1)',
            'preprocessing': 'Gaussian blur with 5x5 kernel'
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
                    'general_purpose': 'threshold1=50, threshold2=150',
                    'fine_details': 'threshold1=30, threshold2=90',
                    'major_edges_only': 'threshold1=100, threshold2=200'
                },
                'tips': [
                    'Lower thresholds detect more edges but may include noise',
                    'Higher thresholds detect only strong edges',
                    'Typical ratio between thresholds is 2:1 or 3:1',
                    'Larger aperture size provides smoother gradients',
                    'L2 gradient is more accurate but computationally expensive'
                ]
            }
        }
