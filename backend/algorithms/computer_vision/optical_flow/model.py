"""Optical Flow model implementation."""

import time
import base64
from io import BytesIO
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import cv2

from .schema import (
    OpticalFlowRequest,
    OpticalFlowResponse,
    FlowStatistics,
    FlowVector
)
from .data import download_sample_image_pair, get_dataset_info


class OpticalFlowModel:
    """Optical Flow model using Farneback or Lucas-Kanade algorithm.

    Optical flow estimates motion between consecutive video frames by analyzing
    pixel intensity changes. It produces a vector field representing the apparent
    motion of objects, surfaces, and edges in a visual scene.

    Algorithms:
        - Farneback: Dense optical flow (computes flow for every pixel)
        - Lucas-Kanade: Sparse optical flow (computes flow for feature points)
    """

    def __init__(self):
        """Initialize Optical Flow model."""
        self.flow_threshold = 0.5  # Threshold for significant motion
        self.arrow_step = 15  # Step size for arrow visualization (sample every N pixels)
        self.arrow_scale = 3  # Scale factor for arrow length

    def create_synthetic_motion(
        self,
        image: np.ndarray,
        motion_type: str
    ) -> np.ndarray:
        """Create synthetic motion by transforming an image.

        Args:
            image: Input image (BGR)
            motion_type: Type of motion to simulate

        Returns:
            Transformed image with simulated motion
        """
        height, width = image.shape[:2]

        if motion_type == "horizontal_translation":
            # Translate image to the right
            M = np.float32([[1, 0, 10], [0, 1, 0]])
            frame2 = cv2.warpAffine(image, M, (width, height))

        elif motion_type == "mixed_translation":
            # Translate diagonally
            M = np.float32([[1, 0, 8], [0, 1, 5]])
            frame2 = cv2.warpAffine(image, M, (width, height))

        elif motion_type == "zoom":
            # Simulate zoom by scaling
            scale = 1.05
            M = cv2.getRotationMatrix2D((width/2, height/2), 0, scale)
            frame2 = cv2.warpAffine(image, M, (width, height))

        else:  # static
            frame2 = image.copy()

        return frame2

    def compute_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        method: str = "farneback",
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Compute optical flow between two frames.

        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)
            method: Flow algorithm ('farneback' or 'lucas-kanade')
            pyr_scale: Pyramid scale factor
            levels: Number of pyramid levels
            winsize: Window size
            iterations: Number of iterations

        Returns:
            Dictionary containing flow data and visualizations

        Raises:
            RuntimeError: If flow computation fails
        """
        start_time = time.time()

        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            if method == "farneback":
                # Compute dense optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(
                    gray1,
                    gray2,
                    None,
                    pyr_scale=pyr_scale,
                    levels=levels,
                    winsize=winsize,
                    iterations=iterations,
                    poly_n=5,
                    poly_sigma=1.1,
                    flags=0
                )

            elif method == "lucas-kanade":
                # Detect good features to track
                feature_params = dict(
                    maxCorners=500,
                    qualityLevel=0.01,
                    minDistance=10,
                    blockSize=winsize
                )
                p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

                if p0 is None or len(p0) == 0:
                    # If no features detected, return zero flow
                    flow = np.zeros((gray1.shape[0], gray1.shape[1], 2), dtype=np.float32)
                else:
                    # Lucas-Kanade parameters
                    lk_params = dict(
                        winSize=(winsize, winsize),
                        maxLevel=levels,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 0.03)
                    )

                    # Calculate sparse optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

                    # Convert sparse flow to dense representation
                    flow = np.zeros((gray1.shape[0], gray1.shape[1], 2), dtype=np.float32)

                    if p1 is not None:
                        # Select good points
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]

                        # Fill flow field at feature locations
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            x, y = int(c), int(d)
                            if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                                flow[y, x, 0] = a - c  # dx
                                flow[y, x, 1] = b - d  # dy

            else:
                raise ValueError(f"Unsupported method: {method}")

            # Calculate flow magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Create visualizations
            flow_rgb = self._create_flow_rgb_visualization(flow, magnitude, angle)
            magnitude_heatmap = self._create_magnitude_heatmap(magnitude)
            direction_hsv = self._create_direction_visualization(magnitude, angle)
            arrows_overlay = self._create_arrows_overlay(
                cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
                flow
            )

            # Calculate statistics
            flow_stats = self._calculate_flow_statistics(flow, magnitude, angle)

            # Sample flow vectors for frontend arrow visualization
            flow_vectors = self._sample_flow_vectors(flow, magnitude, angle)

            computation_time = (time.time() - start_time) * 1000

            return {
                'frame1_rgb': cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
                'frame2_rgb': cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB),
                'flow': flow,
                'magnitude': magnitude,
                'angle': angle,
                'flow_rgb': flow_rgb,
                'magnitude_heatmap': magnitude_heatmap,
                'direction_hsv': direction_hsv,
                'arrows_overlay': arrows_overlay,
                'statistics': flow_stats,
                'flow_vectors': flow_vectors,
                'computation_time_ms': computation_time
            }

        except Exception as e:
            raise RuntimeError(f"Optical flow computation failed: {str(e)}")

    def _create_flow_rgb_visualization(
        self,
        flow: np.ndarray,
        magnitude: np.ndarray,
        angle: np.ndarray
    ) -> np.ndarray:
        """Create RGB visualization of optical flow.

        Args:
            flow: Flow field (H x W x 2)
            magnitude: Flow magnitude
            angle: Flow angle

        Returns:
            RGB visualization of flow
        """
        # Create HSV image where:
        # - Hue represents direction
        # - Value represents magnitude
        # - Saturation is fixed
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
        hsv[..., 1] = 255  # Saturation: full
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude

        # Convert HSV to RGB
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return flow_rgb

    def _create_magnitude_heatmap(self, magnitude: np.ndarray) -> np.ndarray:
        """Create heatmap visualization of flow magnitude.

        Args:
            magnitude: Flow magnitude

        Returns:
            Heatmap visualization
        """
        # Normalize magnitude to 0-255
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply colormap (hot colors for high magnitude)
        heatmap = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_JET)

        # Convert BGR to RGB
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap_rgb

    def _create_direction_visualization(
        self,
        magnitude: np.ndarray,
        angle: np.ndarray
    ) -> np.ndarray:
        """Create HSV visualization of flow direction.

        Args:
            magnitude: Flow magnitude
            angle: Flow angle

        Returns:
            HSV direction visualization
        """
        hsv = np.zeros((magnitude.shape[0], magnitude.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
        hsv[..., 1] = 255  # Saturation: full
        hsv[..., 2] = 255  # Value: full brightness

        # Mask out low magnitude regions
        mask = magnitude > self.flow_threshold
        hsv[~mask] = 0

        # Convert HSV to RGB
        direction_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return direction_rgb

    def _create_arrows_overlay(
        self,
        frame: np.ndarray,
        flow: np.ndarray
    ) -> np.ndarray:
        """Create overlay visualization with motion arrows.

        Args:
            frame: Input frame (RGB)
            flow: Flow field

        Returns:
            Frame with flow arrows overlaid
        """
        overlay = frame.copy()
        step = self.arrow_step

        # Draw arrows on a subsampled grid
        for y in range(0, flow.shape[0], step):
            for x in range(0, flow.shape[1], step):
                fx, fy = flow[y, x]
                magnitude = np.sqrt(fx**2 + fy**2)

                # Only draw arrows for significant motion
                if magnitude > self.flow_threshold:
                    # Calculate arrow end point
                    x2 = int(x + fx * self.arrow_scale)
                    y2 = int(y + fy * self.arrow_scale)

                    # Draw arrow (cyan color for visibility)
                    cv2.arrowedLine(
                        overlay,
                        (x, y),
                        (x2, y2),
                        (0, 255, 255),
                        1,
                        tipLength=0.3
                    )

        return overlay

    def _calculate_flow_statistics(
        self,
        flow: np.ndarray,
        magnitude: np.ndarray,
        angle: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate statistics about the optical flow.

        Args:
            flow: Flow field
            magnitude: Flow magnitude
            angle: Flow angle

        Returns:
            Dictionary with flow statistics
        """
        # Mask for significant motion
        motion_mask = magnitude > self.flow_threshold

        # Calculate statistics
        avg_magnitude = float(np.mean(magnitude))
        max_magnitude = float(np.max(magnitude))
        min_magnitude = float(np.min(magnitude))
        median_magnitude = float(np.median(magnitude))

        # Flow coverage (percentage of pixels with significant motion)
        flow_coverage = float(np.sum(motion_mask) / motion_mask.size * 100)

        # Primary direction (circular mean of angles weighted by magnitude)
        if np.sum(motion_mask) > 0:
            angles_masked = angle[motion_mask]
            magnitudes_masked = magnitude[motion_mask]

            # Convert to Cartesian for averaging
            x_comp = np.sum(magnitudes_masked * np.cos(angles_masked))
            y_comp = np.sum(magnitudes_masked * np.sin(angles_masked))
            primary_direction = float(np.arctan2(y_comp, x_comp) * 180 / np.pi)

            # Normalize to 0-360
            if primary_direction < 0:
                primary_direction += 360
        else:
            primary_direction = 0.0

        return {
            'average_magnitude': avg_magnitude,
            'max_magnitude': max_magnitude,
            'min_magnitude': min_magnitude,
            'median_magnitude': median_magnitude,
            'flow_coverage': flow_coverage,
            'primary_direction': primary_direction,
            'total_pixels': int(flow.shape[0] * flow.shape[1])
        }

    def _sample_flow_vectors(
        self,
        flow: np.ndarray,
        magnitude: np.ndarray,
        angle: np.ndarray,
        max_vectors: int = 100
    ) -> List[Dict[str, Any]]:
        """Sample flow vectors for frontend visualization.

        Args:
            flow: Flow field
            magnitude: Flow magnitude
            angle: Flow angle
            max_vectors: Maximum number of vectors to sample

        Returns:
            List of sampled flow vectors
        """
        vectors = []
        step = self.arrow_step

        for y in range(0, flow.shape[0], step):
            for x in range(0, flow.shape[1], step):
                fx, fy = flow[y, x]
                mag = float(magnitude[y, x])
                ang = float(angle[y, x] * 180 / np.pi)

                # Only include vectors with significant motion
                if mag > self.flow_threshold:
                    vectors.append({
                        'x': int(x),
                        'y': int(y),
                        'dx': float(fx),
                        'dy': float(fy),
                        'magnitude': mag,
                        'angle': ang
                    })

                    if len(vectors) >= max_vectors:
                        break
            if len(vectors) >= max_vectors:
                break

        return vectors

    def process_request(self, request: OpticalFlowRequest) -> OpticalFlowResponse:
        """Process an optical flow request.

        Args:
            request: OpticalFlowRequest with flow parameters

        Returns:
            OpticalFlowResponse with flow computation results

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If flow computation fails
        """
        start_time = time.time()

        # Validate parameters
        request.validate_method()
        request.validate_winsize()

        # Download sample image (will generate synthetic motion)
        try:
            image_path, motion_type = download_sample_image_pair(request.image_pair_index)
        except Exception as e:
            raise ValueError(f"Failed to load sample image: {str(e)}")

        # Load first frame
        frame1 = cv2.imread(image_path)
        if frame1 is None:
            raise RuntimeError(f"Failed to load image from {image_path}")

        # Create second frame with synthetic motion
        frame2 = self.create_synthetic_motion(frame1, motion_type)

        # Compute optical flow
        flow_result = self.compute_optical_flow(
            frame1,
            frame2,
            method=request.method,
            pyr_scale=request.pyr_scale,
            levels=request.levels,
            winsize=request.winsize,
            iterations=request.iterations
        )

        # Prepare statistics
        image_shape = flow_result['frame1_rgb'].shape
        statistics = FlowStatistics(
            average_magnitude=round(flow_result['statistics']['average_magnitude'], 3),
            max_magnitude=round(flow_result['statistics']['max_magnitude'], 3),
            min_magnitude=round(flow_result['statistics']['min_magnitude'], 3),
            median_magnitude=round(flow_result['statistics']['median_magnitude'], 3),
            flow_coverage=round(flow_result['statistics']['flow_coverage'], 2),
            primary_direction=round(flow_result['statistics']['primary_direction'], 1),
            image_dimensions={
                'width': int(image_shape[1]),
                'height': int(image_shape[0])
            },
            total_pixels=flow_result['statistics']['total_pixels']
        )

        # Convert flow vectors to schema objects
        flow_vectors = [
            FlowVector(**vec) for vec in flow_result['flow_vectors']
        ]

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            flow_result['frame1_rgb'],
            flow_result['frame2_rgb'],
            flow_result['flow_rgb'],
            flow_result['magnitude_heatmap'],
            flow_result['direction_hsv'],
            flow_result['arrows_overlay']
        )

        # Get algorithm info
        algorithm_info = self._get_algorithm_info(request.method)

        # Get image info
        image_info = {
            'width': int(image_shape[1]),
            'height': int(image_shape[0]),
            'channels': int(image_shape[2]),
            'image_pair_index': request.image_pair_index,
            'motion_type': motion_type
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return OpticalFlowResponse(
            success=True,
            statistics=statistics,
            visualization_data=visualization_data,
            flow_vectors=flow_vectors,
            execution_time_ms=execution_time_ms,
            algorithm_info=algorithm_info,
            parameters_used={
                'method': request.method,
                'pyr_scale': request.pyr_scale,
                'levels': request.levels,
                'winsize': request.winsize,
                'iterations': request.iterations,
                'image_pair_index': request.image_pair_index
            },
            image_info=image_info
        )

    def _prepare_visualization_data(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        flow_rgb: np.ndarray,
        magnitude_heatmap: np.ndarray,
        direction_hsv: np.ndarray,
        arrows_overlay: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            frame1: First frame (RGB)
            frame2: Second frame (RGB)
            flow_rgb: Flow RGB visualization
            magnitude_heatmap: Magnitude heatmap
            direction_hsv: Direction visualization
            arrows_overlay: Arrows overlay on frame1

        Returns:
            Dictionary with visualization data
        """
        return {
            'frame1': self._image_to_base64(frame1),
            'frame2': self._image_to_base64(frame2),
            'flow_rgb': self._image_to_base64(flow_rgb),
            'magnitude_heatmap': self._image_to_base64(magnitude_heatmap),
            'direction_visualization': self._image_to_base64(direction_hsv),
            'arrows_overlay': self._image_to_base64(arrows_overlay)
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
        pil_image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def _get_algorithm_info(self, method: str) -> Dict[str, Any]:
        """Get information about the optical flow algorithm.

        Args:
            method: Flow algorithm used

        Returns:
            Dictionary with algorithm information
        """
        if method == "farneback":
            return {
                'algorithm_name': 'Farneback Optical Flow',
                'type': 'Dense optical flow',
                'description': 'Computes flow vectors for every pixel using polynomial expansion',
                'library': 'OpenCV (cv2.calcOpticalFlowFarneback)',
                'characteristics': [
                    'Dense flow field (every pixel)',
                    'Good for general motion estimation',
                    'Relatively robust to noise',
                    'Computationally intensive'
                ],
                'complexity': {
                    'time': 'O(width × height × pyramid_levels)',
                    'space': 'O(width × height)'
                },
                'parameters': {
                    'pyr_scale': 'Pyramid scale - smaller values detect larger motions',
                    'levels': 'Number of pyramid levels - more levels = larger motion range',
                    'winsize': 'Window size - larger windows = more robust but less precise',
                    'iterations': 'Number of iterations at each level - more = better accuracy'
                }
            }
        else:  # lucas-kanade
            return {
                'algorithm_name': 'Lucas-Kanade Optical Flow',
                'type': 'Sparse optical flow',
                'description': 'Computes flow at feature points using local brightness constancy',
                'library': 'OpenCV (cv2.calcOpticalFlowPyrLK)',
                'characteristics': [
                    'Sparse flow field (feature points only)',
                    'Fast computation',
                    'Good for tracking specific features',
                    'Assumes brightness constancy'
                ],
                'complexity': {
                    'time': 'O(num_features × window_size × pyramid_levels)',
                    'space': 'O(num_features)'
                },
                'parameters': {
                    'levels': 'Number of pyramid levels for coarse-to-fine search',
                    'winsize': 'Window size for local optimization',
                    'iterations': 'Termination criteria for optimization'
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
                'methods': {
                    'farneback': {
                        'name': 'Farneback Dense Flow',
                        'best_for': 'General motion estimation, video stabilization',
                        'speed': 'Moderate',
                        'accuracy': 'High'
                    },
                    'lucas-kanade': {
                        'name': 'Lucas-Kanade Sparse Flow',
                        'best_for': 'Feature tracking, object following',
                        'speed': 'Fast',
                        'accuracy': 'Good for sparse features'
                    }
                },
                'visualization_guide': {
                    'flow_rgb': 'Color encodes direction (hue) and magnitude (brightness)',
                    'magnitude_heatmap': 'Hot colors indicate faster motion',
                    'direction_viz': 'Color wheel shows motion direction',
                    'arrows': 'Arrow length and direction show motion vectors'
                },
                'tips': [
                    'Use Farneback for dense, smooth motion fields',
                    'Use Lucas-Kanade for tracking specific features',
                    'Larger window size is more robust but less precise',
                    'More pyramid levels help detect larger motions',
                    'Higher iterations improve accuracy but take longer'
                ]
            }
        }
