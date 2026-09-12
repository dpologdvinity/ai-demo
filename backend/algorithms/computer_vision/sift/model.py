"""SIFT (Scale-Invariant Feature Transform) model implementation."""

import time
import base64
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2

from .schema import (
    SIFTRequest,
    SIFTResponse,
    SIFTStatistics,
    KeypointData,
    MatchStatistics
)
from .data import download_sample_image, get_dataset_info


class SIFTModel:
    """SIFT feature detection and matching model.

    SIFT (Scale-Invariant Feature Transform) detects and describes local
    features in images that are invariant to scale and rotation. It's widely
    used for image matching, object recognition, and 3D reconstruction.

    Algorithm stages:
        1. Scale-space extrema detection
        2. Keypoint localization and filtering
        3. Orientation assignment
        4. Descriptor generation (128-dimensional)
    """

    def __init__(
        self,
        nfeatures: int = 500,
        nOctaveLayers: int = 3,
        contrastThreshold: float = 0.04,
        edgeThreshold: float = 10,
        sigma: float = 1.6
    ):
        """Initialize SIFT detector.

        Args:
            nfeatures: Maximum number of features to detect
            nOctaveLayers: Number of layers in each octave
            contrastThreshold: Contrast threshold for filtering weak features
            edgeThreshold: Edge threshold for filtering edge-like features
            sigma: Gaussian sigma for the first octave
        """
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma
        )

        # For feature matching
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def detect_and_compute(
        self,
        image_path: str
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray, np.ndarray]:
        """Detect keypoints and compute descriptors.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (keypoints, descriptors, image_rgb)

        Raises:
            RuntimeError: If detection fails
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"Failed to load image from {image_path}")

            # Convert to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for SIFT
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)

            return keypoints, descriptors, image_rgb

        except Exception as e:
            raise RuntimeError(f"SIFT detection failed: {str(e)}")

    def draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        detailed: bool = True
    ) -> np.ndarray:
        """Draw keypoints on image.

        Args:
            image: Input image (RGB)
            keypoints: List of detected keypoints
            detailed: If True, draw keypoint size and orientation

        Returns:
            Image with keypoints drawn
        """
        if detailed:
            # Draw keypoints with size and orientation
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        else:
            # Draw simple circles
            flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT

        image_with_keypoints = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),  # Green
            flags=flags
        )

        return image_with_keypoints

    def match_features(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        ratio_threshold: float = 0.75
    ) -> List[cv2.DMatch]:
        """Match features between two descriptor sets using Lowe's ratio test.

        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            ratio_threshold: Ratio threshold for Lowe's ratio test

        Returns:
            List of good matches
        """
        # Find k=2 nearest neighbors for each descriptor
        matches = self.bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def draw_matches(
        self,
        image1: np.ndarray,
        keypoints1: List[cv2.KeyPoint],
        image2: np.ndarray,
        keypoints2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_matches_to_draw: int = 50
    ) -> np.ndarray:
        """Draw matches between two images.

        Args:
            image1: First image (RGB)
            keypoints1: Keypoints from first image
            image2: Second image (RGB)
            keypoints2: Keypoints from second image
            matches: List of matches
            max_matches_to_draw: Maximum number of matches to visualize

        Returns:
            Image showing matched keypoints
        """
        # Limit matches for clarity
        matches_to_draw = matches[:max_matches_to_draw]

        # Draw matches
        match_image = cv2.drawMatches(
            image1, keypoints1,
            image2, keypoints2,
            matches_to_draw,
            None,
            matchColor=(0, 255, 0),  # Green
            singlePointColor=(255, 0, 0),  # Red
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return match_image

    def compute_statistics(
        self,
        keypoints: List[cv2.KeyPoint],
        image_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Compute statistics about detected keypoints.

        Args:
            keypoints: List of detected keypoints
            image_shape: Shape of the image (height, width, channels)

        Returns:
            Dictionary with statistics
        """
        if not keypoints:
            return {
                'keypoint_count': 0,
                'average_scale': 0.0,
                'average_response': 0.0,
                'scale_distribution': {},
                'orientation_distribution': {},
                'octave_distribution': {}
            }

        # Extract keypoint properties
        scales = [kp.size for kp in keypoints]
        responses = [kp.response for kp in keypoints]
        angles = [kp.angle for kp in keypoints]
        octaves = [kp.octave & 0xFF for kp in keypoints]  # Extract octave from packed value

        # Compute scale distribution (bins)
        scale_hist, scale_bins = np.histogram(scales, bins=10)
        scale_distribution = {
            f"{scale_bins[i]:.1f}-{scale_bins[i+1]:.1f}": int(count)
            for i, count in enumerate(scale_hist)
        }

        # Compute orientation distribution (8 bins, 45 degrees each)
        angle_bins = np.arange(0, 361, 45)
        angle_hist, _ = np.histogram(angles, bins=angle_bins)
        orientation_distribution = {
            f"{angle_bins[i]}-{angle_bins[i+1]}°": int(angle_hist[i])
            for i in range(len(angle_hist))
        }

        # Compute octave distribution
        octave_counts = {}
        for octave in set(octaves):
            octave_counts[f"Octave {octave}"] = octaves.count(octave)

        return {
            'keypoint_count': len(keypoints),
            'average_scale': float(np.mean(scales)),
            'average_response': float(np.mean(responses)),
            'scale_distribution': scale_distribution,
            'orientation_distribution': orientation_distribution,
            'octave_distribution': octave_counts
        }

    def visualize_descriptors(
        self,
        descriptors: np.ndarray,
        num_samples: int = 20
    ) -> np.ndarray:
        """Create a heatmap visualization of sample descriptors.

        Args:
            descriptors: Descriptor array (N x 128)
            num_samples: Number of descriptors to visualize

        Returns:
            Heatmap image of descriptors
        """
        if descriptors is None or len(descriptors) == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Sample descriptors
        num_samples = min(num_samples, len(descriptors))
        sampled_indices = np.linspace(0, len(descriptors) - 1, num_samples, dtype=int)
        sampled_descriptors = descriptors[sampled_indices]

        # Normalize to 0-255 for visualization
        desc_normalized = ((sampled_descriptors - sampled_descriptors.min()) /
                          (sampled_descriptors.max() - sampled_descriptors.min() + 1e-8) * 255)
        desc_normalized = desc_normalized.astype(np.uint8)

        # Resize for better visibility
        heatmap = cv2.resize(desc_normalized, (512, num_samples * 10),
                            interpolation=cv2.INTER_NEAREST)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        return heatmap_rgb

    def process_request(self, request: SIFTRequest) -> SIFTResponse:
        """Process a SIFT feature detection request.

        Args:
            request: SIFTRequest with detection parameters

        Returns:
            SIFTResponse with detection results

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If detection fails
        """
        start_time = time.time()

        # Re-initialize SIFT with request parameters
        self.sift = cv2.SIFT_create(
            nfeatures=request.nfeatures,
            nOctaveLayers=request.nOctaveLayers,
            contrastThreshold=request.contrastThreshold,
            edgeThreshold=request.edgeThreshold,
            sigma=request.sigma
        )

        # Download sample image(s)
        try:
            image_path1 = download_sample_image(request.image_index)
        except Exception as e:
            raise ValueError(f"Failed to load sample image: {str(e)}")

        # Detect features in first image
        keypoints1, descriptors1, image1_rgb = self.detect_and_compute(image_path1)

        # Handle matching mode
        match_statistics = None
        if request.match_mode and request.match_image_index is not None:
            try:
                image_path2 = download_sample_image(request.match_image_index)
                keypoints2, descriptors2, image2_rgb = self.detect_and_compute(image_path2)

                # Match features
                if descriptors1 is not None and descriptors2 is not None:
                    matches = self.match_features(descriptors1, descriptors2)

                    # Compute match statistics
                    match_distances = [m.distance for m in matches]
                    match_statistics = MatchStatistics(
                        match_count=len(matches),
                        total_matches=len(matches),
                        keypoints_image1=len(keypoints1),
                        keypoints_image2=len(keypoints2),
                        match_ratio=len(matches) / max(len(keypoints1), 1),
                        average_match_distance=float(np.mean(match_distances)) if match_distances else 0.0
                    )
                else:
                    match_statistics = None

            except Exception as e:
                print(f"Warning: Matching failed: {str(e)}")
                match_statistics = None

        # Compute statistics
        stats_data = self.compute_statistics(keypoints1, image1_rgb.shape)

        statistics = SIFTStatistics(
            keypoint_count=stats_data['keypoint_count'],
            image_dimensions={
                'width': int(image1_rgb.shape[1]),
                'height': int(image1_rgb.shape[0])
            },
            average_scale=round(stats_data['average_scale'], 2),
            average_response=round(stats_data['average_response'], 4),
            scale_distribution=stats_data['scale_distribution'],
            orientation_distribution=stats_data['orientation_distribution'],
            octave_distribution=stats_data['octave_distribution']
        )

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            image1_rgb,
            keypoints1,
            descriptors1,
            image2_rgb if request.match_mode and request.match_image_index is not None else None,
            keypoints2 if request.match_mode and request.match_image_index is not None else None,
            matches if request.match_mode and match_statistics else None
        )

        # Get algorithm info
        algorithm_info = self._get_algorithm_info()

        # Convert keypoints to serializable format (sample top N)
        keypoint_list = self._keypoints_to_list(keypoints1, max_keypoints=100)

        # Get image info
        image_info = {
            'width': int(image1_rgb.shape[1]),
            'height': int(image1_rgb.shape[0]),
            'channels': int(image1_rgb.shape[2]),
            'image_index': request.image_index
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return SIFTResponse(
            success=True,
            statistics=statistics,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            algorithm_info=algorithm_info,
            parameters_used={
                'nfeatures': request.nfeatures,
                'nOctaveLayers': request.nOctaveLayers,
                'contrastThreshold': request.contrastThreshold,
                'edgeThreshold': request.edgeThreshold,
                'sigma': request.sigma,
                'image_index': request.image_index,
                'match_mode': request.match_mode,
                'match_image_index': request.match_image_index
            },
            image_info=image_info,
            keypoints=keypoint_list,
            match_statistics=match_statistics
        )

    def _keypoints_to_list(
        self,
        keypoints: List[cv2.KeyPoint],
        max_keypoints: int = 100
    ) -> List[KeypointData]:
        """Convert OpenCV keypoints to serializable list.

        Args:
            keypoints: List of OpenCV keypoints
            max_keypoints: Maximum number of keypoints to include

        Returns:
            List of KeypointData objects
        """
        # Sort by response (strongest features first)
        sorted_keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)

        keypoint_list = []
        for kp in sorted_keypoints[:max_keypoints]:
            keypoint_list.append(KeypointData(
                x=float(kp.pt[0]),
                y=float(kp.pt[1]),
                size=float(kp.size),
                angle=float(kp.angle),
                response=float(kp.response),
                octave=int(kp.octave & 0xFF)
            ))

        return keypoint_list

    def _prepare_visualization_data(
        self,
        image1: np.ndarray,
        keypoints1: List[cv2.KeyPoint],
        descriptors1: Optional[np.ndarray],
        image2: Optional[np.ndarray],
        keypoints2: Optional[List[cv2.KeyPoint]],
        matches: Optional[List[cv2.DMatch]]
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            image1: Original first image (RGB)
            keypoints1: Keypoints from first image
            descriptors1: Descriptors from first image
            image2: Second image (RGB) if in match mode
            keypoints2: Keypoints from second image if in match mode
            matches: Matches between images if in match mode

        Returns:
            Dictionary with visualization data
        """
        # Draw keypoints on first image
        image_with_keypoints = self.draw_keypoints(image1, keypoints1, detailed=True)

        # Create descriptor heatmap
        descriptor_heatmap = self.visualize_descriptors(descriptors1)

        viz_data = {
            'original_image': self._image_to_base64(image1),
            'keypoints_image': self._image_to_base64(image_with_keypoints),
            'descriptor_heatmap': self._image_to_base64(descriptor_heatmap)
        }

        # Add matching visualization if applicable
        if image2 is not None and keypoints2 is not None and matches is not None:
            match_image = self.draw_matches(
                image1, keypoints1,
                image2, keypoints2,
                matches,
                max_matches_to_draw=50
            )
            viz_data['match_image'] = self._image_to_base64(match_image)
            viz_data['image2_with_keypoints'] = self._image_to_base64(
                self.draw_keypoints(image2, keypoints2, detailed=True)
            )

        return viz_data

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

    def _get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the SIFT algorithm.

        Returns:
            Dictionary with algorithm information
        """
        return {
            'algorithm_name': 'SIFT (Scale-Invariant Feature Transform)',
            'inventor': 'David Lowe (1999)',
            'library': 'OpenCV (cv2.SIFT_create)',
            'stages': [
                '1. Scale-space extrema detection (DoG pyramid)',
                '2. Keypoint localization and filtering',
                '3. Orientation assignment (gradient histograms)',
                '4. Descriptor generation (128-dimensional vector)'
            ],
            'descriptor_size': '128 floating-point values',
            'complexity': {
                'time': 'O(n*log(n)) where n = number of pixels',
                'space': 'O(k*128) where k = number of keypoints'
            },
            'parameters': {
                'nfeatures': 'Maximum number of best features to retain',
                'nOctaveLayers': 'Number of layers in each octave (scale levels)',
                'contrastThreshold': 'Threshold for filtering low-contrast keypoints',
                'edgeThreshold': 'Threshold for filtering edge-like features',
                'sigma': 'Gaussian blur sigma for the first octave'
            },
            'invariance': [
                'Scale invariant (detects features at multiple scales)',
                'Rotation invariant (assigns dominant orientation)',
                'Partially illumination invariant',
                'Partially viewpoint invariant'
            ],
            'applications': [
                'Image matching and registration',
                'Object recognition',
                'Panorama stitching',
                '3D reconstruction',
                'Augmented reality',
                'Visual odometry'
            ]
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
                    'general_purpose': 'nfeatures=500, contrastThreshold=0.04',
                    'detailed_matching': 'nfeatures=1000, contrastThreshold=0.03',
                    'fast_detection': 'nfeatures=200, contrastThreshold=0.06'
                },
                'tips': [
                    'More features = better matching but slower processing',
                    'Lower contrast threshold = more features but may include noise',
                    'Higher edge threshold = fewer edge-like features',
                    'SIFT works best on textured images with clear features',
                    'Use matching mode to compare features between images',
                    'Descriptors are 128-dimensional for robust matching'
                ],
                'matching': {
                    'method': "Lowe's ratio test with BFMatcher",
                    'ratio_threshold': 0.75,
                    'distance_metric': 'L2 norm (Euclidean distance)'
                }
            }
        }
