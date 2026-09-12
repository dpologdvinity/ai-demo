"""Semantic Segmentation model implementation using DeepLabV3."""

import time
import base64
from io import BytesIO
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from .schema import (
    SegmentationRequest,
    SegmentationResponse,
    ClassInfo,
    SegmentationStatistics
)
from .data import (
    download_sample_image,
    get_dataset_info,
    get_class_info
)


class SegmentationModel:
    """Semantic Segmentation model using DeepLabV3.

    This class provides a wrapper around the torchvision DeepLabV3 model
    for semantic segmentation tasks.

    Attributes:
        model: The DeepLabV3 model instance
        model_backbone: Backbone architecture (resnet50 or mobilenet)
        device: Device to run the model on (cuda or cpu)
        class_names: List of class names
        class_colors: RGB colors for each class
    """

    def __init__(self, model_backbone: str = 'resnet50', num_classes: int = 21):
        """Initialize Segmentation model.

        Args:
            model_backbone: Backbone architecture - 'resnet50' or 'mobilenet'
            num_classes: Number of classes for segmentation

        Raises:
            ValueError: If model_backbone is invalid
        """
        if model_backbone not in ['resnet50', 'mobilenet']:
            raise ValueError("model_backbone must be 'resnet50' or 'mobilenet'")

        self.model_backbone = model_backbone
        self.num_classes = num_classes
        self.model = None
        self.device = None
        self.class_names, self.class_colors = get_class_info(num_classes)

    def load_model(self):
        """Load the DeepLabV3 model.

        Loads the model on first use to avoid unnecessary initialization.
        """
        if self.model is None:
            try:
                import torchvision.models.segmentation as segmentation

                # Set device
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {self.device}")

                # Load pretrained model
                if self.model_backbone == 'resnet50':
                    print("Loading DeepLabV3 with ResNet50 backbone...")
                    self.model = segmentation.deeplabv3_resnet50(
                        weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
                    )
                else:  # mobilenet
                    print("Loading DeepLabV3 with MobileNetV3 backbone...")
                    self.model = segmentation.deeplabv3_mobilenet_v3_large(
                        weights=segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                    )

                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"Model loaded successfully on {self.device}")

            except Exception as e:
                raise RuntimeError(f"Failed to load segmentation model: {str(e)}")

    def segment(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        image_size: int = 512
    ) -> Dict[str, Any]:
        """Run semantic segmentation on an image.

        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence for predictions
            image_size: Target size for input image

        Returns:
            Dictionary containing segmentation results and metadata

        Raises:
            RuntimeError: If segmentation fails
        """
        # Load model if not already loaded
        self.load_model()

        start_time = time.time()

        try:
            # Load and preprocess image
            original_image = Image.open(image_path).convert('RGB')
            original_size = original_image.size

            # Prepare transform
            transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # Transform image
            input_tensor = transform(original_image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]

            # Get predictions
            output_predictions = output.argmax(0).cpu().numpy()

            # Get confidence scores (softmax probabilities)
            output_probs = torch.nn.functional.softmax(output, dim=0)
            max_probs = output_probs.max(0)[0].cpu().numpy()

            # Apply confidence threshold
            mask = max_probs >= confidence_threshold
            output_predictions = output_predictions * mask

            # Resize predictions back to original size
            pred_image = Image.fromarray(output_predictions.astype(np.uint8))
            pred_image = pred_image.resize(original_size, Image.NEAREST)
            segmentation_mask = np.array(pred_image)

            # Create colored segmentation
            colored_mask = self._create_colored_mask(segmentation_mask)

            # Create overlay
            overlay_image = self._create_overlay(
                np.array(original_image),
                colored_mask,
                alpha=0.6
            )

            segmentation_time = (time.time() - start_time) * 1000

            return {
                'segmentation_mask': segmentation_mask,
                'original_image': np.array(original_image),
                'colored_mask': colored_mask,
                'overlay_image': overlay_image,
                'confidence_map': max_probs,
                'image_shape': original_image.size,
                'segmentation_time_ms': segmentation_time
            }

        except Exception as e:
            raise RuntimeError(f"Segmentation failed: {str(e)}")

    def process_request(self, request: SegmentationRequest) -> SegmentationResponse:
        """Process a segmentation request.

        Args:
            request: SegmentationRequest with segmentation parameters

        Returns:
            SegmentationResponse with segmentation results

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If segmentation fails
        """
        start_time = time.time()

        # Download sample image
        try:
            image_path = download_sample_image(request.image_index)
        except Exception as e:
            raise ValueError(f"Failed to load sample image: {str(e)}")

        # Update model if parameters changed
        if (self.model is None or
            self.model_backbone != request.model_backbone or
            self.num_classes != request.num_classes):
            self.model_backbone = request.model_backbone
            self.num_classes = request.num_classes
            self.class_names, self.class_colors = get_class_info(request.num_classes)
            self.model = None  # Force reload

        # Run segmentation
        seg_result = self.segment(
            image_path,
            confidence_threshold=request.confidence_threshold,
            image_size=request.image_size
        )

        # Calculate statistics
        statistics = self._calculate_statistics(
            seg_result['segmentation_mask'],
            seg_result['confidence_map']
        )

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            seg_result['original_image'],
            seg_result['colored_mask'],
            seg_result['overlay_image'],
            seg_result['segmentation_mask'],
            statistics
        )

        # Get model info
        model_info = self._get_model_info()

        # Get image info
        image_info = {
            'width': seg_result['image_shape'][0],
            'height': seg_result['image_shape'][1],
            'image_index': request.image_index
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return SegmentationResponse(
            success=True,
            statistics=statistics,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'num_classes': request.num_classes,
                'confidence_threshold': request.confidence_threshold,
                'model_backbone': request.model_backbone,
                'image_size': request.image_size,
                'image_index': request.image_index
            },
            image_info=image_info
        )

    def _create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create a colored segmentation mask.

        Args:
            mask: Segmentation mask with class IDs

        Returns:
            RGB image with colored segmentation
        """
        height, width = mask.shape
        colored = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id in range(min(len(self.class_colors), self.num_classes)):
            colored[mask == class_id] = self.class_colors[class_id]

        return colored

    def _create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """Create an overlay of the segmentation mask on the original image.

        Args:
            image: Original RGB image
            mask: Colored segmentation mask
            alpha: Transparency of the overlay (0=transparent, 1=opaque)

        Returns:
            Overlay image
        """
        return (alpha * mask + (1 - alpha) * image).astype(np.uint8)

    def _calculate_statistics(
        self,
        mask: np.ndarray,
        confidence_map: np.ndarray
    ) -> SegmentationStatistics:
        """Calculate statistics from segmentation mask.

        Args:
            mask: Segmentation mask with class IDs
            confidence_map: Confidence scores for predictions

        Returns:
            SegmentationStatistics object
        """
        total_pixels = mask.size
        unique_classes = np.unique(mask)

        class_info_list = []
        total_confidence = 0
        pixels_counted = 0

        for class_id in unique_classes:
            class_id = int(class_id)
            if class_id >= len(self.class_names):
                continue

            class_mask = mask == class_id
            pixel_count = int(np.sum(class_mask))
            percentage = (pixel_count / total_pixels) * 100

            # Calculate mean confidence for this class
            if confidence_map is not None:
                class_confidence = float(np.mean(confidence_map[class_mask]))
                total_confidence += class_confidence * pixel_count
                pixels_counted += pixel_count
            else:
                class_confidence = None

            # Skip background class with very few pixels
            if class_id == 0 and percentage < 1.0:
                continue

            class_info_list.append(ClassInfo(
                class_id=class_id,
                class_name=self.class_names[class_id],
                color=self.class_colors[class_id],
                pixel_count=pixel_count,
                percentage=round(percentage, 2),
                iou_score=None  # Would need ground truth
            ))

        # Sort by pixel count
        class_info_list.sort(key=lambda x: x.pixel_count, reverse=True)

        mean_confidence = total_confidence / pixels_counted if pixels_counted > 0 else 0.0

        return SegmentationStatistics(
            total_classes=len(class_info_list),
            total_pixels=total_pixels,
            mean_confidence=float(mean_confidence),
            class_info=class_info_list
        )

    def _prepare_visualization_data(
        self,
        original_image: np.ndarray,
        colored_mask: np.ndarray,
        overlay_image: np.ndarray,
        segmentation_mask: np.ndarray,
        statistics: SegmentationStatistics
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            original_image: Original input image
            colored_mask: Colored segmentation mask
            overlay_image: Overlay of mask on original
            segmentation_mask: Raw segmentation mask
            statistics: Segmentation statistics

        Returns:
            Dictionary with visualization data
        """
        # Convert images to base64
        original_base64 = self._image_to_base64(original_image)
        mask_base64 = self._image_to_base64(colored_mask)
        overlay_base64 = self._image_to_base64(overlay_image)

        # Prepare class legend
        class_legend = [
            {
                'class_id': info.class_id,
                'class_name': info.class_name,
                'color': f'rgb({info.color[0]}, {info.color[1]}, {info.color[2]})',
                'pixel_count': info.pixel_count,
                'percentage': info.percentage
            }
            for info in statistics.class_info
        ]

        # Prepare per-class statistics chart
        class_distribution = [
            {
                'class': info.class_name,
                'percentage': info.percentage,
                'pixels': info.pixel_count
            }
            for info in statistics.class_info[:10]  # Top 10 classes
        ]

        return {
            'original_image': original_base64,
            'colored_mask': mask_base64,
            'overlay_image': overlay_base64,
            'class_legend': class_legend,
            'class_distribution': class_distribution,
            'show_overlay': True
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

    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': f'DeepLabV3-{self.model_backbone}',
            'backbone': self.model_backbone,
            'num_classes': self.num_classes,
            'framework': 'PyTorch torchvision',
            'device': str(self.device) if self.device else 'cpu',
            'architecture': 'DeepLabV3 (Atrous Spatial Pyramid Pooling)',
            'pretrained': 'COCO train2017 + VOC2012'
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
                'resnet50': 'ResNet50 - Higher accuracy, slower inference',
                'mobilenet': 'MobileNetV3 - Faster inference, lower accuracy'
            }
        }
