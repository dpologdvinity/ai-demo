"""Instance Segmentation model implementation using Mask R-CNN."""

import time
import base64
from io import BytesIO
from typing import Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

from .schema import (
    InstanceSegmentationRequest,
    InstanceSegmentationResponse,
    Instance,
    InstanceStatistics
)
from .data import (
    download_sample_image,
    get_dataset_info,
    get_algorithm_info,
    get_instance_color,
    get_class_name,
    COCO_CLASS_NAMES
)


class InstanceSegmentationModel:
    """Instance Segmentation model using Mask R-CNN.

    This class provides a wrapper around the torchvision Mask R-CNN model
    for instance segmentation tasks.

    Attributes:
        model: The Mask R-CNN model instance
        model_backbone: Backbone architecture (resnet50 or resnet101)
        device: Device to run the model on (cuda or cpu)
    """

    def __init__(self, model_backbone: str = 'resnet50'):
        """Initialize Instance Segmentation model.

        Args:
            model_backbone: Backbone architecture - 'resnet50' or 'resnet101'

        Raises:
            ValueError: If model_backbone is invalid
        """
        if model_backbone not in ['resnet50', 'resnet101']:
            raise ValueError("model_backbone must be 'resnet50' or 'resnet101'")

        self.model_backbone = model_backbone
        self.model = None
        self.device = None

    def load_model(self):
        """Load the Mask R-CNN model.

        Loads the model on first use to avoid unnecessary initialization.
        """
        if self.model is None:
            try:
                # Set device
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {self.device}")

                # Load pretrained model
                if self.model_backbone == 'resnet50':
                    print("Loading Mask R-CNN with ResNet50-FPN backbone...")
                    self.model = maskrcnn_resnet50_fpn(
                        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                    )
                else:  # resnet101
                    # Note: torchvision doesn't have a direct resnet101 Mask R-CNN
                    # We'll use resnet50 as fallback
                    print("Loading Mask R-CNN with ResNet50-FPN backbone (ResNet101 not available)...")
                    self.model = maskrcnn_resnet50_fpn(
                        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                    )

                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"Model loaded successfully on {self.device}")

            except Exception as e:
                raise RuntimeError(f"Failed to load Mask R-CNN model: {str(e)}")

    def segment(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_instances: int = 100,
        nms_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Run instance segmentation on an image.

        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence for detections
            mask_threshold: Binary threshold for masks
            max_instances: Maximum number of instances
            nms_threshold: NMS IoU threshold

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
            image_array = np.array(original_image)

            # Prepare transform
            transform = T.Compose([
                T.ToTensor()
            ])

            # Transform image
            input_tensor = transform(original_image).to(self.device)

            # Run inference
            with torch.no_grad():
                predictions = self.model([input_tensor])[0]

            # Filter by confidence threshold
            keep_indices = predictions['scores'] >= confidence_threshold

            boxes = predictions['boxes'][keep_indices].cpu().numpy()
            labels = predictions['labels'][keep_indices].cpu().numpy()
            scores = predictions['scores'][keep_indices].cpu().numpy()
            masks = predictions['masks'][keep_indices].cpu().numpy()

            # Limit to max_instances
            if len(boxes) > max_instances:
                boxes = boxes[:max_instances]
                labels = labels[:max_instances]
                scores = scores[:max_instances]
                masks = masks[:max_instances]

            # Process masks (apply threshold)
            binary_masks = masks > mask_threshold

            # Create visualizations
            colored_mask_image = self._create_colored_masks(
                image_array, binary_masks, scores
            )

            overlay_image = self._create_overlay(
                image_array, colored_mask_image, alpha=0.5
            )

            annotated_image = self._create_annotated_image(
                image_array, boxes, labels, scores, binary_masks
            )

            segmentation_time = (time.time() - start_time) * 1000

            return {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                'masks': binary_masks,
                'original_image': image_array,
                'colored_masks': colored_mask_image,
                'overlay_image': overlay_image,
                'annotated_image': annotated_image,
                'image_shape': original_size,
                'segmentation_time_ms': segmentation_time
            }

        except Exception as e:
            raise RuntimeError(f"Instance segmentation failed: {str(e)}")

    def _create_colored_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray
    ) -> np.ndarray:
        """Create colored instance masks.

        Args:
            image: Original image array
            masks: Binary masks for each instance
            scores: Confidence scores

        Returns:
            Colored mask image
        """
        colored = np.zeros_like(image)

        # Sort by confidence (draw lower confidence first)
        sorted_indices = np.argsort(scores)

        for idx in sorted_indices:
            mask = masks[idx][0]  # Remove channel dimension
            color = get_instance_color(idx)

            # Apply color to mask region
            for c in range(3):
                colored[:, :, c] = np.where(
                    mask > 0.5,
                    color[c],
                    colored[:, :, c]
                )

        return colored

    def _create_overlay(
        self,
        image: np.ndarray,
        colored_masks: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create overlay of image and colored masks.

        Args:
            image: Original image
            colored_masks: Colored mask image
            alpha: Transparency factor

        Returns:
            Overlaid image
        """
        # Blend images
        overlay = (image * (1 - alpha) + colored_masks * alpha).astype(np.uint8)
        return overlay

    def _create_annotated_image(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        masks: np.ndarray
    ) -> np.ndarray:
        """Create annotated image with bounding boxes and labels.

        Args:
            image: Original image
            boxes: Bounding boxes
            labels: Class labels
            scores: Confidence scores
            masks: Binary masks

        Returns:
            Annotated image
        """
        annotated = image.copy()
        pil_image = Image.fromarray(annotated)
        draw = ImageDraw.Draw(pil_image, 'RGBA')

        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            color = tuple(get_instance_color(idx))

            # Draw bounding box
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color + (255,),
                width=2
            )

            # Draw label with background
            class_name = get_class_name(int(label))
            label_text = f"{class_name}: {score:.2f}"

            # Get text size
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw background rectangle for text
            draw.rectangle(
                [(x1, y1 - text_height - 4), (x1 + text_width + 4, y1)],
                fill=color + (200,)
            )

            # Draw text
            draw.text(
                (x1 + 2, y1 - text_height - 2),
                label_text,
                fill=(255, 255, 255, 255),
                font=font
            )

        return np.array(pil_image)

    def process_request(
        self,
        request: InstanceSegmentationRequest
    ) -> InstanceSegmentationResponse:
        """Process an instance segmentation request.

        Args:
            request: InstanceSegmentationRequest with segmentation parameters

        Returns:
            InstanceSegmentationResponse with segmentation results

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
        if self.model is None or self.model_backbone != request.model_backbone:
            self.model_backbone = request.model_backbone
            self.model = None  # Force reload

        # Run segmentation
        result = self.segment(
            image_path=image_path,
            confidence_threshold=request.confidence_threshold,
            mask_threshold=request.mask_threshold,
            max_instances=request.max_instances,
            nms_threshold=request.nms_threshold
        )

        # Extract results
        boxes = result['boxes']
        labels = result['labels']
        scores = result['scores']
        masks = result['masks']

        # Build instance list
        instances = []
        for idx, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            mask_area = int(np.sum(mask[0] > 0.5))

            instances.append(Instance(
                bbox=box.tolist(),
                class_name=get_class_name(int(label)),
                class_id=int(label),
                confidence=float(score),
                mask_area=mask_area,
                instance_id=idx
            ))

        # Calculate statistics
        statistics = self._calculate_statistics(instances, result['image_shape'])

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            result['original_image'],
            result['colored_masks'],
            result['overlay_image'],
            result['annotated_image'],
            instances
        )

        total_time = (time.time() - start_time) * 1000

        return InstanceSegmentationResponse(
            success=True,
            instances=instances,
            statistics=statistics,
            visualization_data=visualization_data,
            execution_time_ms=total_time,
            model_info=self._get_model_info(),
            parameters_used=request.model_dump(),
            image_info={
                'size': result['image_shape'],
                'index': request.image_index
            }
        )

    def _calculate_statistics(
        self,
        instances: List[Instance],
        image_size: tuple
    ) -> InstanceStatistics:
        """Calculate statistics from detected instances.

        Args:
            instances: List of detected instances
            image_size: Image size (width, height)

        Returns:
            InstanceStatistics object
        """
        if not instances:
            return InstanceStatistics(
                total_instances=0,
                unique_classes=0,
                class_counts={},
                avg_confidence=0.0,
                total_mask_area=0,
                coverage_percentage=0.0,
                avg_instance_size=0.0,
                confidence_distribution={}
            )

        # Class counts
        class_counts = {}
        for instance in instances:
            class_counts[instance.class_name] = class_counts.get(instance.class_name, 0) + 1

        # Confidence statistics
        confidences = [inst.confidence for inst in instances]
        avg_confidence = sum(confidences) / len(confidences)

        # Confidence distribution
        confidence_distribution = {
            "0.0-0.3": 0,
            "0.3-0.5": 0,
            "0.5-0.7": 0,
            "0.7-0.9": 0,
            "0.9-1.0": 0
        }

        for conf in confidences:
            if conf < 0.3:
                confidence_distribution["0.0-0.3"] += 1
            elif conf < 0.5:
                confidence_distribution["0.3-0.5"] += 1
            elif conf < 0.7:
                confidence_distribution["0.5-0.7"] += 1
            elif conf < 0.9:
                confidence_distribution["0.7-0.9"] += 1
            else:
                confidence_distribution["0.9-1.0"] += 1

        # Mask area statistics
        total_mask_area = sum(inst.mask_area for inst in instances)
        image_area = image_size[0] * image_size[1]
        coverage_percentage = (total_mask_area / image_area) * 100
        avg_instance_size = total_mask_area / len(instances)

        return InstanceStatistics(
            total_instances=len(instances),
            unique_classes=len(class_counts),
            class_counts=class_counts,
            avg_confidence=avg_confidence,
            total_mask_area=total_mask_area,
            coverage_percentage=coverage_percentage,
            avg_instance_size=avg_instance_size,
            confidence_distribution=confidence_distribution
        )

    def _prepare_visualization_data(
        self,
        original_image: np.ndarray,
        colored_masks: np.ndarray,
        overlay_image: np.ndarray,
        annotated_image: np.ndarray,
        instances: List[Instance]
    ) -> Dict[str, Any]:
        """Prepare visualization data for frontend.

        Args:
            original_image: Original image
            colored_masks: Colored instance masks
            overlay_image: Overlay of image and masks
            annotated_image: Image with bounding boxes and labels
            instances: List of instances

        Returns:
            Dictionary with base64-encoded images and visualization metadata
        """
        def encode_image(img_array: np.ndarray) -> str:
            """Encode image array to base64."""
            img = Image.fromarray(img_array.astype(np.uint8))
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode()

        # Prepare instance colors for legend
        instance_colors = []
        for inst in instances:
            color = get_instance_color(inst.instance_id)
            instance_colors.append({
                'instance_id': inst.instance_id,
                'class_name': inst.class_name,
                'color': f"rgb({color[0]}, {color[1]}, {color[2]})"
            })

        return {
            'original_image': encode_image(original_image),
            'colored_masks': encode_image(colored_masks),
            'overlay_image': encode_image(overlay_image),
            'annotated_image': encode_image(annotated_image),
            'image_size': {
                'width': original_image.shape[1],
                'height': original_image.shape[0]
            },
            'instance_colors': instance_colors
        }

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model metadata
        """
        return {
            'name': 'Mask R-CNN',
            'backbone': self.model_backbone,
            'framework': 'PyTorch + torchvision',
            'pretrained_on': 'COCO',
            'num_classes': 80,
            'device': str(self.device) if self.device else 'not loaded'
        }

    @staticmethod
    def get_algorithm_info() -> Dict[str, Any]:
        """Get algorithm information.

        Returns:
            Dictionary with algorithm details and dataset info
        """
        return {
            'dataset': get_dataset_info(),
            'algorithm_details': get_algorithm_info()
        }
