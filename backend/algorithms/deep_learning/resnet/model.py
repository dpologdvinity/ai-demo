"""ResNet (Residual Network) implementation using PyTorch and torchvision."""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64

from .schema import ResNetRequest, ResNetResponse, PredictionResult, ResidualBlockInfo
from .data import (
    get_imagenet_labels,
    get_sample_images,
    preprocess_image,
    postprocess_for_visualization
)


class ResNetModel:
    """ResNet model wrapper for image classification.

    Uses pre-trained torchvision ResNet models (ResNet18/34/50/101) for
    image classification with ImageNet labels. Provides feature extraction
    and visualization capabilities.

    Attributes:
        model_variant: ResNet architecture variant (resnet18, resnet34, etc.)
        model: The PyTorch ResNet model
        device: Device to run model on (cpu or cuda)
        transform: Image preprocessing transform
        labels: ImageNet class labels
        feature_maps: Stored intermediate feature maps for visualization
    """

    def __init__(
        self,
        model_variant: str = "resnet18",
        use_pretrained: bool = True,
        device: Optional[str] = None
    ):
        """Initialize ResNet model.

        Args:
            model_variant: ResNet variant (resnet18, resnet34, resnet50, resnet101)
            use_pretrained: Whether to load pre-trained ImageNet weights
            device: Device to run on ('cpu' or 'cuda'), auto-detected if None
        """
        self.model_variant = model_variant
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_maps = {}

        # Load model
        self.model = self._load_model(model_variant, use_pretrained)
        self.model.to(self.device)
        self.model.eval()

        # Register hooks for feature extraction
        self._register_hooks()

        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load ImageNet labels
        self.labels = get_imagenet_labels()

    def _load_model(self, variant: str, use_pretrained: bool) -> nn.Module:
        """Load ResNet model variant.

        Args:
            variant: Model variant name
            use_pretrained: Whether to load pre-trained weights

        Returns:
            ResNet model

        Raises:
            ValueError: If variant is not supported
        """
        model_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
        }

        if variant not in model_map:
            raise ValueError(
                f"Unsupported model variant: {variant}. "
                f"Choose from {list(model_map.keys())}"
            )

        if use_pretrained:
            weights = {
                'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
                'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
                'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
                'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,
            }
            model = model_map[variant](weights=weights[variant])
        else:
            model = model_map[variant](weights=None)

        return model

    def _register_hooks(self):
        """Register forward hooks to extract feature maps from key layers."""

        def get_activation(name):
            def hook(model, input, output):
                # Store feature maps (detach and move to CPU)
                self.feature_maps[name] = output.detach().cpu()
            return hook

        # Register hooks for key layers
        # Layer1, Layer2, Layer3, Layer4 are the residual block groups
        self.model.layer1.register_forward_hook(get_activation('layer1'))
        self.model.layer2.register_forward_hook(get_activation('layer2'))
        self.model.layer3.register_forward_hook(get_activation('layer3'))
        self.model.layer4.register_forward_hook(get_activation('layer4'))

    def predict(
        self,
        image: Image.Image,
        top_k: int = 5
    ) -> Tuple[List[PredictionResult], torch.Tensor]:
        """Run inference on an image.

        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predictions list, raw logits)
        """
        # Preprocess image
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_batch)

        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top-K predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Format predictions
        predictions = []
        for i in range(top_k):
            class_id = int(top_indices[i].item())
            confidence = float(top_probs[i].item())
            class_name = self.labels.get(class_id, f"Class {class_id}")

            predictions.append(PredictionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence
            ))

        return predictions, output

    def extract_feature_maps(self) -> Dict[str, np.ndarray]:
        """Extract and process stored feature maps.

        Returns:
            Dictionary mapping layer names to feature map arrays
        """
        processed_maps = {}

        for layer_name, feature_map in self.feature_maps.items():
            # feature_map shape: (batch, channels, height, width)
            # Take first batch item and compute mean across channels
            fm = feature_map[0]  # (channels, height, width)

            # Average across channels for visualization
            fm_avg = torch.mean(fm, dim=0)  # (height, width)

            processed_maps[layer_name] = {
                'shape': list(fm.shape),
                'avg_activation': fm_avg.numpy().tolist(),
                'num_channels': fm.shape[0]
            }

        return processed_maps

    def get_residual_blocks_info(self) -> List[ResidualBlockInfo]:
        """Get information about residual blocks in the architecture.

        Returns:
            List of residual block information
        """
        blocks = []

        # Helper to extract block info
        def extract_block_info(layer, layer_name, base_channels):
            for i, block in enumerate(layer):
                # Get input/output channels from first conv layer
                first_conv = block.conv1
                input_ch = first_conv.in_channels
                output_ch = block.conv2.out_channels if hasattr(block, 'conv2') else first_conv.out_channels

                # Check if block has downsampling
                has_downsample = hasattr(block, 'downsample') and block.downsample is not None
                stride = first_conv.stride[0]

                blocks.append(ResidualBlockInfo(
                    block_name=f"{layer_name}_block{i+1}",
                    input_channels=input_ch,
                    output_channels=output_ch,
                    stride=stride,
                    has_downsample=has_downsample
                ))

        # Extract info from each layer
        extract_block_info(self.model.layer1, "layer1", 64)
        extract_block_info(self.model.layer2, "layer2", 128)
        extract_block_info(self.model.layer3, "layer3", 256)
        extract_block_info(self.model.layer4, "layer4", 512)

        return blocks

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information.

        Returns:
            Dictionary containing model details
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Get depth based on variant
        depth_map = {
            'resnet18': 18,
            'resnet34': 34,
            'resnet50': 50,
            'resnet101': 101
        }

        # Count residual blocks
        num_blocks = len(list(self.model.layer1)) + len(list(self.model.layer2)) + \
                     len(list(self.model.layer3)) + len(list(self.model.layer4))

        return {
            'model_variant': self.model_variant,
            'depth': depth_map.get(self.model_variant, 0),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_residual_blocks': num_blocks,
            'device': str(self.device),
            'input_size': '224x224',
            'num_classes': 1000
        }

    def run_inference(self, request: ResNetRequest) -> ResNetResponse:
        """Run complete inference pipeline.

        Args:
            request: ResNet inference request

        Returns:
            ResNet inference response with predictions and visualizations
        """
        start_time = time.time()

        # Load image
        if request.image_path:
            # Load from provided path
            image = Image.open(request.image_path).convert('RGB')
        else:
            # Load sample image from gallery
            sample_images = get_sample_images()
            image_data = sample_images[request.image_index % len(sample_images)]
            image = image_data['image']

        # Store original image for visualization
        original_image = np.array(image)

        # Run prediction
        predictions, logits = self.predict(image, top_k=request.top_k)

        # Extract feature maps
        feature_maps = self.extract_feature_maps()

        # Get residual blocks info
        residual_blocks = self.get_residual_blocks_info()

        # Get model info
        model_info = self.get_model_info()

        # Prepare visualization data
        visualization_data = {
            'confidence_chart': [
                {
                    'class_name': pred.class_name,
                    'confidence': pred.confidence,
                    'rank': i + 1
                }
                for i, pred in enumerate(predictions)
            ],
            'residual_blocks': [
                {
                    'name': block.block_name,
                    'input_channels': block.input_channels,
                    'output_channels': block.output_channels,
                    'stride': block.stride,
                    'has_downsample': block.has_downsample
                }
                for block in residual_blocks
            ],
            'feature_maps': feature_maps,
            'architecture_info': {
                'variant': self.model_variant,
                'depth': model_info['depth'],
                'num_blocks': model_info['num_residual_blocks'],
                'total_params': model_info['total_parameters']
            }
        }

        # Add architecture diagram structure
        visualization_data['architecture_diagram'] = {
            'layers': [
                {'name': 'Input', 'type': 'input', 'shape': [224, 224, 3]},
                {'name': 'Conv1', 'type': 'conv', 'filters': 64, 'kernel': 7, 'stride': 2},
                {'name': 'MaxPool', 'type': 'maxpool', 'kernel': 3, 'stride': 2},
                {'name': 'Layer1', 'type': 'residual_group', 'blocks': len(list(self.model.layer1)), 'channels': 64},
                {'name': 'Layer2', 'type': 'residual_group', 'blocks': len(list(self.model.layer2)), 'channels': 128},
                {'name': 'Layer3', 'type': 'residual_group', 'blocks': len(list(self.model.layer3)), 'channels': 256},
                {'name': 'Layer4', 'type': 'residual_group', 'blocks': len(list(self.model.layer4)), 'channels': 512},
                {'name': 'AvgPool', 'type': 'avgpool', 'kernel': 7},
                {'name': 'FC', 'type': 'fc', 'units': 1000}
            ]
        }

        execution_time_ms = (time.time() - start_time) * 1000

        # Prepare input image for response (downsample for transmission)
        input_image_viz = self._prepare_image_for_response(original_image)

        return ResNetResponse(
            success=True,
            predictions=predictions,
            input_image=input_image_viz,
            input_shape=list(original_image.shape),
            feature_maps=feature_maps,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'model_variant': request.model_variant,
                'top_k': request.top_k,
                'use_pretrained': request.use_pretrained,
                'image_index': request.image_index
            }
        )

    def _prepare_image_for_response(
        self,
        image: np.ndarray,
        max_size: int = 224
    ) -> List[List[List[float]]]:
        """Prepare image for JSON response.

        Args:
            image: Input image array (H, W, C)
            max_size: Maximum dimension size

        Returns:
            Image as nested list, normalized to [0, 1]
        """
        # Resize if needed
        h, w = image.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            from PIL import Image
            img_pil = Image.fromarray(image.astype(np.uint8))
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            image = np.array(img_pil)

        # Normalize to [0, 1]
        image_normalized = image.astype(float) / 255.0

        return image_normalized.tolist()
