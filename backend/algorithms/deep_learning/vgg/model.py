"""VGG Network implementation using PyTorch and torchvision."""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from .schema import VGGRequest, VGGResponse, PredictionResult, ConvBlockInfo
from .data import (
    get_imagenet_labels,
    get_sample_images,
    preprocess_image,
    postprocess_for_visualization
)


class VGGModel:
    """VGG model wrapper for image classification.

    Uses pre-trained torchvision VGG models (VGG16/VGG19, with or without batch norm)
    for image classification with ImageNet labels. Provides feature extraction
    and visualization capabilities.

    VGG networks are characterized by their use of small 3x3 convolution filters
    throughout the architecture, stacked in blocks followed by max pooling.

    Attributes:
        model_variant: VGG architecture variant (vgg16 or vgg19)
        batch_norm: Whether to use batch normalization variant
        model: The PyTorch VGG model
        device: Device to run model on (cpu or cuda)
        transform: Image preprocessing transform
        labels: ImageNet class labels
        feature_maps: Stored intermediate feature maps for visualization
    """

    def __init__(
        self,
        model_variant: str = "vgg16",
        use_pretrained: bool = True,
        batch_norm: bool = True,
        device: Optional[str] = None
    ):
        """Initialize VGG model.

        Args:
            model_variant: VGG variant (vgg16 or vgg19)
            use_pretrained: Whether to load pre-trained ImageNet weights
            batch_norm: Whether to use batch normalization variant
            device: Device to run on ('cpu' or 'cuda'), auto-detected if None
        """
        self.model_variant = model_variant
        self.batch_norm = batch_norm
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_maps = {}

        # Load model
        self.model = self._load_model(model_variant, use_pretrained, batch_norm)
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

    def _load_model(self, variant: str, use_pretrained: bool, batch_norm: bool) -> nn.Module:
        """Load VGG model variant.

        Args:
            variant: Model variant name (vgg16 or vgg19)
            use_pretrained: Whether to load pre-trained weights
            batch_norm: Whether to use batch normalization variant

        Returns:
            VGG model

        Raises:
            ValueError: If variant is not supported
        """
        # Map variant and batch_norm to model function
        model_map = {
            ('vgg16', False): (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
            ('vgg16', True): (models.vgg16_bn, models.VGG16_BN_Weights.IMAGENET1K_V1),
            ('vgg19', False): (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1),
            ('vgg19', True): (models.vgg19_bn, models.VGG19_BN_Weights.IMAGENET1K_V1),
        }

        key = (variant, batch_norm)
        if key not in model_map:
            raise ValueError(
                f"Unsupported model variant: {variant} with batch_norm={batch_norm}. "
                f"Choose from vgg16, vgg19 with batch_norm True or False"
            )

        model_fn, weights = model_map[key]

        if use_pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)

        return model

    def _register_hooks(self):
        """Register forward hooks to extract feature maps from key layers."""

        def get_activation(name):
            def hook(model, input, output):
                # Store feature maps (detach and move to CPU)
                self.feature_maps[name] = output.detach().cpu()
            return hook

        # VGG features are organized as a sequential module
        # Extract features from key positions in the network
        # Typical VGG16 structure: 5 conv blocks (2, 2, 3, 3, 3 conv layers each)
        # Typical VGG19 structure: 5 conv blocks (2, 2, 4, 4, 4 conv layers each)

        # Register hooks at max pooling locations (end of each block)
        # VGG features module contains conv, relu, and pooling layers in sequence
        features = self.model.features

        # Find max pooling layers and register hooks before them
        pool_count = 0
        for i, layer in enumerate(features):
            if isinstance(layer, nn.MaxPool2d):
                pool_count += 1
                # Register hook at the layer before pooling (last conv in block)
                if i > 0:
                    features[i-1].register_forward_hook(
                        get_activation(f'block{pool_count}')
                    )

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

    def extract_feature_maps(self) -> Dict[str, Any]:
        """Extract and process stored feature maps.

        Returns:
            Dictionary mapping layer names to feature map arrays
        """
        processed_maps = {}

        for layer_name, feature_map in self.feature_maps.items():
            # feature_map shape: (batch, channels, height, width)
            # Take first batch item
            fm = feature_map[0]  # (channels, height, width)

            # Average across channels for visualization
            fm_avg = torch.mean(fm, dim=0)  # (height, width)

            # Also get some individual channel activations
            num_channels = min(8, fm.shape[0])
            channel_samples = []
            step = max(1, fm.shape[0] // num_channels)
            for c in range(0, fm.shape[0], step):
                if len(channel_samples) < num_channels:
                    channel_samples.append(fm[c].numpy().tolist())

            processed_maps[layer_name] = {
                'shape': list(fm.shape),
                'avg_activation': fm_avg.numpy().tolist(),
                'num_channels': fm.shape[0],
                'sample_channels': channel_samples
            }

        return processed_maps

    def get_conv_blocks_info(self) -> List[ConvBlockInfo]:
        """Get information about convolutional blocks in VGG architecture.

        Returns:
            List of convolutional block information
        """
        blocks = []

        # VGG architecture has 5 convolutional blocks
        # VGG16: [2, 2, 3, 3, 3] conv layers per block
        # VGG19: [2, 2, 4, 4, 4] conv layers per block

        if self.model_variant == 'vgg16':
            block_structures = [
                (2, 64),   # Block 1: 2 conv layers, 64 filters
                (2, 128),  # Block 2: 2 conv layers, 128 filters
                (3, 256),  # Block 3: 3 conv layers, 256 filters
                (3, 512),  # Block 4: 3 conv layers, 512 filters
                (3, 512),  # Block 5: 3 conv layers, 512 filters
            ]
        else:  # vgg19
            block_structures = [
                (2, 64),   # Block 1: 2 conv layers, 64 filters
                (2, 128),  # Block 2: 2 conv layers, 128 filters
                (4, 256),  # Block 3: 4 conv layers, 256 filters
                (4, 512),  # Block 4: 4 conv layers, 512 filters
                (4, 512),  # Block 5: 4 conv layers, 512 filters
            ]

        for i, (num_layers, num_filters) in enumerate(block_structures, 1):
            blocks.append(ConvBlockInfo(
                block_name=f"block{i}",
                num_layers=num_layers,
                num_filters=num_filters,
                kernel_size=3,  # VGG uses 3x3 kernels throughout
                has_pooling=True  # Each block ends with max pooling
            ))

        return blocks

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information.

        Returns:
            Dictionary containing model details
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Get depth and conv layer count
        if self.model_variant == 'vgg16':
            depth = 16
            num_conv_layers = 13
        else:  # vgg19
            depth = 19
            num_conv_layers = 16

        # Count FC parameters (approximately)
        fc_params = sum(p.numel() for p in self.model.classifier.parameters())

        return {
            'model_variant': self.model_variant,
            'batch_norm': self.batch_norm,
            'depth': depth,
            'num_conv_layers': num_conv_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'fc_parameters': fc_params,
            'conv_parameters': total_params - fc_params,
            'device': str(self.device),
            'input_size': '224x224',
            'num_classes': 1000,
            'kernel_size': '3x3 (all conv layers)',
            'pooling': 'MaxPool 2x2'
        }

    def run_inference(self, request: VGGRequest) -> VGGResponse:
        """Run complete inference pipeline.

        Args:
            request: VGG inference request

        Returns:
            VGG inference response with predictions and visualizations
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

        # Get conv blocks info
        conv_blocks = self.get_conv_blocks_info()

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
            'conv_blocks': [
                {
                    'name': block.block_name,
                    'num_layers': block.num_layers,
                    'num_filters': block.num_filters,
                    'kernel_size': block.kernel_size,
                    'has_pooling': block.has_pooling
                }
                for block in conv_blocks
            ],
            'feature_maps': feature_maps,
            'architecture_info': {
                'variant': self.model_variant,
                'depth': model_info['depth'],
                'num_conv_layers': model_info['num_conv_layers'],
                'total_params': model_info['total_parameters'],
                'batch_norm': self.batch_norm
            }
        }

        # Add architecture diagram structure
        visualization_data['architecture_diagram'] = {
            'layers': [
                {'name': 'Input', 'type': 'input', 'shape': [224, 224, 3]},
            ]
        }

        # Add conv blocks to diagram
        for block in conv_blocks:
            # Add conv layers for this block
            for layer_idx in range(block.num_layers):
                visualization_data['architecture_diagram']['layers'].append({
                    'name': f'{block.block_name}_conv{layer_idx+1}',
                    'type': 'conv',
                    'filters': block.num_filters,
                    'kernel': block.kernel_size
                })
                if self.batch_norm:
                    visualization_data['architecture_diagram']['layers'].append({
                        'name': f'{block.block_name}_bn{layer_idx+1}',
                        'type': 'batch_norm'
                    })
                visualization_data['architecture_diagram']['layers'].append({
                    'name': f'{block.block_name}_relu{layer_idx+1}',
                    'type': 'relu'
                })

            # Add pooling at end of block
            visualization_data['architecture_diagram']['layers'].append({
                'name': f'{block.block_name}_pool',
                'type': 'maxpool',
                'kernel': 2,
                'stride': 2
            })

        # Add FC layers
        visualization_data['architecture_diagram']['layers'].extend([
            {'name': 'Flatten', 'type': 'flatten'},
            {'name': 'FC1', 'type': 'fc', 'units': 4096},
            {'name': 'ReLU1', 'type': 'relu'},
            {'name': 'Dropout1', 'type': 'dropout', 'rate': 0.5},
            {'name': 'FC2', 'type': 'fc', 'units': 4096},
            {'name': 'ReLU2', 'type': 'relu'},
            {'name': 'Dropout2', 'type': 'dropout', 'rate': 0.5},
            {'name': 'FC3', 'type': 'fc', 'units': 1000}
        ])

        execution_time_ms = (time.time() - start_time) * 1000

        # Prepare input image for response
        input_image_viz = self._prepare_image_for_response(original_image)

        return VGGResponse(
            success=True,
            predictions=predictions,
            input_image=input_image_viz,
            input_shape=list(original_image.shape),
            feature_maps=feature_maps,
            conv_blocks=conv_blocks,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'model_variant': request.model_variant,
                'top_k': request.top_k,
                'use_pretrained': request.use_pretrained,
                'batch_norm': request.batch_norm,
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
            img_pil = Image.fromarray(image.astype(np.uint8))
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            image = np.array(img_pil)

        # Normalize to [0, 1]
        image_normalized = image.astype(float) / 255.0

        return image_normalized.tolist()
