"""Neural Style Transfer model implementation using PyTorch and VGG19."""

import time
import base64
import copy
from io import BytesIO
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms

from .schema import (
    StyleTransferRequest,
    StyleTransferResponse,
    LossHistory,
    StyleTransferStatistics,
    FeatureVisualization
)
from .data import download_content_image, download_style_image, get_dataset_info


class VGG19Features(nn.Module):
    """VGG19 feature extractor for style transfer.

    Extracts features from specific layers of VGG19 for content and style representations.
    """

    def __init__(self):
        """Initialize VGG19 and define layer mappings."""
        super(VGG19Features, self).__init__()

        # Load pre-trained VGG19
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Content layer: conv4_2 (captures high-level content)
        self.content_layers = ['conv4_2']

        # Style layers: multiple layers for multi-scale style representation
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        # Build layer mapping
        self.layers = nn.ModuleDict()
        layer_names = {
            '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
            '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
        }

        for idx, layer in enumerate(vgg19):
            layer_name = layer_names.get(str(idx))
            if layer_name:
                self.layers[layer_name] = layer
            else:
                # Include all intermediate layers for proper forward pass
                self.layers[f'layer_{idx}'] = layer

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass to extract features from specific layers.

        Args:
            x: Input tensor (batch_size, 3, H, W)

        Returns:
            Dictionary mapping layer names to feature tensors
        """
        features = {}

        # Process through VGG layers in order
        layer_order = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
        current = x

        # Need to process sequentially through all layers
        idx = 0
        for name, layer in self.layers.items():
            current = layer(current)
            if name in layer_order:
                features[name] = current

        return features


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for style representation.

    The Gram matrix captures texture/style information by computing
    correlations between feature maps.

    Args:
        tensor: Feature tensor (batch_size, channels, height, width)

    Returns:
        Gram matrix (batch_size, channels, channels)
    """
    batch_size, channels, height, width = tensor.size()

    # Reshape to (batch_size, channels, height*width)
    features = tensor.view(batch_size, channels, height * width)

    # Compute Gram matrix: G = F * F^T
    gram = torch.bmm(features, features.transpose(1, 2))

    # Normalize by number of elements
    gram = gram / (channels * height * width)

    return gram


class StyleTransferModel:
    """Neural Style Transfer using VGG19 and Gatys et al. algorithm.

    Implements the seminal style transfer algorithm that optimizes an image
    to match the content of one image and the style of another.

    Attributes:
        vgg: VGG19 feature extractor
        device: Device to run on (CPU or CUDA)
        transform: Image preprocessing transform
        denormalize: Transform to convert back to displayable image
    """

    def __init__(self):
        """Initialize the style transfer model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg = None

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Denormalization for visualization
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def load_model(self):
        """Load the VGG19 model."""
        if self.vgg is None:
            print("Loading VGG19 model...")
            self.vgg = VGG19Features().to(self.device)
            self.vgg.eval()
            print(f"VGG19 loaded successfully on {self.device}")

    def load_image(self, image_path: str, size: int = 512) -> Tuple[torch.Tensor, np.ndarray]:
        """Load and preprocess an image.

        Args:
            image_path: Path to image file
            size: Target size for the longest dimension

        Returns:
            Tuple of (preprocessed tensor, original numpy array)
        """
        image = Image.open(image_path).convert('RGB')
        original = np.array(image)

        # Resize maintaining aspect ratio
        w, h = image.size
        if w > h:
            new_w = size
            new_h = int(size * h / w)
        else:
            new_h = size
            new_w = int(size * w / h)

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        tensor = transform(image).unsqueeze(0).to(self.device)

        return tensor, original

    def compute_content_loss(
        self,
        generated_features: torch.Tensor,
        content_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute content loss (MSE between feature representations).

        Args:
            generated_features: Features from generated image
            content_features: Features from content image

        Returns:
            Content loss value
        """
        return F.mse_loss(generated_features, content_features)

    def compute_style_loss(
        self,
        generated_features: torch.Tensor,
        style_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute style loss (MSE between Gram matrices).

        Args:
            generated_features: Features from generated image
            style_features: Features from style image

        Returns:
            Style loss value
        """
        generated_gram = gram_matrix(generated_features)
        style_gram = gram_matrix(style_features)

        return F.mse_loss(generated_gram, style_gram)

    def stylize(
        self,
        content_path: str,
        style_path: str,
        iterations: int = 300,
        content_weight: float = 1.0,
        style_weight: float = 1000000.0,
        learning_rate: float = 0.003,
        image_size: int = 512
    ) -> Dict[str, Any]:
        """Perform neural style transfer.

        Args:
            content_path: Path to content image
            style_path: Path to style image
            iterations: Number of optimization iterations
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            learning_rate: Optimizer learning rate
            image_size: Output image size

        Returns:
            Dictionary with stylized image and optimization history
        """
        # Load model
        self.load_model()

        start_time = time.time()

        # Load images
        content_tensor, content_original = self.load_image(content_path, image_size)
        style_tensor, style_original = self.load_image(style_path, image_size)

        # Initialize generated image as copy of content
        generated = content_tensor.clone().requires_grad_(True)

        # Use Adam optimizer (can also use LBFGS)
        optimizer = optim.Adam([generated], lr=learning_rate)

        # Extract target features
        content_features = self.vgg(content_tensor)
        style_features = self.vgg(style_tensor)

        # Extract target content representation (conv4_2)
        target_content = content_features['conv4_2'].detach()

        # Extract target style representations (multiple layers)
        target_styles = {
            layer: gram_matrix(style_features[layer]).detach()
            for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        }

        # Optimization loop
        loss_history = []
        print(f"Starting style transfer optimization for {iterations} iterations...")

        for iteration in range(iterations):
            optimizer.zero_grad()

            # Extract features from generated image
            generated_features = self.vgg(generated)

            # Compute content loss
            content_loss = self.compute_content_loss(
                generated_features['conv4_2'],
                target_content
            )

            # Compute style loss (sum over multiple layers)
            style_loss = 0
            for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
                generated_gram = gram_matrix(generated_features[layer])
                style_loss += F.mse_loss(generated_gram, target_styles[layer])

            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Clamp pixel values to valid range
            with torch.no_grad():
                generated.clamp_(-2.5, 2.5)  # Approximate range after normalization

            # Record loss history
            if iteration % 10 == 0 or iteration == iterations - 1:
                loss_history.append({
                    'iteration': iteration,
                    'total_loss': float(total_loss.item()),
                    'content_loss': float(content_loss.item()),
                    'style_loss': float(style_loss.item())
                })

                if iteration % 50 == 0:
                    print(f"Iteration {iteration}/{iterations}: "
                          f"Total Loss = {total_loss.item():.2f}, "
                          f"Content = {content_loss.item():.4f}, "
                          f"Style = {style_loss.item():.2f}")

        stylize_time = (time.time() - start_time) * 1000

        # Convert generated image to numpy array
        generated_image = self.tensor_to_image(generated)

        return {
            'generated_image': generated_image,
            'content_original': content_original,
            'style_original': style_original,
            'loss_history': loss_history,
            'stylize_time_ms': stylize_time
        }

    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable numpy array.

        Args:
            tensor: Image tensor (1, 3, H, W) normalized

        Returns:
            Numpy array (H, W, 3) with values in [0, 255]
        """
        # Denormalize
        image = self.denormalize(tensor.squeeze(0).cpu())

        # Clamp to [0, 1]
        image = torch.clamp(image, 0, 1)

        # Convert to numpy and scale to [0, 255]
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)

        return image

    def extract_feature_visualizations(
        self,
        content_tensor: torch.Tensor,
        generated_tensor: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Extract and visualize feature maps from key layers.

        Args:
            content_tensor: Content image tensor
            generated_tensor: Generated image tensor

        Returns:
            List of feature visualization dictionaries
        """
        content_features = self.vgg(content_tensor)
        generated_features = self.vgg(generated_tensor)

        visualizations = []

        # Visualize a few key layers
        layers_to_visualize = [
            ('conv1_1', 'Early edges and colors'),
            ('conv2_1', 'Textures and patterns'),
            ('conv4_2', 'High-level content features')
        ]

        for layer_name, description in layers_to_visualize:
            # Get first channel of feature map
            feature_map = generated_features[layer_name][0, 0].detach().cpu().numpy()

            # Normalize to [0, 255]
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            feature_map = (feature_map * 255).astype(np.uint8)

            # Convert to RGB for visualization
            feature_image = np.stack([feature_map] * 3, axis=-1)

            # Convert to base64
            feature_base64 = self._image_to_base64(feature_image)

            visualizations.append({
                'layer_name': layer_name,
                'feature_map': feature_base64,
                'description': description
            })

        return visualizations

    def process_request(
        self,
        request: StyleTransferRequest
    ) -> StyleTransferResponse:
        """Process a style transfer request.

        Args:
            request: StyleTransferRequest with parameters

        Returns:
            StyleTransferResponse with results
        """
        start_time = time.time()

        # Download images
        try:
            content_path = download_content_image(request.content_image_index)
            style_path = download_style_image(request.style_image_index)
        except Exception as e:
            raise ValueError(f"Failed to load images: {str(e)}")

        # Perform style transfer
        result = self.stylize(
            content_path,
            style_path,
            iterations=request.iterations,
            content_weight=request.content_weight,
            style_weight=request.style_weight,
            learning_rate=request.learning_rate,
            image_size=request.image_size
        )

        # Parse loss history
        loss_history = [
            LossHistory(**loss_data) for loss_data in result['loss_history']
        ]

        # Calculate statistics
        statistics = self._calculate_statistics(loss_history)

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            result['content_original'],
            result['style_original'],
            result['generated_image'],
            loss_history
        )

        # Get model info
        model_info = self._get_model_info()

        # Get image info
        image_info = {
            'content_image_index': request.content_image_index,
            'style_image_index': request.style_image_index,
            'output_size': result['generated_image'].shape[:2],
            'content_size': result['content_original'].shape[:2],
            'style_size': result['style_original'].shape[:2]
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return StyleTransferResponse(
            success=True,
            statistics=statistics,
            visualization_data=visualization_data,
            loss_history=loss_history,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'content_image_index': request.content_image_index,
                'style_image_index': request.style_image_index,
                'iterations': request.iterations,
                'content_weight': request.content_weight,
                'style_weight': request.style_weight,
                'learning_rate': request.learning_rate,
                'image_size': request.image_size
            },
            image_info=image_info
        )

    def _calculate_statistics(
        self,
        loss_history: List[LossHistory]
    ) -> StyleTransferStatistics:
        """Calculate statistics from loss history.

        Args:
            loss_history: List of LossHistory objects

        Returns:
            StyleTransferStatistics object
        """
        if not loss_history:
            return StyleTransferStatistics(
                total_iterations=0,
                final_total_loss=0.0,
                final_content_loss=0.0,
                final_style_loss=0.0,
                initial_total_loss=0.0,
                loss_reduction=0.0,
                convergence_rate=0.0
            )

        initial_loss = loss_history[0].total_loss
        final_loss = loss_history[-1].total_loss

        # Calculate loss reduction percentage
        loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100

        # Calculate convergence rate (loss reduction per iteration)
        convergence_rate = (initial_loss - final_loss) / len(loss_history)

        return StyleTransferStatistics(
            total_iterations=loss_history[-1].iteration + 1,
            final_total_loss=loss_history[-1].total_loss,
            final_content_loss=loss_history[-1].content_loss,
            final_style_loss=loss_history[-1].style_loss,
            initial_total_loss=initial_loss,
            loss_reduction=loss_reduction,
            convergence_rate=convergence_rate
        )

    def _prepare_visualization_data(
        self,
        content_image: np.ndarray,
        style_image: np.ndarray,
        generated_image: np.ndarray,
        loss_history: List[LossHistory]
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            content_image: Original content image
            style_image: Original style image
            generated_image: Generated stylized image
            loss_history: Loss history during optimization

        Returns:
            Dictionary with visualization data
        """
        # Convert images to base64
        content_base64 = self._image_to_base64(content_image)
        style_base64 = self._image_to_base64(style_image)
        generated_base64 = self._image_to_base64(generated_image)

        # Prepare loss curve data
        loss_curves = {
            'iterations': [loss.iteration for loss in loss_history],
            'total_loss': [loss.total_loss for loss in loss_history],
            'content_loss': [loss.content_loss for loss in loss_history],
            'style_loss': [loss.style_loss for loss in loss_history]
        }

        return {
            'content_image': content_base64,
            'style_image': style_base64,
            'generated_image': generated_base64,
            'loss_curves': loss_curves,
            'three_panel_display': {
                'content': content_base64,
                'style': style_base64,
                'result': generated_base64
            }
        }

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string.

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
        """Get information about the VGG19 model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': 'VGG19',
            'description': 'Pre-trained VGG19 for feature extraction',
            'framework': 'PyTorch',
            'device': str(self.device),
            'content_layers': ['conv4_2'],
            'style_layers': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
            'algorithm': 'Gatys et al. Neural Style Transfer',
            'paper': 'A Neural Algorithm of Artistic Style (2015)',
            'optimization': 'Adam optimizer with image-space optimization'
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
                'method': 'Gatys et al. Neural Style Transfer',
                'backbone': 'VGG19 pre-trained on ImageNet',
                'content_representation': 'High-level feature maps from conv4_2',
                'style_representation': 'Gram matrices from multiple convolutional layers',
                'optimization': 'Image-space gradient descent (Adam or LBFGS)',
                'loss_function': 'Weighted sum of content loss (MSE) and style loss (Gram matrix MSE)'
            }
        }
