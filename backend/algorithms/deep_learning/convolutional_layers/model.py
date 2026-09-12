"""Convolutional Layers model for demonstrating convolution operations and filters."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from .schema import ConvolutionalLayersRequest, ConvolutionalLayersResponse
from .data import load_sample_image, get_common_filters


class ConvolutionalLayersModel:
    """Model demonstrating convolutional layer operations for feature extraction.

    This model showcases:
    - Convolution operations with learnable filters
    - Common predefined filters (Sobel, Gaussian, etc.)
    - Feature map visualization
    - Effect of stride and padding
    - Activation functions
    """

    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = 'same',
        activation: str = 'relu',
        random_state: int = 42
    ):
        """Initialize the convolutional layers model.

        Args:
            num_filters: Number of convolutional filters
            kernel_size: Size of convolution kernel
            stride: Stride for convolution
            padding: Padding type ('same' or 'valid')
            activation: Activation function ('relu', 'tanh', 'none')
            random_state: Random seed for reproducibility
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_type = padding
        self.activation = activation
        self.random_state = random_state

        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # Calculate padding
        if padding == 'same':
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = 0

        # Create convolutional layer
        self.conv_layer = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding
        )

        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.conv_layer.weight)
        nn.init.zeros_(self.conv_layer.bias)

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function to tensor.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        else:  # 'none'
            return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through convolutional layer.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Tuple of (pre-activation output, post-activation output)
        """
        # Apply convolution
        conv_out = self.conv_layer(x)

        # Apply activation
        activated_out = self.apply_activation(conv_out)

        return conv_out, activated_out

    def extract_filter_kernels(self) -> List[np.ndarray]:
        """Extract filter kernel weights from the convolutional layer.

        Returns:
            List of filter kernels as numpy arrays
        """
        weights = self.conv_layer.weight.detach().cpu().numpy()
        # Shape: (num_filters, in_channels, kernel_size, kernel_size)

        kernels = []
        for i in range(self.num_filters):
            kernel = weights[i, 0, :, :]  # Extract single channel kernel
            kernels.append(kernel)

        return kernels

    def compute_output_dimensions(self, input_height: int, input_width: int) -> Dict[str, int]:
        """Compute output dimensions after convolution.

        Args:
            input_height: Height of input image
            input_width: Width of input image

        Returns:
            Dictionary with output dimensions
        """
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        return {
            'height': output_height,
            'width': output_width,
            'channels': self.num_filters
        }

    def compute_activation_stats(self, feature_maps: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about activation patterns.

        Args:
            feature_maps: Output feature maps tensor

        Returns:
            Dictionary of activation statistics
        """
        fm_numpy = feature_maps.detach().cpu().numpy()

        return {
            'mean_activation': float(np.mean(fm_numpy)),
            'std_activation': float(np.std(fm_numpy)),
            'max_activation': float(np.max(fm_numpy)),
            'min_activation': float(np.min(fm_numpy)),
            'sparsity': float(np.mean(fm_numpy == 0))  # Fraction of zero activations
        }

    def apply_common_filters(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply common predefined filters to the image.

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary of filtered outputs
        """
        common_filters = get_common_filters()
        results = {}

        # Convert image to tensor
        img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

        for filter_name, filter_kernel in common_filters.items():
            # Create filter as Conv2d layer
            kernel_size = filter_kernel.shape[0]
            padding = (kernel_size - 1) // 2

            # Create temporary conv layer with this filter
            temp_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

            # Set the filter weights
            with torch.no_grad():
                temp_conv.weight.data = torch.from_numpy(filter_kernel).float().unsqueeze(0).unsqueeze(0)

            # Apply filter
            filtered = temp_conv(img_tensor)
            results[filter_name] = filtered.squeeze().detach().cpu().numpy()

        return results

    def run_demonstration(self, request: ConvolutionalLayersRequest) -> ConvolutionalLayersResponse:
        """Run the convolutional layers demonstration.

        Args:
            request: Request parameters

        Returns:
            Response with results and visualizations
        """
        import time
        start_time = time.time()

        # Load sample image
        image = load_sample_image()
        input_height, input_width = image.shape

        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

        # Forward pass
        conv_out, feature_maps = self.forward(image_tensor)

        # Extract filter kernels
        filter_kernels = self.extract_filter_kernels()

        # Apply common filters
        common_filter_results = self.apply_common_filters(image)

        # Compute output dimensions
        output_dims = self.compute_output_dimensions(input_height, input_width)

        # Compute activation statistics
        activation_stats = self.compute_activation_stats(feature_maps)

        # Prepare filter kernel data (show first 8 filters for visualization)
        num_visualize = min(8, self.num_filters)
        filter_kernel_data = [
            {
                'filter_id': i,
                'weights': filter_kernels[i].tolist(),
                'shape': list(filter_kernels[i].shape),
                'min_weight': float(np.min(filter_kernels[i])),
                'max_weight': float(np.max(filter_kernels[i])),
                'mean_weight': float(np.mean(filter_kernels[i]))
            }
            for i in range(num_visualize)
        ]

        # Prepare feature map data (show first 8 feature maps)
        feature_maps_numpy = feature_maps.squeeze(0).detach().cpu().numpy()
        feature_map_data = [
            {
                'filter_id': i,
                'output': feature_maps_numpy[i].tolist(),
                'shape': list(feature_maps_numpy[i].shape),
                'min_value': float(np.min(feature_maps_numpy[i])),
                'max_value': float(np.max(feature_maps_numpy[i])),
                'mean_value': float(np.mean(feature_maps_numpy[i]))
            }
            for i in range(num_visualize)
        ]

        # Prepare common filters data
        common_filters_data = {
            'sobel_x': {
                'output': common_filter_results['sobel_x'].tolist(),
                'description': 'Detects vertical edges'
            },
            'sobel_y': {
                'output': common_filter_results['sobel_y'].tolist(),
                'description': 'Detects horizontal edges'
            },
            'gaussian_blur': {
                'output': common_filter_results['gaussian_blur'].tolist(),
                'description': 'Smooths the image'
            },
            'sharpen': {
                'output': common_filter_results['sharpen'].tolist(),
                'description': 'Enhances edges and details'
            },
            'edge_detect': {
                'output': common_filter_results['edge_detect'].tolist(),
                'description': 'Detects all edges'
            }
        }

        # Prepare visualization data
        visualization_data = {
            'input_shape': [input_height, input_width],
            'output_shape': [output_dims['height'], output_dims['width']],
            'num_filters': self.num_filters,
            'num_visualized': num_visualize,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation,
            'dimension_calculation': {
                'formula': '(input_size + 2*padding - kernel_size) / stride + 1',
                'input_height': input_height,
                'input_width': input_width,
                'padding': self.padding,
                'kernel_size': self.kernel_size,
                'stride': self.stride,
                'output_height': output_dims['height'],
                'output_width': output_dims['width']
            }
        }

        # Model info
        model_info = {
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding_type,
            'activation': self.activation,
            'total_parameters': sum(p.numel() for p in self.conv_layer.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.conv_layer.parameters() if p.requires_grad)
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return ConvolutionalLayersResponse(
            success=True,
            input_image=image.tolist(),
            filter_kernels=filter_kernel_data,
            feature_maps=feature_map_data,
            common_filters=common_filters_data,
            output_dimensions=output_dims,
            activation_stats=activation_stats,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'num_filters': self.num_filters,
                'kernel_size': self.kernel_size,
                'stride': self.stride,
                'padding': self.padding_type,
                'activation': self.activation
            }
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information.

        Returns:
            Dictionary with model details
        """
        return {
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding_type,
            'activation': self.activation,
            'total_parameters': sum(p.numel() for p in self.conv_layer.parameters())
        }
