"""Pooling Layers model implementation."""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time


class PoolingLayersModel:
    """Model for demonstrating various pooling operations in CNNs.

    This class implements common pooling operations (max, average, global pooling)
    for educational visualization purposes.
    """

    def __init__(
        self,
        pool_type: str = 'max',
        pool_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        random_state: int = 42
    ):
        """Initialize the pooling layers model.

        Args:
            pool_type: Type of pooling ('max', 'average', 'global_max', 'global_average')
            pool_size: Size of pooling window
            stride: Stride for pooling operation
            padding: Padding to add to input
            random_state: Random seed for reproducibility
        """
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.random_state = random_state
        np.random.seed(random_state)

    def _calculate_output_size(self, input_size: int) -> int:
        """Calculate output size after pooling.

        Args:
            input_size: Input dimension size

        Returns:
            Output dimension size
        """
        return ((input_size + 2 * self.padding - self.pool_size) // self.stride) + 1

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Add padding to input feature map.

        Args:
            x: Input array of shape [channels, height, width]

        Returns:
            Padded array
        """
        if self.padding == 0:
            return x

        return np.pad(
            x,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant',
            constant_values=0
        )

    def max_pool(
        self,
        x: np.ndarray,
        return_positions: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply max pooling to input feature map.

        Args:
            x: Input array of shape [channels, height, width]
            return_positions: Whether to return positions of max values

        Returns:
            Tuple of (pooled output, max positions)
        """
        channels, height, width = x.shape
        x_padded = self._pad_input(x)

        out_h = self._calculate_output_size(height)
        out_w = self._calculate_output_size(width)

        output = np.zeros((channels, out_h, out_w))
        max_positions = np.zeros((channels, out_h, out_w, 2), dtype=int) if return_positions else None

        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.pool_size
                    w_start = j * self.stride
                    w_end = w_start + self.pool_size

                    window = x_padded[c, h_start:h_end, w_start:w_end]
                    output[c, i, j] = np.max(window)

                    if return_positions:
                        # Find position of max value in window
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_positions[c, i, j] = [h_start + max_idx[0], w_start + max_idx[1]]

        return output, max_positions

    def average_pool(self, x: np.ndarray) -> np.ndarray:
        """Apply average pooling to input feature map.

        Args:
            x: Input array of shape [channels, height, width]

        Returns:
            Pooled output
        """
        channels, height, width = x.shape
        x_padded = self._pad_input(x)

        out_h = self._calculate_output_size(height)
        out_w = self._calculate_output_size(width)

        output = np.zeros((channels, out_h, out_w))

        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.pool_size
                    w_start = j * self.stride
                    w_end = w_start + self.pool_size

                    window = x_padded[c, h_start:h_end, w_start:w_end]
                    output[c, i, j] = np.mean(window)

        return output

    def global_max_pool(self, x: np.ndarray) -> np.ndarray:
        """Apply global max pooling to input feature map.

        Args:
            x: Input array of shape [channels, height, width]

        Returns:
            Pooled output of shape [channels, 1, 1]
        """
        channels = x.shape[0]
        output = np.zeros((channels, 1, 1))

        for c in range(channels):
            output[c, 0, 0] = np.max(x[c])

        return output

    def global_average_pool(self, x: np.ndarray) -> np.ndarray:
        """Apply global average pooling to input feature map.

        Args:
            x: Input array of shape [channels, height, width]

        Returns:
            Pooled output of shape [channels, 1, 1]
        """
        channels = x.shape[0]
        output = np.zeros((channels, 1, 1))

        for c in range(channels):
            output[c, 0, 0] = np.mean(x[c])

        return output

    def apply_pooling(
        self,
        x: np.ndarray,
        pool_type: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply specified pooling operation.

        Args:
            x: Input array of shape [channels, height, width]
            pool_type: Type of pooling (uses self.pool_type if None)

        Returns:
            Tuple of (pooled output, max positions if applicable)
        """
        pool_type = pool_type or self.pool_type

        if pool_type == 'max':
            return self.max_pool(x, return_positions=True)
        elif pool_type == 'average':
            return self.average_pool(x), None
        elif pool_type == 'global_max':
            return self.global_max_pool(x), None
        elif pool_type == 'global_average':
            return self.global_average_pool(x), None
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")

    def extract_pooling_windows(
        self,
        x: np.ndarray,
        num_windows: int = 4
    ) -> List[Dict[str, Any]]:
        """Extract sample pooling windows to show the operation.

        Args:
            x: Input array of shape [channels, height, width]
            num_windows: Number of sample windows to extract

        Returns:
            List of window dictionaries with input, output, and positions
        """
        channels, height, width = x.shape
        x_padded = self._pad_input(x)

        out_h = self._calculate_output_size(height)
        out_w = self._calculate_output_size(width)

        windows = []
        window_count = 0

        # Extract windows from first channel only for visualization
        c = 0
        for i in range(min(2, out_h)):  # Sample from first 2 rows
            for j in range(min(2, out_w)):  # Sample from first 2 cols
                if window_count >= num_windows:
                    break

                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                window = x_padded[c, h_start:h_end, w_start:w_end]

                if self.pool_type == 'max':
                    output_value = np.max(window)
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    operation = f"max({window.flatten().tolist()}) = {output_value:.2f}"
                    max_position = [int(max_idx[0]), int(max_idx[1])]
                elif self.pool_type == 'average':
                    output_value = np.mean(window)
                    operation = f"mean({window.flatten().tolist()}) = {output_value:.2f}"
                    max_position = None
                elif self.pool_type.startswith('global'):
                    output_value = np.max(x[c]) if self.pool_type == 'global_max' else np.mean(x[c])
                    operation = f"{self.pool_type}(entire map) = {output_value:.2f}"
                    max_position = None
                else:
                    output_value = 0.0
                    operation = "unknown"
                    max_position = None

                windows.append({
                    'window_data': window.tolist(),
                    'output_value': float(output_value),
                    'position': [int(i), int(j)],
                    'input_region': [int(h_start), int(h_end), int(w_start), int(w_end)],
                    'max_position': max_position,
                    'operation': operation
                })

                window_count += 1

            if window_count >= num_windows:
                break

        return windows

    def compare_pooling_types(
        self,
        x: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different pooling types on the same input.

        Args:
            x: Input array of shape [channels, height, width]

        Returns:
            Dictionary with results for each pooling type
        """
        comparison = {}

        for pool_type in ['max', 'average']:
            # Temporarily change pool type
            original_pool_type = self.pool_type
            self.pool_type = pool_type

            output, max_pos = self.apply_pooling(x)

            comparison[pool_type] = {
                'output': output.tolist(),
                'output_shape': list(output.shape),
                'max_value': float(np.max(output)),
                'min_value': float(np.min(output)),
                'mean_value': float(np.mean(output)),
                'description': self._get_pooling_description(pool_type)
            }

            # Restore original pool type
            self.pool_type = original_pool_type

        # Add global pooling comparison
        for pool_type in ['global_max', 'global_average']:
            original_pool_type = self.pool_type
            self.pool_type = pool_type

            output, _ = self.apply_pooling(x)

            comparison[pool_type] = {
                'output': output.tolist(),
                'output_shape': list(output.shape),
                'value': float(output[0, 0, 0]),
                'description': self._get_pooling_description(pool_type)
            }

            self.pool_type = original_pool_type

        return comparison

    def _get_pooling_description(self, pool_type: str) -> str:
        """Get description of pooling type.

        Args:
            pool_type: Type of pooling

        Returns:
            Description string
        """
        descriptions = {
            'max': 'Selects maximum value from each pooling window. Preserves strongest activations.',
            'average': 'Computes average of values in each pooling window. Smooths features.',
            'global_max': 'Selects maximum value across entire feature map. One value per channel.',
            'global_average': 'Computes average across entire feature map. One value per channel.'
        }
        return descriptions.get(pool_type, 'Unknown pooling type')

    def get_dimension_info(
        self,
        input_shape: Tuple[int, int, int],
        output_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Get information about dimension changes.

        Args:
            input_shape: Input tensor shape [channels, height, width]
            output_shape: Output tensor shape [channels, out_h, out_w]

        Returns:
            Dictionary with dimension information
        """
        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        reduction_factor = input_size / output_size if output_size > 0 else 0

        return {
            'input_shape': list(input_shape),
            'output_shape': list(output_shape),
            'input_size': int(input_size),
            'output_size': int(output_size),
            'reduction_factor': float(reduction_factor),
            'spatial_reduction': f"{input_shape[1]}×{input_shape[2]} → {output_shape[1]}×{output_shape[2]}",
            'formula': f"output_size = ((input_size + 2*padding - pool_size) / stride) + 1",
            'calculation': f"(({input_shape[1]} + 2*{self.padding} - {self.pool_size}) / {self.stride}) + 1 = {output_shape[1]}"
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            'pool_type': self.pool_type,
            'pool_size': self.pool_size,
            'stride': self.stride,
            'padding': self.padding,
            'available_pool_types': ['max', 'average', 'global_max', 'global_average'],
            'complexity': {
                'time': f'O(H*W*K²) where H,W=input size, K=pool_size',
                'space': f'O(H/K * W/K) for output'
            },
            'properties': {
                'translation_invariance': 'Pooling provides local translation invariance',
                'dimensionality_reduction': f'Reduces spatial dimensions by factor of ~{self.stride}²',
                'parameter_free': 'Pooling has no learnable parameters',
                'backprop': 'Max pooling routes gradients to max positions; average pooling distributes equally'
            }
        }


def compute_pooling(
    pool_type: str = 'max',
    pool_size: int = 2,
    stride: int = 2,
    padding: int = 0,
    input_size: int = 8,
    num_channels: int = 1,
    random_state: int = 42
) -> Dict[str, Any]:
    """Compute pooling operation and generate visualization data.

    Args:
        pool_type: Type of pooling operation
        pool_size: Size of pooling window
        stride: Stride for pooling
        padding: Padding to add
        input_size: Size of input feature map
        num_channels: Number of input channels
        random_state: Random seed

    Returns:
        Dictionary with pooling results and visualization data
    """
    start_time = time.time()

    # Initialize model
    model = PoolingLayersModel(
        pool_type=pool_type,
        pool_size=pool_size,
        stride=stride,
        padding=padding,
        random_state=random_state
    )

    # Generate random input feature map
    np.random.seed(random_state)
    input_feature_map = np.random.randn(num_channels, input_size, input_size) * 2 + 5
    input_feature_map = np.clip(input_feature_map, 0, 10)  # Keep values in reasonable range

    # Apply pooling
    pooled_output, max_positions = model.apply_pooling(input_feature_map)

    # Get dimension information
    dimension_info = model.get_dimension_info(
        input_feature_map.shape,
        pooled_output.shape
    )

    # Extract sample pooling windows
    pooling_windows = model.extract_pooling_windows(input_feature_map, num_windows=4)

    # Compare different pooling types
    comparison_data = model.compare_pooling_types(input_feature_map)

    # Prepare visualization data
    visualization_data = {
        'input_heatmap': {
            'data': input_feature_map[0].tolist(),  # First channel
            'shape': list(input_feature_map[0].shape),
            'title': 'Input Feature Map'
        },
        'output_heatmap': {
            'data': pooled_output[0].tolist(),  # First channel
            'shape': list(pooled_output[0].shape),
            'title': f'{pool_type.title()} Pooled Output'
        },
        'dimension_calculator': dimension_info,
        'pooling_windows_visualization': pooling_windows,
        'comparison_charts': comparison_data
    }

    # Get model info
    model_info = model.get_model_info()

    execution_time_ms = (time.time() - start_time) * 1000

    return {
        'success': True,
        'input_feature_map': input_feature_map.tolist(),
        'pooled_output': pooled_output.tolist(),
        'max_positions': max_positions.tolist() if max_positions is not None else None,
        'dimension_info': dimension_info,
        'pooling_windows': pooling_windows,
        'comparison_data': comparison_data,
        'visualization_data': visualization_data,
        'execution_time_ms': execution_time_ms,
        'model_info': model_info,
        'parameters_used': {
            'pool_type': pool_type,
            'pool_size': pool_size,
            'stride': stride,
            'padding': padding,
            'input_size': input_size,
            'num_channels': num_channels,
            'random_state': random_state
        }
    }
