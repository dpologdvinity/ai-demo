"""Data utilities for activation functions demonstration."""

from typing import Dict, Any


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the synthetic dataset used for activation functions.

    Returns:
        Dictionary containing dataset metadata
    """
    return {
        'name': 'synthetic_range',
        'description': 'Synthetic data generated over a specified range for visualization',
        'type': 'continuous',
        'features': 'Input values (x)',
        'targets': 'Activation function outputs (y) and derivatives (dy/dx)',
        'default_range': [-10.0, 10.0],
        'default_points': 200,
        'purpose': 'Educational demonstration of activation function properties'
    }
