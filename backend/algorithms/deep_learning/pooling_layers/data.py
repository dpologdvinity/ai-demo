"""Data utilities for Pooling Layers demonstration."""

from typing import Dict, Any


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the pooling demonstration dataset.

    Returns:
        Dictionary containing dataset information
    """
    return {
        "name": "Sample Feature Maps",
        "description": (
            "Randomly generated feature maps for demonstrating pooling operations. "
            "Values are sampled from a normal distribution and clipped to [0, 10] "
            "to simulate typical CNN activation values."
        ),
        "type": "synthetic",
        "input_format": "Feature maps with shape [channels, height, width]",
        "value_range": "[0, 10]",
        "default_size": "8×8",
        "channels": "1-3 channels supported",
        "use_cases": [
            "Understanding pooling operations in CNNs",
            "Visualizing dimension reduction",
            "Comparing max vs average pooling",
            "Demonstrating global pooling"
        ]
    }
