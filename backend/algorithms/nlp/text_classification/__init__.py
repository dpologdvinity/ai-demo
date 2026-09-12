"""Text Classification algorithm module."""

from .model import TextClassificationModel
from .schema import (
    TextClassificationParameters,
    TextClassificationResponse,
    TextPrediction,
    ClassMetrics,
    TopFeature
)
from .data import (
    get_default_dataset,
    get_category_descriptions,
    prepare_custom_dataset,
    get_dataset_info
)

__all__ = [
    "TextClassificationModel",
    "TextClassificationParameters",
    "TextClassificationResponse",
    "TextPrediction",
    "ClassMetrics",
    "TopFeature",
    "get_default_dataset",
    "get_category_descriptions",
    "prepare_custom_dataset",
    "get_dataset_info"
]
