"""Image Classification model implementation using PyTorch."""

import time
import base64
import json
from io import BytesIO
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from .schema import (
    ImageClassificationRequest,
    ImageClassificationResponse,
    Prediction,
    PredictionStatistics
)
from .data import download_sample_image, get_dataset_info


# ImageNet class labels
def load_imagenet_labels() -> Dict[int, str]:
    """Load ImageNet class labels.

    Returns:
        Dictionary mapping class IDs to class names
    """
    # This is a subset of ImageNet classes for demonstration
    # In production, you'd load the full 1000-class mapping
    labels = {
        0: "tench", 1: "goldfish", 2: "great white shark", 3: "tiger shark",
        4: "hammerhead", 5: "electric ray", 6: "stingray", 7: "rooster",
        8: "hen", 9: "ostrich", 10: "brambling", 11: "goldfinch", 12: "house finch",
        13: "junco", 14: "indigo bunting", 15: "robin", 16: "bulbul",
        17: "jay", 18: "magpie", 19: "chickadee", 20: "water ouzel",
        144: "pelican", 145: "king penguin", 146: "albatross",
        207: "golden retriever", 208: "Labrador retriever", 209: "cocker spaniel",
        281: "tabby cat", 282: "tiger cat", 283: "Persian cat", 284: "Siamese cat",
        285: "Egyptian cat", 286: "mountain lion", 287: "lynx", 288: "leopard",
        289: "snow leopard", 290: "jaguar", 291: "lion", 292: "tiger",
        293: "cheetah", 294: "brown bear", 295: "American black bear",
        296: "ice bear", 297: "sloth bear", 298: "mongoose", 299: "meerkat",
        300: "tiger beetle", 301: "ladybug", 302: "ground beetle",
        386: "African elephant", 387: "Indian elephant",
        388: "lesser panda", 389: "giant panda",
        404: "airliner", 405: "warplane", 406: "space shuttle", 407: "balloon",
        408: "airship", 409: "ambulance", 410: "amphibian", 411: "analog clock",
        468: "bullet train", 469: "cab", 470: "cab", 471: "convertible",
        474: "crane", 475: "croquet ball", 476: "crutch", 477: "dam",
        511: "convertible", 555: "fire engine", 556: "garbage truck",
        565: "golf cart", 566: "gondola", 569: "Grand Piano",
        609: "hook", 614: "iPod", 627: "lawn mower", 628: "lens cap",
        654: "military uniform", 656: "miniskirt", 661: "Model T",
        670: "mountain bike", 671: "mountain tent", 672: "mouse",
        675: "nail", 717: "pickup truck", 734: "pole", 751: "racer",
        779: "school bus", 829: "streetcar", 847: "tank",
        864: "tow truck", 866: "trailer truck", 867: "tray",
        874: "trolleybus", 895: "warplane", 947: "strawberry",
        948: "orange", 949: "lemon", 950: "fig", 951: "pineapple",
        952: "banana", 953: "jackfruit", 954: "custard apple",
        955: "pomegranate", 956: "hay", 957: "carbonara", 958: "chocolate sauce",
        959: "dough", 960: "meat loaf", 961: "pizza", 962: "potpie",
        963: "burrito", 964: "red wine", 965: "espresso", 966: "cup",
        967: "eggnog", 968: "alp", 969: "bubble", 970: "cliff"
    }

    # Fill in missing indices with placeholder
    full_labels = {}
    for i in range(1000):
        full_labels[i] = labels.get(i, f"class_{i}")

    return full_labels


class ImageClassificationModel:
    """Image Classification model using pre-trained CNNs.

    This class provides a wrapper around PyTorch pre-trained models
    for image classification tasks.

    Attributes:
        model: The PyTorch model instance
        model_name: Name of the model (resnet18, resnet50, mobilenet_v2)
        class_labels: ImageNet class labels
        transform: Image preprocessing transform
        device: Device to run inference on (CPU or CUDA)
    """

    def __init__(self, model_name: str = 'resnet18'):
        """Initialize the image classification model.

        Args:
            model_name: Model to use - 'resnet18', 'resnet50', or 'mobilenet_v2'

        Raises:
            ValueError: If model_name is invalid
        """
        if model_name not in ['resnet18', 'resnet50', 'mobilenet_v2']:
            raise ValueError("model_name must be 'resnet18', 'resnet50', or 'mobilenet_v2'")

        self.model_name = model_name
        self.model = None
        self.class_labels = load_imagenet_labels()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ImageNet preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_model(self):
        """Load the pre-trained model.

        Loads the model on first use to avoid unnecessary initialization.
        """
        if self.model is None:
            try:
                print(f"Loading model: {self.model_name}")

                if self.model_name == 'resnet18':
                    self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                elif self.model_name == 'resnet50':
                    self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                elif self.model_name == 'mobilenet_v2':
                    self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

                self.model.to(self.device)
                self.model.eval()

                print(f"Model loaded successfully on {self.device}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")

    def classify(
        self,
        image_path: str,
        top_k: int = 5,
        confidence_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Run classification on an image.

        Args:
            image_path: Path to the input image
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence for predictions

        Returns:
            Dictionary containing classification results and metadata

        Raises:
            RuntimeError: If classification fails
        """
        # Load model if not already loaded
        self.load_model()

        start_time = time.time()

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_image = np.array(image)

            # Transform for model input
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs[0], dim=0)

            # Get top-K predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, 1000))

            # Parse predictions
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                prob_value = float(prob.cpu().numpy())
                idx_value = int(idx.cpu().numpy())

                # Apply confidence threshold
                if prob_value >= confidence_threshold:
                    predictions.append({
                        'class_name': self.class_labels[idx_value],
                        'class_id': idx_value,
                        'confidence': prob_value,
                        'probability': prob_value * 100
                    })

            classification_time = (time.time() - start_time) * 1000

            return {
                'predictions': predictions,
                'original_image': original_image,
                'image_shape': original_image.shape,
                'classification_time_ms': classification_time,
                'all_probabilities': probabilities.cpu().numpy()
            }

        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")

    def process_request(
        self,
        request: ImageClassificationRequest
    ) -> ImageClassificationResponse:
        """Process an image classification request.

        Args:
            request: ImageClassificationRequest with classification parameters

        Returns:
            ImageClassificationResponse with classification results

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If classification fails
        """
        start_time = time.time()

        # Download sample image
        try:
            image_path = download_sample_image(request.image_index)
        except Exception as e:
            raise ValueError(f"Failed to load sample image: {str(e)}")

        # Update model if needed
        if self.model_name != request.model_name:
            self.model_name = request.model_name
            self.model = None  # Force reload

        # Run classification
        classification_result = self.classify(
            image_path,
            top_k=request.top_k,
            confidence_threshold=request.confidence_threshold
        )

        # Parse predictions
        predictions = [
            Prediction(**pred) for pred in classification_result['predictions']
        ]

        # Calculate statistics
        statistics = self._calculate_statistics(
            predictions,
            classification_result['all_probabilities']
        )

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            classification_result['original_image'],
            predictions
        )

        # Get model info
        model_info = self._get_model_info()

        # Get image info
        image_shape = classification_result['image_shape']
        image_info = {
            'width': int(image_shape[1]),
            'height': int(image_shape[0]),
            'channels': int(image_shape[2]),
            'image_index': request.image_index
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return ImageClassificationResponse(
            success=True,
            predictions=predictions,
            statistics=statistics,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'model_name': request.model_name,
                'top_k': request.top_k,
                'confidence_threshold': request.confidence_threshold,
                'image_index': request.image_index
            },
            image_info=image_info
        )

    def _calculate_statistics(
        self,
        predictions: List[Prediction],
        all_probabilities: np.ndarray
    ) -> PredictionStatistics:
        """Calculate statistics from predictions.

        Args:
            predictions: List of Prediction objects
            all_probabilities: All class probabilities

        Returns:
            PredictionStatistics object
        """
        if not predictions:
            return PredictionStatistics(
                total_predictions=0,
                top_confidence=0.0,
                confidence_spread=0.0,
                entropy=0.0
            )

        confidences = [pred.confidence for pred in predictions]

        # Calculate entropy (uncertainty measure)
        # Higher entropy = more uncertain
        epsilon = 1e-10
        probs = all_probabilities + epsilon
        entropy = -np.sum(probs * np.log2(probs))

        return PredictionStatistics(
            total_predictions=len(predictions),
            top_confidence=max(confidences),
            confidence_spread=max(confidences) - min(confidences),
            entropy=float(entropy)
        )

    def _prepare_visualization_data(
        self,
        original_image: np.ndarray,
        predictions: List[Prediction]
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            original_image: Original input image
            predictions: List of Prediction objects

        Returns:
            Dictionary with visualization data
        """
        # Convert image to base64 for frontend
        image_base64 = self._image_to_base64(original_image)

        # Prepare prediction list for display
        prediction_list = [
            {
                'class_name': pred.class_name,
                'confidence': round(pred.confidence, 4),
                'probability': round(pred.probability, 2)
            }
            for pred in predictions
        ]

        # Prepare bar chart data
        bar_chart_data = [
            {
                'class': pred.class_name,
                'probability': round(pred.probability, 2)
            }
            for pred in predictions
        ]

        # Color coding for confidence levels
        confidence_colors = []
        for pred in predictions:
            if pred.confidence >= 0.7:
                confidence_colors.append('high')  # Green
            elif pred.confidence >= 0.4:
                confidence_colors.append('medium')  # Yellow
            else:
                confidence_colors.append('low')  # Red

        return {
            'image': image_base64,
            'prediction_list': prediction_list,
            'bar_chart_data': bar_chart_data,
            'confidence_colors': confidence_colors
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
        pil_image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {}

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        model_descriptions = {
            'resnet18': 'ResNet-18 (18 layers, ~11M params)',
            'resnet50': 'ResNet-50 (50 layers, ~25M params)',
            'mobilenet_v2': 'MobileNetV2 (Efficient mobile architecture, ~3.5M params)'
        }

        return {
            'model_name': self.model_name,
            'description': model_descriptions[self.model_name],
            'num_classes': 1000,
            'framework': 'PyTorch',
            'device': str(self.device),
            'input_size': '224x224',
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'architecture': self._get_architecture_description()
        }

    def _get_architecture_description(self) -> str:
        """Get architecture description for the current model.

        Returns:
            Architecture description string
        """
        descriptions = {
            'resnet18': 'ResNet-18: Residual connections with 18 layers (4 residual blocks)',
            'resnet50': 'ResNet-50: Deeper residual network with 50 layers (bottleneck design)',
            'mobilenet_v2': 'MobileNetV2: Inverted residual structure with linear bottlenecks'
        }
        return descriptions[self.model_name]

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
                'resnet18': 'ResNet-18 - Fast, good accuracy (11M params)',
                'resnet50': 'ResNet-50 - Higher accuracy, slower (25M params)',
                'mobilenet_v2': 'MobileNetV2 - Fastest, optimized for mobile (3.5M params)'
            }
        }
