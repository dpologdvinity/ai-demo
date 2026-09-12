# Image Classification using CNN

This module implements image classification using pre-trained Convolutional Neural Networks (CNNs) from PyTorch/torchvision.

## Overview

The image classification algorithm uses pre-trained models (ResNet-18, ResNet-50, MobileNetV2) to classify images into one of 1000 ImageNet classes. The models are trained on the ImageNet dataset and provide high-accuracy predictions for a wide variety of objects.

## Features

- **Multiple Model Options**: Choose between ResNet-18 (fast), ResNet-50 (accurate), or MobileNetV2 (fastest)
- **Top-K Predictions**: Get the top N most confident predictions
- **Confidence Threshold**: Filter predictions by minimum confidence
- **15 Sample Images**: Pre-selected images across various categories
- **Rich Visualization Data**: Includes bar charts, confidence colors, and prediction statistics

## Architecture

### Models Available

1. **ResNet-18** (Default)
   - 18 layers deep
   - ~11M parameters
   - Good balance of speed and accuracy
   - Uses residual connections

2. **ResNet-50**
   - 50 layers deep
   - ~25M parameters
   - Higher accuracy, slower inference
   - Bottleneck architecture

3. **MobileNetV2**
   - Efficient mobile architecture
   - ~3.5M parameters
   - Fastest inference
   - Inverted residual structure with linear bottlenecks

### Processing Pipeline

1. **Image Loading**: Load image from file
2. **Preprocessing**:
   - Resize to 256x256
   - Center crop to 224x224
   - Convert to tensor
   - Normalize with ImageNet mean/std
3. **Inference**: Forward pass through CNN
4. **Post-processing**:
   - Apply softmax to get probabilities
   - Extract top-K predictions
   - Filter by confidence threshold
5. **Visualization**: Generate bar charts and statistics

## API Endpoints

### POST `/computer-vision/image-classification/classify`

Classify an image using a pre-trained CNN.

**Request Body**:
```json
{
  "model_name": "resnet18",
  "top_k": 5,
  "confidence_threshold": 0.1,
  "image_index": 0
}
```

**Response**:
```json
{
  "success": true,
  "predictions": [
    {
      "class_name": "tabby cat",
      "class_id": 281,
      "confidence": 0.8234,
      "probability": 82.34
    }
  ],
  "statistics": {
    "total_predictions": 5,
    "top_confidence": 0.8234,
    "confidence_spread": 0.6123,
    "entropy": 2.34
  },
  "visualization_data": {
    "image": "data:image/jpeg;base64,...",
    "bar_chart_data": [...],
    "confidence_colors": ["high", "high", "medium", "low", "low"]
  },
  "execution_time_ms": 123.45,
  "model_info": {...},
  "parameters_used": {...},
  "image_info": {...}
}
```

### GET `/computer-vision/image-classification/info`

Get algorithm metadata, available models, and sample images.

## Sample Images

The module includes 15 diverse sample images:

0. Cat portrait
1. Golden Labrador
2. Commercial airplane
3. Red sports car
4. Goose with goslings
5. Koala on tree
6. Green apple
7. Red apple
8. Red double-decker bus
9. Cup of coffee
10. Giant panda
11. African elephant
12. Strawberry
13. Margherita pizza
14. Bunch of bananas

## Parameters

### model_name
- **Type**: Select
- **Default**: `resnet18`
- **Options**: `resnet18`, `resnet50`, `mobilenet_v2`
- **Description**: Pre-trained model architecture

### top_k
- **Type**: Number
- **Default**: 5
- **Range**: 1-10
- **Description**: Number of top predictions to return

### confidence_threshold
- **Type**: Range
- **Default**: 0.1
- **Range**: 0.0-1.0
- **Step**: 0.05
- **Description**: Minimum confidence for predictions

### image_index
- **Type**: Number
- **Default**: 0
- **Range**: 0-14
- **Description**: Sample image to classify

## Statistics

The response includes several statistics:

- **total_predictions**: Number of predictions above threshold
- **top_confidence**: Highest confidence score
- **confidence_spread**: Difference between top and bottom confidence
- **entropy**: Prediction uncertainty measure (higher = more uncertain)

## Visualization

The visualization data includes:

1. **Image**: Base64-encoded original image
2. **Prediction List**: Class names with confidence percentages
3. **Bar Chart Data**: Data for horizontal bar chart
4. **Confidence Colors**: Color coding for predictions
   - Green: High confidence (≥70%)
   - Yellow: Medium confidence (40-70%)
   - Red: Low confidence (<40%)

## Use Cases

1. **Photo Organization**: Automatically tag and organize photos
2. **Medical Diagnosis**: Classify medical images (X-rays, MRIs)
3. **Quality Control**: Detect defects in manufacturing
4. **Wildlife Monitoring**: Identify animals in camera trap images
5. **Content Moderation**: Flag inappropriate images

## Theory

Convolutional Neural Networks learn hierarchical feature representations through layers of convolutions, pooling, and fully connected operations. ResNet uses residual connections (skip connections) to enable training of very deep networks, while MobileNetV2 uses efficient depthwise separable convolutions.

### Key Concepts

- **Convolution**: Apply learned filters to extract features
- **Pooling**: Downsample to reduce spatial dimensions
- **Residual Connections**: Skip connections that help gradient flow
- **Transfer Learning**: Use pre-trained weights from ImageNet
- **Softmax**: Convert logits to probabilities

## Complexity

- **Time Complexity**: O(image_size × filters × layers)
- **Space Complexity**: O(model_params)
- **Typical Inference Time**: 50-200ms (CPU), 10-50ms (GPU)

## Dependencies

- `torch>=2.0.0`: PyTorch deep learning framework
- `torchvision>=0.15.0`: Pre-trained models and transforms
- `pillow>=9.0.0`: Image loading and processing
- `numpy>=1.20.0`: Numerical operations

## Files

- `model.py`: Main model implementation
- `schema.py`: Pydantic schemas for request/response
- `data.py`: Sample image management and dataset info
- `__init__.py`: Module exports

## Testing

Run the test script to verify the implementation:

```bash
cd backend
python test_image_classification.py
```

This will test:
- ResNet-18 model
- MobileNetV2 model
- Different sample images
- Metadata retrieval

## Notes

- Models are downloaded automatically on first use (~50-100MB per model)
- Images are downloaded and cached in `data/` directory
- GPU is used automatically if available
- ImageNet preprocessing is applied (resize, crop, normalize)
