# VGG Network Implementation

## Overview
VGG (Visual Geometry Group) networks are deep convolutional neural networks characterized by their use of very small (3×3) convolution filters throughout the entire architecture. This implementation provides VGG-16 and VGG-19 variants for ImageNet classification.

## Files Structure

```
vgg/
├── __init__.py          # Module exports
├── schema.py            # Pydantic request/response schemas
├── data.py             # ImageNet labels and data utilities
├── model.py            # VGG model implementation
└── README.md           # This file
```

## Quick Start

### Basic Usage

```python
from algorithms.deep_learning.vgg import VGGModel, VGGRequest

# Create request
request = VGGRequest(
    model_variant="vgg16",
    top_k=5,
    use_pretrained=True,
    batch_norm=True,
    image_index=0
)

# Initialize model
model = VGGModel(
    model_variant=request.model_variant,
    use_pretrained=request.use_pretrained,
    batch_norm=request.batch_norm
)

# Run inference
response = model.run_inference(request)

# Access results
print(f"Top prediction: {response.predictions[0].class_name}")
print(f"Confidence: {response.predictions[0].confidence:.2%}")
```

### API Endpoints

#### Predict
```bash
POST /api/deep-learning/vgg/predict
Content-Type: application/json

{
  "model_variant": "vgg16",
  "top_k": 5,
  "use_pretrained": true,
  "batch_norm": true,
  "image_index": 0
}
```

#### Get Info
```bash
GET /api/deep-learning/vgg/info
```

## Model Variants

### VGG-16
- **Depth**: 16 layers (13 conv + 3 FC)
- **Parameters**: ~138M
- **Structure**: [2, 2, 3, 3, 3] conv layers per block

### VGG-19
- **Depth**: 19 layers (16 conv + 3 FC)
- **Parameters**: ~144M
- **Structure**: [2, 2, 4, 4, 4] conv layers per block

### Batch Normalization
Both variants support batch normalization versions:
- `vgg16` / `vgg16_bn`
- `vgg19` / `vgg19_bn`

## Architecture Details

### Convolutional Blocks
```
Block 1: Input(224×224×3) → Conv3×3(64) → Conv3×3(64) → MaxPool → (112×112×64)
Block 2: Conv3×3(128) → Conv3×3(128) → MaxPool → (56×56×128)
Block 3: Conv3×3(256) × 3 → MaxPool → (28×28×256)
Block 4: Conv3×3(512) × 3 → MaxPool → (14×14×512)
Block 5: Conv3×3(512) × 3 → MaxPool → (7×7×512)
```

### Fully Connected Layers
```
Flatten → FC(4096) → ReLU → Dropout(0.5)
       → FC(4096) → ReLU → Dropout(0.5)
       → FC(1000)
```

## Features

### Feature Maps
Extracts activations from the end of each convolutional block:
- `block1`: After 2nd conv layer (64 channels)
- `block2`: After 2nd conv layer (128 channels)
- `block3`: After 3rd/4th conv layer (256 channels)
- `block4`: After 3rd/4th conv layer (512 channels)
- `block5`: After 3rd/4th conv layer (512 channels)

### Visualization Data
- Top-K predictions with confidence scores
- Feature maps from all 5 blocks
- Convolutional block information
- Architecture diagram
- Model statistics

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_variant | str | "vgg16" | VGG variant (vgg16 or vgg19) |
| top_k | int | 5 | Number of top predictions (1-10) |
| use_pretrained | bool | True | Use pre-trained ImageNet weights |
| batch_norm | bool | True | Use batch normalization variant |
| image_index | int | 0 | Sample image index (0-9) |
| image_path | str | None | Custom image file path |

## Response Schema

```python
{
    "success": bool,
    "predictions": [
        {
            "class_id": int,
            "class_name": str,
            "confidence": float
        }
    ],
    "input_image": List[List[List[float]]],  # H×W×C normalized
    "input_shape": [H, W, C],
    "feature_maps": {
        "block1": {
            "shape": [channels, height, width],
            "avg_activation": List[List[float]],
            "num_channels": int,
            "sample_channels": List[...]
        },
        ...
    },
    "conv_blocks": [
        {
            "block_name": str,
            "num_layers": int,
            "num_filters": int,
            "kernel_size": int,
            "has_pooling": bool
        }
    ],
    "visualization_data": {...},
    "execution_time_ms": float,
    "model_info": {
        "model_variant": str,
        "batch_norm": bool,
        "depth": int,
        "num_conv_layers": int,
        "total_parameters": int,
        "trainable_parameters": int,
        "fc_parameters": int,
        "conv_parameters": int,
        "device": str,
        "input_size": "224x224",
        "num_classes": 1000,
        "kernel_size": "3x3 (all conv layers)",
        "pooling": "MaxPool 2x2"
    },
    "parameters_used": {...}
}
```

## ImageNet Preprocessing

All images are preprocessed using ImageNet normalization:
```python
# Resize to 256×256
# Center crop to 224×224
# Normalize with:
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## Sample Images

The implementation includes 10 sample images for testing:
0. Cat (Orange tabby)
1. Dog (Golden retriever)
2. Bird (Blue bird)
3. Car (Red sports car)
4. Flower (Pink rose)
5. Landscape (Green forest)
6. Food (Pizza)
7. Person (Portrait)
8. Building (Modern architecture)
9. Animal (Wild animal)

## Theory

### Why 3×3 Filters?
- Smallest size to capture spatial patterns (left/right, up/down, center)
- Stacking multiple 3×3 layers achieves larger receptive fields
- Fewer parameters than single large filters
- More non-linearity (more ReLU activations)

### Receptive Field
- 1 conv layer (3×3): 3×3 receptive field
- 2 conv layers (3×3): 5×5 receptive field
- 3 conv layers (3×3): 7×7 receptive field

### Advantages
- Simple, uniform architecture
- Deep networks (16-19 layers)
- Strong feature extraction
- Excellent for transfer learning
- Good ImageNet performance (92.7% top-5)

### Limitations
- Large model size (~138M parameters)
- Most parameters in FC layers
- Slow inference vs modern architectures
- No skip connections (vanishing gradient issues)

## Use Cases

1. **Image Classification**: Direct ImageNet classification
2. **Transfer Learning**: Pre-trained features for other tasks
3. **Style Transfer**: Deep features for style representation
4. **Feature Extraction**: Conv5 features for similarity search
5. **Object Detection**: Backbone for detection networks

## Dependencies

- PyTorch (torch)
- torchvision (models, transforms)
- PIL/Pillow (image handling)
- NumPy (array operations)

## References

- Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2014)
- arXiv: https://arxiv.org/abs/1409.1556
- Original implementation: https://www.robots.ox.ac.uk/~vgg/research/very_deep/

## Performance

### VGG-16 ImageNet Results
- Top-1 Accuracy: 71.5%
- Top-5 Accuracy: 90.1%

### VGG-19 ImageNet Results
- Top-1 Accuracy: 71.1%
- Top-5 Accuracy: 90.0%

(Note: Batch normalization variants typically improve accuracy by 1-2%)

## Notes

- The implementation uses torchvision's pre-trained models
- Feature maps are extracted using forward hooks
- All convolutional layers use 3×3 kernels with stride 1, padding 1
- All max pooling layers use 2×2 windows with stride 2
- ReLU activation is used throughout
- Batch normalization is applied after conv, before ReLU (when enabled)
