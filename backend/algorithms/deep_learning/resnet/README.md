# ResNet (Residual Network) Module

## Quick Start

```python
from algorithms.deep_learning.resnet import ResNetModel, ResNetRequest

# Initialize model
model = ResNetModel(model_variant='resnet18', use_pretrained=True)

# Run inference
request = ResNetRequest(model_variant='resnet18', top_k=5, image_index=0)
response = model.run_inference(request)

# Access predictions
for pred in response.predictions:
    print(f"{pred.class_name}: {pred.confidence:.2%}")
```

## Module Structure

```
resnet/
├── __init__.py          # Module exports
├── model.py             # ResNet model implementation
├── schema.py            # Pydantic schemas (request/response)
├── data.py              # Data utilities (labels, samples)
└── README.md            # This file
```

## Key Classes

### ResNetModel
Main model class for ResNet inference.

**Methods:**
- `predict(image, top_k)` - Run inference on an image
- `extract_feature_maps()` - Get feature maps from residual layers
- `get_residual_blocks_info()` - Get residual block architecture details
- `get_model_info()` - Get model metadata and parameter counts
- `run_inference(request)` - Complete inference pipeline

### ResNetRequest
Pydantic schema for API requests.

**Fields:**
- `model_variant` (str): resnet18, resnet34, resnet50, resnet101
- `top_k` (int): Number of top predictions (1-10)
- `use_pretrained` (bool): Use pre-trained ImageNet weights
- `image_index` (int): Sample image index (0-9)
- `random_state` (int): Random seed

### ResNetResponse
Pydantic schema for API responses.

**Fields:**
- `success` (bool): Inference success status
- `predictions` (List[PredictionResult]): Top-K predictions
- `input_image` (List): Input image data for visualization
- `input_shape` (List[int]): Image shape [H, W, C]
- `feature_maps` (Dict): Extracted feature maps
- `visualization_data` (Dict): Frontend visualization data
- `execution_time_ms` (float): Inference execution time
- `model_info` (Dict): Model architecture information
- `parameters_used` (Dict): Request parameters

## Model Variants

| Variant   | Depth | Parameters | Block Type | Use Case |
|-----------|-------|------------|------------|----------|
| resnet18  | 18    | 11.7M      | Basic      | Fast inference, prototyping |
| resnet34  | 34    | 21.8M      | Basic      | Better accuracy, still fast |
| resnet50  | 50    | 25.6M      | Bottleneck | High accuracy, production |
| resnet101 | 101   | 44.5M      | Bottleneck | Maximum accuracy |

**Recommendation:** Use ResNet-18 for demos and prototyping, ResNet-50 for production.

## ImageNet Labels

The model uses ImageNet-1K labels (1000 classes):
- Animals (mammals, birds, fish, insects)
- Objects (furniture, vehicles, tools)
- Plants (flowers, trees)
- Scenes (landscapes, buildings)
- Food items

Labels are automatically loaded from `data.py`.

## Sample Images

10 sample images are provided for testing:
- Index 0: Cat (orange tabby)
- Index 1: Dog (golden retriever)
- Index 2: Bird (blue bird)
- Index 3: Car (red sports car)
- Index 4: Flower (pink rose)
- Index 5: Landscape (green forest)
- Index 6: Food (pizza)
- Index 7: Person (portrait)
- Index 8: Building (architecture)
- Index 9: Animal (wild animal)

Access via: `get_sample_images()` in `data.py`

## Feature Extraction

Feature maps are extracted from 4 key layers:
- **layer1**: 64 channels, 56×56 spatial
- **layer2**: 128 channels, 28×28 spatial
- **layer3**: 256 channels, 14×14 spatial
- **layer4**: 512 channels, 7×7 spatial

These correspond to the 4 residual block groups in ResNet.

## Residual Block Structure

**Basic Block** (ResNet-18/34):
```
x --> Conv3x3 --> BN --> ReLU --> Conv3x3 --> BN --> (+) --> ReLU
|                                                      ^
+------------------------------------------------------+
```

**Bottleneck Block** (ResNet-50/101):
```
x --> Conv1x1 --> BN --> ReLU --> Conv3x3 --> BN --> ReLU --> Conv1x1 --> BN --> (+) --> ReLU
|                                                                              ^
+------------------------------------------------------------------------------+
```

## Image Preprocessing

Input images are preprocessed following ImageNet standards:
1. Resize to 256×256
2. Center crop to 224×224
3. Convert to tensor (0-1 range)
4. Normalize with ImageNet mean/std:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

## API Integration

The module integrates with FastAPI routes in `/api/deep-learning/resnet/`:

**Endpoints:**
- `POST /predict` - Run inference
- `GET /info` - Get algorithm metadata

See `api/routes/deep_learning.py` for implementation.

## Testing

Run tests with:
```bash
cd backend
source venv/bin/activate
python test_resnet_basic.py
```

Tests cover:
- Model initialization
- Prediction generation
- Feature extraction
- Residual block analysis
- Full inference pipeline

## Performance

**GPU (CUDA):**
- ResNet-18: ~18ms per image
- ResNet-50: ~35ms per image

**CPU:**
- ResNet-18: ~200ms per image
- ResNet-50: ~500ms per image

Note: First inference may be slower due to model initialization.

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'torch'`
**Solution:** Install PyTorch: `pip install torch torchvision`

**Issue:** Out of memory error
**Solution:** Use smaller model variant (resnet18) or reduce batch size

**Issue:** Slow inference
**Solution:** Ensure CUDA is available, or use CPU with smaller model

**Issue:** Predictions are random/wrong
**Solution:** Ensure `use_pretrained=True` for ImageNet weights

## References

- **Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Published:** CVPR 2016
- **Citation:** He et al., "Deep Residual Learning for Image Recognition," 2016

## License

Implementation uses torchvision models (BSD 3-Clause License).
Pre-trained weights are provided by PyTorch/torchvision.
