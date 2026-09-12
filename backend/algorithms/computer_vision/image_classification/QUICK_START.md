# Image Classification - Quick Start Guide

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Dependencies already in requirements.txt:
- torch==2.4.1
- torchvision==0.19.1
- pillow==10.4.0
- numpy==2.1.1

## Basic Usage

### Python Code

```python
from algorithms.computer_vision.image_classification import (
    ImageClassificationModel,
    ImageClassificationRequest
)

# Create request
request = ImageClassificationRequest(
    model_name="resnet18",      # or "resnet50", "mobilenet_v2"
    top_k=5,                    # top 5 predictions
    confidence_threshold=0.1,   # minimum 10% confidence
    image_index=0               # cat image
)

# Create model and classify
model = ImageClassificationModel(model_name=request.model_name)
response = model.process_request(request)

# Print results
print(f"Execution time: {response.execution_time_ms:.2f}ms")
for pred in response.predictions:
    print(f"{pred.class_name}: {pred.probability:.2f}%")
```

### API Request

```bash
# Classify image
curl -X POST "http://localhost:8000/computer-vision/image-classification/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet18",
    "top_k": 5,
    "confidence_threshold": 0.1,
    "image_index": 0
  }'

# Get algorithm info
curl "http://localhost:8000/computer-vision/image-classification/info"
```

## Model Options

### ResNet-18 (Default)
- **Speed**: Fast
- **Accuracy**: Good
- **Parameters**: ~11M
- **Use for**: General purpose, balanced performance

```python
model = ImageClassificationModel(model_name="resnet18")
```

### ResNet-50
- **Speed**: Slower
- **Accuracy**: Best
- **Parameters**: ~25M
- **Use for**: Maximum accuracy, less time-critical

```python
model = ImageClassificationModel(model_name="resnet50")
```

### MobileNetV2
- **Speed**: Fastest
- **Accuracy**: Good
- **Parameters**: ~3.5M
- **Use for**: Real-time, mobile/edge devices

```python
model = ImageClassificationModel(model_name="mobilenet_v2")
```

## Sample Images

```python
# Available images (0-14):
0  - cat.jpg          # Domestic cat portrait
1  - dog.jpg          # Golden Labrador
2  - airplane.jpg     # Commercial airplane
3  - car.jpg          # Red sports car
4  - bird.jpg         # Goose with goslings
5  - koala.jpg        # Koala on tree
6  - apple.jpg        # Green apple
7  - apple2.jpg       # Red apple
8  - bus.jpg          # Double-decker bus
9  - coffee.jpg       # Cup of coffee
10 - panda.jpg        # Giant panda
11 - elephant.jpg     # African elephant
12 - strawberry.jpg   # Fresh strawberry
13 - pizza.jpg        # Margherita pizza
14 - banana.jpg       # Bunch of bananas
```

## Response Structure

```python
response = model.process_request(request)

# Top predictions
for pred in response.predictions:
    print(f"{pred.class_name}: {pred.confidence:.4f}")
    # pred.class_id: ImageNet class ID
    # pred.probability: Percentage (0-100)

# Statistics
stats = response.statistics
print(f"Total predictions: {stats.total_predictions}")
print(f"Top confidence: {stats.top_confidence:.4f}")
print(f"Confidence spread: {stats.confidence_spread:.4f}")
print(f"Entropy: {stats.entropy:.2f}")

# Model info
info = response.model_info
print(f"Model: {info['model_name']}")
print(f"Device: {info['device']}")
print(f"Parameters: {info['total_parameters']:,}")

# Visualization data
viz = response.visualization_data
# viz['image']: base64 image
# viz['bar_chart_data']: for plotting
# viz['confidence_colors']: color coding
```

## Testing

Run the test script:

```bash
cd backend
python test_image_classification.py
```

Expected output:
- ResNet-18 test with cat image
- MobileNetV2 test with dog image
- ResNet-18 test with airplane image
- Metadata validation

## Common Tasks

### Classify with High Confidence Only

```python
request = ImageClassificationRequest(
    model_name="resnet18",
    top_k=3,
    confidence_threshold=0.5,  # 50% minimum
    image_index=0
)
```

### Get More Predictions

```python
request = ImageClassificationRequest(
    model_name="resnet18",
    top_k=10,  # up to 10 predictions
    confidence_threshold=0.01,  # lower threshold
    image_index=0
)
```

### Use Fastest Model

```python
request = ImageClassificationRequest(
    model_name="mobilenet_v2",  # fastest
    top_k=5,
    confidence_threshold=0.1,
    image_index=0
)
```

### Use Most Accurate Model

```python
request = ImageClassificationRequest(
    model_name="resnet50",  # most accurate
    top_k=5,
    confidence_threshold=0.1,
    image_index=0
)
```

## Performance Tips

1. **Model Loading**: Models are loaded once and cached
2. **GPU Usage**: Automatically uses CUDA if available
3. **Image Caching**: Sample images are downloaded once and cached
4. **Batch Processing**: Process multiple images with same model

```python
# Efficient: reuse model
model = ImageClassificationModel(model_name="resnet18")

for image_idx in range(5):
    request = ImageClassificationRequest(
        model_name="resnet18",
        image_index=image_idx
    )
    response = model.process_request(request)
    # Process response...
```

## Troubleshooting

### Model Download Issues
Models are downloaded automatically from PyTorch Hub. If download fails:
- Check internet connection
- Models are ~50-100MB each
- Stored in `~/.cache/torch/hub/checkpoints/`

### Out of Memory
If you get OOM errors:
- Use MobileNetV2 (smallest model)
- Ensure no other GPU processes running
- Reduce batch processing

### Slow Inference
- First run is slower (model download + loading)
- Use GPU if available (10x faster)
- Use MobileNetV2 for fastest inference
- Subsequent runs are faster (model cached)

## API Endpoints

### POST /computer-vision/image-classification/classify
Classify an image

**Request**: `ImageClassificationRequest`
**Response**: `ImageClassificationResponse`
**Status**: 200 (success), 400 (invalid params), 500 (error)

### GET /computer-vision/image-classification/info
Get algorithm metadata

**Response**: Algorithm metadata, dataset info, model variants
**Status**: 200 (success), 404 (not found)

## File Locations

```
backend/
├── algorithms/computer_vision/image_classification/
│   ├── __init__.py      # Module exports
│   ├── schema.py        # Pydantic schemas
│   ├── data.py          # Sample images
│   ├── model.py         # CNN implementation
│   ├── README.md        # Full documentation
│   ├── QUICK_START.md   # This file
│   └── data/            # Cached images (auto-created)
├── api/routes/
│   └── computer_vision.py  # API endpoints
└── test_image_classification.py  # Test script
```

## Next Steps

1. Start the backend server: `uvicorn main:app --reload`
2. Test the API: `curl http://localhost:8000/computer-vision/image-classification/info`
3. Integrate with frontend
4. Add custom images (modify `data.py`)
5. Experiment with different models and parameters
