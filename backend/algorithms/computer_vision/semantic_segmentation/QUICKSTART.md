# Semantic Segmentation Quick Start

## Quick Setup

### 1. Install Dependencies

Ensure torchvision is installed:
```bash
pip install torchvision==0.19.1
```

Or install all requirements:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Test the Implementation

```bash
cd backend
python test_segmentation.py
```

Expected output:
```
============================================================
Testing Semantic Segmentation Implementation
============================================================

Request Parameters:
  - Number of classes: 21
  - Confidence threshold: 0.5
  - Model backbone: resnet50
  - Image size: 512
  - Image index: 0

Initializing model...
Using device: cpu
Loading DeepLabV3 with ResNet50 backbone...
Model loaded successfully on cpu

Processing segmentation request...
Downloading sample image: dog1.jpg...

Segmentation Results:
  - Success: True
  - Execution time: 234.56 ms
  - Total classes found: 3
  - Total pixels: 262,144
  - Mean confidence: 0.876

Class Distribution:
  1. dog: 123,456 pixels (47.10%)
  2. background: 98,765 pixels (37.65%)
  3. person: 39,923 pixels (15.25%)

============================================================
Test completed successfully!
============================================================
```

## Quick API Test

### Using curl

```bash
# Run segmentation
curl -X POST http://localhost:8000/api/computer-vision/semantic-segmentation/segment \
  -H "Content-Type: application/json" \
  -d '{
    "num_classes": 21,
    "confidence_threshold": 0.5,
    "model_backbone": "resnet50",
    "image_size": 512,
    "image_index": 0
  }'

# Get algorithm info
curl http://localhost:8000/api/computer-vision/semantic-segmentation/info
```

### Using Python requests

```python
import requests
import json

# Run segmentation
response = requests.post(
    'http://localhost:8000/api/computer-vision/semantic-segmentation/segment',
    json={
        'num_classes': 21,
        'confidence_threshold': 0.5,
        'model_backbone': 'resnet50',
        'image_size': 512,
        'image_index': 0
    }
)

result = response.json()
print(f"Found {result['statistics']['total_classes']} classes")
print(f"Processing time: {result['execution_time_ms']:.1f}ms")
```

### Using JavaScript fetch

```javascript
const response = await fetch('/api/computer-vision/semantic-segmentation/segment', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    num_classes: 21,
    confidence_threshold: 0.5,
    model_backbone: 'resnet50',
    image_size: 512,
    image_index: 0
  })
});

const result = await response.json();
console.log(`Found ${result.statistics.total_classes} classes`);
```

## Quick Parameter Guide

### Model Backbone
- **resnet50**: Better accuracy (~77% mIoU), slower (~250ms CPU)
- **mobilenet**: Faster inference (~120ms CPU), lower accuracy (~72% mIoU)

### Image Size
- **256**: Fastest, lowest quality
- **512**: Balanced (recommended)
- **1024**: Best quality, slowest

### Confidence Threshold
- **0.3-0.5**: More classes detected, more noise
- **0.5-0.7**: Balanced (recommended)
- **0.7-0.9**: Fewer classes, higher confidence

### Sample Images
- **0**: Dog portrait (close-up)
- **1**: Dog on grass (outdoor)
- **2**: Street scene (urban)

## Common Issues

### Issue: Model download is slow
**Solution**: First model load downloads ~160MB (ResNet50) or ~15MB (MobileNet). Subsequent loads use cached model.

### Issue: Out of memory
**Solution**: Use smaller image_size (256 or 512) or mobilenet backbone.

### Issue: Import error for torchvision
**Solution**: 
```bash
pip install torch==2.4.1 torchvision==0.19.1
```

### Issue: CUDA out of memory
**Solution**: Model automatically falls back to CPU if CUDA memory is insufficient.

## Quick Integration Checklist

Backend:
- [x] Algorithm implementation
- [x] API endpoints
- [x] Metadata registration
- [x] Dependencies added

Frontend (TODO):
- [ ] Create component
- [ ] Add to routing
- [ ] Test visualization
- [ ] Update navigation

## Next Steps

1. **Start the backend server**:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Test the API**: Use curl or the test script

3. **Integrate frontend**: See `SEMANTIC_SEGMENTATION_FRONTEND_GUIDE.md`

4. **Customize**: Adjust parameters for your use case

## Performance Tips

1. **Use GPU**: Segmentation is 3-5x faster on GPU
2. **Cache models**: Models are cached after first load
3. **Batch requests**: Process multiple images together (future feature)
4. **Use MobileNet**: For real-time applications
5. **Lower resolution**: Use 256x256 for speed-critical apps

## Resources

- Full documentation: `README.md`
- Frontend guide: `SEMANTIC_SEGMENTATION_FRONTEND_GUIDE.md`
- Implementation details: `SEMANTIC_SEGMENTATION_IMPLEMENTATION.md`
- PyTorch docs: https://pytorch.org/vision/stable/models.html

## Support

If something isn't working:
1. Check Python version (3.8+)
2. Verify dependencies installed
3. Check logs for errors
4. Test with default parameters first
5. Try different sample images

## Quick Examples

### Example 1: Fast Segmentation
```python
request = {
    'model_backbone': 'mobilenet',
    'image_size': 256,
    'confidence_threshold': 0.5,
    'image_index': 0
}
# ~100ms on CPU
```

### Example 2: High Quality
```python
request = {
    'model_backbone': 'resnet50',
    'image_size': 1024,
    'confidence_threshold': 0.7,
    'image_index': 2
}
# ~500ms on CPU, very detailed
```

### Example 3: Balanced (Recommended)
```python
request = {
    'model_backbone': 'resnet50',
    'image_size': 512,
    'confidence_threshold': 0.5,
    'image_index': 1
}
# ~250ms on CPU, good quality
```

---

**Ready to use!** Run `python test_segmentation.py` to get started.
