# Edge Detection - Quick Reference

## API Endpoints

### Detect Edges
```bash
POST http://localhost:8000/computer-vision/edge-detection/detect
Content-Type: application/json

{
  "threshold1": 50,
  "threshold2": 150,
  "aperture_size": 3,
  "l2gradient": false,
  "image_index": 0
}
```

### Get Algorithm Info
```bash
GET http://localhost:8000/computer-vision/edge-detection/info
```

### List All CV Algorithms
```bash
GET http://localhost:8000/computer-vision/algorithms
```

## Python Usage

```python
from algorithms.computer_vision.edge_detection import (
    EdgeDetectionModel,
    EdgeDetectionRequest
)

# Create request
request = EdgeDetectionRequest(
    threshold1=50,
    threshold2=150,
    aperture_size=3,
    l2gradient=False,
    image_index=0
)

# Run detection
model = EdgeDetectionModel()
response = model.process_request(request)

# Access results
print(f"Edge pixels: {response.statistics.edge_pixel_count}")
print(f"Edge density: {response.statistics.edge_density}%")
print(f"Execution time: {response.execution_time_ms}ms")
```

## Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| threshold1 | 0-255 | 50 | Lower hysteresis threshold |
| threshold2 | 0-255 | 150 | Upper hysteresis threshold |
| aperture_size | 3, 5, 7 | 3 | Sobel kernel size |
| l2gradient | true/false | false | Use L2 norm |
| image_index | 0-7 | 0 | Sample image selector |

## Sample Images

| Index | Name | Description |
|-------|------|-------------|
| 0 | bus.jpg | Street scene - Urban elements |
| 1 | zidane.jpg | Sports - Soccer players |
| 2 | lena.jpg | Portrait - Classic test image |
| 3 | building.jpg | Architecture - Geometric patterns |
| 4 | fruits.jpg | Still life - Varied textures |
| 5 | chicky.png | Cartoon - High contrast |
| 6 | messi.jpg | Sports portrait |
| 7 | home.jpg | Indoor scene |

## Preset Configurations

### General Purpose
```json
{"threshold1": 50, "threshold2": 150, "aperture_size": 3}
```

### Fine Details
```json
{"threshold1": 30, "threshold2": 90, "aperture_size": 3}
```

### Major Edges Only
```json
{"threshold1": 100, "threshold2": 200, "aperture_size": 5}
```

## Response Structure

```javascript
{
  success: boolean,
  statistics: {
    edge_pixel_count: number,
    total_pixels: number,
    edge_density: number,      // 0-100%
    image_dimensions: {width, height},
    threshold_ratio: number
  },
  visualization_data: {
    original_image: "data:image/jpeg;base64,...",
    edge_image: "data:image/jpeg;base64,...",
    overlay_image: "data:image/jpeg;base64,...",
    gray_image: "data:image/jpeg;base64,..."
  },
  execution_time_ms: number,
  algorithm_info: {...},
  parameters_used: {...},
  image_info: {...}
}
```

## Testing Commands

### Start Backend Server
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```

### Test with curl
```bash
curl -X POST http://localhost:8000/computer-vision/edge-detection/detect \
  -H "Content-Type: application/json" \
  -d '{"threshold1":50,"threshold2":150,"aperture_size":3,"l2gradient":false,"image_index":0}'
```

### Get Info
```bash
curl http://localhost:8000/computer-vision/edge-detection/info | jq
```

## File Locations

### Backend Module
```
backend/algorithms/computer_vision/edge_detection/
├── __init__.py       # Module exports
├── schema.py         # Request/Response models
├── model.py          # Edge detection implementation
├── data.py           # Sample images
└── data/             # Downloaded images (cached)
```

### API Routes
```
backend/api/routes/computer_vision.py
  - Lines 6-9: Import edge detection module
  - Lines 158-238: Edge detection metadata
  - Lines 535-608: Edge detection endpoints
```

## Error Codes

| Code | Error | Cause |
|------|-------|-------|
| 400 | Bad Request | Invalid parameters (threshold1 >= threshold2, etc.) |
| 404 | Not Found | Algorithm metadata not found |
| 500 | Server Error | OpenCV processing error, image download failure |

## Tips

- **Threshold Ratio**: Keep threshold2/threshold1 between 2:1 and 3:1
- **Lower Thresholds**: More edges, more noise
- **Higher Thresholds**: Fewer edges, only strong ones
- **Aperture Size**: Larger = smoother gradients but slower
- **L2 Gradient**: More accurate but 1.5-2x slower

## Frontend Integration Checklist

- [ ] Create EdgeDetection component
- [ ] Add threshold sliders (1 and 2)
- [ ] Add aperture size dropdown
- [ ] Add L2 gradient checkbox
- [ ] Add image selector/gallery
- [ ] Display original and edge images
- [ ] Show statistics (edge count, density)
- [ ] Add download edge image button
- [ ] Add loading/error states
- [ ] Add algorithm info section
- [ ] Test all 8 sample images
- [ ] Test parameter variations

## Dependencies

Already in requirements.txt:
- opencv-python==4.10.0.84
- pillow==10.4.0
- numpy (via OpenCV)
- pydantic (via FastAPI)

## Verification

Run this to verify installation:
```bash
cd backend
source venv/bin/activate
python3 -c "from api.routes.computer_vision import router; print('✓ Edge detection routes:', [r.path for r in router.routes if 'edge' in r.path.lower()])"
```

Expected output:
```
✓ Edge detection routes: ['/computer-vision/edge-detection/detect', '/computer-vision/edge-detection/info']
```
