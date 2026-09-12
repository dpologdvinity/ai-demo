# Face Detection Quick Start Guide

## Quick Overview
Face Detection using Haar Cascades has been successfully implemented as a beginner-level computer vision algorithm.

## API Endpoints

### Detect Faces
```bash
curl -X POST "http://localhost:8000/api/computer-vision/face-detection/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_size": 30,
    "max_size": null,
    "image_index": 0
  }'
```

### Get Algorithm Info
```bash
curl "http://localhost:8000/api/computer-vision/face-detection/info"
```

## Key Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| scale_factor | float | 1.05-1.3 | 1.1 | Scale reduction between scans (lower = more accurate) |
| min_neighbors | int | 1-10 | 5 | Minimum neighbors for detection (higher = fewer false positives) |
| min_size | int | 20-100 | 30 | Minimum face size in pixels |
| max_size | int\|null | 50+ | null | Maximum face size (null = no limit) |
| image_index | int | 0-7 | 0 | Sample image selector |

## Sample Images

| Index | Description | Best For |
|-------|-------------|----------|
| 0 | Lena - Classic portrait | Single face detection |
| 1 | Small group (2-3 faces) | Group photos |
| 2 | Family group (3-5 faces) | Multiple faces |
| 3 | Business setting | Professional photos |
| 4 | Outdoor portrait | Natural lighting |
| 5 | Close-up portrait | Large faces |
| 6 | Studio portrait | Professional lighting |
| 7 | Diverse group | Multiple ethnicities |

## Preset Configurations

### High Accuracy
```json
{
  "scale_factor": 1.05,
  "min_neighbors": 6,
  "min_size": 30
}
```

### Balanced (Default)
```json
{
  "scale_factor": 1.1,
  "min_neighbors": 5,
  "min_size": 30
}
```

### Fast Detection
```json
{
  "scale_factor": 1.2,
  "min_neighbors": 3,
  "min_size": 40
}
```

## Response Structure

### Faces Array
Each detected face includes:
- **x, y**: Top-left corner coordinates
- **width, height**: Bounding box dimensions
- **confidence**: Detection confidence (0-1)
- **center**: Face center point {x, y}
- **area**: Face area in pixels

### Statistics
- **face_count**: Number of faces detected
- **average_face_size**: Average face area
- **largest_face_size**: Largest face area
- **smallest_face_size**: Smallest face area
- **total_face_area**: Total area covered by faces
- **face_density**: Percentage of image covered by faces
- **image_dimensions**: Image width and height
- **detection_quality**: "High", "Medium", or "Low"

### Visualization Data
4 base64-encoded images:
1. **original_image**: Unmodified input
2. **annotated_image**: Green bounding boxes with labels
3. **gray_image**: Histogram-equalized grayscale
4. **centers_image**: Red dots marking face centers

## Testing

### Run Tests
```bash
cd backend
source venv/bin/activate
pytest test_face_detection.py -v
```

### Test Coverage
- Schema validation
- Model initialization
- Face detection with various parameters
- Response structure validation
- Statistics accuracy
- Visualization data format

## File Locations

### Backend
```
backend/
├── algorithms/computer_vision/face_detection/
│   ├── __init__.py
│   ├── schema.py          # Pydantic models
│   ├── data.py            # Sample image management
│   └── model.py           # Face detection implementation
├── api/routes/computer_vision.py  # API routes (updated)
└── test_face_detection.py         # Tests
```

### Documentation
```
FACE_DETECTION_IMPLEMENTATION.md  # Full implementation guide
FACE_DETECTION_QUICK_START.md     # This file
```

## Algorithm Details

### How It Works
1. Load pre-trained Haar Cascade classifier
2. Convert image to grayscale
3. Apply histogram equalization for contrast
4. Detect faces using sliding window at multiple scales
5. Draw bounding boxes and labels
6. Calculate statistics and prepare visualizations

### Complexity
- **Time**: O(image_size × scales)
- **Space**: O(cascade_size)

### Use Cases
- Photo organization and tagging
- Security and surveillance systems
- Attendance tracking
- Social media auto-tagging
- Demographics analysis

## Tips for Best Results

### Parameter Tuning
- **For high accuracy**: Use scale_factor=1.05-1.1, min_neighbors=5-6
- **For speed**: Use scale_factor=1.15-1.2, min_neighbors=3-4
- **For small faces**: Decrease min_size to 20-25
- **To reduce false positives**: Increase min_neighbors

### Image Requirements
- Works best with frontal faces
- Good lighting conditions preferred
- Faces should be upright (not tilted)
- Clear, unobstructed faces

### Limitations
- Struggles with profile views or tilted faces
- Performance degrades in poor lighting
- Not robust to occlusions (glasses, masks)
- Less accurate than modern deep learning methods

## Next Steps

### Backend
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest test_face_detection.py -v`
3. Start server: `uvicorn main:app --reload`
4. Test endpoints with curl or Postman

### Frontend
1. Create `FaceDetection.tsx` component
2. Implement parameter controls (sliders, inputs)
3. Create 2x2 image grid for visualizations
4. Add statistics panel with badges
5. Implement image selector with previews
6. Add download annotated image feature

## Related Algorithms
- **dlib-face**: More accurate deep learning approach
- **MTCNN**: Multi-task Cascaded CNN for face detection
- **YOLO-Face**: Real-time face detection with YOLO
- **RetinaFace**: State-of-the-art face detection

## Support
For issues or questions:
- Check full implementation guide: `FACE_DETECTION_IMPLEMENTATION.md`
- Review test file: `test_face_detection.py`
- Check OpenCV docs: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html

---

**Status**: ✅ Ready to use
**Last Updated**: 2026-08-07
