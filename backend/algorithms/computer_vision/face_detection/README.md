# Face Detection (Haar Cascades)

## Overview
Face Detection implementation using OpenCV's Haar Cascade classifiers. This beginner-level computer vision algorithm detects faces in images using the Viola-Jones object detection framework.

## Features
- Pre-trained Haar Cascade classifier for frontal face detection
- Multi-scale sliding window detection
- Adjustable detection parameters for speed/accuracy trade-offs
- Comprehensive statistics and visualization
- Support for multiple sample images (portraits, groups, various lighting)

## Files
- **`__init__.py`**: Module exports
- **`schema.py`**: Pydantic models (Request, Response, FaceBox, Statistics)
- **`data.py`**: Sample image management (8 diverse images)
- **`model.py`**: Face detection model implementation (696 total lines)

## Usage

### Basic Detection
```python
from algorithms.computer_vision.face_detection import FaceDetectionModel, FaceDetectionRequest

# Initialize model
model = FaceDetectionModel()

# Create request
request = FaceDetectionRequest(
    scale_factor=1.1,
    min_neighbors=5,
    min_size=30,
    image_index=0
)

# Run detection
response = model.process_request(request)

# Access results
print(f"Detected {response.statistics.face_count} faces")
for i, face in enumerate(response.faces):
    print(f"Face {i+1}: {face.width}x{face.height} at ({face.x}, {face.y})")
```

## Parameters

### scale_factor (float)
- **Range**: 1.05 - 1.3
- **Default**: 1.1
- **Description**: Scale reduction factor between successive scans
- **Effect**: Lower = more thorough but slower; Higher = faster but may miss faces

### min_neighbors (int)
- **Range**: 1 - 10
- **Default**: 5
- **Description**: Minimum neighbors required for detection
- **Effect**: Higher = fewer false positives but may miss faces

### min_size (int)
- **Range**: 20 - 100 pixels
- **Default**: 30
- **Description**: Minimum face size to detect
- **Effect**: Faces smaller than this are ignored

### max_size (int | None)
- **Range**: 50+ pixels or None
- **Default**: None (no limit)
- **Description**: Maximum face size to detect
- **Effect**: Faces larger than this are ignored

### image_index (int)
- **Range**: 0 - 7
- **Default**: 0
- **Description**: Sample image selector

## Sample Images
1. **Lena** - Classic portrait, single face
2. **Small Group** - 2-3 faces, group photo
3. **Family Group** - 3-5 faces, family photo
4. **Business** - Multiple faces, professional setting
5. **Outdoor** - Natural lighting portrait
6. **Close-up** - Large face, close-up shot
7. **Studio** - Professional studio portrait
8. **Diverse** - Multiple ethnicities and ages

## Response Structure

### FaceBox
Each detected face includes:
```python
{
    "x": int,              # Top-left X coordinate
    "y": int,              # Top-left Y coordinate
    "width": int,          # Bounding box width
    "height": int,         # Bounding box height
    "confidence": float,   # Detection confidence (0-1)
    "center": {            # Face center point
        "x": int,
        "y": int
    },
    "area": int            # Face area in pixels
}
```

### FaceStatistics
```python
{
    "face_count": int,              # Total faces detected
    "average_face_size": float,     # Average face area
    "largest_face_size": int,       # Largest face area
    "smallest_face_size": int,      # Smallest face area
    "total_face_area": int,         # Total area covered
    "face_density": float,          # % of image covered
    "image_dimensions": {           # Image size
        "width": int,
        "height": int
    },
    "detection_quality": str        # "High", "Medium", or "Low"
}
```

### Visualization Data
4 base64-encoded JPEG images:
1. **original_image**: Unmodified input image
2. **annotated_image**: Green bounding boxes with face numbers
3. **gray_image**: Histogram-equalized grayscale (used for detection)
4. **centers_image**: Red dots marking face centers

## Algorithm Details

### Detection Process
1. Load image and convert to RGB
2. Convert to grayscale
3. Apply histogram equalization for contrast enhancement
4. Run Haar Cascade detection with sliding window
5. Non-maximum suppression (handled by OpenCV)
6. Draw bounding boxes and labels
7. Calculate statistics
8. Prepare visualizations

### Haar Cascade Classifier
- **File**: `haarcascade_frontalface_default.xml`
- **Method**: Viola-Jones object detection framework (2001)
- **Features**: Haar-like features (rectangular intensity differences)
- **Training**: AdaBoost with cascade structure
- **Detection**: Multi-scale sliding window

### Complexity
- **Time**: O(image_size × scales)
- **Space**: O(cascade_size)

### Preprocessing
- Grayscale conversion (Haar features work on intensity)
- Histogram equalization (improves detection in varying lighting)

## Quality Assessment

Detection quality is rated based on parameters:

### High Quality
- `scale_factor <= 1.1` AND `min_neighbors >= 5`
- Most thorough search, best accuracy
- Slowest performance

### Medium Quality
- `scale_factor <= 1.15` AND `min_neighbors >= 4`
- Balanced speed and accuracy

### Low Quality
- Other parameter combinations
- Fastest performance
- May miss faces or produce false positives

## Use Cases
1. **Photo Organization**: Automatically tag and organize photos by detecting faces
2. **Security Systems**: Monitor and detect faces in surveillance footage
3. **Attendance Tracking**: Automated attendance in schools/offices
4. **Social Media**: Auto-tagging friends in photos
5. **Demographics Analysis**: Analyze crowd composition and density

## Advantages
- Very fast detection (suitable for real-time applications)
- Works well for frontal faces with good lighting
- Pre-trained models readily available in OpenCV
- Low computational requirements (no GPU needed)
- Simple to implement and use
- Small model size (~1MB)

## Limitations
- Struggles with profile views or tilted faces (>15° rotation)
- Performance degrades in poor lighting conditions
- Can produce false positives on face-like patterns
- Not robust to occlusions (glasses, masks, hair covering face)
- Less accurate than modern deep learning methods (MTCNN, RetinaFace)
- Not rotation-invariant (faces must be mostly upright)

## Optimal Settings

### High Accuracy (Recommended for photos)
```python
FaceDetectionRequest(
    scale_factor=1.05,
    min_neighbors=6,
    min_size=30
)
```

### Balanced (Default, good for most cases)
```python
FaceDetectionRequest(
    scale_factor=1.1,
    min_neighbors=5,
    min_size=30
)
```

### Fast (For real-time or large images)
```python
FaceDetectionRequest(
    scale_factor=1.2,
    min_neighbors=3,
    min_size=40
)
```

## Tips for Best Results

### Parameter Tuning
- Start with default parameters (scale_factor=1.1, min_neighbors=5)
- If missing faces: decrease scale_factor, decrease min_neighbors
- If too many false positives: increase min_neighbors
- If only detecting large faces: decrease min_size
- For faster detection: increase scale_factor, increase min_size

### Image Requirements
- Frontal faces work best (±15° rotation tolerated)
- Good lighting conditions (avoid heavy shadows)
- Faces should be upright (not heavily tilted)
- Clear, unobstructed faces (avoid occlusions)
- Minimum face size ~30x30 pixels

## Related Algorithms
- **dlib Face Detection**: More accurate, uses HOG + SVM or CNN
- **MTCNN**: Multi-task Cascaded CNN, state-of-the-art accuracy
- **YOLO-Face**: Real-time face detection with YOLO
- **RetinaFace**: Current SOTA, very accurate but slower
- **MediaPipe Face Detection**: Google's efficient face detector

## References
- Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. CVPR.
- OpenCV Documentation: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
- Haar Cascade Training: https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html

## Testing
Run tests with:
```bash
pytest test_face_detection.py -v
```

Test coverage includes:
- Schema validation
- Model initialization
- Detection with various parameters
- Response structure validation
- Statistics accuracy
- Visualization data format
- Quality rating logic

## API Endpoints
- `POST /api/computer-vision/face-detection/detect` - Run face detection
- `GET /api/computer-vision/face-detection/info` - Get algorithm metadata

---

**Implementation Date**: 2026-08-07
**OpenCV Version**: 4.10.0.84
**License**: MIT
