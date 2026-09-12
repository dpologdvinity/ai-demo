# YOLO Object Detection - Usage Examples

## Python API Usage

### Basic Detection

```python
from algorithms.computer_vision.yolo import YOLOModel, YOLORequest

# Create a request with default parameters
request = YOLORequest(
    confidence_threshold=0.5,
    iou_threshold=0.4,
    model_size='n',
    max_detections=100,
    image_index=0
)

# Initialize and run model
model = YOLOModel(model_size='n')
response = model.process_request(request)

# Access results
print(f"Detected {response.statistics.total_detections} objects")
for detection in response.detections:
    print(f"{detection.class_name}: {detection.confidence:.3f}")
```

### High Confidence Detection

```python
# Detect only high-confidence objects
request = YOLORequest(
    confidence_threshold=0.8,  # Higher threshold
    iou_threshold=0.4,
    model_size='n',
    max_detections=50,
    image_index=0
)

response = model.process_request(request)
```

### Using Larger Model for Better Accuracy

```python
# Use medium model for better accuracy
request = YOLORequest(
    confidence_threshold=0.5,
    iou_threshold=0.4,
    model_size='m',  # Medium model
    max_detections=100,
    image_index=1  # Different sample image
)

model = YOLOModel(model_size='m')
response = model.process_request(request)
```

## REST API Usage

### cURL Examples

#### Basic Detection Request

```bash
curl -X POST "http://localhost:8000/computer-vision/yolo/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "confidence_threshold": 0.5,
    "iou_threshold": 0.4,
    "model_size": "n",
    "max_detections": 100,
    "image_index": 0
  }'
```

#### Get Algorithm Info

```bash
curl -X GET "http://localhost:8000/computer-vision/yolo/info"
```

#### List All Computer Vision Algorithms

```bash
curl -X GET "http://localhost:8000/computer-vision/algorithms"
```

### JavaScript/TypeScript Examples

#### Using Fetch API

```javascript
async function detectObjects(params) {
  const response = await fetch('http://localhost:8000/computer-vision/yolo/detect', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      confidence_threshold: params.confidence || 0.5,
      iou_threshold: params.iou || 0.4,
      model_size: params.modelSize || 'n',
      max_detections: params.maxDetections || 100,
      image_index: params.imageIndex || 0,
    }),
  });

  const data = await response.json();
  return data;
}

// Usage
const result = await detectObjects({
  confidence: 0.6,
  modelSize: 's',
  imageIndex: 1,
});

console.log(`Found ${result.statistics.total_detections} objects`);
console.log('Classes:', result.statistics.class_counts);
```

#### Using Axios

```javascript
import axios from 'axios';

const detectObjects = async (params) => {
  try {
    const response = await axios.post(
      'http://localhost:8000/computer-vision/yolo/detect',
      {
        confidence_threshold: params.confidence || 0.5,
        iou_threshold: params.iou || 0.4,
        model_size: params.modelSize || 'n',
        max_detections: params.maxDetections || 100,
        image_index: params.imageIndex || 0,
      }
    );

    return response.data;
  } catch (error) {
    console.error('Detection failed:', error.response?.data || error.message);
    throw error;
  }
};
```

### React Component Example

```jsx
import React, { useState } from 'react';

function YOLODetector() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState({
    confidence: 0.5,
    iou: 0.4,
    modelSize: 'n',
    maxDetections: 100,
    imageIndex: 0,
  });

  const runDetection = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/computer-vision/yolo/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          confidence_threshold: params.confidence,
          iou_threshold: params.iou,
          model_size: params.modelSize,
          max_detections: params.maxDetections,
          image_index: params.imageIndex,
        }),
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Detection failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>YOLO Object Detection</h2>
      
      {/* Parameter Controls */}
      <div>
        <label>
          Confidence Threshold:
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={params.confidence}
            onChange={(e) => setParams({...params, confidence: parseFloat(e.target.value)})}
          />
          {params.confidence}
        </label>
      </div>

      <div>
        <label>
          Model Size:
          <select
            value={params.modelSize}
            onChange={(e) => setParams({...params, modelSize: e.target.value})}
          >
            <option value="n">Nano (Fastest)</option>
            <option value="s">Small (Balanced)</option>
            <option value="m">Medium (Most Accurate)</option>
          </select>
        </label>
      </div>

      <button onClick={runDetection} disabled={loading}>
        {loading ? 'Detecting...' : 'Run Detection'}
      </button>

      {/* Results Display */}
      {result && (
        <div>
          <h3>Results</h3>
          <p>Detected {result.statistics.total_detections} objects in {result.execution_time_ms.toFixed(2)}ms</p>
          
          {/* Display annotated image */}
          <img src={result.visualization_data.annotated_image} alt="Detected objects" />
          
          {/* Display detections table */}
          <table>
            <thead>
              <tr>
                <th>Class</th>
                <th>Confidence</th>
                <th>Bounding Box</th>
              </tr>
            </thead>
            <tbody>
              {result.detections.map((det, idx) => (
                <tr key={idx}>
                  <td>{det.class_name}</td>
                  <td>{(det.confidence * 100).toFixed(1)}%</td>
                  <td>{det.bbox.map(v => v.toFixed(0)).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Display class counts */}
          <h4>Class Distribution</h4>
          <ul>
            {Object.entries(result.statistics.class_counts).map(([cls, count]) => (
              <li key={cls}>{cls}: {count}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default YOLODetector;
```

## Response Structure Example

### Successful Detection Response

```json
{
  "success": true,
  "detections": [
    {
      "bbox": [184.5, 123.2, 456.8, 678.9],
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.945
    },
    {
      "bbox": [12.3, 234.5, 234.6, 456.7],
      "class_name": "car",
      "class_id": 2,
      "confidence": 0.876
    }
  ],
  "statistics": {
    "total_detections": 5,
    "class_counts": {
      "person": 3,
      "car": 2
    },
    "avg_confidence": 0.852,
    "confidence_distribution": {
      "0.0-0.3": 0,
      "0.3-0.5": 0,
      "0.5-0.7": 1,
      "0.7-0.9": 2,
      "0.9-1.0": 2
    }
  },
  "visualization_data": {
    "original_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "detection_list": [
      {
        "bbox": [184.5, 123.2, 456.8, 678.9],
        "class_name": "person",
        "confidence": 0.945
      }
    ],
    "confidence_chart": [
      {"class": "person", "confidence": 0.945},
      {"class": "car", "confidence": 0.876}
    ]
  },
  "execution_time_ms": 150.5,
  "model_info": {
    "model_name": "YOLOv8n",
    "model_size": "n",
    "num_classes": 80,
    "framework": "Ultralytics YOLOv8",
    "device": "cpu",
    "input_size": "640x640 (default)",
    "architecture": "YOLOv8 (CSPDarknet + PANet + Detection Head)"
  },
  "parameters_used": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.4,
    "model_size": "n",
    "max_detections": 100,
    "image_index": 0
  },
  "image_info": {
    "width": 1024,
    "height": 768,
    "channels": 3,
    "image_index": 0
  }
}
```

## Common Use Cases

### 1. Real-time Surveillance

```python
# High confidence, medium model for accuracy
request = YOLORequest(
    confidence_threshold=0.7,
    iou_threshold=0.5,
    model_size='m',
    max_detections=50
)
```

### 2. Fast Prototype/Demo

```python
# Fast detection with nano model
request = YOLORequest(
    confidence_threshold=0.5,
    iou_threshold=0.4,
    model_size='n',
    max_detections=100
)
```

### 3. Dense Object Detection

```python
# Lower thresholds for detecting more objects
request = YOLORequest(
    confidence_threshold=0.3,
    iou_threshold=0.3,
    model_size='s',
    max_detections=300
)
```

### 4. Precise Detection

```python
# High thresholds, fewer but more accurate detections
request = YOLORequest(
    confidence_threshold=0.8,
    iou_threshold=0.6,
    model_size='m',
    max_detections=50
)
```

## Error Handling

```python
from fastapi import HTTPException

try:
    model = YOLOModel(model_size='n')
    response = model.process_request(request)
    
    if not response.success:
        print("Detection failed")
    else:
        print(f"Success! Found {len(response.detections)} objects")
        
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"Detection error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Model Selection**
   - Use `'n'` (nano) for fastest inference (~5-10ms)
   - Use `'s'` (small) for balanced performance (~15-30ms)
   - Use `'m'` (medium) for best accuracy (~40-80ms)

2. **Threshold Tuning**
   - Higher confidence_threshold = fewer false positives
   - Lower iou_threshold = fewer overlapping boxes
   - Adjust based on your specific use case

3. **GPU Acceleration**
   - Install CUDA-enabled PyTorch for GPU support
   - Speeds up inference by 5-10x

4. **Batch Processing**
   - Process multiple images in sequence
   - Model stays loaded in memory
   - Subsequent detections are faster

## Testing Checklist

- [ ] Test with default parameters
- [ ] Test with high confidence threshold (0.8+)
- [ ] Test with low confidence threshold (0.2-0.3)
- [ ] Test with each model size (n, s, m)
- [ ] Test with different sample images (0, 1)
- [ ] Test with max_detections limit
- [ ] Verify base64 images display correctly
- [ ] Check statistics are calculated correctly
- [ ] Verify error handling with invalid parameters
- [ ] Test API response time
