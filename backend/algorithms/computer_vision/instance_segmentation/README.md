# Instance Segmentation (Mask R-CNN)

## Overview

Instance Segmentation using Mask R-CNN detects and segments individual object instances with pixel-level precision. Unlike semantic segmentation which assigns a class to each pixel, instance segmentation distinguishes between different instances of the same class.

## Algorithm Details

- **Name**: Instance Segmentation (Mask R-CNN)
- **Category**: Computer Vision
- **Difficulty**: Advanced
- **Framework**: PyTorch + torchvision
- **Pretrained Model**: COCO dataset (80 classes)

## Features

- Pixel-accurate instance segmentation masks
- Bounding box detection for each instance
- Multi-instance detection (same class)
- Confidence scoring per instance
- Handles overlapping objects
- Real-time visualization with colored masks

## Architecture

Mask R-CNN extends Faster R-CNN by adding a mask prediction branch:

1. **Backbone CNN** (ResNet-50-FPN): Feature extraction
2. **Region Proposal Network (RPN)**: Generates object proposals
3. **RoI Align**: Extracts features without quantization
4. **Box Head**: Predicts class labels and bounding boxes
5. **Mask Head**: Predicts binary masks for each instance (FCN)

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `confidence_threshold` | float | 0.5 | 0.1-0.9 | Minimum confidence for detections |
| `model_backbone` | string | 'resnet50' | resnet50, resnet101 | Backbone architecture |
| `mask_threshold` | float | 0.5 | 0.1-0.9 | Binary threshold for masks |
| `max_instances` | int | 100 | 10-200 | Maximum instances to detect |
| `image_index` | int | 0 | 0-9 | Sample image index |
| `nms_threshold` | float | 0.5 | 0.1-0.9 | NMS IoU threshold |

## Sample Images

The implementation includes 10 diverse sample images:

0. Street scene with bus and people
1. Soccer players (multiple people)
2. City street with vehicles and pedestrians
3. Living room with furniture and objects
4. Multiple people outdoors
5. Dogs in park
6. Horses in field
7. Cats indoors
8. Airport scene with airplanes
9. Kitchen with appliances and objects

## Usage

### Python API

```python
from algorithms.computer_vision.instance_segmentation import (
    InstanceSegmentationModel,
    InstanceSegmentationRequest
)

# Create request
request = InstanceSegmentationRequest(
    confidence_threshold=0.5,
    model_backbone='resnet50',
    mask_threshold=0.5,
    max_instances=100,
    image_index=0,
    nms_threshold=0.5
)

# Initialize model
model = InstanceSegmentationModel(model_backbone='resnet50')

# Process request
response = model.process_request(request)

# Access results
print(f"Total instances: {response.statistics.total_instances}")
for inst in response.instances:
    print(f"{inst.class_name}: {inst.confidence:.3f}")
```

### REST API

**Segment Instances**
```bash
POST /api/computer-vision/instance-segmentation/segment
Content-Type: application/json

{
  "confidence_threshold": 0.5,
  "model_backbone": "resnet50",
  "mask_threshold": 0.5,
  "max_instances": 100,
  "image_index": 0,
  "nms_threshold": 0.5
}
```

**Get Algorithm Info**
```bash
GET /api/computer-vision/instance-segmentation/info
```

## Response Format

```json
{
  "success": true,
  "instances": [
    {
      "bbox": [x1, y1, x2, y2],
      "class_name": "person",
      "class_id": 1,
      "confidence": 0.95,
      "mask_area": 12500,
      "instance_id": 0
    }
  ],
  "statistics": {
    "total_instances": 5,
    "unique_classes": 3,
    "class_counts": {"person": 3, "car": 2},
    "avg_confidence": 0.92,
    "total_mask_area": 150000,
    "coverage_percentage": 35.2,
    "avg_instance_size": 30000,
    "confidence_distribution": {
      "0.9-1.0": 4,
      "0.7-0.9": 1
    }
  },
  "visualization_data": {
    "original_image": "base64_encoded_image",
    "colored_masks": "base64_encoded_masks",
    "overlay_image": "base64_encoded_overlay",
    "annotated_image": "base64_encoded_annotated",
    "instance_colors": [...]
  },
  "execution_time_ms": 1250.5,
  "model_info": {
    "name": "Mask R-CNN",
    "backbone": "resnet50",
    "framework": "PyTorch + torchvision",
    "pretrained_on": "COCO",
    "device": "cuda"
  }
}
```

## Visualizations

The implementation generates four visualization types:

1. **Original Image**: Input image
2. **Colored Masks**: Each instance rendered in a unique color
3. **Overlay Image**: Transparent mask overlay on original image
4. **Annotated Image**: Original image with bounding boxes and labels

## COCO Classes

The model is trained on 80 COCO classes including:
- People: person
- Vehicles: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- Animals: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- Objects: backpack, umbrella, handbag, tie, suitcase, bottle, cup, fork, knife, spoon, bowl
- Food: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- Furniture: chair, couch, potted plant, bed, dining table, toilet
- Electronics: tv, laptop, mouse, remote, keyboard, cell phone
- And more...

## Performance

- **Time Complexity**: O(image_size * instances)
- **Space Complexity**: O(model_params)
- **Typical Inference Time**: 1-3 seconds on GPU, 5-15 seconds on CPU
- **Model Size**: ~170 MB (ResNet50-FPN)

## Use Cases

1. **Autonomous Driving**: Detect and segment pedestrians, vehicles, road signs
2. **Medical Imaging**: Segment organs, tumors, cells in medical scans
3. **Robotics**: Object recognition and manipulation in robotic systems
4. **Video Analytics**: Track and analyze individual objects in video streams
5. **Retail Automation**: Product detection and counting in stores
6. **Agricultural Monitoring**: Crop monitoring, pest detection, yield estimation

## Advantages

- Pixel-accurate instance segmentation
- Handles multiple instances of same class
- Unified detection and segmentation
- State-of-the-art accuracy
- Pre-trained models available
- Robust to overlapping objects

## Limitations

- Computationally expensive (requires GPU)
- Slower than object detection alone
- Requires significant memory
- May struggle with very small objects
- Sensitive to heavy occlusion

## References

- **Paper**: [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870)
- **Framework**: [PyTorch torchvision](https://pytorch.org/vision/stable/models.html#mask-r-cnn)
- **Dataset**: [COCO Dataset](https://cocodataset.org/)

## Testing

Run the test suite:

```bash
cd backend
source venv/bin/activate
python test_instance_segmentation.py
python test_instance_segmentation_api.py
```

## File Structure

```
instance_segmentation/
├── __init__.py              # Module exports
├── model.py                 # Mask R-CNN implementation
├── schema.py                # Pydantic request/response schemas
├── data.py                  # Sample data and COCO class names
├── sample_images/           # Cached sample images
└── README.md               # This file
```
