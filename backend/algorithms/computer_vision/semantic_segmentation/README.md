# Semantic Segmentation Implementation

This module implements semantic segmentation using DeepLabV3 with PyTorch for the AI Algorithms Demo website.

## Overview

Semantic segmentation is a computer vision task that assigns a class label to each pixel in an image, enabling detailed scene understanding. This implementation uses the pre-trained DeepLabV3 model from torchvision.

## Algorithm Details

- **Name**: Semantic Segmentation (U-Net)
- **Slug**: `semantic-segmentation`
- **Category**: Computer Vision
- **Difficulty**: Advanced
- **Framework**: PyTorch (torchvision)
- **Model**: DeepLabV3 with ResNet50 or MobileNetV3 backbone

## Architecture

DeepLabV3 uses **Atrous Spatial Pyramid Pooling (ASPP)** to capture multi-scale context:

1. **Backbone Network**: ResNet50 or MobileNetV3 for feature extraction
2. **ASPP Module**: Captures multi-scale information using dilated convolutions
3. **Decoder**: Gradually recovers spatial information
4. **Classification Head**: Produces per-pixel class predictions

## Features

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| num_classes | int | 21 | 2-150 | Number of segmentation classes |
| confidence_threshold | float | 0.5 | 0.1-0.95 | Minimum confidence for predictions |
| model_backbone | string | 'resnet50' | resnet50, mobilenet | Backbone architecture |
| image_size | int | 512 | 256, 512, 1024 | Input image resolution |
| image_index | int | 0 | 0-2 | Sample image selector |

### Sample Images

1. **Image 0**: Dog portrait (close-up)
2. **Image 1**: Dog on grass (outdoor scene)
3. **Image 2**: Street scene with vehicles

### Output

The API returns:

- **Segmentation Mask**: Per-pixel class predictions
- **Colored Mask**: Visualization with class colors
- **Overlay Image**: Semi-transparent mask over original
- **Class Statistics**: 
  - Pixel count per class
  - Percentage of image
  - Class names and colors
- **Confidence Scores**: Mean prediction confidence
- **Execution Time**: Processing time in milliseconds

## API Endpoints

### POST `/api/computer-vision/semantic-segmentation/segment`

Run semantic segmentation on a sample image.

**Request Body**:
```json
{
  "num_classes": 21,
  "confidence_threshold": 0.5,
  "model_backbone": "resnet50",
  "image_size": 512,
  "image_index": 0
}
```

**Response**:
```json
{
  "success": true,
  "statistics": {
    "total_classes": 5,
    "total_pixels": 262144,
    "mean_confidence": 0.87,
    "class_info": [
      {
        "class_id": 12,
        "class_name": "dog",
        "color": [64, 0, 128],
        "pixel_count": 123456,
        "percentage": 47.1,
        "iou_score": null
      }
    ]
  },
  "visualization_data": {
    "original_image": "data:image/jpeg;base64,...",
    "colored_mask": "data:image/jpeg;base64,...",
    "overlay_image": "data:image/jpeg;base64,...",
    "class_legend": [...],
    "class_distribution": [...],
    "show_overlay": true
  },
  "execution_time_ms": 234.5,
  "model_info": {...},
  "parameters_used": {...},
  "image_info": {...}
}
```

### GET `/api/computer-vision/semantic-segmentation/info`

Get algorithm metadata and information.

## Classes (PASCAL VOC 2012)

The default configuration uses 21 classes from PASCAL VOC 2012:

1. background
2. aeroplane
3. bicycle
4. bird
5. boat
6. bottle
7. bus
8. car
9. cat
10. chair
11. cow
12. diningtable
13. dog
14. horse
15. motorbike
16. person
17. pottedplant
18. sheep
19. sofa
20. train
21. tvmonitor

Each class has a distinct color for visualization.

## Use Cases

1. **Autonomous Driving**: Road scene parsing (lanes, vehicles, pedestrians, signs)
2. **Medical Imaging**: Organ segmentation, tumor detection
3. **Satellite Imagery**: Land use classification, urban planning
4. **Augmented Reality**: Scene understanding for AR overlays
5. **Video Background Removal**: Person segmentation for virtual backgrounds

## Complexity

- **Time Complexity**: O(H × W × C) where H=height, W=width, C=channels
- **Space Complexity**: O(H × W) for storing segmentation mask
- **Inference Time**: 
  - ResNet50: ~200-300ms (CPU), ~50-100ms (GPU)
  - MobileNet: ~100-150ms (CPU), ~30-50ms (GPU)

## Pros & Cons

### Advantages

- Dense pixel-level predictions for detailed scene understanding
- Multi-scale context capture via ASPP
- Preserves spatial information better than object detection
- Pre-trained models available for transfer learning
- Applicable to diverse domains

### Limitations

- Computationally expensive (processes every pixel)
- Requires pixel-level annotations for training
- Slower inference than object detection
- May struggle with fine boundaries
- Limited to predefined classes

## Frontend Integration

The frontend should display:

1. **Image Viewer**: 
   - Original image
   - Segmentation mask (colored)
   - Overlay toggle

2. **Class Legend**:
   - Color-coded class labels
   - Pixel count per class
   - Percentage of image

3. **Statistics Panel**:
   - Total classes detected
   - Mean confidence score
   - Processing time

4. **Interactive Controls**:
   - Overlay opacity slider
   - Class visibility toggles
   - Zoom/pan for detailed inspection

5. **Class Distribution Chart**:
   - Bar chart showing top classes by pixel count

## Testing

Run the test script:

```bash
cd backend
python test_segmentation.py
```

## Dependencies

- PyTorch >= 2.4.1
- torchvision >= 0.19.1
- PIL (Pillow) >= 10.4.0
- NumPy >= 2.1.1
- FastAPI >= 0.115.0
- Pydantic >= 2.9.2

## Model Details

### ResNet50 Backbone
- **Accuracy**: Higher (mIoU ~77% on PASCAL VOC)
- **Speed**: Slower (~200-300ms on CPU)
- **Model Size**: ~160MB
- **Use Case**: Best for quality over speed

### MobileNetV3 Backbone
- **Accuracy**: Lower (mIoU ~72% on PASCAL VOC)
- **Speed**: Faster (~100-150ms on CPU)
- **Model Size**: ~15MB
- **Use Case**: Best for speed over quality

## References

- [DeepLabV3 Paper](https://arxiv.org/abs/1706.05587)
- [PyTorch Semantic Segmentation](https://pytorch.org/vision/stable/models.html#semantic-segmentation)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)

## Implementation Notes

1. **Pre-trained Weights**: Uses models trained on COCO and PASCAL VOC
2. **Image Preprocessing**: Resizes to target size, normalizes with ImageNet stats
3. **Post-processing**: Applies confidence threshold, resizes back to original size
4. **Color Palette**: Uses standard PASCAL VOC colors, generates additional colors if needed
5. **Memory Optimization**: Model loaded lazily on first request

## Future Enhancements

- [ ] Add support for custom image upload
- [ ] Implement U-Net architecture option
- [ ] Add instance segmentation (Mask R-CNN)
- [ ] Support for video segmentation
- [ ] Real-time webcam segmentation
- [ ] Export segmentation masks
- [ ] Fine-tuning on custom datasets
- [ ] Multi-scale inference for better accuracy
