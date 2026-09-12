# Instance Segmentation - Frontend Integration Guide

## Quick Start

Instance Segmentation is now available in the Computer Vision category with slug `instance-segmentation`.

## API Endpoints

### Base URL
```
http://localhost:8000/api/computer-vision
```

### 1. Get Algorithm Information
```http
GET /instance-segmentation/info
```

**Response:**
```json
{
  "metadata": {
    "id": "instance-segmentation",
    "name": "Instance Segmentation (Mask R-CNN)",
    "slug": "instance-segmentation",
    "category": "computer_vision",
    "difficulty": "Advanced",
    "description": "Detect and segment individual object instances with pixel-level masks",
    "tags": ["computer-vision", "instance-segmentation", "mask-rcnn", ...],
    "use_cases": ["Autonomous driving", "Medical imaging", ...],
    "parameters": [...]
  },
  "dataset": {
    "name": "COCO Instance Segmentation",
    "num_classes": 80,
    "num_samples": 10,
    "sample_descriptions": [...]
  }
}
```

### 2. Run Segmentation
```http
POST /instance-segmentation/segment
Content-Type: application/json
```

**Request Body:**
```json
{
  "confidence_threshold": 0.5,
  "model_backbone": "resnet50",
  "mask_threshold": 0.5,
  "max_instances": 100,
  "image_index": 0,
  "nms_threshold": 0.5
}
```

**Response:**
```json
{
  "success": true,
  "instances": [
    {
      "bbox": [x1, y1, x2, y2],
      "class_name": "person",
      "class_id": 1,
      "confidence": 0.999,
      "mask_area": 33820,
      "instance_id": 0
    }
  ],
  "statistics": {
    "total_instances": 5,
    "unique_classes": 2,
    "class_counts": {"person": 4, "bus": 1},
    "avg_confidence": 0.989,
    "total_mask_area": 369277,
    "coverage_percentage": 42.21,
    "avg_instance_size": 73855.4,
    "confidence_distribution": {
      "0.0-0.3": 0,
      "0.3-0.5": 0,
      "0.5-0.7": 0,
      "0.7-0.9": 0,
      "0.9-1.0": 5
    }
  },
  "visualization_data": {
    "original_image": "base64_encoded_jpeg",
    "colored_masks": "base64_encoded_jpeg",
    "overlay_image": "base64_encoded_jpeg",
    "annotated_image": "base64_encoded_jpeg",
    "image_size": {"width": 810, "height": 1080},
    "instance_colors": [
      {
        "instance_id": 0,
        "class_name": "person",
        "color": "rgb(255, 0, 0)"
      }
    ]
  },
  "execution_time_ms": 1327.94,
  "model_info": {
    "name": "Mask R-CNN",
    "backbone": "resnet50",
    "framework": "PyTorch + torchvision",
    "pretrained_on": "COCO",
    "device": "cuda"
  },
  "image_info": {
    "size": [810, 1080],
    "index": 0
  }
}
```

## UI Components to Implement

### 1. Parameter Controls

```javascript
const parameters = [
  {
    name: "confidence_threshold",
    label: "Confidence Threshold",
    type: "range",
    default: 0.5,
    min: 0.1,
    max: 0.9,
    step: 0.05,
    description: "Minimum confidence for detections (higher = fewer but more confident instances)"
  },
  {
    name: "model_backbone",
    label: "Model Backbone",
    type: "select",
    default: "resnet50",
    options: [
      {label: "ResNet50 (Balanced)", value: "resnet50"},
      {label: "ResNet101 (More Accurate)", value: "resnet101"}
    ]
  },
  {
    name: "mask_threshold",
    label: "Mask Threshold",
    type: "range",
    default: 0.5,
    min: 0.1,
    max: 0.9,
    step: 0.05,
    description: "Binary threshold for mask generation"
  },
  {
    name: "max_instances",
    label: "Max Instances",
    type: "number",
    default: 100,
    min: 10,
    max: 200,
    step: 10
  },
  {
    name: "image_index",
    label: "Sample Image",
    type: "number",
    default: 0,
    min: 0,
    max: 9,
    step: 1,
    description: "Select sample image (0-9)"
  },
  {
    name: "nms_threshold",
    label: "NMS Threshold",
    type: "range",
    default: 0.5,
    min: 0.1,
    max: 0.9,
    step: 0.05,
    description: "Non-Maximum Suppression IoU threshold"
  }
];
```

### 2. Image Visualization

Display multiple views:

```jsx
// Main visualization tabs
const views = [
  { id: "original", label: "Original Image", key: "original_image" },
  { id: "masks", label: "Colored Masks", key: "colored_masks" },
  { id: "overlay", label: "Overlay", key: "overlay_image" },
  { id: "annotated", label: "Annotated", key: "annotated_image" }
];

// Render images
{views.map(view => (
  <img 
    src={`data:image/jpeg;base64,${response.visualization_data[view.key]}`}
    alt={view.label}
  />
))}
```

### 3. Instance List Table

```jsx
<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Class</th>
      <th>Confidence</th>
      <th>Area (px)</th>
      <th>BBox</th>
    </tr>
  </thead>
  <tbody>
    {response.instances.map(inst => (
      <tr key={inst.instance_id} style={{
        borderLeft: `4px solid ${getInstanceColor(inst.instance_id)}`
      }}>
        <td>{inst.instance_id}</td>
        <td>{inst.class_name}</td>
        <td>{(inst.confidence * 100).toFixed(1)}%</td>
        <td>{inst.mask_area.toLocaleString()}</td>
        <td>
          [{inst.bbox[0].toFixed(0)}, {inst.bbox[1].toFixed(0)}, 
           {inst.bbox[2].toFixed(0)}, {inst.bbox[3].toFixed(0)}]
        </td>
      </tr>
    ))}
  </tbody>
</table>
```

### 4. Statistics Display

```jsx
<div className="statistics">
  <div className="stat-card">
    <div className="stat-label">Total Instances</div>
    <div className="stat-value">{statistics.total_instances}</div>
  </div>
  
  <div className="stat-card">
    <div className="stat-label">Unique Classes</div>
    <div className="stat-value">{statistics.unique_classes}</div>
  </div>
  
  <div className="stat-card">
    <div className="stat-label">Average Confidence</div>
    <div className="stat-value">{(statistics.avg_confidence * 100).toFixed(1)}%</div>
  </div>
  
  <div className="stat-card">
    <div className="stat-label">Coverage</div>
    <div className="stat-value">{statistics.coverage_percentage.toFixed(1)}%</div>
  </div>
</div>
```

### 5. Class Distribution Chart

```jsx
// Pie chart data
const chartData = Object.entries(statistics.class_counts).map(([name, count]) => ({
  name,
  value: count
}));

// Using any chart library (recharts, chart.js, etc.)
<PieChart width={400} height={300}>
  <Pie 
    data={chartData}
    dataKey="value"
    nameKey="name"
    cx="50%"
    cy="50%"
    label
  />
  <Tooltip />
  <Legend />
</PieChart>
```

### 6. Confidence Distribution Chart

```jsx
// Bar chart data
const confidenceData = Object.entries(statistics.confidence_distribution).map(
  ([range, count]) => ({
    range,
    count
  })
);

<BarChart width={500} height={300} data={confidenceData}>
  <XAxis dataKey="range" />
  <YAxis />
  <Tooltip />
  <Bar dataKey="count" fill="#8884d8" />
</BarChart>
```

### 7. Instance Color Legend

```jsx
<div className="instance-legend">
  <h3>Instance Colors</h3>
  {response.visualization_data.instance_colors.map(item => (
    <div key={item.instance_id} className="legend-item">
      <div 
        className="color-box" 
        style={{backgroundColor: item.color}}
      />
      <span>Instance {item.instance_id}: {item.class_name}</span>
    </div>
  ))}
</div>
```

## Sample Images

10 available sample images (index 0-9):
- 0: Street scene with bus and people
- 1: Soccer players (multiple people)
- 2: City street with vehicles and pedestrians
- 3: Living room with furniture and objects
- 4: Multiple people outdoors
- 5: Dogs in park
- 6: Horses in field
- 7: Cats indoors
- 8: Airport scene with airplanes
- 9: Kitchen with appliances and objects

## Example API Call

```javascript
async function runInstanceSegmentation(params) {
  try {
    const response = await fetch(
      'http://localhost:8000/api/computer-vision/instance-segmentation/segment',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Segmentation failed:', error);
    throw error;
  }
}

// Usage
const params = {
  confidence_threshold: 0.5,
  model_backbone: 'resnet50',
  mask_threshold: 0.5,
  max_instances: 100,
  image_index: 0,
  nms_threshold: 0.5
};

const result = await runInstanceSegmentation(params);
console.log(`Detected ${result.statistics.total_instances} instances`);
```

## Visualization Options

You can provide users with mask display options:
- **Outline mode**: Show only mask boundaries
- **Filled mode**: Show fully colored masks
- **Transparent mode**: Blend masks with original image (overlay)
- **Annotated mode**: Show bounding boxes + labels + masks

## Performance Notes

- First request takes ~7-8 seconds (model loading)
- Subsequent requests take ~1-2 seconds on GPU
- Consider showing loading indicator during segmentation
- Model runs on GPU if available, falls back to CPU

## COCO Classes Supported

80 classes including:
- **People**: person
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Furniture**: chair, couch, potted plant, bed, dining table
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone
- **Food**: banana, apple, sandwich, orange, pizza, donut, cake
- And many more...

## Error Handling

```javascript
try {
  const result = await runInstanceSegmentation(params);
  // Handle success
} catch (error) {
  if (error.response?.status === 400) {
    // Invalid parameters
    console.error('Invalid parameters:', error.response.data.detail);
  } else if (error.response?.status === 500) {
    // Server error
    console.error('Segmentation failed:', error.response.data.detail);
  } else {
    // Network or other error
    console.error('Request failed:', error.message);
  }
}
```

## Tips for Best Results

1. **Confidence threshold**: 0.5-0.7 works well for most images
2. **Max instances**: Set to 50-100 for typical scenes
3. **Mask threshold**: 0.5 provides balanced mask quality
4. **Sample images**: Images 0, 1, 3, 5 have multiple instances
5. **Visualization**: Overlay mode shows context, annotated shows details

## Next Steps

1. Add instance segmentation to the algorithm list UI
2. Create parameter controls form
3. Implement image display with view switching
4. Add instance list table
5. Add statistics cards
6. Add class distribution pie chart
7. Add confidence distribution bar chart
8. Add instance color legend
9. Test with all sample images
10. Add loading states and error handling
