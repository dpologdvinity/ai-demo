# Semantic Segmentation Frontend Integration Guide

This guide explains how to integrate the Semantic Segmentation algorithm into the frontend of the AI Algorithms Demo website.

## Backend Implementation Complete

The backend implementation is complete with the following endpoints:

- **POST** `/api/computer-vision/semantic-segmentation/segment` - Run segmentation
- **GET** `/api/computer-vision/semantic-segmentation/info` - Get algorithm metadata

## Frontend Component Structure

Create a new component at:
```
frontend/src/components/algorithm-demos/computer-vision/SemanticSegmentation.tsx
```

## Key Features to Implement

### 1. Image Display Area

Display three views with toggle controls:

```tsx
<div className="image-grid">
  {/* Original Image */}
  <div className="image-container">
    <h3>Original Image</h3>
    <img src={response.visualization_data.original_image} alt="Original" />
  </div>

  {/* Segmentation Mask */}
  <div className="image-container">
    <h3>Segmentation Mask</h3>
    <img src={response.visualization_data.colored_mask} alt="Mask" />
  </div>

  {/* Overlay */}
  <div className="image-container">
    <h3>Overlay</h3>
    <img src={response.visualization_data.overlay_image} alt="Overlay" />
    <input 
      type="range" 
      min="0" 
      max="100" 
      value={overlayOpacity}
      onChange={(e) => setOverlayOpacity(e.target.value)}
      className="opacity-slider"
    />
  </div>
</div>
```

### 2. Class Legend

Display color-coded classes with statistics:

```tsx
<div className="class-legend">
  <h3>Detected Classes</h3>
  <div className="legend-items">
    {response.visualization_data.class_legend.map((classInfo) => (
      <div key={classInfo.class_id} className="legend-item">
        <div 
          className="color-box" 
          style={{ backgroundColor: classInfo.color }}
        />
        <span className="class-name">{classInfo.class_name}</span>
        <span className="percentage">{classInfo.percentage}%</span>
        <span className="pixel-count">{classInfo.pixel_count.toLocaleString()} px</span>
      </div>
    ))}
  </div>
</div>
```

### 3. Parameter Controls

```tsx
<div className="parameter-controls">
  {/* Number of Classes */}
  <div className="control-group">
    <label>Number of Classes</label>
    <input 
      type="number"
      min={2}
      max={150}
      value={params.num_classes}
      onChange={(e) => setParams({...params, num_classes: parseInt(e.target.value)})}
    />
  </div>

  {/* Confidence Threshold */}
  <div className="control-group">
    <label>Confidence Threshold: {params.confidence_threshold}</label>
    <input 
      type="range"
      min={0.1}
      max={0.95}
      step={0.05}
      value={params.confidence_threshold}
      onChange={(e) => setParams({...params, confidence_threshold: parseFloat(e.target.value)})}
    />
  </div>

  {/* Model Backbone */}
  <div className="control-group">
    <label>Model Backbone</label>
    <select 
      value={params.model_backbone}
      onChange={(e) => setParams({...params, model_backbone: e.target.value})}
    >
      <option value="resnet50">ResNet50 (More Accurate)</option>
      <option value="mobilenet">MobileNet (Faster)</option>
    </select>
  </div>

  {/* Image Size */}
  <div className="control-group">
    <label>Image Size</label>
    <select 
      value={params.image_size}
      onChange={(e) => setParams({...params, image_size: parseInt(e.target.value)})}
    >
      <option value={256}>256x256 (Fastest)</option>
      <option value={512}>512x512 (Balanced)</option>
      <option value={1024}>1024x1024 (Best Quality)</option>
    </select>
  </div>

  {/* Sample Image Selector */}
  <div className="control-group">
    <label>Sample Image</label>
    <select 
      value={params.image_index}
      onChange={(e) => setParams({...params, image_index: parseInt(e.target.value)})}
    >
      <option value={0}>Dog Portrait</option>
      <option value={1}>Dog on Grass</option>
      <option value={2}>Street Scene</option>
    </select>
  </div>

  <button onClick={runSegmentation} disabled={loading}>
    {loading ? 'Processing...' : 'Run Segmentation'}
  </button>
</div>
```

### 4. Statistics Panel

```tsx
<div className="statistics-panel">
  <h3>Segmentation Statistics</h3>
  <div className="stat-grid">
    <div className="stat-item">
      <label>Total Classes</label>
      <value>{response.statistics.total_classes}</value>
    </div>
    <div className="stat-item">
      <label>Total Pixels</label>
      <value>{response.statistics.total_pixels.toLocaleString()}</value>
    </div>
    <div className="stat-item">
      <label>Mean Confidence</label>
      <value>{(response.statistics.mean_confidence * 100).toFixed(1)}%</value>
    </div>
    <div className="stat-item">
      <label>Execution Time</label>
      <value>{response.execution_time_ms.toFixed(1)}ms</value>
    </div>
  </div>
</div>
```

### 5. Class Distribution Chart

Use a bar chart library (Recharts, Chart.js, etc.) to visualize class distribution:

```tsx
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

<div className="distribution-chart">
  <h3>Class Distribution</h3>
  <BarChart 
    width={600} 
    height={300} 
    data={response.visualization_data.class_distribution}
  >
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="class" />
    <YAxis label={{ value: 'Percentage', angle: -90 }} />
    <Tooltip />
    <Legend />
    <Bar dataKey="percentage" fill="#8884d8" />
  </BarChart>
</div>
```

### 6. Zoom/Pan Controls

For detailed inspection, add zoom and pan functionality:

```tsx
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

<TransformWrapper
  initialScale={1}
  minScale={0.5}
  maxScale={4}
>
  {({ zoomIn, zoomOut, resetTransform }) => (
    <>
      <div className="zoom-controls">
        <button onClick={() => zoomIn()}>Zoom In</button>
        <button onClick={() => zoomOut()}>Zoom Out</button>
        <button onClick={() => resetTransform()}>Reset</button>
      </div>
      <TransformComponent>
        <img src={currentImage} alt="Segmentation" />
      </TransformComponent>
    </>
  )}
</TransformWrapper>
```

## API Integration

### Making API Calls

```tsx
const runSegmentation = async () => {
  setLoading(true);
  setError(null);

  try {
    const response = await fetch('/api/computer-vision/semantic-segmentation/segment', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        num_classes: params.num_classes,
        confidence_threshold: params.confidence_threshold,
        model_backbone: params.model_backbone,
        image_size: params.image_size,
        image_index: params.image_index
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    setResult(data);
  } catch (err) {
    setError(err.message);
    console.error('Segmentation error:', err);
  } finally {
    setLoading(false);
  }
};
```

### Fetching Algorithm Info

```tsx
useEffect(() => {
  const fetchInfo = async () => {
    try {
      const response = await fetch('/api/computer-vision/semantic-segmentation/info');
      const data = await response.json();
      setAlgorithmInfo(data);
    } catch (err) {
      console.error('Failed to fetch algorithm info:', err);
    }
  };

  fetchInfo();
}, []);
```

## TypeScript Types

```typescript
interface SegmentationRequest {
  num_classes: number;
  confidence_threshold: number;
  model_backbone: 'resnet50' | 'mobilenet';
  image_size: 256 | 512 | 1024;
  image_index: number;
}

interface ClassInfo {
  class_id: number;
  class_name: string;
  color: string;
  pixel_count: number;
  percentage: number;
  iou_score?: number;
}

interface SegmentationStatistics {
  total_classes: number;
  total_pixels: number;
  mean_confidence: number;
  class_info: ClassInfo[];
}

interface VisualizationData {
  original_image: string;
  colored_mask: string;
  overlay_image: string;
  class_legend: ClassInfo[];
  class_distribution: {
    class: string;
    percentage: number;
    pixels: number;
  }[];
  show_overlay: boolean;
}

interface SegmentationResponse {
  success: boolean;
  statistics: SegmentationStatistics;
  visualization_data: VisualizationData;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: SegmentationRequest;
  image_info: {
    width: number;
    height: number;
    image_index: number;
  };
}
```

## Styling Suggestions

### Color Legend Item
```css
.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 8px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.05);
}

.color-box {
  width: 20px;
  height: 20px;
  border-radius: 4px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.class-name {
  flex: 1;
  font-weight: 500;
}

.percentage {
  font-weight: 600;
  color: #4ade80;
}

.pixel-count {
  font-size: 0.9em;
  color: rgba(255, 255, 255, 0.6);
}
```

### Image Container
```css
.image-container {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.2);
}

.image-container img {
  width: 100%;
  height: auto;
  display: block;
}

.opacity-slider {
  position: absolute;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  width: 80%;
}
```

## User Experience Enhancements

1. **Loading States**: Show skeleton loaders or spinners during processing
2. **Error Handling**: Display user-friendly error messages
3. **Responsive Design**: Ensure images scale properly on different screen sizes
4. **Keyboard Shortcuts**: Add shortcuts for common actions (R for run, Z for zoom)
5. **Export Options**: Allow users to download segmentation results
6. **Comparison Mode**: Side-by-side comparison of different parameters
7. **Tutorial Overlay**: First-time user guidance

## Performance Optimization

1. **Lazy Loading**: Load images only when needed
2. **Caching**: Cache algorithm info to reduce API calls
3. **Debouncing**: Debounce slider inputs to avoid excessive API calls
4. **Image Compression**: Consider compressing displayed images for faster loading
5. **Progressive Loading**: Show lower quality preview while processing

## Accessibility

1. Use semantic HTML elements
2. Add ARIA labels for interactive controls
3. Ensure sufficient color contrast
4. Support keyboard navigation
5. Provide text alternatives for visual information

## Testing Checklist

- [ ] All parameters update correctly
- [ ] Images display properly
- [ ] Class legend shows all detected classes
- [ ] Statistics calculate correctly
- [ ] Zoom/pan works smoothly
- [ ] Error states display appropriately
- [ ] Loading states show during processing
- [ ] Responsive on mobile devices
- [ ] Works with all sample images
- [ ] Overlay toggle functions correctly

## Related Algorithms

Link to related computer vision algorithms:
- YOLO Object Detection
- Edge Detection (Canny)
- Image Classification
- Instance Segmentation (future)

## Next Steps

1. Create the React component following this guide
2. Test with all sample images
3. Adjust styling to match website theme
4. Add to routing configuration
5. Update navigation to include semantic segmentation
6. Write unit tests for component
7. Add integration tests for API calls
