# YOLO Frontend Integration Guide

## Overview
This guide provides instructions for integrating the YOLO Object Detection backend API with the frontend React application.

## API Endpoints

### 1. Run Detection
**Endpoint**: `POST /api/computer-vision/yolo/detect`

**Request**:
```typescript
interface YOLORequest {
  model_version: 'yolov8n' | 'yolov8s' | 'yolov8m' | 'yolov5s';
  confidence_threshold: number; // 0.1 - 0.9
  iou_threshold: number; // 0.1 - 0.9
  max_detections: number; // 10 - 300
  image_index: number; // 0 - 14
  class_filter: 'all' | 'person' | 'vehicle' | 'animal';
}
```

**Usage**:
```typescript
import { apiService } from '@/services/api';

const runYOLODetection = async (params: YOLORequest) => {
  const response = await apiService.post(
    '/api/computer-vision/yolo/detect',
    params
  );
  return response;
};
```

### 2. Get Algorithm Info
**Endpoint**: `GET /api/computer-vision/yolo/info`

**Usage**:
```typescript
const getYOLOInfo = async () => {
  const info = await apiService.getAlgorithmInfo(
    'computer-vision',
    'yolo'
  );
  return info;
};
```

### 3. List Computer Vision Algorithms
**Endpoint**: `GET /api/computer-vision/algorithms`

**Usage**:
```typescript
const listCVAlgorithms = async () => {
  const algorithms = await apiService.listAlgorithms('computer-vision');
  return algorithms;
};
```

## Response Types

```typescript
interface Detection {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  class_name: string;
  class_id: number;
  confidence: number;
}

interface DetectionStatistics {
  total_detections: number;
  class_counts: Record<string, number>;
  avg_confidence: number;
  confidence_distribution: Record<string, number>;
}

interface YOLOResponse {
  success: boolean;
  detections: Detection[];
  statistics: DetectionStatistics;
  visualization_data: {
    original_image: string; // base64 data URL
    annotated_image: string; // base64 data URL
    detection_list: Array<{
      bbox: number[];
      class_name: string;
      confidence: number;
    }>;
    confidence_chart: Array<{
      class: string;
      confidence: number;
    }>;
  };
  execution_time_ms: number;
  model_info: {
    model_name: string;
    model_version: string;
    num_classes: number;
    framework: string;
    device: string;
    input_size: string;
    architecture: string;
  };
  parameters_used: YOLORequest;
  image_info: {
    width: number;
    height: number;
    channels: number;
    image_index: number;
  };
}
```

## Component Structure

### Recommended Component Hierarchy

```
YOLODemo/
├── index.tsx                    # Main component
├── YOLOControls.tsx            # Parameter controls
├── YOLOVisualization.tsx       # Image display
├── YOLODetectionTable.tsx      # Detection results table
├── YOLOStatistics.tsx          # Statistics display
└── YOLOCharts.tsx              # Class distribution & confidence charts
```

## Implementation Example

### Main Component (YOLODemo.tsx)

```typescript
import React, { useState, useEffect } from 'react';
import { apiService } from '@/services/api';
import YOLOControls from './YOLOControls';
import YOLOVisualization from './YOLOVisualization';
import YOLODetectionTable from './YOLODetectionTable';
import YOLOStatistics from './YOLOStatistics';
import YOLOCharts from './YOLOCharts';

interface YOLOParams {
  model_version: string;
  confidence_threshold: number;
  iou_threshold: number;
  max_detections: number;
  image_index: number;
  class_filter: string;
}

export const YOLODemo: React.FC = () => {
  const [params, setParams] = useState<YOLOParams>({
    model_version: 'yolov8n',
    confidence_threshold: 0.25,
    iou_threshold: 0.45,
    max_detections: 100,
    image_index: 0,
    class_filter: 'all'
  });
  
  const [response, setResponse] = useState<YOLOResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<any>(null);

  // Load algorithm metadata on mount
  useEffect(() => {
    const loadMetadata = async () => {
      try {
        const info = await apiService.getAlgorithmInfo('computer-vision', 'yolo');
        setMetadata(info.metadata);
      } catch (err) {
        console.error('Failed to load metadata:', err);
      }
    };
    loadMetadata();
  }, []);

  // Run detection
  const runDetection = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await apiService.post(
        '/api/computer-vision/yolo/detect',
        params
      );
      setResponse(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Detection failed');
      console.error('Detection error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="yolo-demo">
      <h1>YOLO Object Detection</h1>
      
      {/* Parameter Controls */}
      <YOLOControls
        params={params}
        onChange={setParams}
        onRun={runDetection}
        loading={loading}
        metadata={metadata}
      />

      {/* Error Display */}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* Results */}
      {response && (
        <>
          {/* Images Side-by-Side */}
          <YOLOVisualization
            originalImage={response.visualization_data.original_image}
            annotatedImage={response.visualization_data.annotated_image}
            imageInfo={response.image_info}
          />

          {/* Statistics Summary */}
          <YOLOStatistics
            statistics={response.statistics}
            executionTime={response.execution_time_ms}
            modelInfo={response.model_info}
          />

          {/* Detection Results Table */}
          <YOLODetectionTable
            detections={response.detections}
          />

          {/* Charts */}
          <YOLOCharts
            classCounts={response.statistics.class_counts}
            confidenceDistribution={response.statistics.confidence_distribution}
            confidenceChart={response.visualization_data.confidence_chart}
          />
        </>
      )}
    </div>
  );
};
```

### Controls Component (YOLOControls.tsx)

```typescript
import React from 'react';

interface YOLOControlsProps {
  params: YOLOParams;
  onChange: (params: YOLOParams) => void;
  onRun: () => void;
  loading: boolean;
  metadata: any;
}

export const YOLOControls: React.FC<YOLOControlsProps> = ({
  params,
  onChange,
  onRun,
  loading,
  metadata
}) => {
  const updateParam = (key: string, value: any) => {
    onChange({ ...params, [key]: value });
  };

  return (
    <div className="yolo-controls">
      {/* Model Version */}
      <div className="control-group">
        <label>Model Version</label>
        <select
          value={params.model_version}
          onChange={(e) => updateParam('model_version', e.target.value)}
        >
          <option value="yolov8n">YOLOv8 Nano (Fastest)</option>
          <option value="yolov8s">YOLOv8 Small (Balanced)</option>
          <option value="yolov8m">YOLOv8 Medium (Most Accurate)</option>
          <option value="yolov5s">YOLOv5 Small</option>
        </select>
      </div>

      {/* Confidence Threshold */}
      <div className="control-group">
        <label>
          Confidence Threshold: {params.confidence_threshold.toFixed(2)}
        </label>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={params.confidence_threshold}
          onChange={(e) => updateParam('confidence_threshold', parseFloat(e.target.value))}
        />
      </div>

      {/* IoU Threshold */}
      <div className="control-group">
        <label>
          IoU Threshold: {params.iou_threshold.toFixed(2)}
        </label>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={params.iou_threshold}
          onChange={(e) => updateParam('iou_threshold', parseFloat(e.target.value))}
        />
      </div>

      {/* Max Detections */}
      <div className="control-group">
        <label>Max Detections: {params.max_detections}</label>
        <input
          type="number"
          min="10"
          max="300"
          step="10"
          value={params.max_detections}
          onChange={(e) => updateParam('max_detections', parseInt(e.target.value))}
        />
      </div>

      {/* Image Index */}
      <div className="control-group">
        <label>Sample Image: {params.image_index}</label>
        <input
          type="range"
          min="0"
          max="14"
          step="1"
          value={params.image_index}
          onChange={(e) => updateParam('image_index', parseInt(e.target.value))}
        />
        <span className="image-description">
          {getImageDescription(params.image_index)}
        </span>
      </div>

      {/* Class Filter */}
      <div className="control-group">
        <label>Class Filter</label>
        <select
          value={params.class_filter}
          onChange={(e) => updateParam('class_filter', e.target.value)}
        >
          <option value="all">All Classes</option>
          <option value="person">Person</option>
          <option value="vehicle">Vehicle</option>
          <option value="animal">Animal</option>
        </select>
      </div>

      {/* Run Button */}
      <button
        onClick={onRun}
        disabled={loading}
        className="run-button"
      >
        {loading ? 'Detecting...' : 'Run Detection'}
      </button>
    </div>
  );
};

function getImageDescription(index: number): string {
  const descriptions = [
    'Street scene with bus',
    'Soccer players',
    'Sports action',
    'Traffic scene',
    'City street',
    'Parking lot',
    'Dogs in park',
    'Living room',
    'Kitchen',
    'Horses in field',
    'Office desk',
    'Airport',
    'Beach scene',
    'Indoor cats',
    'Train station'
  ];
  return descriptions[index] || 'Unknown';
}
```

### Visualization Component (YOLOVisualization.tsx)

```typescript
import React from 'react';

interface YOLOVisualizationProps {
  originalImage: string;
  annotatedImage: string;
  imageInfo: {
    width: number;
    height: number;
    channels: number;
    image_index: number;
  };
}

export const YOLOVisualization: React.FC<YOLOVisualizationProps> = ({
  originalImage,
  annotatedImage,
  imageInfo
}) => {
  return (
    <div className="yolo-visualization">
      <div className="image-comparison">
        {/* Original Image */}
        <div className="image-container">
          <h3>Original Image</h3>
          <img src={originalImage} alt="Original" />
          <div className="image-info">
            {imageInfo.width} × {imageInfo.height} px
          </div>
        </div>

        {/* Annotated Image */}
        <div className="image-container">
          <h3>Detected Objects</h3>
          <img src={annotatedImage} alt="Annotated with detections" />
          <div className="image-info">
            With bounding boxes and labels
          </div>
        </div>
      </div>
    </div>
  );
};
```

### Detection Table Component (YOLODetectionTable.tsx)

```typescript
import React from 'react';

interface Detection {
  bbox: number[];
  class_name: string;
  class_id: number;
  confidence: number;
}

interface YOLODetectionTableProps {
  detections: Detection[];
}

export const YOLODetectionTable: React.FC<YOLODetectionTableProps> = ({
  detections
}) => {
  return (
    <div className="yolo-detection-table">
      <h3>Detection Results ({detections.length} objects)</h3>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Class</th>
            <th>Confidence</th>
            <th>Bounding Box (x1, y1, x2, y2)</th>
          </tr>
        </thead>
        <tbody>
          {detections.map((det, idx) => (
            <tr key={idx}>
              <td>{idx + 1}</td>
              <td>
                <span className="class-badge">{det.class_name}</span>
              </td>
              <td>
                <span className="confidence-value">
                  {(det.confidence * 100).toFixed(1)}%
                </span>
              </td>
              <td className="bbox-coords">
                {det.bbox.map(v => v.toFixed(1)).join(', ')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
```

### Statistics Component (YOLOStatistics.tsx)

```typescript
import React from 'react';

interface YOLOStatisticsProps {
  statistics: {
    total_detections: number;
    class_counts: Record<string, number>;
    avg_confidence: number;
    confidence_distribution: Record<string, number>;
  };
  executionTime: number;
  modelInfo: {
    model_name: string;
    framework: string;
    device: string;
  };
}

export const YOLOStatistics: React.FC<YOLOStatisticsProps> = ({
  statistics,
  executionTime,
  modelInfo
}) => {
  const uniqueClasses = Object.keys(statistics.class_counts).length;

  return (
    <div className="yolo-statistics">
      <h3>Detection Statistics</h3>
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Total Objects</div>
          <div className="stat-value">{statistics.total_detections}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Unique Classes</div>
          <div className="stat-value">{uniqueClasses}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Avg Confidence</div>
          <div className="stat-value">
            {(statistics.avg_confidence * 100).toFixed(1)}%
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Execution Time</div>
          <div className="stat-value">{executionTime.toFixed(1)}ms</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Model</div>
          <div className="stat-value">{modelInfo.model_name}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Device</div>
          <div className="stat-value">{modelInfo.device}</div>
        </div>
      </div>
    </div>
  );
};
```

### Charts Component (YOLOCharts.tsx)

```typescript
import React from 'react';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

interface YOLOChartsProps {
  classCounts: Record<string, number>;
  confidenceDistribution: Record<string, number>;
  confidenceChart: Array<{ class: string; confidence: number }>;
}

export const YOLOCharts: React.FC<YOLOChartsProps> = ({
  classCounts,
  confidenceDistribution,
  confidenceChart
}) => {
  // Prepare data for class distribution pie chart
  const classData = Object.entries(classCounts).map(([name, count]) => ({
    name,
    value: count
  }));

  // Prepare data for confidence histogram
  const confidenceData = Object.entries(confidenceDistribution).map(([range, count]) => ({
    range,
    count
  }));

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div className="yolo-charts">
      {/* Class Distribution Pie Chart */}
      <div className="chart-container">
        <h3>Class Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={classData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {classData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Confidence Histogram */}
      <div className="chart-container">
        <h3>Confidence Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={confidenceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top Detections by Confidence */}
      <div className="chart-container">
        <h3>Top Detections by Confidence</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={confidenceChart.slice(0, 10)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="class" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="confidence" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
```

## Styling Recommendations

### CSS Example (yolo-demo.css)

```css
.yolo-demo {
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
}

.yolo-controls {
  background: #f5f5f5;
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.control-group {
  margin-bottom: 1rem;
}

.control-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
}

.control-group input[type="range"] {
  width: 100%;
}

.run-button {
  background: #4CAF50;
  color: white;
  border: none;
  padding: 0.75rem 2rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  margin-top: 1rem;
}

.run-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.image-comparison {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin: 2rem 0;
}

.image-container img {
  width: 100%;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.yolo-detection-table table {
  width: 100%;
  border-collapse: collapse;
}

.yolo-detection-table th,
.yolo-detection-table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

.class-badge {
  background: #2196F3;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
}

.confidence-value {
  font-weight: 600;
  color: #4CAF50;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}

.stat-card {
  background: white;
  padding: 1rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  text-align: center;
}

.stat-label {
  font-size: 0.875rem;
  color: #666;
  margin-bottom: 0.5rem;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: #333;
}

.yolo-charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
}

.chart-container {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
```

## Dependencies

Add these to your `package.json`:

```json
{
  "dependencies": {
    "recharts": "^2.x.x",
    "axios": "^1.x.x"
  }
}
```

## Testing

1. Test all model versions (yolov8n, yolov8s, yolov8m, yolov5s)
2. Test all sample images (0-14)
3. Test class filters (all, person, vehicle, animal)
4. Test parameter ranges (confidence, IoU, max_detections)
5. Verify images load correctly
6. Verify charts render properly
7. Test error handling

## Performance Considerations

1. Images are base64 encoded - may be large
2. Detection can take 10-100ms depending on model
3. Consider adding loading states
4. Consider caching results for same parameters
5. Add error boundaries for robustness

## Next Steps

1. Create the component files in `frontend/src/components/algorithm-demos/yolo/`
2. Add routing in the main app
3. Style components to match existing design system
4. Add responsive design for mobile
5. Add export functionality for results
6. Add comparison mode for different parameters
