# Convolutional Layers Implementation

## Overview

This document describes the implementation of the Convolutional Layers demonstration for the AI algorithms demonstration website. The implementation provides an educational tool for understanding how convolution operations work in CNNs.

## Implementation Details

### Files Created

1. **`algorithms/deep_learning/convolutional_layers/__init__.py`**
   - Module initialization
   - Exports: `ConvolutionalLayersModel`, `ConvolutionalLayersRequest`, `ConvolutionalLayersResponse`

2. **`algorithms/deep_learning/convolutional_layers/schema.py`**
   - Pydantic request/response schemas
   - Request parameters: num_filters, kernel_size, stride, padding, activation
   - Response includes: input image, filter kernels, feature maps, common filters, output dimensions

3. **`algorithms/deep_learning/convolutional_layers/model.py`**
   - Main model implementation using PyTorch
   - Demonstrates convolution operations with learnable filters
   - Applies common predefined filters (Sobel, Gaussian, Sharpen, Laplacian)
   - Computes output dimensions based on stride/padding
   - Calculates activation statistics

4. **`algorithms/deep_learning/convolutional_layers/data.py`**
   - Generates synthetic sample image with various features
   - Defines common filter kernels (Sobel X/Y, Gaussian blur, Sharpen, Edge detect)
   - Provides dataset information and filter descriptions

5. **`api/routes/deep_learning.py`** (updated)
   - Added imports for convolutional layers module
   - Registered algorithm metadata with AlgorithmRegistry
   - Added POST `/convolutional-layers/demo` endpoint
   - Added GET `/convolutional-layers/info` endpoint

6. **`test_convolutional_layers.py`**
   - Comprehensive unit tests for all components
   - Tests: image loading, filters, model initialization, forward pass, activations
   - All tests pass successfully

7. **`test_convolutional_layers_api.py`**
   - API endpoint tests (requires running server)

## Algorithm Metadata

- **ID**: `convolutional-layers`
- **Name**: Convolutional Layers
- **Category**: Deep Learning
- **Difficulty**: Beginner
- **Slug**: `convolutional-layers`

### Parameters

1. **num_filters** (range: 8-128, default: 32)
   - Number of convolutional filters to learn

2. **kernel_size** (options: 3, 5, 7, default: 3)
   - Size of the convolution kernel

3. **stride** (options: 1, 2, 3, default: 1)
   - Stride for convolution operation

4. **padding** (options: 'same', 'valid', default: 'same')
   - Padding type for convolution

5. **activation** (options: 'relu', 'tanh', 'none', default: 'relu')
   - Activation function to apply

### Use Cases

- Image feature extraction
- Edge detection
- Pattern recognition
- Object detection preprocessing
- CNNs building block

### Complexity

- **Time**: O(H×W×K²×F) where H,W = image size, K = kernel size, F = filters
- **Space**: O(H×W×F)

## Features Implemented

### Core Functionality

1. **Convolution Operations**
   - Applies learnable filters to input image
   - Supports different kernel sizes (3x3, 5x5, 7x7)
   - Configurable stride and padding
   - Multiple filters in parallel

2. **Filter Visualization**
   - Extracts and visualizes learned filter weights
   - Shows first 8 filters for UI display
   - Includes statistics (min, max, mean weights)

3. **Feature Maps**
   - Outputs feature maps for each filter
   - Shows activation patterns
   - Includes statistics per feature map

4. **Common Filters Demo**
   - **Sobel X**: Detects vertical edges
   - **Sobel Y**: Detects horizontal edges
   - **Gaussian Blur**: Smooths the image
   - **Sharpen**: Enhances edges and details
   - **Edge Detect (Laplacian)**: Detects all edges

5. **Activation Functions**
   - ReLU: max(0, x)
   - Tanh: hyperbolic tangent
   - None: linear (no activation)

6. **Output Dimension Calculation**
   - Formula: (input_size + 2×padding - kernel_size) / stride + 1
   - Demonstrates effect of stride and padding
   - Shows dimension changes in visualization

7. **Activation Statistics**
   - Mean activation across feature maps
   - Standard deviation
   - Max and min activations
   - Sparsity (fraction of zero activations)

### Sample Image

The demonstration uses a synthetic 28×28 grayscale image with:
- Rectangular shapes for edge detection
- Diagonal lines for orientation-specific filters
- Gradients for texture analysis
- Circular patterns for radial features
- Added noise for realistic texture

## API Endpoints

### POST `/api/deep-learning/convolutional-layers/demo`

Runs the convolutional layers demonstration.

**Request Body:**
```json
{
  "num_filters": 32,
  "kernel_size": 3,
  "stride": 1,
  "padding": "same",
  "activation": "relu",
  "random_state": 42
}
```

**Response:**
```json
{
  "success": true,
  "input_image": [[...]],
  "filter_kernels": [
    {
      "filter_id": 0,
      "weights": [[...]],
      "shape": [3, 3],
      "min_weight": -0.5,
      "max_weight": 0.5,
      "mean_weight": 0.01
    }
  ],
  "feature_maps": [
    {
      "filter_id": 0,
      "output": [[...]],
      "shape": [28, 28],
      "min_value": 0.0,
      "max_value": 1.5,
      "mean_value": 0.3
    }
  ],
  "common_filters": {
    "sobel_x": {
      "output": [[...]],
      "description": "Detects vertical edges"
    },
    "sobel_y": {...},
    "gaussian_blur": {...},
    "sharpen": {...},
    "edge_detect": {...}
  },
  "output_dimensions": {
    "height": 28,
    "width": 28,
    "channels": 32
  },
  "activation_stats": {
    "mean_activation": 0.024,
    "std_activation": 0.087,
    "max_activation": 0.497,
    "min_activation": 0.0,
    "sparsity": 0.612
  },
  "visualization_data": {
    "input_shape": [28, 28],
    "output_shape": [28, 28],
    "num_filters": 32,
    "num_visualized": 8,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "activation": "relu",
    "dimension_calculation": {
      "formula": "(input_size + 2*padding - kernel_size) / stride + 1",
      "input_height": 28,
      "input_width": 28,
      "padding": 1,
      "kernel_size": 3,
      "stride": 1,
      "output_height": 28,
      "output_width": 28
    }
  },
  "execution_time_ms": 2.48,
  "model_info": {
    "num_filters": 32,
    "kernel_size": 3,
    "stride": 1,
    "padding": "same",
    "activation": "relu",
    "total_parameters": 320,
    "trainable_parameters": 320
  },
  "parameters_used": {
    "num_filters": 32,
    "kernel_size": 3,
    "stride": 1,
    "padding": "same",
    "activation": "relu"
  }
}
```

### GET `/api/deep-learning/convolutional-layers/info`

Returns algorithm metadata and dataset information.

**Response:**
```json
{
  "metadata": {
    "id": "convolutional-layers",
    "name": "Convolutional Layers",
    "slug": "convolutional-layers",
    "category": "deep_learning",
    "description": "Demonstrate convolution operation and filters for feature extraction",
    "difficulty": "beginner",
    "tags": ["deep-learning", "cnn", "convolution", "feature-extraction"],
    "use_cases": [...],
    "complexity": {
      "time": "O(H*W*K²*F)",
      "space": "O(H*W*F)"
    },
    "parameters": [...],
    "theory": "...",
    "pros": [...],
    "cons": [...],
    "related_algorithms": [...]
  },
  "dataset": {
    "name": "Sample Image",
    "description": "Synthetic grayscale image with various features for demonstrating convolution",
    "image_size": "28x28 pixels",
    "channels": 1,
    "features": [...],
    "value_range": "0.0 to 1.0 (normalized)"
  }
}
```

## Frontend Integration

The frontend should implement the following visualizations:

### 1. Input Image Display
- Show the 28×28 grayscale input image
- Display pixel values on hover

### 2. Filter Kernels Grid
- 2×4 grid showing 8 filter kernels
- Visualize weights as heatmaps (negative=blue, positive=red)
- Show weight statistics below each kernel

### 3. Feature Maps Grid
- 2×4 grid showing 8 feature maps
- Visualize activations as heatmaps
- Show activation statistics below each map
- Highlight sparse activations (ReLU effect)

### 4. Common Filters Demo
- Side-by-side comparison of common filters:
  - Original image
  - Sobel X output (vertical edges)
  - Sobel Y output (horizontal edges)
  - Gaussian blur output
  - Sharpen output
  - Edge detect output
- Include filter descriptions

### 5. Output Size Calculator
- Interactive calculator showing dimension changes
- Formula visualization: (input_size + 2×padding - kernel_size) / stride + 1
- Update dynamically as parameters change

### 6. Parameter Controls
- Slider for num_filters (8-128)
- Dropdown for kernel_size (3, 5, 7)
- Dropdown for stride (1, 2, 3)
- Toggle for padding (same/valid)
- Dropdown for activation (relu/tanh/none)
- Real-time updates when parameters change

### 7. Activation Statistics Panel
- Bar chart or metrics panel showing:
  - Mean activation
  - Max activation
  - Sparsity percentage
- Explain sparsity in context of ReLU

## Testing

### Unit Tests

Run unit tests:
```bash
cd backend
source venv/bin/activate
python test_convolutional_layers.py
```

**Test Coverage:**
- ✅ Sample image loading
- ✅ Common filters
- ✅ Dataset info
- ✅ Model initialization
- ✅ Output dimension calculations
- ✅ Forward pass
- ✅ Filter kernel extraction
- ✅ Activation functions
- ✅ Common filters application
- ✅ Full demonstration

### API Tests

Start the server and run API tests:
```bash
# Terminal 1: Start server
cd backend
source venv/bin/activate
uvicorn main:app --reload

# Terminal 2: Run tests
cd backend
source venv/bin/activate
python test_convolutional_layers_api.py
```

## Educational Value

This demonstration teaches:

1. **Convolution Basics**: How filters slide across images
2. **Feature Detection**: What different filters detect (edges, textures)
3. **Hyperparameters**: Effect of kernel size, stride, and padding
4. **Activation Functions**: How ReLU introduces non-linearity and sparsity
5. **Output Dimensions**: Mathematical relationship between parameters and output size
6. **Common Filters**: Standard image processing operations
7. **CNNs Foundation**: Building block for understanding deeper architectures

## Related Algorithms

- CNN (full network)
- Pooling layers
- Batch Normalization
- ResNet (residual connections)

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- FastAPI >= 0.68.0
- Pydantic >= 1.8.0

## Next Steps

To enhance the demonstration:

1. Add more common filters (Prewitt, Scharr, etc.)
2. Show stride animation
3. Visualize receptive fields
4. Add 3D visualization of filter weights
5. Compare learned vs. handcrafted filters
6. Show intermediate activations during forward pass
7. Add interactive filter designer
8. Demonstrate on real images (upload functionality)

## Notes

- Filter weights are randomly initialized with Xavier initialization
- Output shows first 8 filters/feature maps for UI clarity
- All 32 (or configured number) filters are computed internally
- Execution is fast (~2-5ms) due to simple forward pass
- No training loop - demonstrates single forward pass
- Suitable for beginners learning CNN basics
