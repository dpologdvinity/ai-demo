# Activation Functions

Interactive demonstration and comparison of neural network activation functions.

## Overview

This module provides a comprehensive visualization and comparison of common activation functions used in neural networks, including their mathematical properties, derivatives, and practical implications for deep learning.

## Implemented Functions

### 1. ReLU (Rectified Linear Unit)
- **Formula**: `f(x) = max(0, x)`
- **Range**: `[0, ∞)`
- **Derivative**: `f'(x) = 1 if x > 0 else 0`
- **Properties**: Fast, helps with vanishing gradients
- **Issue**: Dead neurons (zero gradient for x < 0)

### 2. Leaky ReLU
- **Formula**: `f(x) = x if x > 0 else αx` (typically α = 0.01)
- **Range**: `(-∞, ∞)`
- **Derivative**: `f'(x) = 1 if x > 0 else α`
- **Properties**: Fixes dead neuron problem, maintains efficiency
- **Advantage**: Non-zero gradient for negative inputs

### 3. Sigmoid
- **Formula**: `f(x) = 1 / (1 + e^(-x))`
- **Range**: `(0, 1)`
- **Derivative**: `f'(x) = f(x) * (1 - f(x))`
- **Properties**: Smooth, output bounded to (0, 1)
- **Issue**: Vanishing gradients for large |x|

### 4. Tanh (Hyperbolic Tangent)
- **Formula**: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- **Range**: `(-1, 1)`
- **Derivative**: `f'(x) = 1 - f(x)²`
- **Properties**: Zero-centered, stronger gradients than sigmoid
- **Issue**: Still suffers from vanishing gradients

### 5. ELU (Exponential Linear Unit)
- **Formula**: `f(x) = x if x > 0 else α(e^x - 1)`
- **Range**: `(-α, ∞)`
- **Derivative**: `f'(x) = 1 if x > 0 else α*e^x`
- **Properties**: Smooth, can produce negative values
- **Cost**: Exponential computation for x < 0

### 6. Swish
- **Formula**: `f(x) = x * sigmoid(x)`
- **Range**: `(-∞, ∞)`
- **Derivative**: `f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`
- **Properties**: Self-gated, smooth, non-monotonic
- **Advantage**: Can outperform ReLU in deeper networks

## Features

### Function Visualization
- **Function Plots**: Overlay multiple activation functions on the same chart
- **Derivative Plots**: Compare gradient behavior across functions
- **Interactive Range**: Adjust input range dynamically
- **Smooth Curves**: Configurable number of points for visualization

### Dead Neuron Demonstration
Demonstrates the "dying ReLU" problem:
- Shows how ReLU neurons can become inactive (zero gradient)
- Compares with Leaky ReLU's solution (small non-zero gradient)
- Interactive visualization with different input values

### Comparison Table
Comprehensive property comparison:
- Mathematical formulas
- Output ranges
- Derivative ranges
- Monotonicity
- Zero-centered property
- Gradient behavior for negative inputs
- Pros and cons

## API Endpoints

### POST `/deep-learning/activation-functions/compute`

Compute activation functions and their derivatives.

**Request Body**:
```json
{
  "function_type": "leaky_relu",
  "alpha": 0.01,
  "input_range": [-10.0, 10.0],
  "compare_all": true,
  "num_points": 200
}
```

**Parameters**:
- `function_type`: Primary function to demonstrate (`relu`, `leaky_relu`, `sigmoid`, `tanh`, `elu`, `swish`)
- `alpha`: Negative slope for Leaky ReLU (0.0 - 0.3, default: 0.01)
- `input_range`: Range for x-axis `[min, max]` (default: `[-10, 10]`)
- `compare_all`: Show all functions together (default: `true`)
- `num_points`: Number of points for visualization (50-1000, default: 200)

**Response**:
```json
{
  "success": true,
  "function_data": {
    "relu": {
      "x": [...],
      "y": [...],
      "derivative": [...]
    },
    ...
  },
  "comparison_table": [...],
  "dead_neuron_demo": {
    "input_value": -5.0,
    "relu": {
      "output": 0.0,
      "gradient": 0.0,
      "is_dead": true
    },
    "leaky_relu": {
      "output": -0.05,
      "gradient": 0.01,
      "is_dead": false
    }
  },
  "visualization_data": {...},
  "execution_time_ms": 1.23
}
```

### GET `/deep-learning/activation-functions/info`

Get algorithm metadata and dataset information.

## Usage Example

```python
from algorithms.deep_learning.activation_functions import ActivationFunctionsModel

# Initialize model
model = ActivationFunctionsModel(alpha=0.01)

# Compute functions
import numpy as np
x = np.linspace(-10, 10, 200)

# Single function
relu_data = model.compute_function_data(x, 'relu')
y = relu_data['y']
dy = relu_data['derivative']

# All functions
x, function_data = model.generate_comparison_data([-10.0, 10.0], num_points=200)

# Dead neuron demonstration
demo = model.demonstrate_dead_neurons(x_negative=-5.0)
print(f"ReLU gradient: {demo['relu']['gradient']}")  # 0.0
print(f"Leaky ReLU gradient: {demo['leaky_relu']['gradient']}")  # 0.01
```

## Educational Use Cases

1. **Neural Network Design**: Understand which activation to use for different layers
2. **Gradient Flow**: Visualize how gradients propagate through different functions
3. **Vanishing Gradients**: See why sigmoid/tanh cause problems in deep networks
4. **Dead Neurons**: Understand and prevent the dying ReLU problem
5. **Function Properties**: Compare ranges, derivatives, and behavior

## Complexity

- **Time Complexity**: O(n) where n is the number of points
- **Space Complexity**: O(n) for storing function values and derivatives

## Related Algorithms

- Feedforward Neural Networks (MLP)
- Convolutional Neural Networks (CNN)
- Backpropagation
- Gradient Descent

## References

1. Glorot et al. (2011) - "Deep Sparse Rectifier Neural Networks"
2. He et al. (2015) - "Delving Deep into Rectifiers"
3. Maas et al. (2013) - "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
4. Clevert et al. (2015) - "Fast and Accurate Deep Network Learning by ELU"
5. Ramachandran et al. (2017) - "Searching for Activation Functions"

## Testing

Run the test suite:

```bash
cd backend
source venv/bin/activate
python test_activation_functions.py
```

All tests should pass, validating:
- Function computations
- Derivative calculations
- Dead neuron demonstration
- Comparison data generation
- Schema validation
