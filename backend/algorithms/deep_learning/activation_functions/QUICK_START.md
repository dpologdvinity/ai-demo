# Activation Functions - Quick Start Guide

## TL;DR

Interactive demonstration of 6 neural network activation functions with visualization and comparison.

## Quick Test

```bash
cd backend
source venv/bin/activate
python test_activation_functions.py
```

## Quick Example

```python
from algorithms.deep_learning.activation_functions.model import compute_activation_functions

# Compute all functions
result = compute_activation_functions(
    function_type='leaky_relu',
    alpha=0.01,
    input_range=[-10, 10],
    compare_all=True,
    num_points=200
)

print(f"Functions: {list(result['function_data'].keys())}")
# Output: ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu', 'swish']

print(f"Dead ReLU gradient: {result['dead_neuron_demo']['relu']['gradient']}")
# Output: 0.0
```

## API Endpoints

### Compute Functions
```bash
POST /deep-learning/activation-functions/compute

# Example request
{
  "function_type": "leaky_relu",
  "alpha": 0.01,
  "input_range": [-10.0, 10.0],
  "compare_all": true,
  "num_points": 200
}
```

### Get Info
```bash
GET /deep-learning/activation-functions/info
```

## Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| function_type | string | - | leaky_relu | Primary function to demo |
| alpha | float | 0.0-0.3 | 0.01 | Leaky ReLU negative slope |
| input_range | array | [-100,100] | [-10,10] | X-axis range |
| compare_all | boolean | - | true | Show all functions |
| num_points | int | 50-1000 | 200 | Visualization points |

## Function Options

- `relu` - Rectified Linear Unit
- `leaky_relu` - Leaky ReLU (fixes dead neurons)
- `sigmoid` - Sigmoid (output in 0-1)
- `tanh` - Hyperbolic tangent (output in -1 to 1)
- `elu` - Exponential Linear Unit
- `swish` - Self-gated activation

## Response Structure

```json
{
  "success": true,
  "function_data": {
    "relu": {
      "x": [...],          // Input values
      "y": [...],          // Function outputs
      "derivative": [...]  // Gradients
    }
  },
  "comparison_table": [...],     // Properties comparison
  "dead_neuron_demo": {...},     // Dead ReLU demonstration
  "visualization_data": {...},   // Chart-ready data
  "execution_time_ms": 1.23
}
```

## Common Use Cases

### 1. Compare All Functions
```python
result = compute_activation_functions(compare_all=True)
```

### 2. Focus on One Function
```python
result = compute_activation_functions(
    function_type='relu',
    compare_all=False
)
```

### 3. Adjust Leaky ReLU Slope
```python
result = compute_activation_functions(
    function_type='leaky_relu',
    alpha=0.1,  # Larger negative slope
    compare_all=True
)
```

### 4. High-Resolution Visualization
```python
result = compute_activation_functions(
    num_points=1000,  # Smoother curves
    input_range=[-20, 20]  # Wider range
)
```

## Dead Neuron Demo

```python
demo = result['dead_neuron_demo']

print(f"Input: {demo['input_value']}")
# Output: -5.0

print(f"ReLU gradient: {demo['relu']['gradient']}")
# Output: 0.0 (dead neuron!)

print(f"Leaky ReLU gradient: {demo['leaky_relu']['gradient']}")
# Output: 0.01 (still learning!)
```

## Frontend Integration

### Plot Functions
```javascript
const functionsChart = response.visualization_data.functions_chart;
functionsChart.forEach(func => {
  plotLine(func.name, func.data); // [{x: -10, y: 0}, ...]
});
```

### Plot Derivatives
```javascript
const derivativesChart = response.visualization_data.derivatives_chart;
derivativesChart.forEach(func => {
  plotLine(func.name, func.data);
});
```

### Show Comparison Table
```javascript
const table = response.comparison_table;
table.forEach(func => {
  displayRow(
    func.function,    // "ReLU"
    func.formula,     // "f(x) = max(0, x)"
    func.range,       // "[0, ∞)"
    func.pros,        // "Fast computation..."
    func.cons         // "Dead neurons..."
  );
});
```

## File Structure

```
activation_functions/
├── __init__.py       # Exports
├── model.py          # Core implementation
├── schema.py         # Request/Response schemas
├── data.py           # Dataset info
├── README.md         # Full documentation
└── QUICK_START.md    # This file
```

## Performance

- **Computation**: < 1ms for 200 points
- **Memory**: O(n) where n = num_points
- **Scalability**: Handles 1000 points smoothly

## Testing

Run the test suite:
```bash
python test_activation_functions.py
```

Expected output:
```
✓ All activation functions working correctly
✓ Dead neuron demonstration working correctly
✓ Comparison data generation working correctly
✓ Computation function working correctly
✓ Request schema working correctly
✓ Dataset info working correctly
```

## Troubleshooting

### Import Error
```bash
# Activate virtual environment
cd backend
source venv/bin/activate
```

### Validation Error
Check parameter ranges:
- `alpha`: Must be 0.0-0.3
- `input_range`: Must be 2 values [min, max]
- `num_points`: Must be 50-1000
- `function_type`: Must be valid function name

### Wrong Output
Verify request parameters match expected types:
```python
from algorithms.deep_learning.activation_functions import ActivationFunctionsRequest

request = ActivationFunctionsRequest(
    function_type='relu',  # string
    alpha=0.01,            # float
    input_range=[-10, 10], # list of 2 floats
    compare_all=True,      # boolean
    num_points=200         # integer
)
```

## Resources

- **Full Documentation**: See `README.md`
- **Implementation**: See `ACTIVATION_FUNCTIONS_IMPLEMENTATION.md`
- **Test Suite**: See `test_activation_functions.py`
- **API Route**: See `/backend/api/routes/deep_learning.py`

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Run test suite to verify functionality
3. Review example code in this guide
4. Check parameter validation in `schema.py`
