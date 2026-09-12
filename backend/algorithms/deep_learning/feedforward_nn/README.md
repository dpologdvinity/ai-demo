# Feedforward Neural Network (MLP) Implementation

## Overview
This module implements a Multilayer Perceptron (MLP) classifier using scikit-learn's `MLPClassifier`. It provides a simple yet effective neural network for classification tasks on the Iris dataset.

## Files Structure
- `__init__.py` - Module initialization and exports
- `model.py` - MLPModel implementation with training, prediction, and evaluation
- `schema.py` - Pydantic schemas for API requests and responses
- `data.py` - Data loading and preprocessing utilities for the Iris dataset

## Key Features

### Model (`model.py`)
- **MLPModel class**: Wrapper around scikit-learn's MLPClassifier
- Configurable architecture with variable hidden layer sizes
- Multiple activation functions (ReLU, tanh, sigmoid)
- Training history tracking (loss and accuracy per iteration)
- Comprehensive evaluation metrics
- Network architecture visualization support

### Data (`data.py`)
- **load_iris_data()**: Loads and preprocesses the classic Iris dataset
- **get_dataset_info()**: Provides dataset metadata
- **prepare_visualization_data()**: Formats data for frontend visualization

### Schema (`schema.py`)
- **MLPRequest**: Input parameters for training
  - `hidden_layers`: List of hidden layer sizes (default: [64, 32])
  - `learning_rate`: Learning rate for optimizer (default: 0.001)
  - `epochs`: Maximum training iterations (default: 100)
  - `batch_size`: Minibatch size (default: 32)
  - `activation`: Activation function (default: 'relu')
  
- **MLPResponse**: Training results and visualization data
  - Metrics (accuracy, precision, recall, F1 score)
  - Predictions and actual labels
  - Training history (loss and accuracy curves)
  - Confusion matrix
  - Network architecture visualization

## Dataset
The Iris dataset contains 150 samples of iris flowers with 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

Three classes:
- Setosa
- Versicolor
- Virginica

## API Endpoints

### POST `/deep-learning/feedforward-nn/train`
Trains an MLP classifier on the Iris dataset.

**Request Body**:
```json
{
  "hidden_layers": [64, 32],
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 32,
  "activation": "relu",
  "dataset_name": "iris",
  "random_state": 42
}
```

**Response**:
```json
{
  "success": true,
  "metrics": {
    "accuracy": 0.967,
    "precision": 0.969,
    "recall": 0.967,
    "f1_score": 0.967,
    "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 0, 10]]
  },
  "predictions": [0, 1, 2, ...],
  "actual": [0, 1, 2, ...],
  "training_history": {
    "iterations": [1, 2, 3, ...],
    "loss": [1.2, 0.8, 0.5, ...],
    "accuracy": [0.5, 0.7, 0.85, ...]
  },
  "visualization_data": {
    "training_curves": {...},
    "confusion_matrix": [...],
    "network_architecture": {...}
  },
  "execution_time_ms": 1234.56,
  "model_info": {...},
  "parameters_used": {...}
}
```

### GET `/deep-learning/feedforward-nn/info`
Returns algorithm metadata and dataset information.

## Visualization Components

1. **Network Architecture Diagram**
   - Shows input layer → hidden layers → output layer
   - Displays neuron counts and connections
   - Includes activation function information

2. **Training Curves**
   - Loss over iterations
   - Accuracy over iterations

3. **Confusion Matrix**
   - Heatmap showing predicted vs actual classifications

4. **Sample Predictions Table**
   - Shows first 10 test samples with features, predictions, and correctness

## Example Usage

```python
from algorithms.deep_learning.feedforward_nn import MLPModel
from algorithms.deep_learning.feedforward_nn.data import load_iris_data

# Load data
data = load_iris_data(test_size=0.2, normalize=True)

# Initialize model
model = MLPModel(
    hidden_layers=[64, 32],
    learning_rate=0.001,
    activation='relu',
    batch_size=32
)

# Train model
training_info = model.train(
    X_train=data['X_train'],
    y_train=data['y_train'],
    X_test=data['X_test'],
    y_test=data['y_test'],
    max_iter=100
)

# Make predictions
predictions = model.predict(data['X_test'])

# Evaluate
metrics = model.evaluate(data['X_test'], data['y_test'])
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Algorithm Theory

Feedforward Neural Networks (MLPs) are the foundation of deep learning:

1. **Architecture**: Input → Hidden Layers → Output
2. **Forward Pass**: Data flows through weighted connections and activation functions
3. **Backpropagation**: Gradients computed via chain rule to update weights
4. **Training**: Iterative weight updates to minimize loss function

### Key Concepts
- **Universal Approximation**: MLPs can approximate any continuous function
- **Non-linearity**: Activation functions enable learning complex patterns
- **Gradient Descent**: Optimization algorithm for weight updates
- **Regularization**: Prevents overfitting (implicit in scikit-learn)

## Complexity
- **Time Complexity**: O(L × n × m²)
  - L = number of layers
  - n = number of samples
  - m = neurons per layer
  
- **Space Complexity**: O(L × m²)
  - Stores weights and biases for all layers

## Dependencies
- scikit-learn >= 1.5.2
- numpy
- pydantic (for API schemas)
