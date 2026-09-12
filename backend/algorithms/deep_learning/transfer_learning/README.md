# Transfer Learning

Transfer Learning implementation for the AI Algorithms Demo website.

## Overview

This module demonstrates transfer learning by fine-tuning pre-trained deep learning models on small custom datasets. It showcases how models pre-trained on large datasets (ImageNet) can be adapted to new tasks with limited data.

## Features

### Supported Pre-trained Models
- **ResNet-18**: 18-layer residual network (11.7M parameters)
- **ResNet-50**: 50-layer residual network (25.6M parameters)
- **MobileNet-V2**: Efficient mobile architecture (3.5M parameters)
- **EfficientNet-B0**: Compound scaled efficient network (5.3M parameters)

### Transfer Learning Strategies

1. **Feature Extraction**
   - Freezes all pre-trained layers
   - Trains only the final classification layer
   - Fastest training, best for very small datasets
   - Treats model as fixed feature extractor

2. **Fine-tuning**
   - Freezes early layers (general features)
   - Trains later layers + classifier
   - Balanced approach between speed and adaptation
   - Most commonly used strategy

3. **Full Training**
   - Trains all layers from pre-trained initialization
   - Baseline comparison to other strategies
   - Requires more data and computation

### Layer Freezing Options

- **Auto**: Automatically determines optimal freezing based on strategy
- **None**: No layers frozen (full training)
- **Early**: Freeze early convolutional layers
- **Most**: Freeze most layers, train only last few blocks
- **All But Last**: Feature extraction mode

## Datasets

The implementation uses synthetic small-scale datasets to simulate transfer learning scenarios:

- **Flowers**: 5 classes (Rose, Daisy, Tulip, Sunflower, Orchid)
- **Animals**: 5 classes (Cat, Dog, Bird, Fish, Rabbit)
- **Food**: 5 classes (Pizza, Burger, Sushi, Salad, Pasta)

Each dataset contains:
- 150 total samples (30 per class)
- 224×224×3 RGB images
- 60% train, 20% validation, 20% test split
- ImageNet-compatible preprocessing

## API Endpoints

### Train Transfer Learning Model
```
POST /api/deep-learning/transfer-learning/train
```

**Request Body:**
```json
{
  "base_model": "resnet18",
  "strategy": "fine_tune",
  "freeze_layers": "auto",
  "learning_rate": 0.001,
  "epochs": 10,
  "batch_size": 16,
  "dataset": "flowers",
  "random_state": 42
}
```

**Response:**
- Training and validation metrics
- Confusion matrix
- Layer freezing information
- Sample predictions with confidence scores
- Training curves (loss, accuracy)
- Model architecture details
- Feature maps from frozen layers

### Get Algorithm Info
```
GET /api/deep-learning/transfer-learning/info
```

Returns metadata, parameters, theory, and dataset information.

## Implementation Details

### Architecture

```
transfer_learning/
├── __init__.py          # Module exports
├── schema.py            # Pydantic request/response schemas
├── model.py             # Transfer learning model implementation
├── data.py              # Dataset generation and preprocessing
└── README.md            # Documentation
```

### Key Components

**TransferLearningModel**: Main model class
- Loads pre-trained models from torchvision
- Applies layer freezing based on strategy
- Replaces final layer for custom classes
- Handles training with frozen/unfrozen parameters

**Layer Freezing Logic**: 
- Sets `requires_grad=False` for frozen layers
- Optimizer only updates trainable parameters
- Different strategies freeze different layer groups

**Data Preprocessing**:
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Image resizing to 224×224
- Synthetic data generation with class-specific patterns

## Visualization Data

The endpoint returns comprehensive visualization data:

1. **Training Curves**
   - Training/validation loss over epochs
   - Training/validation accuracy over epochs
   - Convergence comparison

2. **Confusion Matrix**
   - Per-class prediction accuracy
   - Misclassification patterns

3. **Layer Freezing Diagram**
   - Which layers are frozen vs trainable
   - Parameter counts per layer
   - Visual representation of freeze strategy

4. **Feature Maps**
   - Extracted features from frozen layers
   - Visualization of learned representations

5. **Sample Predictions**
   - Top predictions with confidence scores
   - Correct and incorrect classifications
   - Confidence distributions

6. **Strategy Comparison** (optional)
   - Accuracy comparison across strategies
   - Training time comparison
   - Parameters updated per strategy
   - Convergence speed analysis

## Usage Examples

### Feature Extraction (Small Dataset)
```python
{
  "base_model": "resnet18",
  "strategy": "feature_extraction",
  "freeze_layers": "all_but_last",
  "learning_rate": 0.001,
  "epochs": 10,
  "batch_size": 16,
  "dataset": "flowers"
}
```

### Fine-tuning (Medium Dataset)
```python
{
  "base_model": "resnet50",
  "strategy": "fine_tune",
  "freeze_layers": "early",
  "learning_rate": 0.0001,
  "epochs": 20,
  "batch_size": 32,
  "dataset": "animals"
}
```

### Full Training (Baseline)
```python
{
  "base_model": "mobilenet_v2",
  "strategy": "full_train",
  "freeze_layers": "none",
  "learning_rate": 0.001,
  "epochs": 30,
  "batch_size": 16,
  "dataset": "food"
}
```

## Performance Expectations

With default parameters (10 epochs, 150 samples):

| Strategy | Expected Accuracy | Training Time | Params Updated |
|----------|------------------|---------------|----------------|
| Feature Extraction | 75-85% | ~30s | 2-5K |
| Fine-tuning (early) | 80-90% | ~60s | 1-3M |
| Fine-tuning (most) | 85-92% | ~90s | 3-5M |
| Full Training | 70-85% | ~120s | 11-25M |

## Theory

Transfer learning works because:
1. **Feature Hierarchy**: Early layers learn universal features (edges, textures), later layers learn task-specific features
2. **Good Initialization**: Pre-trained weights provide better starting point than random initialization
3. **Data Efficiency**: Leverages knowledge from millions of images to work with hundreds
4. **Regularization**: Frozen layers act as regularization, reducing overfitting

## Related Algorithms

- ResNet (base architecture)
- VGG (alternative base model)
- CNN (building blocks)
- Feature Extraction (related technique)
- Domain Adaptation (related problem)

## References

- "How transferable are features in deep neural networks?" - Yosinski et al., 2014
- "Deep Residual Learning for Image Recognition" - He et al., 2016
- "Rethinking ImageNet Pre-training" - He et al., 2018
