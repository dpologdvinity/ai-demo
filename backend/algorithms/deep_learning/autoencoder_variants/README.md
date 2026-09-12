# Autoencoder Variants Implementation

This document describes the implementation of the Autoencoder Variants algorithm for the AI Algorithms Demo website.

## Overview

The implementation provides four autoencoder variants for comparison:
1. **Vanilla Autoencoder**: Standard encoder-decoder architecture
2. **Denoising Autoencoder**: Trained to remove noise from corrupted inputs
3. **Sparse Autoencoder**: Uses L1 regularization on latent activations
4. **Contractive Autoencoder**: Penalizes Jacobian for input robustness

## Files Created

### Module Structure
```
backend/algorithms/deep_learning/autoencoder_variants/
├── __init__.py          # Module initialization
├── schema.py            # Pydantic request/response schemas
├── data.py              # Data loading and preprocessing utilities
└── model.py             # PyTorch model implementations
```

### API Integration
- **Modified**: `backend/api/routes/deep_learning.py`
  - Added import statements
  - Registered metadata in AlgorithmRegistry
  - Added POST `/autoencoder-variants/train` endpoint
  - Added GET `/autoencoder-variants/info` endpoint

### Test Script
- **Created**: `backend/test_autoencoder_variants.py`

## Architecture

### Encoder: 64 → 256 → 128 → latent_dim
- Input: Flattened 8x8 images (64 dimensions)
- Hidden layers: 256, 128 neurons with ReLU activation
- Output: latent_dim dimensions (bottleneck)

### Decoder: latent_dim → 128 → 256 → 64
- Input: latent_dim dimensions
- Hidden layers: 128, 256 neurons with ReLU activation
- Output: 64 dimensions with Sigmoid activation (range [0, 1])

## Variant Details

### 1. Vanilla Autoencoder
- **Loss**: MSE(x, x_reconstructed)
- **Use Case**: Standard dimensionality reduction and feature learning
- **Training**: Clean inputs → encode → decode → compare with original

### 2. Denoising Autoencoder
- **Loss**: MSE(x_clean, decode(encode(x_noisy)))
- **Noise**: Gaussian noise with configurable noise_factor (0.0-0.5)
- **Use Case**: Image denoising, robust feature learning
- **Training**: Noisy inputs → encode → decode → compare with clean original

### 3. Sparse Autoencoder
- **Loss**: MSE(x, x_recon) + λ * ||z||₁
- **Regularization**: L1 penalty on latent activations
- **Use Case**: Interpretable feature learning, sparse representations
- **Training**: Encourages only a few latent units to activate

### 4. Contractive Autoencoder
- **Loss**: MSE(x, x_recon) + λ * ||∂h/∂x||²_F
- **Regularization**: Frobenius norm of Jacobian
- **Use Case**: Learning manifold structure, robust features
- **Training**: Penalizes sensitivity to input perturbations

## API Endpoints

### POST `/deep-learning/autoencoder-variants/train`

Trains an autoencoder variant on MNIST digits dataset.

**Request Body**:
```json
{
  "variant": "vanilla",           // "vanilla" | "denoising" | "sparse" | "contractive"
  "latent_dim": 32,                // 2-128
  "epochs": 10,                    // 5-50
  "learning_rate": 0.001,          // 0.0001-0.01
  "noise_factor": 0.3,             // 0.0-0.5 (for denoising)
  "sparsity_weight": 0.001,        // 0.0-0.1 (for sparse/contractive)
  "batch_size": 128,               // 32, 64, 128, 256
  "random_state": 42
}
```

**Response**:
```json
{
  "success": true,
  "variant": "vanilla",
  "metrics": {
    "final_loss": 0.042,
    "reconstruction_mse": 0.040,
    "avg_reconstruction_error": 0.035
  },
  "loss_history": [
    {"epoch": 0, "train_loss": 0.15, "val_loss": 0.14},
    {"epoch": 1, "train_loss": 0.08, "val_loss": 0.09}
  ],
  "original_images": [[...]],           // 10 samples × 64 pixels
  "noisy_images": null,                 // Only for denoising variant
  "reconstructed_images": [[...]],      // 10 samples × 64 pixels
  "latent_space": [[x, y], ...],        // 2D coordinates (PCA/t-SNE)
  "latent_labels": [0, 1, 2, ...],      // Digit labels
  "reconstruction_errors": [0.035, ...], // Per-sample MSE
  "learned_filters": [[...]],           // 16 filters × 64 pixels
  "visualization_data": {
    "n_samples": 200,
    "image_shape": [8, 8],
    "latent_dim": 32,
    "projection_method": "pca",
    "variant": "vanilla"
  },
  "execution_time_ms": 8532.5,
  "model_info": {
    "variant": "vanilla",
    "latent_dim": 32,
    "total_params": 100000,
    "encoder_params": 50000,
    "decoder_params": 50000,
    "device": "cpu"
  },
  "parameters_used": {...}
}
```

### GET `/deep-learning/autoencoder-variants/info`

Returns algorithm metadata, parameters, theory, and dataset information.

## Frontend Visualizations

The response data supports the following visualizations:

1. **Grid Comparison**: original | noisy (if denoising) | reconstructed
   - Display 10 sample images side-by-side
   - Show reconstruction quality visually

2. **Training Loss Curve**
   - Plot train_loss and val_loss over epochs
   - Show convergence behavior

3. **Latent Space 2D Scatter Plot**
   - Colored by digit class (0-9)
   - Shows clustering in learned representation
   - Uses PCA or t-SNE projection if latent_dim > 2

4. **Reconstruction Error Heatmap**
   - Visualize per-sample reconstruction error
   - Identify which samples are hard to reconstruct

5. **Learned Filters Visualization**
   - Display first 16 encoder filters as 8×8 images
   - Shows what features the network learned

6. **Variant Comparison Table**
   - Compare metrics across variants
   - Show relative strengths/weaknesses

## Dataset

- **Name**: MNIST Digits (8x8)
- **Source**: `sklearn.datasets.load_digits`
- **Size**: 1797 samples (1000 used for speed)
- **Classes**: 10 digits (0-9)
- **Features**: 64 (8×8 grayscale images)
- **Pixel Range**: [0, 1] (normalized)
- **Split**: 80% train, 20% test

## Complexity

- **Time**: O(epochs × batch_size × parameters)
- **Space**: O(encoder_params + decoder_params)
- **Parameters**: ~100K (depends on latent_dim)

## Algorithm Metadata

Registered in `AlgorithmRegistry` with:
- **ID**: `autoencoder-variants`
- **Slug**: `autoencoder-variants`
- **Category**: `DEEP_LEARNING`
- **Difficulty**: `INTERMEDIATE`
- **Tags**: deep-learning, autoencoder, unsupervised, dimensionality-reduction, denoising

## Testing

Run the test suite:
```bash
cd /home/kaitlyn/git/ai-demo/backend
python test_autoencoder_variants.py
```

Tests cover:
- Module imports
- Schema validation
- Dataset loading
- Noise addition utility
- Model initialization (all 4 variants)
- Training pipeline
- Latent space visualization

## Usage Example

```python
from algorithms.deep_learning.autoencoder_variants import (
    AutoencoderVariantsModel,
    load_mnist_data
)

# Load data
data = load_mnist_data(n_samples=1000, test_size=0.2)

# Train denoising autoencoder
model = AutoencoderVariantsModel(
    variant='denoising',
    latent_dim=32,
    learning_rate=0.001,
    noise_factor=0.3
)

results = model.train(
    X_train=data['X_train'],
    X_test=data['X_test'],
    epochs=10,
    batch_size=128
)

# Encode to latent space
latent = model.encode(data['X_test'])

# Reconstruct images
reconstructed = model.reconstruct(data['X_test'])

# Evaluate
metrics = model.evaluate(data['X_test'])
print(f"Reconstruction MSE: {metrics['reconstruction_mse']:.4f}")
```

## Theory

### Autoencoder Loss Function

All variants minimize reconstruction loss plus optional regularization:

**General Form**: L = L_recon + λ * L_reg

1. **Vanilla**: L = MSE(x, x̂)
2. **Denoising**: L = MSE(x_clean, decoder(encoder(x_noisy)))
3. **Sparse**: L = MSE(x, x̂) + λ * ||z||₁
4. **Contractive**: L = MSE(x, x̂) + λ * ||∂z/∂x||²_F

### Why Different Variants?

- **Vanilla**: Simple, fast, good baseline
- **Denoising**: Robust to noise, useful for real-world data
- **Sparse**: Interpretable features, better for visualization
- **Contractive**: Smooth latent space, good for manifold learning

## Future Enhancements

Potential improvements:
1. Add variational autoencoder (VAE) variant
2. Support for full MNIST 28×28 images
3. Convolutional autoencoder variants
4. Latent space interpolation demo
5. Anomaly detection use case
6. Semi-supervised learning with labeled data
