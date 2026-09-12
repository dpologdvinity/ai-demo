# GAN API Quick Reference

## Endpoints

### 1. Train GAN
**POST** `/api/deep-learning/gan/train`

Train a Generative Adversarial Network to generate digit-like images.

#### Request Body
```json
{
  "latent_dim": 100,
  "g_hidden": 128,
  "d_hidden": 128,
  "learning_rate": 0.0002,
  "epochs": 100,
  "batch_size": 64,
  "random_state": 42
}
```

#### Parameters
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `latent_dim` | int | 50-200 | 100 | Noise vector dimension |
| `g_hidden` | int | 64-256 | 128 | Generator hidden layer size |
| `d_hidden` | int | 64-256 | 128 | Discriminator hidden layer size |
| `learning_rate` | float | 0.0001-0.001 | 0.0002 | Learning rate for both networks |
| `epochs` | int | 20-300 | 100 | Number of training epochs |
| `batch_size` | int | 32/64/128 | 64 | Batch size for training |
| `random_state` | int | any | 42 | Random seed for reproducibility |

#### Response
```json
{
  "loss_history": [
    {
      "epoch": 0,
      "g_loss": 2.35,
      "d_loss": 0.82,
      "d_real_loss": 0.45,
      "d_fake_loss": 0.37
    }
  ],
  "generated_samples": [
    {
      "epoch": 0,
      "samples": [[0.1, 0.2, ...]]
    }
  ],
  "final_samples": [[0.1, 0.2, ...]],
  "visualization_data": {
    "sample_epochs": [0, 10, 20],
    "loss_epochs": [0, 1, 2],
    "n_samples": 1797,
    "image_shape": [8, 8]
  },
  "execution_time_ms": 15423.5,
  "model_info": {
    "latent_dim": 100,
    "g_hidden": 128,
    "d_hidden": 128,
    "total_epochs": 100,
    "generator_params": 20864,
    "discriminator_params": 16641,
    "total_params": 37505,
    "device": "cpu"
  }
}
```

#### Status Codes
- `200`: Success
- `400`: Invalid parameters
- `500`: Training failed

---

### 2. Get GAN Info
**GET** `/api/deep-learning/gan/info`

Get algorithm metadata and dataset information.

#### Response
```json
{
  "metadata": {
    "id": "gan",
    "name": "Generative Adversarial Network",
    "slug": "gan",
    "category": "deep_learning",
    "description": "Two neural networks trained adversarially",
    "difficulty": "advanced",
    "tags": ["deep-learning", "generative", "adversarial", "unsupervised", "gan"],
    "use_cases": [
      "Image generation",
      "Data augmentation",
      "Style transfer",
      "Face generation",
      "Art creation"
    ],
    "complexity": {
      "time": "O(n*(G_params+D_params)*epochs)",
      "space": "O(G_params+D_params)"
    },
    "parameters": [...],
    "theory": "...",
    "pros": [...],
    "cons": [...],
    "related_algorithms": ["vae", "diffusion", "autoencoder", "cnn"]
  },
  "dataset": {
    "name": "Digits Dataset (8x8)",
    "type": "real-world",
    "n_samples": 1797,
    "n_features": 64,
    "image_shape": [8, 8],
    "pixel_range": [0.0, 1.0],
    "description": "...",
    "use_case": "..."
  }
}
```

---

## Usage Examples

### cURL

#### Train GAN
```bash
curl -X POST http://localhost:8000/api/deep-learning/gan/train \
  -H "Content-Type: application/json" \
  -d '{
    "latent_dim": 100,
    "g_hidden": 128,
    "d_hidden": 128,
    "learning_rate": 0.0002,
    "epochs": 100,
    "batch_size": 64,
    "random_state": 42
  }'
```

#### Get Info
```bash
curl http://localhost:8000/api/deep-learning/gan/info
```

### Python (requests)

```python
import requests

# Train GAN
response = requests.post(
    'http://localhost:8000/api/deep-learning/gan/train',
    json={
        'latent_dim': 100,
        'g_hidden': 128,
        'd_hidden': 128,
        'learning_rate': 0.0002,
        'epochs': 100,
        'batch_size': 64,
        'random_state': 42
    }
)
result = response.json()

# Access results
loss_history = result['loss_history']
final_samples = result['final_samples']
print(f"Training completed in {result['execution_time_ms']:.2f}ms")

# Get algorithm info
info_response = requests.get('http://localhost:8000/api/deep-learning/gan/info')
info = info_response.json()
print(f"Algorithm: {info['metadata']['name']}")
print(f"Dataset: {info['dataset']['name']}")
```

### JavaScript (fetch)

```javascript
// Train GAN
const response = await fetch('http://localhost:8000/api/deep-learning/gan/train', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    latent_dim: 100,
    g_hidden: 128,
    d_hidden: 128,
    learning_rate: 0.0002,
    epochs: 100,
    batch_size: 64,
    random_state: 42
  })
});

const result = await response.json();

// Access results
const lossHistory = result.loss_history;
const finalSamples = result.final_samples;
console.log(`Training completed in ${result.execution_time_ms.toFixed(2)}ms`);

// Get algorithm info
const infoResponse = await fetch('http://localhost:8000/api/deep-learning/gan/info');
const info = await infoResponse.json();
console.log(`Algorithm: ${info.metadata.name}`);
console.log(`Dataset: ${info.dataset.name}`);
```

---

## Visualizing Results

### Display Generated Images

Each generated sample is a 64-element array representing an 8x8 grayscale image.

```python
import numpy as np
import matplotlib.pyplot as plt

# Get final samples from API response
final_samples = result['final_samples']  # List of 16 samples

# Display as 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for idx, sample in enumerate(final_samples):
    row = idx // 4
    col = idx % 4
    # Reshape from 64 to 8x8
    image = np.array(sample).reshape(8, 8)
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()
```

### Plot Loss Curves

```python
import matplotlib.pyplot as plt

# Get loss history from API response
loss_history = result['loss_history']

# Extract loss values
epochs = [h['epoch'] for h in loss_history]
g_losses = [h['g_loss'] for h in loss_history]
d_losses = [h['d_loss'] for h in loss_history]
d_real_losses = [h['d_real_loss'] for h in loss_history]
d_fake_losses = [h['d_fake_loss'] for h in loss_history]

# Plot
plt.figure(figsize=(12, 5))

# Generator vs Discriminator
plt.subplot(1, 2, 1)
plt.plot(epochs, g_losses, label='Generator Loss', color='blue')
plt.plot(epochs, d_losses, label='Discriminator Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator vs Discriminator Loss')
plt.legend()
plt.grid(True)

# Discriminator Real vs Fake
plt.subplot(1, 2, 2)
plt.plot(epochs, d_real_losses, label='D Loss (Real)', color='green')
plt.plot(epochs, d_fake_losses, label='D Loss (Fake)', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator: Real vs Fake Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Animate Training Progress

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Get generated samples at different epochs
generated_samples = result['generated_samples']

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('GAN Training Progress')

def update(frame):
    samples = generated_samples[frame]['samples']
    epoch = generated_samples[frame]['epoch']
    
    for idx, sample in enumerate(samples[:16]):
        row = idx // 4
        col = idx % 4
        image = np.array(sample).reshape(8, 8)
        axes[row, col].clear()
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].axis('off')
    
    fig.suptitle(f'GAN Training Progress - Epoch {epoch}')

anim = FuncAnimation(
    fig, 
    update, 
    frames=len(generated_samples),
    interval=500,  # 500ms per frame
    repeat=True
)

plt.tight_layout()
plt.show()
```

---

## Frontend Integration

### React Example

```tsx
import React, { useState } from 'react';

interface GANRequest {
  latent_dim: number;
  g_hidden: number;
  d_hidden: number;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  random_state: number;
}

const GANDemo: React.FC = () => {
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<any>(null);

  const trainGAN = async (params: GANRequest) => {
    setTraining(true);
    try {
      const response = await fetch('/api/deep-learning/gan/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Training failed:', error);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div>
      <h1>GAN Image Generator</h1>
      <button 
        onClick={() => trainGAN({
          latent_dim: 100,
          g_hidden: 128,
          d_hidden: 128,
          learning_rate: 0.0002,
          epochs: 100,
          batch_size: 64,
          random_state: 42
        })}
        disabled={training}
      >
        {training ? 'Training...' : 'Train GAN'}
      </button>
      
      {result && (
        <div>
          <h2>Results</h2>
          <p>Training time: {result.execution_time_ms.toFixed(2)}ms</p>
          <p>Final G loss: {result.loss_history[result.loss_history.length - 1].g_loss.toFixed(4)}</p>
          <p>Final D loss: {result.loss_history[result.loss_history.length - 1].d_loss.toFixed(4)}</p>
          
          <div className="samples-grid">
            {result.final_samples.map((sample: number[], idx: number) => (
              <img 
                key={idx}
                src={arrayToImageUrl(sample)}
                alt={`Generated sample ${idx}`}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
```

---

## Error Handling

### Common Errors

**400 Bad Request**
```json
{
  "detail": "latent_dim must be between 50 and 200, got 300"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Training failed: CUDA out of memory"
}
```

### Best Practices

1. **Validate parameters** before sending request
2. **Handle timeouts** for long training runs
3. **Show progress** during training (use WebSocket if available)
4. **Cache results** to avoid retraining
5. **Display errors** gracefully to users

---

## Performance Tips

1. **Start small**: Use fewer epochs for initial tests
2. **GPU acceleration**: Training is much faster on GPU
3. **Batch size**: Larger batches = faster but more memory
4. **Monitoring**: Watch loss curves for training stability
5. **Early stopping**: Stop if losses diverge or plateau

---

## Related Endpoints

- **List all algorithms**: `GET /api/deep-learning/algorithms`
- **CNN training**: `POST /api/deep-learning/cnn/train`
- **RNN training**: `POST /api/deep-learning/rnn/train`
- **LSTM training**: `POST /api/deep-learning/lstm/train`
