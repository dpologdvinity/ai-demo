# Neural Style Transfer - Quick Start Guide

Quick reference for implementing and using Neural Style Transfer.

## 🚀 Quick Setup

```python
from algorithms.computer_vision.style_transfer import (
    StyleTransferModel,
    StyleTransferRequest
)

# Initialize model
model = StyleTransferModel()

# Create request
request = StyleTransferRequest(
    content_image_index=0,      # Mountain landscape
    style_image_index=0,        # Van Gogh's Starry Night
    iterations=300,              # Optimization steps
    content_weight=1.0,          # Content preservation
    style_weight=1000000.0,      # Style strength
    learning_rate=0.003,         # Optimizer LR
    image_size=512               # Output size
)

# Run style transfer
response = model.process_request(request)
```

## 📝 Common Use Cases

### 1. Basic Style Transfer
```python
request = StyleTransferRequest(
    content_image_index=0,
    style_image_index=0,
    iterations=300
)
```

### 2. Strong Stylization
```python
request = StyleTransferRequest(
    content_image_index=2,          # Portrait
    style_image_index=1,            # Picasso cubism
    style_weight=5000000.0,         # Strong style
    content_weight=0.5,             # Less content
    iterations=500
)
```

### 3. Subtle Style (Content Preservation)
```python
request = StyleTransferRequest(
    content_image_index=4,          # Architecture
    style_image_index=3,            # Monet impressionism
    style_weight=500000.0,          # Weak style
    content_weight=5.0,             # Strong content
    iterations=200
)
```

### 4. Fast Preview
```python
request = StyleTransferRequest(
    content_image_index=1,
    style_image_index=2,
    image_size=256,                 # Small size
    iterations=100,                 # Few iterations
    learning_rate=0.005             # Faster convergence
)
```

### 5. High Quality
```python
request = StyleTransferRequest(
    content_image_index=5,
    style_image_index=4,
    image_size=1024,                # Large size
    iterations=1000,                # Many iterations
    learning_rate=0.002             # Stable convergence
)
```

## 🎨 Available Images

### Content Images (0-9)
- **0**: Mountain landscape
- **1**: Alpine scene with lake
- **2**: Person portrait
- **3**: Cat portrait
- **4**: Milan cathedral
- **5**: Eiffel Tower
- **6**: Urban street scene
- **7**: Nature scene with birds
- **8**: Outdoor path
- **9**: Coastal aerial view

### Style Images (0-9)
- **0**: Van Gogh - Starry Night (swirling post-impressionism)
- **1**: Picasso - Les Demoiselles (cubism)
- **2**: Munch - The Scream (expressionism)
- **3**: Monet - Impression Sunrise (soft impressionism)
- **4**: Kandinsky - Composition VII (bold abstract)
- **5**: Hokusai - The Great Wave (Japanese woodblock)
- **6**: Klimt - The Kiss (gold art nouveau)
- **7**: Mondrian - Composition (geometric)
- **8**: Goya - The Colossus (romanticism)
- **9**: Kandinsky - On White II (abstract expressionism)

## ⚙️ Parameter Guide

### Iterations
- **50-100**: Fast preview, rough result
- **200-300**: Standard quality (default)
- **500-1000**: High quality, well-converged

### Content Weight (α)
- **0.1-0.5**: Minimal content, heavy stylization
- **1.0**: Balanced (default)
- **5.0-10.0**: Strong content preservation

### Style Weight (β)
- **100,000-500,000**: Subtle style transfer
- **1,000,000**: Balanced (default)
- **5,000,000-10,000,000**: Strong stylization

### Learning Rate
- **0.001**: Slow, stable convergence
- **0.003**: Balanced (default)
- **0.01**: Fast, may be unstable

### Image Size
- **256**: Fast (~5s), lower quality
- **512**: Balanced (~20s) (default)
- **1024**: High quality (~60s)

## 📊 Response Structure

```python
response.statistics
├── total_iterations: 300
├── final_total_loss: 125000.5
├── final_content_loss: 0.05
├── final_style_loss: 0.12
├── initial_total_loss: 500000.0
├── loss_reduction: 75.0
└── convergence_rate: 1250.0

response.visualization_data
├── content_image: "data:image/jpeg;base64,..."
├── style_image: "data:image/jpeg;base64,..."
├── generated_image: "data:image/jpeg;base64,..."
└── loss_curves
    ├── iterations: [0, 10, 20, ...]
    ├── total_loss: [500000, 450000, ...]
    ├── content_loss: [0.1, 0.08, ...]
    └── style_loss: [5.0, 4.5, ...]
```

## 🎯 Best Practices

### 1. Start with Defaults
```python
request = StyleTransferRequest(
    content_image_index=0,
    style_image_index=0
)
# Then adjust parameters based on result
```

### 2. Balance Content/Style
```
Typical ratio: style_weight / content_weight = 1,000,000
```

### 3. Iterate Progressively
```python
# First: Quick preview
response = model.process_request(StyleTransferRequest(
    iterations=100, image_size=256
))

# Then: Full quality
response = model.process_request(StyleTransferRequest(
    iterations=500, image_size=512
))
```

### 4. Match Style to Content
- **Portraits**: Try expressionism, cubism, art nouveau
- **Landscapes**: Try impressionism, post-impressionism
- **Architecture**: Try geometric, abstract
- **Nature**: Try Japanese woodblock, impressionism

## 🔧 Troubleshooting

### Problem: Result too abstract, content lost
**Solution**: 
- Increase `content_weight` to 5.0-10.0
- Decrease `style_weight` to 500,000
- Use fewer iterations (100-200)

### Problem: Style not visible enough
**Solution**:
- Increase `style_weight` to 5,000,000
- Decrease `content_weight` to 0.5
- Use more iterations (500-1000)

### Problem: Too slow
**Solution**:
- Reduce `image_size` to 256
- Reduce `iterations` to 100
- Use GPU if available

### Problem: Result noisy/unstable
**Solution**:
- Decrease `learning_rate` to 0.001-0.002
- Use more iterations for smoothing
- Check content/style weight balance

## 📈 Performance Tips

1. **Use GPU**: 5-10x faster than CPU
2. **Start Small**: Preview at 256px before full resolution
3. **Batch Processing**: Cache VGG19 model between requests
4. **Progressive Refinement**: Start with low iterations, refine if needed
5. **Preset Combinations**: Pre-compute good content/style pairs

## 🌐 API Example

```bash
curl -X POST "http://localhost:8000/api/computer-vision/style-transfer/stylize" \
  -H "Content-Type: application/json" \
  -d '{
    "content_image_index": 0,
    "style_image_index": 0,
    "iterations": 300,
    "content_weight": 1.0,
    "style_weight": 1000000.0,
    "learning_rate": 0.003,
    "image_size": 512
  }'
```

## 💡 Pro Tips

1. **Van Gogh (style 0)** works well with landscapes (content 0, 1)
2. **Picasso (style 1)** creates dramatic effects on portraits (content 2, 3)
3. **Monet (style 3)** adds softness to any scene
4. **Kandinsky (style 4, 9)** creates bold abstract effects
5. **The Great Wave (style 5)** works beautifully with water/coastal scenes (content 9)
6. **Mondrian (style 7)** creates geometric abstractions from architecture (content 4, 5)

## 🎨 Recommended Combinations

| Content | Style | Effect | Parameters |
|---------|-------|--------|------------|
| 0 (Mountain) | 0 (Starry Night) | Swirling landscape | Default |
| 2 (Portrait) | 1 (Picasso) | Cubist portrait | style_weight=5M |
| 4 (Cathedral) | 7 (Mondrian) | Geometric building | content_weight=3.0 |
| 7 (Nature) | 3 (Monet) | Impressionist scene | iterations=500 |
| 9 (Coastal) | 5 (Wave) | Japanese aesthetic | Default |

## 📚 Next Steps

1. Experiment with different content/style combinations
2. Adjust weights to find your preferred balance
3. Try progressive refinement (preview → full quality)
4. Monitor loss curves to understand convergence
5. Save successful parameter combinations as presets
