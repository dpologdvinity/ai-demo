# Neural Style Transfer

Implementation of the seminal neural style transfer algorithm by Gatys et al. (2015) using PyTorch and VGG19.

## Overview

Neural Style Transfer combines the **content** of one image with the **artistic style** of another by optimizing a generated image to match content features from one image and style features from another.

## Algorithm Details

### Architecture
- **Backbone**: Pre-trained VGG19 on ImageNet
- **Content Representation**: High-level features from `conv4_2`
- **Style Representation**: Gram matrices from multiple layers: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`

### Loss Functions

1. **Content Loss**: MSE between feature maps
   ```
   L_content = ||F_generated^l - F_content^l||^2
   ```

2. **Style Loss**: MSE between Gram matrices
   ```
   L_style = Σ_l ||G_generated^l - G_style^l||^2
   ```
   where G is the Gram matrix: `G_ij = Σ_k F_ik * F_jk`

3. **Total Loss**:
   ```
   L_total = α * L_content + β * L_style
   ```

### Optimization
- **Method**: Adam optimizer with image-space optimization
- **Initial Image**: Copy of content image
- **Iterations**: 50-1000 (default 300)
- **Learning Rate**: 0.001-0.01 (default 0.003)

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `content_image_index` | 0 | 0-9 | Content image selection |
| `style_image_index` | 0 | 0-9 | Style image selection |
| `iterations` | 300 | 50-1000 | Optimization steps |
| `content_weight` | 1.0 | 0.1-10.0 | Content loss weight (α) |
| `style_weight` | 1,000,000 | 100,000-10,000,000 | Style loss weight (β) |
| `learning_rate` | 0.003 | 0.001-0.01 | Optimizer LR |
| `image_size` | 512 | 256-1024 | Output image size |

## Dataset

### Content Images (10)
- Landscapes (mountains, lakes, coastal)
- Portraits (people, animals)
- Architecture (landmarks, buildings)
- Urban scenes
- Nature scenes

### Style Images (10)
Famous artistic styles including:
- Van Gogh - Starry Night (post-impressionism)
- Picasso - Les Demoiselles d'Avignon (cubism)
- Munch - The Scream (expressionism)
- Monet - Impression Sunrise (impressionism)
- Kandinsky - Composition VII (abstract)
- Hokusai - The Great Wave (Japanese woodblock)
- Klimt - The Kiss (art nouveau)
- Mondrian - Composition (geometric)
- And more...

## Usage

### API Endpoint
```bash
POST /api/computer-vision/style-transfer/stylize
```

### Request Example
```json
{
  "content_image_index": 0,
  "style_image_index": 0,
  "iterations": 300,
  "content_weight": 1.0,
  "style_weight": 1000000.0,
  "learning_rate": 0.003,
  "image_size": 512
}
```

### Response
```json
{
  "success": true,
  "statistics": {
    "total_iterations": 300,
    "final_total_loss": 125000.5,
    "final_content_loss": 0.05,
    "final_style_loss": 0.12,
    "initial_total_loss": 500000.0,
    "loss_reduction": 75.0,
    "convergence_rate": 1250.0
  },
  "visualization_data": {
    "content_image": "data:image/jpeg;base64,...",
    "style_image": "data:image/jpeg;base64,...",
    "generated_image": "data:image/jpeg;base64,...",
    "loss_curves": {
      "iterations": [0, 10, 20, ...],
      "total_loss": [500000, 450000, ...],
      "content_loss": [0.1, 0.08, ...],
      "style_loss": [5.0, 4.5, ...]
    }
  },
  "loss_history": [...],
  "execution_time_ms": 45000.0,
  "model_info": {...},
  "parameters_used": {...},
  "image_info": {...}
}
```

## Implementation Notes

### Key Features
1. **VGG19 Feature Extraction**: Uses pre-trained VGG19 to extract multi-scale features
2. **Gram Matrix Style**: Captures texture/style through feature correlations
3. **Multi-layer Style Loss**: Combines style from multiple convolutional layers
4. **Image Normalization**: Uses ImageNet mean/std normalization
5. **Loss Tracking**: Records loss history for visualization

### Optimization Details
- **Optimizer**: Adam (alternative: LBFGS)
- **Pixel Clamping**: Constrains pixel values to valid range
- **Loss Recording**: Every 10 iterations
- **Progress Logging**: Every 50 iterations

### Performance
- **256x256**: ~5-10 seconds (CPU), ~2-3 seconds (GPU)
- **512x512**: ~20-30 seconds (CPU), ~5-8 seconds (GPU)
- **1024x1024**: ~60-90 seconds (CPU), ~15-20 seconds (GPU)

*Times for 300 iterations*

## Tuning Guide

### For More Content Preservation
- Increase `content_weight` (e.g., 5.0-10.0)
- Decrease `style_weight` (e.g., 100,000-500,000)
- Use fewer iterations (100-200)

### For Stronger Stylization
- Decrease `content_weight` (e.g., 0.1-0.5)
- Increase `style_weight` (e.g., 5,000,000-10,000,000)
- Use more iterations (500-1000)

### For Faster Results
- Reduce `image_size` to 256
- Use fewer iterations (50-100)
- Increase `learning_rate` to 0.005-0.01

### For Best Quality
- Use `image_size` of 1024
- Use 500-1000 iterations
- Fine-tune weights based on style/content combination

## Frontend Visualization

The response includes data for:
1. **Three-Panel Display**: Content | Style | Result
2. **Loss Curves**: Content loss, style loss, total loss over iterations
3. **Progress Indicator**: Real-time optimization progress
4. **Before/After Comparison**: Side-by-side comparison
5. **Download Button**: Save stylized image

## References

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style*. arXiv:1508.06576
2. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556

## Use Cases

- **Artistic Creation**: Generate unique art from photos
- **Photo Enhancement**: Apply artistic filters to photos
- **Video Stylization**: Frame-by-frame style transfer for videos
- **Game Assets**: Create stylized textures and environments
- **Design Tools**: Provide style transfer in creative applications
- **Social Media**: Instagram/Snapchat-style artistic filters
