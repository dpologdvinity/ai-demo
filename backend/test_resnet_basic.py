"""Basic test for ResNet implementation."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.deep_learning.resnet import ResNetModel, ResNetRequest
from algorithms.deep_learning.resnet.data import get_sample_images, get_dataset_info
import numpy as np


def test_resnet_model():
    """Test ResNet model initialization and inference."""
    print("=" * 60)
    print("Testing ResNet Implementation")
    print("=" * 60)

    # Test 1: Model initialization
    print("\n1. Testing model initialization...")
    model = ResNetModel(model_variant='resnet18', use_pretrained=False)
    print(f"   ✓ Model initialized: {model.model_variant}")
    print(f"   ✓ Device: {model.device}")
    print(f"   ✓ Labels loaded: {len(model.labels)} classes")

    # Test 2: Model info
    print("\n2. Testing model info...")
    info = model.get_model_info()
    print(f"   ✓ Depth: {info['depth']}")
    print(f"   ✓ Total parameters: {info['total_parameters']:,}")
    print(f"   ✓ Residual blocks: {info['num_residual_blocks']}")

    # Test 3: Sample images
    print("\n3. Testing sample images...")
    samples = get_sample_images()
    print(f"   ✓ Sample images loaded: {len(samples)}")
    print(f"   ✓ First sample: {samples[0]['name']}")

    # Test 4: Prediction
    print("\n4. Testing prediction...")
    test_image = samples[0]['image']
    predictions, logits = model.predict(test_image, top_k=5)
    print(f"   ✓ Predictions generated: {len(predictions)}")
    print(f"   ✓ Top prediction: {predictions[0].class_name}")
    print(f"   ✓ Confidence: {predictions[0].confidence:.4f}")

    # Test 5: Feature maps
    print("\n5. Testing feature extraction...")
    feature_maps = model.extract_feature_maps()
    print(f"   ✓ Feature maps extracted: {len(feature_maps)} layers")
    for layer_name, fm_data in feature_maps.items():
        print(f"   ✓ {layer_name}: {fm_data['shape']} ({fm_data['num_channels']} channels)")

    # Test 6: Residual blocks info
    print("\n6. Testing residual blocks info...")
    blocks = model.get_residual_blocks_info()
    print(f"   ✓ Residual blocks: {len(blocks)}")
    print(f"   ✓ First block: {blocks[0].block_name}")
    print(f"   ✓ Input channels: {blocks[0].input_channels}")
    print(f"   ✓ Output channels: {blocks[0].output_channels}")

    # Test 7: Full inference pipeline
    print("\n7. Testing full inference pipeline...")
    request = ResNetRequest(
        model_variant='resnet18',
        top_k=5,
        use_pretrained=False,
        image_index=0
    )
    response = model.run_inference(request)
    print(f"   ✓ Inference completed in {response.execution_time_ms:.2f}ms")
    print(f"   ✓ Success: {response.success}")
    print(f"   ✓ Predictions: {len(response.predictions)}")
    print(f"   ✓ Input shape: {response.input_shape}")

    # Test 8: Dataset info
    print("\n8. Testing dataset info...")
    dataset_info = get_dataset_info()
    print(f"   ✓ Dataset: {dataset_info['name']}")
    print(f"   ✓ Classes: {dataset_info['num_classes']}")
    print(f"   ✓ Image size: {dataset_info['image_size']}")

    # Test 9: Different variants
    print("\n9. Testing different ResNet variants...")
    for variant in ['resnet18', 'resnet34', 'resnet50']:
        model_var = ResNetModel(model_variant=variant, use_pretrained=False)
        info_var = model_var.get_model_info()
        print(f"   ✓ {variant}: depth={info_var['depth']}, params={info_var['total_parameters']:,}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_resnet_model()
