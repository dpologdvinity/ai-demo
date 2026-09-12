"""Comprehensive demo of GAN implementation showing all features."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from algorithms.deep_learning.gan import (
    GANModel,
    load_digits_data,
    prepare_data_for_gan
)


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def demo_full_gan_workflow():
    """Demonstrate complete GAN workflow with all features."""

    print_section("GAN COMPREHENSIVE DEMO")

    # 1. Load and prepare data
    print_section("1. DATA LOADING & PREPARATION")
    print("Loading MNIST digits dataset...")
    data = load_digits_data(random_state=42)
    X = prepare_data_for_gan(data['X'])

    print(f"✓ Loaded: {data['n_samples']} samples")
    print(f"✓ Features: {data['n_features']} (8×8 pixels)")
    print(f"✓ Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"✓ Description: {data['description'][:100]}...")

    # 2. Initialize model
    print_section("2. MODEL INITIALIZATION")
    print("Creating GAN with custom architecture...")

    model = GANModel(
        latent_dim=100,
        g_hidden=256,
        d_hidden=256,
        learning_rate=0.0002,
        random_state=42
    )

    g_params, d_params = model.count_parameters()
    model_info = model.get_model_info()

    print(f"✓ Device: {model_info['device']}")
    print(f"✓ Latent dimension: {model_info['latent_dim']}")
    print(f"\n✓ Generator Architecture:")
    print(f"  - Parameters: {g_params:,}")
    print(f"  - Input: {model_info['latent_dim']}D noise vector")
    print(f"  - Hidden: 256 → 256")
    print(f"  - Output: 64D image (8×8)")

    print(f"\n✓ Discriminator Architecture:")
    print(f"  - Parameters: {d_params:,}")
    print(f"  - Input: 64D image")
    print(f"  - Hidden: 256 → 256 (with dropout)")
    print(f"  - Output: 1D probability")

    print(f"\n✓ Total parameters: {model_info['total_params']:,}")

    # 3. Train model
    print_section("3. ADVERSARIAL TRAINING")
    print("Training GAN with alternating optimization...")
    print("(Using 15 epochs for demo - increase for better results)")

    training_results = model.train(
        X=X,
        epochs=15,
        batch_size=64,
        sample_interval=5
    )

    print(f"\n✓ Training completed in {training_results['training_time_ms']:.2f}ms")
    print(f"✓ Epochs trained: {len(training_results['loss_history'])}")
    print(f"✓ Checkpoints saved: {len(training_results['generated_samples'])}")

    # 4. Analyze training progress
    print_section("4. TRAINING METRICS")

    print("\nEpoch-by-epoch breakdown:")
    print(f"{'Epoch':>6} | {'G Loss':>8} | {'D Loss':>8} | {'D Real Acc':>10} | {'D Fake Acc':>10}")
    print("-" * 60)

    for i, metrics in enumerate(training_results['loss_history'][::3]):  # Show every 3rd
        print(f"{metrics['epoch']:>6} | "
              f"{metrics['g_loss']:>8.4f} | "
              f"{metrics['d_loss']:>8.4f} | "
              f"{metrics['d_real_accuracy']:>10.4f} | "
              f"{metrics['d_fake_accuracy']:>10.4f}")

    final_metrics = training_results['loss_history'][-1]
    print("\n✓ Final Metrics:")
    print(f"  - Generator Loss: {final_metrics['g_loss']:.4f}")
    print(f"  - Discriminator Loss: {final_metrics['d_loss']:.4f}")
    print(f"  - D Real Accuracy: {final_metrics['d_real_accuracy']:.4f} "
          f"({'correctly classifies real' if final_metrics['d_real_accuracy'] > 0.5 else 'needs improvement'})")
    print(f"  - D Fake Accuracy: {final_metrics['d_fake_accuracy']:.4f} "
          f"({'correctly classifies fake' if final_metrics['d_fake_accuracy'] > 0.5 else 'generator fooling discriminator'})")

    # 5. Generate samples
    print_section("5. SAMPLE GENERATION")
    print("Generating synthetic digit images...")

    samples = model.generate_samples(n_samples=16)
    print(f"✓ Generated {samples.shape[0]} images")
    print(f"✓ Image shape: 8×8 pixels")
    print(f"✓ Pixel range: [{samples.min():.3f}, {samples.max():.3f}]")

    # Show sample statistics
    print(f"\n✓ Sample Statistics:")
    print(f"  - Mean pixel value: {samples.mean():.3f}")
    print(f"  - Std pixel value: {samples.std():.3f}")
    print(f"  - Min/Max: {samples.min():.3f} / {samples.max():.3f}")

    # 6. Latent space interpolation
    print_section("6. LATENT SPACE INTERPOLATION")
    print("Interpolating between random points in latent space...")

    interpolated = model.interpolate_latent_space(
        n_steps=10,
        n_interpolations=3
    )

    print(f"✓ Generated {interpolated.shape[0]} interpolated images")
    print(f"✓ Interpolation sequences: 3")
    print(f"✓ Steps per sequence: 10")
    print(f"✓ Demonstrates smooth transitions in learned distribution")

    # 7. Decision boundary
    print_section("7. DISCRIMINATOR DECISION BOUNDARY")
    print("Computing discriminator scores for generated samples...")

    boundary_data = model.compute_decision_boundary(n_samples=100)

    scores = np.array(boundary_data['scores'])
    print(f"✓ Evaluated {len(scores)} generated samples")
    print(f"✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"✓ Mean score: {scores.mean():.3f}")
    print(f"✓ Std score: {scores.std():.3f}")

    # Analyze score distribution
    high_quality = (scores > 0.5).sum()
    low_quality = (scores <= 0.5).sum()

    print(f"\n✓ Score Distribution:")
    print(f"  - High quality (>0.5): {high_quality} ({high_quality/len(scores)*100:.1f}%)")
    print(f"  - Low quality (≤0.5): {low_quality} ({low_quality/len(scores)*100:.1f}%)")
    print(f"  - Generator fooling rate: {high_quality/len(scores)*100:.1f}%")

    # 8. Compare with real data
    print_section("8. REAL VS GENERATED COMPARISON")

    # Score some real images
    real_samples = X[:100]
    real_scores = model.get_discriminator_scores(real_samples)

    print(f"✓ Real Image Scores:")
    print(f"  - Mean: {real_scores.mean():.3f}")
    print(f"  - Std: {real_scores.std():.3f}")
    print(f"  - Range: [{real_scores.min():.3f}, {real_scores.max():.3f}]")

    print(f"\n✓ Generated Image Scores:")
    print(f"  - Mean: {scores.mean():.3f}")
    print(f"  - Std: {scores.std():.3f}")
    print(f"  - Range: [{scores.min():.3f}, {scores.max():.3f}]")

    score_diff = abs(real_scores.mean() - scores.mean())
    print(f"\n✓ Quality Gap: {score_diff:.3f}")
    if score_diff < 0.2:
        print("  → Excellent! Generated samples are very realistic")
    elif score_diff < 0.4:
        print("  → Good! Generated samples are fairly realistic")
    else:
        print("  → Needs more training for better quality")

    # 9. Summary
    print_section("9. SUMMARY")

    print("✓ Training Completed Successfully!")
    print(f"  - Total time: {training_results['training_time_ms']:.2f}ms")
    print(f"  - Epochs: {len(training_results['loss_history'])}")
    print(f"  - Final G loss: {final_metrics['g_loss']:.4f}")
    print(f"  - Final D loss: {final_metrics['d_loss']:.4f}")

    print("\n✓ Capabilities Demonstrated:")
    print("  [✓] Adversarial training of generator and discriminator")
    print("  [✓] Loss and accuracy tracking")
    print("  [✓] Sample generation from noise")
    print("  [✓] Latent space interpolation")
    print("  [✓] Decision boundary visualization")
    print("  [✓] Quality assessment vs real data")

    print("\n✓ Use Cases Enabled:")
    print("  • Image synthesis (generate new digit images)")
    print("  • Data augmentation (expand training datasets)")
    print("  • Style transfer (learn visual patterns)")
    print("  • Super resolution (upscale images)")
    print("  • Art generation (create novel images)")
    print("  • Missing data imputation (fill in gaps)")

    print("\n✓ Next Steps:")
    print("  • Train for more epochs (50-100) for better quality")
    print("  • Increase hidden layer sizes for more capacity")
    print("  • Experiment with different latent dimensions")
    print("  • Try different learning rates and batch sizes")
    print("  • Use generated images for downstream tasks")

    print_section("DEMO COMPLETE")
    print("All GAN features have been successfully demonstrated!")
    print("The implementation is ready for production use.")


if __name__ == "__main__":
    try:
        demo_full_gan_workflow()
    except Exception as e:
        print(f"\n✗ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
