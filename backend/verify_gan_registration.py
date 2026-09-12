"""Verify GAN is properly registered in the algorithm registry."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils.algorithm_metadata import AlgorithmRegistry, AlgorithmCategory

# Import the routes to ensure metadata is registered
from api.routes import deep_learning

def verify_gan_registration():
    """Verify GAN is registered correctly."""
    print("\nVerifying GAN registration...")

    # Get all deep learning algorithms
    dl_algorithms = AlgorithmRegistry.get_by_category(AlgorithmCategory.DEEP_LEARNING)
    print(f"\n✓ Found {len(dl_algorithms)} deep learning algorithms")

    # Find GAN
    gan = AlgorithmRegistry.get("gan")
    if not gan:
        print("✗ GAN not found in registry!")
        return False

    print(f"\n✓ GAN found in registry")
    print(f"  - ID: {gan.id}")
    print(f"  - Name: {gan.name}")
    print(f"  - Slug: {gan.slug}")
    print(f"  - Category: {gan.category}")
    print(f"  - Difficulty: {gan.difficulty}")
    print(f"  - Description: {gan.description}")
    print(f"  - Parameters: {len(gan.parameters)}")
    print(f"  - Use Cases: {len(gan.use_cases)}")
    print(f"  - Tags: {', '.join(gan.tags)}")
    print(f"  - Dataset: {gan.dataset_name}")
    print(f"  - Visualization Types: {gan.visualization_type}")

    # Verify parameters
    print(f"\n✓ Parameters:")
    for param in gan.parameters:
        print(f"  - {param.name}: {param.label} (default: {param.default})")

    # Verify use cases
    print(f"\n✓ Use Cases:")
    for uc in gan.use_cases:
        print(f"  - {uc}")

    # Verify complexity
    print(f"\n✓ Complexity:")
    print(f"  - Time: {gan.complexity.time}")
    print(f"  - Space: {gan.complexity.space}")

    # List all deep learning algorithms
    print(f"\n✓ All Deep Learning Algorithms:")
    for algo in dl_algorithms:
        print(f"  - {algo.id}: {algo.name}")

    return True


if __name__ == "__main__":
    try:
        if verify_gan_registration():
            print("\n✓ GAN verification successful!")
        else:
            print("\n✗ GAN verification failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
