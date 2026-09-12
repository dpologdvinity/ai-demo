"""Quick test script for CNN implementation."""

import sys
import numpy as np
from algorithms.deep_learning.cnn import CNNModel
from algorithms.deep_learning.cnn.data import load_digits_data, get_dataset_info

def test_cnn():
    """Test CNN implementation."""
    print("Testing CNN implementation...")

    # Test data loading
    print("\n1. Testing data loading...")
    data = load_digits_data(test_size=0.2, random_state=42)
    print(f"   Training samples: {data['X_train'].shape}")
    print(f"   Test samples: {data['X_test'].shape}")
    print(f"   Number of classes: {data['num_classes']}")

    # Test dataset info
    print("\n2. Testing dataset info...")
    info = get_dataset_info()
    print(f"   Dataset: {info['name']}")
    print(f"   Description: {info['description']}")
    print(f"   Total samples: {info['num_samples']}")

    # Test model initialization
    print("\n3. Testing model initialization...")
    model = CNNModel(
        conv_filters=[8, 16],
        kernel_size=3,
        learning_rate=0.001,
        dropout=0.3
    )
    print("   Model initialized successfully")

    # Test training with small dataset and few epochs
    print("\n4. Testing model training (3 epochs, small dataset)...")
    # Use only 200 samples for quick test
    train_samples = 200
    test_samples = 50

    training_info = model.train(
        X_train=data['X_train'][:train_samples],
        y_train=data['y_train'][:train_samples],
        X_test=data['X_test'][:test_samples],
        y_test=data['y_test'][:test_samples],
        epochs=3,
        batch_size=32
    )
    print(f"   Training time: {training_info['training_time_ms']:.2f}ms")
    print(f"   Final train accuracy: {training_info['final_train_accuracy']:.4f}")
    print(f"   Final val accuracy: {training_info['final_val_accuracy']:.4f}")

    # Test predictions
    print("\n5. Testing predictions...")
    predictions = model.predict(data['X_test'][:test_samples])
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions[:10]}")

    # Test evaluation
    print("\n6. Testing evaluation...")
    metrics = model.evaluate(data['X_test'][:test_samples], data['y_test'][:test_samples])
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")

    # Test feature map extraction
    print("\n7. Testing feature map extraction...")
    feature_maps = model.extract_feature_maps(data['X_test'][:5], num_samples=3)
    print(f"   Feature maps shape: {feature_maps.shape}")

    # Test model info
    print("\n8. Testing model info...")
    model_info = model.get_model_info()
    print(f"   Total parameters: {model_info['total_parameters']}")
    print(f"   Device: {model_info['device']}")

    print("\n✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_cnn()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
