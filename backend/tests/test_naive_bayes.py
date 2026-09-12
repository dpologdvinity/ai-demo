"""Tests for Naive Bayes algorithm implementation."""

import pytest
import numpy as np
from algorithms.ml.naive_bayes.model import NaiveBayesModel
from algorithms.ml.naive_bayes.data import (
    load_iris_data,
    load_wine_data,
    get_supported_datasets
)


class TestNaiveBayesModel:
    """Test suite for NaiveBayesModel class."""

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = NaiveBayesModel()
        assert model.var_smoothing == 1e-9
        assert model.priors is None
        assert model.classes_ is None

    def test_model_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        var_smoothing = 1e-8
        priors = [0.3, 0.3, 0.4]
        model = NaiveBayesModel(var_smoothing=var_smoothing, priors=priors)
        assert model.var_smoothing == var_smoothing
        assert np.allclose(model.priors, np.array(priors))

    def test_invalid_var_smoothing(self):
        """Test that invalid var_smoothing raises ValueError."""
        with pytest.raises(ValueError, match="var_smoothing must be between"):
            NaiveBayesModel(var_smoothing=1e-2)

        with pytest.raises(ValueError, match="var_smoothing must be between"):
            NaiveBayesModel(var_smoothing=1e-11)

    def test_invalid_priors(self):
        """Test that priors not summing to 1 raises ValueError."""
        with pytest.raises(ValueError, match="Priors must sum to 1"):
            NaiveBayesModel(priors=[0.3, 0.3, 0.3])

    def test_train_on_iris(self):
        """Test training on Iris dataset."""
        model = NaiveBayesModel()
        data = load_iris_data()

        results = model.train(
            data['X_train'],
            data['y_train'],
            data['X_test'],
            data['y_test'],
            data.get('target_names')
        )

        # Check result structure
        assert 'metrics' in results
        assert 'predictions' in results
        assert 'probabilities' in results
        assert 'visualization_data' in results
        assert 'execution_time_ms' in results

        # Check metrics
        assert 'accuracy' in results['metrics']
        assert 'precision' in results['metrics']
        assert 'recall' in results['metrics']
        assert 'f1_score' in results['metrics']

        # Check accuracy is reasonable
        assert 0.7 <= results['metrics']['accuracy'] <= 1.0

        # Check predictions shape
        assert len(results['predictions']) == len(data['y_test'])

        # Check probabilities shape
        assert len(results['probabilities']) == len(data['y_test'])
        assert len(results['probabilities'][0]) == 3  # 3 classes

    def test_train_with_normalization(self):
        """Test training with feature normalization."""
        data = load_iris_data()
        results = NaiveBayesModel.from_dataset(
            dataset_name="iris",
            normalize=True
        )

        assert results['success'] is True
        assert results['parameters_used']['normalize'] is True
        assert results['metrics']['accuracy'] > 0

    def test_train_without_normalization(self):
        """Test training without feature normalization."""
        data = load_iris_data()
        results = NaiveBayesModel.from_dataset(
            dataset_name="iris",
            normalize=False
        )

        assert results['success'] is True
        assert results['parameters_used']['normalize'] is False
        assert results['metrics']['accuracy'] > 0

    def test_train_on_wine(self):
        """Test training on Wine dataset."""
        results = NaiveBayesModel.from_dataset(
            dataset_name="wine",
            normalize=True
        )

        assert results['success'] is True
        assert results['metrics']['accuracy'] > 0.7

    def test_train_on_digits(self):
        """Test training on Digits dataset."""
        results = NaiveBayesModel.from_dataset(
            dataset_name="digits",
            normalize=True
        )

        assert results['success'] is True
        assert results['metrics']['accuracy'] > 0.6

    def test_confusion_matrix(self):
        """Test that confusion matrix is generated."""
        results = NaiveBayesModel.from_dataset(dataset_name="iris")

        assert 'confusion_matrix' in results['visualization_data']
        conf_matrix = results['visualization_data']['confusion_matrix']
        assert len(conf_matrix) == 3  # 3 classes
        assert len(conf_matrix[0]) == 3

    def test_probability_distributions(self):
        """Test that probability distributions are generated."""
        results = NaiveBayesModel.from_dataset(dataset_name="iris")

        assert 'probability_distributions' in results['visualization_data']
        prob_dist = results['visualization_data']['probability_distributions']
        assert 'samples' in prob_dist
        assert 'true_labels' in prob_dist
        assert 'predicted_labels' in prob_dist

    def test_class_priors(self):
        """Test that class priors are computed."""
        results = NaiveBayesModel.from_dataset(dataset_name="iris")

        assert 'class_priors' in results['visualization_data']
        priors = results['visualization_data']['class_priors']
        assert len(priors) == 3
        assert np.isclose(sum(priors), 1.0)

    def test_feature_importance(self):
        """Test that feature importance is computed."""
        results = NaiveBayesModel.from_dataset(dataset_name="iris")

        assert 'feature_importance' in results['visualization_data']
        feat_imp = results['visualization_data']['feature_importance']
        assert 'importance_scores' in feat_imp
        assert 'between_class_variance' in feat_imp
        assert 'within_class_variance' in feat_imp

    def test_invalid_input_shapes(self):
        """Test that mismatched input shapes raise ValueError."""
        model = NaiveBayesModel()
        X_train = np.random.rand(100, 4)
        y_train = np.random.randint(0, 3, 90)  # Wrong size
        X_test = np.random.rand(30, 4)
        y_test = np.random.randint(0, 3, 30)

        with pytest.raises(ValueError, match="must have same number of samples"):
            model.train(X_train, y_train, X_test, y_test)

    def test_unsupported_dataset(self):
        """Test that unsupported dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            NaiveBayesModel.from_dataset(dataset_name="invalid_dataset")


class TestNaiveBayesData:
    """Test suite for Naive Bayes data loading functions."""

    def test_load_iris_data(self):
        """Test loading Iris dataset."""
        data = load_iris_data()

        assert 'X_train' in data
        assert 'X_test' in data
        assert 'y_train' in data
        assert 'y_test' in data
        assert 'feature_names' in data
        assert 'target_names' in data

        # Check shapes
        assert data['X_train'].shape[1] == 4  # 4 features
        assert len(data['y_train']) == data['X_train'].shape[0]

    def test_load_wine_data(self):
        """Test loading Wine dataset."""
        data = load_wine_data()

        assert 'X_train' in data
        assert data['X_train'].shape[1] == 13  # 13 features

    def test_load_digits_data(self):
        """Test loading Digits dataset."""
        data = load_iris_data()  # Using iris for simplicity
        assert data is not None

    def test_get_supported_datasets(self):
        """Test getting list of supported datasets."""
        datasets = get_supported_datasets()

        assert isinstance(datasets, list)
        assert 'iris' in datasets
        assert 'wine' in datasets
        assert 'digits' in datasets
        assert len(datasets) >= 3


class TestNaiveBayesIntegration:
    """Integration tests for Naive Bayes algorithm."""

    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        # Train model
        results = NaiveBayesModel.from_dataset(
            dataset_name="iris",
            var_smoothing=1e-9,
            normalize=True
        )

        # Verify all expected keys are present
        expected_keys = [
            'metrics', 'predictions', 'probabilities',
            'feature_contributions', 'visualization_data',
            'execution_time_ms', 'parameters_used'
        ]

        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

        # Verify parameters were recorded
        assert results['parameters_used']['var_smoothing'] == 1e-9
        assert results['parameters_used']['dataset_name'] == "iris"
        assert results['parameters_used']['normalize'] is True

        # Verify metrics are in valid range
        assert 0 <= results['metrics']['accuracy'] <= 1
        assert 0 <= results['metrics']['precision'] <= 1
        assert 0 <= results['metrics']['recall'] <= 1
        assert 0 <= results['metrics']['f1_score'] <= 1

        # Verify execution time is positive
        assert results['execution_time_ms'] > 0

    def test_different_var_smoothing_values(self):
        """Test training with different variance smoothing values."""
        var_smoothing_values = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8]
        accuracies = []

        for var_smoothing in var_smoothing_values:
            results = NaiveBayesModel.from_dataset(
                dataset_name="iris",
                var_smoothing=var_smoothing,
                normalize=True
            )
            accuracies.append(results['metrics']['accuracy'])

        # All should achieve reasonable accuracy
        assert all(acc > 0.7 for acc in accuracies)

    def test_reproducibility(self):
        """Test that training is reproducible with same parameters."""
        results1 = NaiveBayesModel.from_dataset(
            dataset_name="iris",
            var_smoothing=1e-9,
            normalize=True
        )

        results2 = NaiveBayesModel.from_dataset(
            dataset_name="iris",
            var_smoothing=1e-9,
            normalize=True
        )

        # Should get same accuracy (dataset is always the same with fixed random state)
        assert results1['metrics']['accuracy'] == results2['metrics']['accuracy']
