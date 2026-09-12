"""
Tests for Decision Tree Classifier implementation.
"""

import pytest
import numpy as np
from algorithms.ml.decision_tree import DecisionTreeModel


class TestDecisionTreeModel:
    """Test suite for DecisionTreeModel class."""

    def test_initialization(self):
        """Test model initialization."""
        model = DecisionTreeModel()
        assert model.model is None
        assert model.feature_names == []
        assert model.target_names == []
        assert model.training_time_ms == 0.0

    def test_train_default_parameters(self):
        """Test training with default parameters."""
        model = DecisionTreeModel()
        result = model.train()

        assert result['success'] is True
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert 'precision' in result['metrics']
        assert 'recall' in result['metrics']
        assert 'f1_score' in result['metrics']
        assert result['metrics']['accuracy'] > 0.8  # Should achieve decent accuracy on iris
        assert result['execution_time_ms'] > 0

    def test_train_with_max_depth(self):
        """Test training with specific max_depth."""
        model = DecisionTreeModel()
        result = model.train(max_depth=3)

        assert result['success'] is True
        assert result['metrics']['max_depth_achieved'] <= 3
        assert result['parameters_used']['max_depth'] == 3

    def test_train_with_criterion_entropy(self):
        """Test training with entropy criterion."""
        model = DecisionTreeModel()
        result = model.train(criterion='entropy')

        assert result['success'] is True
        assert result['parameters_used']['criterion'] == 'entropy'

    def test_train_with_min_samples_split(self):
        """Test training with min_samples_split parameter."""
        model = DecisionTreeModel()
        result = model.train(min_samples_split=10)

        assert result['success'] is True
        assert result['parameters_used']['min_samples_split'] == 10

    def test_train_with_min_samples_leaf(self):
        """Test training with min_samples_leaf parameter."""
        model = DecisionTreeModel()
        result = model.train(min_samples_leaf=5)

        assert result['success'] is True
        assert result['parameters_used']['min_samples_leaf'] == 5

    def test_visualization_data_structure(self):
        """Test that visualization data has correct structure."""
        model = DecisionTreeModel()
        result = model.train()

        assert 'visualization_data' in result
        viz_data = result['visualization_data']

        # Check confusion matrix
        assert 'confusion_matrix' in viz_data
        assert len(viz_data['confusion_matrix']) == 3  # Iris has 3 classes

        # Check labels
        assert 'labels' in viz_data
        assert len(viz_data['labels']) == 3

        # Check tree structure
        assert 'tree_structure' in viz_data
        tree_structure = viz_data['tree_structure']
        assert 'id' in tree_structure
        assert 'type' in tree_structure
        assert 'samples' in tree_structure

        # Check tree text
        assert 'tree_text' in viz_data
        assert isinstance(viz_data['tree_text'], str)
        assert len(viz_data['tree_text']) > 0

        # Check feature importance
        assert 'feature_importance' in viz_data
        assert 'features' in viz_data['feature_importance']
        assert 'importance' in viz_data['feature_importance']

    def test_predictions_structure(self):
        """Test that predictions have correct structure."""
        model = DecisionTreeModel()
        result = model.train()

        assert 'predictions' in result
        predictions = result['predictions']

        assert 'y_test' in predictions
        assert 'y_pred' in predictions
        assert 'y_pred_proba' in predictions

        # Check shapes match
        assert len(predictions['y_test']) == len(predictions['y_pred'])
        assert len(predictions['y_test']) == len(predictions['y_pred_proba'])

        # Check probability predictions
        for proba in predictions['y_pred_proba']:
            assert len(proba) == 3  # Iris has 3 classes
            assert abs(sum(proba) - 1.0) < 0.01  # Probabilities should sum to 1

    def test_metrics_structure(self):
        """Test that metrics have correct structure and reasonable values."""
        model = DecisionTreeModel()
        result = model.train()

        metrics = result['metrics']

        # Check all required metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'n_nodes' in metrics
        assert 'n_leaves' in metrics
        assert 'max_depth_achieved' in metrics

        # Check metrics are in valid range [0, 1]
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1

        # Check tree structure metrics
        assert metrics['n_nodes'] > 0
        assert metrics['n_leaves'] > 0
        assert metrics['n_leaves'] <= metrics['n_nodes']
        assert metrics['max_depth_achieved'] > 0

    def test_feature_importance_sum(self):
        """Test that feature importance values sum to 1."""
        model = DecisionTreeModel()
        result = model.train()

        importance = result['visualization_data']['feature_importance']['importance']
        total_importance = sum(importance)

        assert abs(total_importance - 1.0) < 0.01  # Should sum to approximately 1

    def test_invalid_min_samples_split(self):
        """Test that invalid min_samples_split raises error."""
        model = DecisionTreeModel()
        result = model.train(min_samples_split=1)

        assert result['success'] is False
        assert 'error' in result

    def test_invalid_min_samples_leaf(self):
        """Test that invalid min_samples_leaf raises error."""
        model = DecisionTreeModel()
        result = model.train(min_samples_leaf=0)

        assert result['success'] is False
        assert 'error' in result

    def test_invalid_criterion(self):
        """Test that invalid criterion raises error."""
        model = DecisionTreeModel()
        result = model.train(criterion='invalid')

        assert result['success'] is False
        assert 'error' in result

    def test_predict_before_training(self):
        """Test that predict raises error before training."""
        model = DecisionTreeModel()
        X_test = np.array([[1, 2, 3, 4]])

        with pytest.raises(RuntimeError):
            model.predict(X_test)

    def test_predict_proba_before_training(self):
        """Test that predict_proba raises error before training."""
        model = DecisionTreeModel()
        X_test = np.array([[1, 2, 3, 4]])

        with pytest.raises(RuntimeError):
            model.predict_proba(X_test)

    def test_tree_structure_recursive(self):
        """Test that tree structure is properly recursive."""
        model = DecisionTreeModel()
        result = model.train(max_depth=2)

        tree = result['visualization_data']['tree_structure']

        def check_tree_node(node, current_depth=0):
            assert 'id' in node
            assert 'type' in node
            assert 'samples' in node
            assert 'impurity' in node

            if node['type'] == 'leaf':
                assert 'class' in node
                assert 'class_name' in node
                assert 'value' in node
            else:
                assert node['type'] == 'internal'
                assert 'feature' in node
                assert 'feature_name' in node
                assert 'threshold' in node
                assert 'left' in node
                assert 'right' in node

                # Recursively check children
                check_tree_node(node['left'], current_depth + 1)
                check_tree_node(node['right'], current_depth + 1)

        check_tree_node(tree)

    def test_reproducibility(self):
        """Test that training with same random_state produces same results."""
        model1 = DecisionTreeModel()
        result1 = model1.train(random_state=42)

        model2 = DecisionTreeModel()
        result2 = model2.train(random_state=42)

        assert result1['metrics']['accuracy'] == result2['metrics']['accuracy']
        assert result1['metrics']['n_nodes'] == result2['metrics']['n_nodes']
        assert result1['metrics']['n_leaves'] == result2['metrics']['n_leaves']

    def test_different_random_states(self):
        """Test that different random_states can produce different results."""
        model1 = DecisionTreeModel()
        result1 = model1.train(random_state=42)

        model2 = DecisionTreeModel()
        result2 = model2.train(random_state=123)

        # Results might be the same but usually will differ slightly
        # We just verify both succeed
        assert result1['success'] is True
        assert result2['success'] is True
