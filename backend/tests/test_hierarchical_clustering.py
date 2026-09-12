"""Tests for Hierarchical Clustering algorithm implementation."""

import pytest
import numpy as np
from algorithms.ml.hierarchical_clustering.model import HierarchicalClusteringModel
from algorithms.ml.hierarchical_clustering.data import get_sample_data


class TestHierarchicalClusteringModel:
    """Test cases for HierarchicalClusteringModel."""

    def test_model_initialization(self):
        """Test model initializes with correct parameters."""
        model = HierarchicalClusteringModel(n_clusters=3, linkage='ward', affinity='euclidean')
        assert model.n_clusters == 3
        assert model.linkage == 'ward'
        assert model.affinity == 'euclidean'

    def test_model_initialization_invalid_clusters(self):
        """Test model raises error with invalid n_clusters."""
        with pytest.raises(ValueError, match="n_clusters must be at least 2"):
            HierarchicalClusteringModel(n_clusters=1)

    def test_model_initialization_invalid_ward_affinity(self):
        """Test model raises error when ward linkage used with non-euclidean affinity."""
        with pytest.raises(ValueError, match="Ward linkage requires euclidean affinity"):
            HierarchicalClusteringModel(n_clusters=3, linkage='ward', affinity='manhattan')

    def test_train_basic(self):
        """Test basic training functionality."""
        data = get_sample_data(n_samples=100, n_clusters=3)
        X = data['X']

        model = HierarchicalClusteringModel(n_clusters=3)
        result = model.train(X)

        assert 'n_clusters' in result
        assert 'n_samples' in result
        assert 'training_time_ms' in result
        assert result['n_clusters'] == 3
        assert result['n_samples'] == 100

    def test_train_empty_data(self):
        """Test training fails with empty data."""
        model = HierarchicalClusteringModel(n_clusters=3)
        X = np.array([])

        with pytest.raises(ValueError, match="X cannot be empty"):
            model.train(X)

    def test_train_insufficient_samples(self):
        """Test training fails when n_samples < n_clusters."""
        model = HierarchicalClusteringModel(n_clusters=5)
        X = np.array([[1, 2], [3, 4], [5, 6]])  # Only 3 samples

        with pytest.raises(ValueError, match="n_samples .* must be >= n_clusters"):
            model.train(X)

    def test_get_cluster_labels(self):
        """Test getting cluster labels after training."""
        data = get_sample_data(n_samples=100, n_clusters=3)
        X = data['X']

        model = HierarchicalClusteringModel(n_clusters=3)
        model.train(X)
        labels = model.get_cluster_labels()

        assert len(labels) == 100
        assert len(np.unique(labels)) == 3
        assert labels.min() >= 0
        assert labels.max() < 3

    def test_get_cluster_labels_before_training(self):
        """Test getting labels before training raises error."""
        model = HierarchicalClusteringModel(n_clusters=3)

        with pytest.raises(ValueError, match="Model must be trained"):
            model.get_cluster_labels()

    def test_evaluate(self):
        """Test evaluation metrics."""
        data = get_sample_data(n_samples=100, n_clusters=3)
        X = data['X']

        model = HierarchicalClusteringModel(n_clusters=3)
        model.train(X)
        metrics = model.evaluate(X)

        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert -1 <= metrics['silhouette_score'] <= 1
        assert metrics['davies_bouldin_score'] >= 0
        assert metrics['calinski_harabasz_score'] >= 0

    def test_get_dendrogram_data(self):
        """Test getting dendrogram data."""
        data = get_sample_data(n_samples=50, n_clusters=3)
        X = data['X']

        model = HierarchicalClusteringModel(n_clusters=3)
        model.train(X)
        dendrogram_data = model.get_dendrogram_data()

        assert 'linkage_matrix' in dendrogram_data
        assert 'n_clusters' in dendrogram_data
        assert 'linkage_method' in dendrogram_data
        assert 'affinity' in dendrogram_data
        assert len(dendrogram_data['linkage_matrix']) == 49  # n_samples - 1

    def test_get_model_info(self):
        """Test getting model information."""
        data = get_sample_data(n_samples=100, n_clusters=3)
        X = data['X']

        model = HierarchicalClusteringModel(n_clusters=3)
        model.train(X)
        info = model.get_model_info()

        assert 'n_clusters' in info
        assert 'linkage' in info
        assert 'affinity' in info
        assert 'n_samples' in info
        assert 'cluster_sizes' in info
        assert sum(info['cluster_sizes'].values()) == 100

    def test_different_linkage_methods(self):
        """Test different linkage methods produce different results."""
        data = get_sample_data(n_samples=100, n_clusters=3)
        X = data['X']

        # Train with ward linkage
        model_ward = HierarchicalClusteringModel(n_clusters=3, linkage='ward')
        model_ward.train(X)
        labels_ward = model_ward.get_cluster_labels()

        # Train with complete linkage
        model_complete = HierarchicalClusteringModel(n_clusters=3, linkage='complete')
        model_complete.train(X)
        labels_complete = model_complete.get_cluster_labels()

        # Labels should be different (though not guaranteed for all datasets)
        # We just verify both complete successfully
        assert len(labels_ward) == len(labels_complete)
        assert len(np.unique(labels_ward)) == 3
        assert len(np.unique(labels_complete)) == 3


class TestHierarchicalClusteringData:
    """Test cases for data loading functions."""

    def test_get_sample_data(self):
        """Test getting sample data."""
        data = get_sample_data(n_samples=200, n_clusters=4)

        assert 'X' in data
        assert 'y' in data
        assert data['X'].shape[0] == 200
        assert data['X'].shape[1] == 2
        assert len(np.unique(data['y'])) == 4

    def test_get_sample_data_invalid_samples(self):
        """Test getting sample data with invalid n_samples."""
        with pytest.raises(ValueError, match="n_samples must be at least 1"):
            get_sample_data(n_samples=0, n_clusters=3)

    def test_get_sample_data_invalid_clusters(self):
        """Test getting sample data with invalid n_clusters."""
        with pytest.raises(ValueError, match="n_centers must be at least 1"):
            get_sample_data(n_samples=100, n_clusters=0)
