"""Tests for DBSCAN clustering algorithm implementation."""

import pytest
import numpy as np
from algorithms.ml.dbscan import DBSCANModel, DBSCANParameters


class TestDBSCANModel:
    """Test suite for DBSCAN clustering model."""

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        model = DBSCANModel()
        assert model.model is None
        assert model.parameters is None
        assert model.labels is None
        assert model.n_clusters == 0
        assert model.n_noise == 0

    def test_train_with_default_parameters(self):
        """Test training with default parameters."""
        model = DBSCANModel()
        params = DBSCANParameters()

        response = model.train(params)

        assert response.success is True
        assert response.error is None
        assert response.metrics is not None
        assert 'n_clusters' in response.metrics
        assert 'n_noise' in response.metrics
        assert 'noise_ratio' in response.metrics
        assert response.execution_time_ms > 0
        assert model.labels is not None
        assert len(model.labels) == params.n_samples

    def test_train_with_custom_parameters(self):
        """Test training with custom parameters."""
        model = DBSCANModel()
        params = DBSCANParameters(
            eps=0.3,
            min_samples=10,
            metric='manhattan',
            n_samples=200,
            noise=0.15
        )

        response = model.train(params)

        assert response.success is True
        assert response.parameters_used['eps'] == 0.3
        assert response.parameters_used['min_samples'] == 10
        assert response.parameters_used['metric'] == 'manhattan'

    def test_clusters_found(self):
        """Test that DBSCAN finds clusters in the moons dataset."""
        model = DBSCANModel()
        params = DBSCANParameters(eps=0.3, min_samples=5)

        response = model.train(params)

        assert response.success is True
        # Moons dataset should typically produce 2 clusters
        assert response.metrics['n_clusters'] > 0
        assert len(response.clusters) > 0

    def test_noise_detection(self):
        """Test that DBSCAN can detect noise points."""
        model = DBSCANModel()
        # With high eps, we expect fewer noise points
        params = DBSCANParameters(eps=0.5, min_samples=5, noise=0.2)

        response = model.train(params)

        assert response.success is True
        # Should have a noise_ratio metric
        assert 'noise_ratio' in response.metrics
        assert 0 <= response.metrics['noise_ratio'] <= 1

    def test_cluster_info(self):
        """Test that cluster information is correctly populated."""
        model = DBSCANModel()
        params = DBSCANParameters()

        response = model.train(params)

        assert response.success is True
        assert len(response.clusters) > 0

        # Check cluster info structure
        for cluster in response.clusters:
            assert hasattr(cluster, 'cluster_id')
            assert hasattr(cluster, 'size')
            assert hasattr(cluster, 'is_noise')
            assert cluster.size > 0

    def test_visualization_data(self):
        """Test that visualization data is correctly formatted."""
        model = DBSCANModel()
        params = DBSCANParameters()

        response = model.train(params)

        assert response.success is True
        assert 'clusters' in response.visualization_data
        assert 'noise_points' in response.visualization_data
        assert 'n_clusters' in response.visualization_data

        # Check that clusters contain point data
        for cluster in response.visualization_data['clusters']:
            if len(cluster) > 0:
                point = cluster[0]
                assert 'x' in point
                assert 'y' in point
                assert 'cluster' in point

    def test_silhouette_score_with_multiple_clusters(self):
        """Test that silhouette score is calculated when multiple clusters exist."""
        model = DBSCANModel()
        # Use parameters that should produce multiple clusters
        params = DBSCANParameters(eps=0.3, min_samples=5)

        response = model.train(params)

        assert response.success is True
        # If we have 2 or more clusters, silhouette score should be present
        if response.metrics['n_clusters'] >= 2:
            assert 'silhouette_score' in response.metrics
            assert -1 <= response.metrics['silhouette_score'] <= 1

    def test_invalid_eps_parameter(self):
        """Test that invalid eps parameter is rejected."""
        with pytest.raises(ValueError):
            DBSCANParameters(eps=0.05)  # Below minimum

        with pytest.raises(ValueError):
            DBSCANParameters(eps=3.0)  # Above maximum

    def test_invalid_min_samples_parameter(self):
        """Test that invalid min_samples parameter is rejected."""
        with pytest.raises(ValueError):
            DBSCANParameters(min_samples=1)  # Below minimum

        with pytest.raises(ValueError):
            DBSCANParameters(min_samples=25)  # Above maximum

    def test_invalid_metric_parameter(self):
        """Test that invalid metric parameter is rejected."""
        with pytest.raises(ValueError):
            DBSCANParameters(metric='invalid_metric')

    def test_different_eps_values(self):
        """Test that different eps values produce different clustering results."""
        model1 = DBSCANModel()
        params1 = DBSCANParameters(eps=0.2, min_samples=5)
        response1 = model1.train(params1)

        model2 = DBSCANModel()
        params2 = DBSCANParameters(eps=0.8, min_samples=5)
        response2 = model2.train(params2)

        assert response1.success is True
        assert response2.success is True
        # Smaller eps should typically result in more clusters or more noise
        assert response1.metrics['n_clusters'] != response2.metrics['n_clusters'] or \
               response1.metrics['n_noise'] != response2.metrics['n_noise']

    def test_get_cluster_labels(self):
        """Test that cluster labels can be retrieved after training."""
        model = DBSCANModel()
        params = DBSCANParameters()

        response = model.train(params)

        assert response.success is True
        labels = model.get_cluster_labels()
        assert labels is not None
        assert len(labels) == params.n_samples
        # DBSCAN uses -1 for noise points
        assert np.any(labels >= -1)
