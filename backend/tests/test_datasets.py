import pytest
import numpy as np
from backend.utils.datasets import DatasetManager, get_dataset


def test_get_iris():
    """Test loading Iris dataset"""
    data = DatasetManager.get_iris()
    assert 'X_train' in data
    assert 'X_test' in data
    assert 'y_train' in data
    assert 'y_test' in data
    assert len(data['target_names']) == 3
    assert 'feature_names' in data


def test_get_iris_custom_split():
    """Test Iris dataset with custom test split"""
    data = DatasetManager.get_iris(test_size=0.2)
    total_samples = len(data['X_train']) + len(data['X_test'])
    assert total_samples == 150  # Iris has 150 samples


def test_get_digits():
    """Test loading digits dataset"""
    data = DatasetManager.get_digits()
    assert 'X_train' in data
    assert 'X_test' in data
    assert 'y_train' in data
    assert 'y_test' in data
    assert 'images' in data


def test_get_wine():
    """Test loading wine dataset"""
    data = DatasetManager.get_wine()
    assert 'X_train' in data
    assert 'X_test' in data
    assert 'y_train' in data
    assert 'y_test' in data
    assert 'feature_names' in data
    assert 'target_names' in data


def test_get_blobs():
    """Test generating blob clusters"""
    data = DatasetManager.get_blobs(n_samples=100, centers=3)
    assert data['X'].shape[0] == 100
    assert data['n_centers'] == 3
    assert 'y' in data


def test_get_circles():
    """Test generating concentric circles"""
    data = DatasetManager.get_circles(n_samples=200)
    assert data['X'].shape[0] == 200
    assert len(np.unique(data['y'])) == 2  # Binary classification


def test_get_moons():
    """Test generating moon shapes"""
    data = DatasetManager.get_moons(n_samples=200)
    assert data['X'].shape[0] == 200
    assert len(np.unique(data['y'])) == 2  # Binary classification


def test_normalize_data():
    """Test data normalization"""
    X_train = np.random.rand(100, 4)
    X_test = np.random.rand(30, 4)

    X_train_norm, X_test_norm = DatasetManager.normalize_data(X_train, X_test)

    # Check that normalization produces roughly zero mean
    assert abs(X_train_norm.mean()) < 0.1
    # Check that standard deviation is close to 1
    assert abs(X_train_norm.std() - 1.0) < 0.1


def test_get_dataset_by_name():
    """Test get_dataset convenience function"""
    data = get_dataset('iris')
    assert 'X_train' in data


def test_invalid_dataset_name():
    """Test error on invalid dataset name"""
    with pytest.raises(ValueError):
        get_dataset('invalid_dataset_name')


def test_invalid_test_size():
    """Test error on invalid test_size parameter"""
    with pytest.raises(ValueError):
        DatasetManager.get_iris(test_size=1.5)

    with pytest.raises(ValueError):
        DatasetManager.get_iris(test_size=-0.1)


def test_invalid_blob_params():
    """Test error on invalid blob parameters"""
    with pytest.raises(ValueError):
        DatasetManager.get_blobs(n_samples=0)

    with pytest.raises(ValueError):
        DatasetManager.get_blobs(centers=0)


def test_normalize_empty_data():
    """Test error when normalizing empty arrays"""
    with pytest.raises(ValueError):
        DatasetManager.normalize_data(np.array([]), np.array([]))


def test_normalize_incompatible_shapes():
    """Test error when normalizing arrays with different feature counts"""
    X_train = np.random.rand(100, 4)
    X_test = np.random.rand(30, 5)  # Different number of features

    with pytest.raises(ValueError):
        DatasetManager.normalize_data(X_train, X_test)


def test_list_available_datasets():
    """Test listing available datasets"""
    datasets = DatasetManager.list_available_datasets()
    assert isinstance(datasets, list)
    assert 'iris' in datasets
    assert 'blobs' in datasets
    assert 'wine' in datasets


def test_clear_cache():
    """Test clearing dataset cache"""
    # Load a dataset to populate cache
    DatasetManager.get_iris()
    assert 'iris_0.3_42' in DatasetManager._datasets_cache

    # Clear cache
    DatasetManager.clear_cache()
    assert len(DatasetManager._datasets_cache) == 0
