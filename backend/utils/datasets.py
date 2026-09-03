"""Dataset management utilities for AI algorithm demonstrations.

This module provides a centralized DatasetManager class for loading and caching
common machine learning datasets from scikit-learn. It supports both real-world
datasets (Iris, California Housing, Digits, Wine) and synthetic datasets (blobs,
circles, moons) for various algorithm demonstrations.
"""

from typing import Dict, Tuple, Any, List
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetManager:
    """Manages pre-loaded datasets for algorithm demonstrations.

    This class provides cached access to various datasets used for demonstrating
    machine learning algorithms. Datasets are loaded once and cached for subsequent
    requests to improve performance.

    Attributes:
        _datasets_cache: Class-level cache for storing loaded datasets.
    """

    _datasets_cache: Dict[str, Any] = {}

    @classmethod
    def get_iris(cls, test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
        """Get Iris dataset for classification tasks.

        The Iris dataset contains 150 samples of iris flowers with 4 features
        (sepal length, sepal width, petal length, petal width) and 3 target
        classes (setosa, versicolor, virginica).

        Args:
            test_size: Proportion of the dataset to include in the test split.
                Must be between 0.0 and 1.0.
            random_state: Random seed for reproducible splits.

        Returns:
            Dictionary containing:
                - X_train: Training features (numpy array)
                - X_test: Test features (numpy array)
                - y_train: Training labels (numpy array)
                - y_test: Test labels (numpy array)
                - feature_names: List of feature names
                - target_names: List of target class names
                - description: Dataset description string

        Raises:
            ValueError: If test_size is not between 0.0 and 1.0.
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

        if 'iris' not in cls._datasets_cache:
            iris = datasets.load_iris()
            X_train, X_test, y_train, y_test = train_test_split(
                iris.data, iris.target, test_size=test_size, random_state=random_state
            )
            cls._datasets_cache['iris'] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': iris.feature_names,
                'target_names': iris.target_names.tolist(),
                'description': iris.DESCR
            }
        return cls._datasets_cache['iris']

    @classmethod
    def get_boston(cls, test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
        """Get dataset for regression tasks.

        Uses the California Housing dataset as a replacement for the deprecated
        Boston Housing dataset. Contains 20,640 samples with 8 features
        predicting median house values.

        Args:
            test_size: Proportion of the dataset to include in the test split.
                Must be between 0.0 and 1.0.
            random_state: Random seed for reproducible splits.

        Returns:
            Dictionary containing:
                - X_train: Training features (numpy array)
                - X_test: Test features (numpy array)
                - y_train: Training target values (numpy array)
                - y_test: Test target values (numpy array)
                - feature_names: List of feature names
                - description: Dataset description string

        Raises:
            ValueError: If test_size is not between 0.0 and 1.0.
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

        if 'boston' not in cls._datasets_cache:
            california = datasets.fetch_california_housing()
            X_train, X_test, y_train, y_test = train_test_split(
                california.data, california.target, test_size=test_size, random_state=random_state
            )
            cls._datasets_cache['boston'] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': california.feature_names,
                'description': california.DESCR
            }
        return cls._datasets_cache['boston']

    @classmethod
    def get_digits(cls, test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
        """Get handwritten digits dataset for image classification.

        Contains 1,797 samples of 8x8 grayscale images of handwritten digits
        (0-9). Each image is represented as a 64-element feature vector.

        Args:
            test_size: Proportion of the dataset to include in the test split.
                Must be between 0.0 and 1.0.
            random_state: Random seed for reproducible splits.

        Returns:
            Dictionary containing:
                - X_train: Training features (numpy array, flattened images)
                - X_test: Test features (numpy array, flattened images)
                - y_train: Training labels (numpy array)
                - y_test: Test labels (numpy array)
                - images: Original 8x8 images (numpy array)
                - description: Dataset description string

        Raises:
            ValueError: If test_size is not between 0.0 and 1.0.
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

        if 'digits' not in cls._datasets_cache:
            digits = datasets.load_digits()
            X_train, X_test, y_train, y_test = train_test_split(
                digits.data, digits.target, test_size=test_size, random_state=random_state
            )
            cls._datasets_cache['digits'] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'images': digits.images,
                'description': digits.DESCR
            }
        return cls._datasets_cache['digits']

    @classmethod
    def get_wine(cls, test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
        """Get wine dataset for classification tasks.

        Contains 178 samples of wines with 13 chemical features and 3 target
        classes representing different wine cultivars.

        Args:
            test_size: Proportion of the dataset to include in the test split.
                Must be between 0.0 and 1.0.
            random_state: Random seed for reproducible splits.

        Returns:
            Dictionary containing:
                - X_train: Training features (numpy array)
                - X_test: Test features (numpy array)
                - y_train: Training labels (numpy array)
                - y_test: Test labels (numpy array)
                - feature_names: List of feature names
                - target_names: List of target class names
                - description: Dataset description string

        Raises:
            ValueError: If test_size is not between 0.0 and 1.0.
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

        if 'wine' not in cls._datasets_cache:
            wine = datasets.load_wine()
            X_train, X_test, y_train, y_test = train_test_split(
                wine.data, wine.target, test_size=test_size, random_state=random_state
            )
            cls._datasets_cache['wine'] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': wine.feature_names,
                'target_names': wine.target_names.tolist(),
                'description': wine.DESCR
            }
        return cls._datasets_cache['wine']

    @classmethod
    def get_blobs(
        cls,
        n_samples: int = 300,
        centers: int = 3,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Generate blob clusters for clustering algorithms.

        Creates isotropic Gaussian blobs for clustering demonstrations.
        Each blob represents a distinct cluster with configurable parameters.

        Args:
            n_samples: Total number of points to generate.
            centers: Number of cluster centers to generate.
            random_state: Random seed for reproducible generation.

        Returns:
            Dictionary containing:
                - X: Feature array (numpy array, shape [n_samples, 2])
                - y: Cluster labels (numpy array)
                - n_samples: Number of samples generated
                - n_centers: Number of cluster centers

        Raises:
            ValueError: If n_samples or centers is less than 1.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {n_samples}")
        if centers < 1:
            raise ValueError(f"centers must be at least 1, got {centers}")

        X, y = datasets.make_blobs(
            n_samples=n_samples,
            centers=centers,
            random_state=random_state,
            cluster_std=1.0
        )
        return {
            'X': X,
            'y': y,
            'n_samples': n_samples,
            'n_centers': centers
        }

    @classmethod
    def get_circles(
        cls,
        n_samples: int = 300,
        noise: float = 0.05,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Generate concentric circles for non-linear classification.

        Creates two concentric circles, useful for demonstrating non-linear
        classification algorithms and kernel methods.

        Args:
            n_samples: Total number of points to generate.
            noise: Standard deviation of Gaussian noise added to the data.
            random_state: Random seed for reproducible generation.

        Returns:
            Dictionary containing:
                - X: Feature array (numpy array, shape [n_samples, 2])
                - y: Binary labels (numpy array, 0=inner circle, 1=outer circle)

        Raises:
            ValueError: If n_samples is less than 1 or noise is negative.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {n_samples}")
        if noise < 0:
            raise ValueError(f"noise must be non-negative, got {noise}")

        X, y = datasets.make_circles(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state
        )
        return {
            'X': X,
            'y': y
        }

    @classmethod
    def get_moons(
        cls,
        n_samples: int = 300,
        noise: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Generate two interleaving half circles.

        Creates a dataset of two interleaving half circles, useful for
        demonstrating non-linear classification and clustering algorithms.

        Args:
            n_samples: Total number of points to generate.
            noise: Standard deviation of Gaussian noise added to the data.
            random_state: Random seed for reproducible generation.

        Returns:
            Dictionary containing:
                - X: Feature array (numpy array, shape [n_samples, 2])
                - y: Binary labels (numpy array, 0=first moon, 1=second moon)

        Raises:
            ValueError: If n_samples is less than 1 or noise is negative.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {n_samples}")
        if noise < 0:
            raise ValueError(f"noise must be non-negative, got {noise}")

        X, y = datasets.make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state
        )
        return {
            'X': X,
            'y': y
        }

    @classmethod
    def normalize_data(
        cls,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features using StandardScaler.

        Applies standardization to features by removing the mean and scaling
        to unit variance. The scaler is fit on training data and applied to
        both training and test data to prevent data leakage.

        Args:
            X_train: Training features to fit and transform.
            X_test: Test features to transform using training statistics.

        Returns:
            Tuple containing:
                - X_train_scaled: Normalized training features
                - X_test_scaled: Normalized test features

        Raises:
            ValueError: If X_train or X_test are empty or have incompatible shapes.
        """
        if X_train.size == 0:
            raise ValueError("X_train cannot be empty")
        if X_test.size == 0:
            raise ValueError("X_test cannot be empty")
        if X_train.ndim != X_test.ndim:
            raise ValueError(
                f"X_train and X_test must have same number of dimensions, "
                f"got {X_train.ndim} and {X_test.ndim}"
            )
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"X_train and X_test must have same number of features, "
                f"got {X_train.shape[1]} and {X_test.shape[1]}"
            )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the datasets cache.

        Removes all cached datasets from memory. Useful for freeing memory
        or forcing a fresh load of datasets with different parameters.
        """
        cls._datasets_cache.clear()

    @classmethod
    def list_available_datasets(cls) -> List[str]:
        """Get list of available dataset names.

        Returns:
            List of dataset names that can be used with get_dataset().
        """
        return [
            'iris',
            'boston',
            'california',
            'digits',
            'wine',
            'blobs',
            'circles',
            'moons'
        ]


# Convenience functions
def get_dataset(name: str, **kwargs) -> Dict[str, Any]:
    """Get dataset by name.

    Convenience function for accessing datasets without directly using the
    DatasetManager class. Supports all datasets available through DatasetManager.

    Args:
        name: Name of the dataset to load. Valid options are:
            - 'iris': Iris flower classification dataset
            - 'boston' or 'california': California housing regression dataset
            - 'digits': Handwritten digits classification dataset
            - 'wine': Wine classification dataset
            - 'blobs': Synthetic blob clusters
            - 'circles': Synthetic concentric circles
            - 'moons': Synthetic interleaving half circles
        **kwargs: Additional keyword arguments passed to the dataset loader.
            Common arguments include:
            - test_size: For split datasets (default 0.3)
            - random_state: For reproducibility (default 42)
            - n_samples: For synthetic datasets (default 300)
            - noise: For noisy synthetic datasets
            - centers: For blob datasets (default 3)

    Returns:
        Dictionary containing dataset arrays and metadata. Structure depends
        on the specific dataset requested.

    Raises:
        ValueError: If the requested dataset name is not recognized.

    Examples:
        >>> # Load Iris dataset with custom test split
        >>> iris = get_dataset('iris', test_size=0.2)
        >>> X_train, y_train = iris['X_train'], iris['y_train']

        >>> # Generate synthetic moons with custom noise
        >>> moons = get_dataset('moons', n_samples=500, noise=0.15)
        >>> X, y = moons['X'], moons['y']
    """
    dataset_functions = {
        'iris': DatasetManager.get_iris,
        'boston': DatasetManager.get_boston,
        'california': DatasetManager.get_boston,
        'digits': DatasetManager.get_digits,
        'wine': DatasetManager.get_wine,
        'blobs': DatasetManager.get_blobs,
        'circles': DatasetManager.get_circles,
        'moons': DatasetManager.get_moons,
    }

    if name not in dataset_functions:
        available = DatasetManager.list_available_datasets()
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {', '.join(available)}"
        )

    return dataset_functions[name](**kwargs)
