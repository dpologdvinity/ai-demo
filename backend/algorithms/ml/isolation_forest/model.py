"""Isolation Forest anomaly detection algorithm implementation."""

import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

from .schema import IsolationForestParameters, IsolationForestResponse, AnomalyInfo
from .data import get_isolation_forest_data, prepare_visualization_data


class IsolationForestModel:
    """Isolation Forest anomaly detection algorithm implementation.

    This class implements the Isolation Forest algorithm using scikit-learn,
    providing methods for training and evaluating the model on synthetic data
    with injected outliers.

    Isolation Forest is an unsupervised anomaly detection algorithm that works
    by isolating anomalies rather than profiling normal points. It builds an
    ensemble of isolation trees where anomalies are points that have short
    average path lengths (are easier to isolate). The algorithm is particularly
    effective for high-dimensional datasets and doesn't require labeled data.

    Key advantages:
    - Efficient for large datasets with linear time complexity
    - Handles high-dimensional data well
    - No need for labeled anomaly data
    - Provides anomaly scores for ranking
    - Robust to irrelevant features

    Attributes:
        model: The underlying scikit-learn IsolationForest model
        parameters: Training parameters used
        predictions: Anomaly predictions (-1 for anomaly, 1 for normal)
        anomaly_scores: Anomaly scores (more negative = more anomalous)
        n_anomalies: Number of detected anomalies
        n_normal: Number of normal points

    Example:
        >>> params = IsolationForestParameters(n_estimators=100, contamination=0.1)
        >>> iforest = IsolationForestModel()
        >>> response = iforest.train(params)
        >>> print(f"Detected {response.anomaly_info.n_anomalies} anomalies")
    """

    def __init__(self):
        """Initialize the Isolation Forest model."""
        self.model: Optional[IsolationForest] = None
        self.parameters: Optional[IsolationForestParameters] = None
        self.predictions: Optional[np.ndarray] = None
        self.anomaly_scores: Optional[np.ndarray] = None
        self.n_anomalies: int = 0
        self.n_normal: int = 0

    def train(self, parameters: IsolationForestParameters) -> IsolationForestResponse:
        """Train Isolation Forest anomaly detection model.

        Generates synthetic data with injected outliers and applies the
        Isolation Forest algorithm to detect anomalies. Computes performance
        metrics and prepares visualization data.

        The algorithm works by building an ensemble of isolation trees. Each
        tree is constructed by randomly selecting a feature and then randomly
        selecting a split value between the minimum and maximum values of the
        selected feature. Anomalies are points that have shorter average path
        lengths in the trees, as they are easier to isolate.

        Args:
            parameters: Training parameters including n_estimators, contamination,
                       max_samples, and data generation parameters

        Returns:
            IsolationForestResponse containing training results, metrics, and
            visualization data

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If training fails

        Example:
            >>> params = IsolationForestParameters(
            ...     n_estimators=150,
            ...     contamination=0.15,
            ...     n_samples=500
            ... )
            >>> model = IsolationForestModel()
            >>> result = model.train(params)
            >>> if result.success:
            ...     print(f"Training completed in {result.execution_time_ms:.2f}ms")
            ...     print(f"Detected {result.anomaly_info.n_anomalies} anomalies")
        """
        try:
            start_time = time.time()
            self.parameters = parameters

            # Generate synthetic data with outliers
            data = get_isolation_forest_data(
                n_samples=parameters.n_samples,
                n_outliers_ratio=parameters.n_outliers_ratio,
                random_state=parameters.random_state
            )
            X = data['X']
            y_true = data['y_true']

            # Convert max_samples to integer if needed
            max_samples_param = parameters.max_samples
            if isinstance(max_samples_param, str) and max_samples_param == 'auto':
                max_samples_param = 'auto'
            elif isinstance(max_samples_param, int):
                max_samples_param = min(max_samples_param, len(X))

            # Initialize and train Isolation Forest model
            self.model = IsolationForest(
                n_estimators=parameters.n_estimators,
                contamination=parameters.contamination,
                max_samples=max_samples_param,
                random_state=parameters.random_state,
                n_jobs=-1  # Use all CPU cores for faster training
            )

            # Fit the model and predict
            self.predictions = self.model.fit_predict(X)

            # Get anomaly scores (more negative = more anomalous)
            self.anomaly_scores = self.model.score_samples(X)

            # Count anomalies and normal points
            self.n_anomalies = int(np.sum(self.predictions == -1))
            self.n_normal = int(np.sum(self.predictions == 1))

            # Calculate metrics
            metrics = self._calculate_metrics(X, y_true)

            # Prepare anomaly information
            anomaly_info = AnomalyInfo(
                n_anomalies=self.n_anomalies,
                n_normal=self.n_normal,
                anomaly_ratio=float(self.n_anomalies) / len(X)
            )

            # Prepare visualization data
            visualization_data = prepare_visualization_data(
                X, self.predictions, self.anomaly_scores, y_true
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return IsolationForestResponse(
                success=True,
                metrics=metrics,
                anomaly_info=anomaly_info,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump()
            )

        except ValueError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return IsolationForestResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Invalid parameters: {str(e)}"
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return IsolationForestResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Training failed: {str(e)}"
            )

    def _calculate_metrics(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Calculate anomaly detection performance metrics.

        Computes various metrics to evaluate Isolation Forest performance:
        - Number and ratio of detected anomalies
        - Confusion matrix (if ground truth available)
        - Precision, recall, F1-score (if ground truth available)
        - Anomaly score statistics

        Args:
            X: Feature array
            y_true: Ground truth labels (1 for normal, -1 for anomaly)

        Returns:
            Dictionary of metric names to values

        Note:
            Anomaly scores are more negative for more anomalous points.
            A threshold can be applied to these scores for binary classification.
        """
        metrics = {
            'n_anomalies': self.n_anomalies,
            'n_normal': self.n_normal,
            'anomaly_ratio': float(self.n_anomalies) / len(X) if len(X) > 0 else 0.0,
            'score_mean': float(self.anomaly_scores.mean()),
            'score_std': float(self.anomaly_scores.std()),
            'score_min': float(self.anomaly_scores.min()),
            'score_max': float(self.anomaly_scores.max())
        }

        # Calculate performance metrics against ground truth if available
        if y_true is not None:
            # Confusion matrix
            cm = confusion_matrix(y_true, self.predictions, labels=[1, -1])
            metrics['confusion_matrix'] = cm.tolist()

            # Classification metrics
            try:
                report = classification_report(
                    y_true,
                    self.predictions,
                    labels=[1, -1],
                    target_names=['Normal', 'Anomaly'],
                    output_dict=True,
                    zero_division=0
                )

                metrics['precision'] = float(report['Anomaly']['precision'])
                metrics['recall'] = float(report['Anomaly']['recall'])
                metrics['f1_score'] = float(report['Anomaly']['f1-score'])
                metrics['accuracy'] = float(report['accuracy'])

                # True/False positives/negatives
                tn, fp, fn, tp = cm.ravel()
                metrics['true_positives'] = int(tp)
                metrics['false_positives'] = int(fp)
                metrics['true_negatives'] = int(tn)
                metrics['false_negatives'] = int(fn)

            except Exception:
                # If classification report fails, skip detailed metrics
                pass

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies for new data.

        Classifies new data points as normal (1) or anomaly (-1) based on
        the trained Isolation Forest model.

        Args:
            X: Feature array for new data points (shape [n_samples, n_features])

        Returns:
            Array of predictions (1 for normal, -1 for anomaly)

        Raises:
            RuntimeError: If model hasn't been trained yet

        Example:
            >>> new_points = np.array([[0.5, 0.5], [10.0, 10.0]])
            >>> predictions = model.predict(new_points)
            >>> print(predictions)  # e.g., [1, -1] (second point is anomaly)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        return self.model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores for data points.

        Computes anomaly scores for data points. More negative scores indicate
        higher likelihood of being an anomaly. These scores can be used for
        ranking points by anomaly likelihood or for applying custom thresholds.

        Args:
            X: Feature array (shape [n_samples, n_features])

        Returns:
            Array of anomaly scores (more negative = more anomalous)

        Raises:
            RuntimeError: If model hasn't been trained yet

        Example:
            >>> scores = model.score_samples(X)
            >>> # Get top 10 most anomalous points
            >>> most_anomalous_indices = np.argsort(scores)[:10]
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before scoring")

        return self.model.score_samples(X)

    def get_predictions(self) -> Optional[np.ndarray]:
        """Get the anomaly predictions from training.

        Returns:
            Array of predictions (1 for normal, -1 for anomaly), or None if not trained
        """
        return self.predictions

    def get_anomaly_scores(self) -> Optional[np.ndarray]:
        """Get the anomaly scores from training.

        Returns:
            Array of anomaly scores (more negative = more anomalous), or None if not trained
        """
        return self.anomaly_scores
