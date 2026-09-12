"""Anomaly Detection algorithm implementation with multiple methods."""

import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

from .schema import (
    AnomalyDetectionParameters,
    AnomalyDetectionResponse,
    AnomalyInfo,
    MethodComparison
)
from .data import get_anomaly_detection_data, prepare_visualization_data


class AnomalyDetectionModel:
    """Anomaly Detection implementation with multiple methods.

    This class implements three popular anomaly detection algorithms:
    1. Isolation Forest: Tree-based path length anomaly scoring
    2. One-Class SVM: Margin-based outlier detection
    3. Local Outlier Factor (LOF): Density-based anomalies

    Each method has different strengths:
    - Isolation Forest: Fast, scalable, works well in high dimensions
    - One-Class SVM: Good for finding global outliers with various kernels
    - LOF: Excellent for local density-based anomalies

    Attributes:
        model: The underlying anomaly detection model
        method: Detection method being used
        predictions: Anomaly predictions (-1 for anomaly, 1 for normal)
        anomaly_scores: Anomaly scores (interpretation varies by method)
    """

    def __init__(self):
        """Initialize the Anomaly Detection model."""
        self.model = None
        self.method = None
        self.predictions = None
        self.anomaly_scores = None

    def train(self, parameters: AnomalyDetectionParameters) -> AnomalyDetectionResponse:
        """Train anomaly detection model with specified method.

        Args:
            parameters: Training parameters including method, contamination, etc.

        Returns:
            AnomalyDetectionResponse containing training results and metrics

        Raises:
            ValueError: If parameters are invalid
        """
        try:
            start_time = time.time()

            # Load dataset
            data = get_anomaly_detection_data(
                dataset_type=parameters.dataset,
                n_samples=parameters.n_samples,
                contamination=parameters.contamination,
                random_state=parameters.random_state
            )
            X = data['X']
            y_true = data['y_true']

            # Train based on method
            if parameters.method == 'compare':
                return self._compare_methods(parameters, X, y_true, start_time)
            else:
                return self._train_single_method(parameters, X, y_true, start_time)

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return AnomalyDetectionResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Training failed: {str(e)}"
            )

    def _train_single_method(
        self,
        parameters: AnomalyDetectionParameters,
        X: np.ndarray,
        y_true: np.ndarray,
        start_time: float
    ) -> AnomalyDetectionResponse:
        """Train a single anomaly detection method.

        Args:
            parameters: Training parameters
            X: Feature array
            y_true: Ground truth labels
            start_time: Training start time

        Returns:
            AnomalyDetectionResponse with results
        """
        # Train the specified method
        if parameters.method == 'isolation_forest':
            predictions, scores = self._train_isolation_forest(parameters, X)
        elif parameters.method == 'one_class_svm':
            predictions, scores = self._train_one_class_svm(parameters, X)
        elif parameters.method == 'lof':
            predictions, scores = self._train_lof(parameters, X)
        else:
            raise ValueError(f"Unknown method: {parameters.method}")

        self.predictions = predictions
        self.anomaly_scores = scores
        self.method = parameters.method

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, scores, y_true)

        # Count anomalies
        n_anomalies = int(np.sum(predictions == -1))
        n_normal = int(np.sum(predictions == 1))

        # Create anomaly info
        anomaly_info = AnomalyInfo(
            n_anomalies=n_anomalies,
            n_normal=n_normal,
            anomaly_ratio=float(n_anomalies) / len(X),
            method_used=parameters.method
        )

        # Prepare visualization data
        visualization_data = prepare_visualization_data(
            X, predictions, scores, y_true, parameters.method
        )

        # Add decision boundary if 2D
        if X.shape[1] == 2:
            decision_boundary = self._generate_decision_boundary(X, parameters)
            visualization_data['decision_boundary'] = decision_boundary

        # Add ROC curve data
        if y_true is not None:
            roc_data = self._calculate_roc_curve(y_true, scores)
            visualization_data['roc_curve'] = roc_data

        execution_time_ms = (time.time() - start_time) * 1000

        return AnomalyDetectionResponse(
            success=True,
            metrics=metrics,
            anomaly_info=anomaly_info,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            parameters_used=parameters.model_dump()
        )

    def _train_isolation_forest(
        self,
        parameters: AnomalyDetectionParameters,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train Isolation Forest model.

        Args:
            parameters: Training parameters
            X: Feature array

        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        self.model = IsolationForest(
            n_estimators=parameters.n_estimators,
            contamination=parameters.contamination,
            random_state=parameters.random_state,
            n_jobs=-1
        )

        predictions = self.model.fit_predict(X)
        scores = self.model.score_samples(X)

        return predictions, scores

    def _train_one_class_svm(
        self,
        parameters: AnomalyDetectionParameters,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train One-Class SVM model.

        Args:
            parameters: Training parameters
            X: Feature array

        Returns:
            Tuple of (predictions, decision_scores)
        """
        # Convert contamination to nu parameter
        nu = min(parameters.contamination, 0.5)

        self.model = OneClassSVM(
            kernel=parameters.kernel,
            nu=nu,
            gamma='auto'
        )

        predictions = self.model.fit_predict(X)
        # Decision function: negative values = outliers
        scores = self.model.decision_function(X)

        return predictions, scores

    def _train_lof(
        self,
        parameters: AnomalyDetectionParameters,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train Local Outlier Factor model.

        Args:
            parameters: Training parameters
            X: Feature array

        Returns:
            Tuple of (predictions, negative_outlier_factors)
        """
        self.model = LocalOutlierFactor(
            n_neighbors=parameters.n_neighbors,
            contamination=parameters.contamination,
            novelty=False  # For training data only
        )

        predictions = self.model.fit_predict(X)
        # Negative outlier factor: more negative = more anomalous
        scores = self.model.negative_outlier_factor_

        return predictions, scores

    def _compare_methods(
        self,
        parameters: AnomalyDetectionParameters,
        X: np.ndarray,
        y_true: np.ndarray,
        start_time: float
    ) -> AnomalyDetectionResponse:
        """Compare all three anomaly detection methods.

        Args:
            parameters: Training parameters
            X: Feature array
            y_true: Ground truth labels
            start_time: Training start time

        Returns:
            AnomalyDetectionResponse with comparison results
        """
        methods = ['isolation_forest', 'one_class_svm', 'lof']
        comparison_results = []
        all_predictions = {}
        all_scores = {}

        # Train each method
        for method in methods:
            method_start = time.time()

            # Set method-specific parameters
            temp_params = parameters.model_copy()
            temp_params.method = method

            # Train
            if method == 'isolation_forest':
                preds, scores = self._train_isolation_forest(temp_params, X)
            elif method == 'one_class_svm':
                preds, scores = self._train_one_class_svm(temp_params, X)
            else:  # lof
                preds, scores = self._train_lof(temp_params, X)

            all_predictions[method] = preds
            all_scores[method] = scores

            # Calculate metrics
            n_anomalies = int(np.sum(preds == -1))
            precision = precision_score(y_true, preds, pos_label=-1, zero_division=0)
            recall = recall_score(y_true, preds, pos_label=-1, zero_division=0)
            f1 = f1_score(y_true, preds, pos_label=-1, zero_division=0)

            method_time = (time.time() - method_start) * 1000

            comparison_results.append(MethodComparison(
                method_name=method,
                n_anomalies=n_anomalies,
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                execution_time_ms=float(method_time)
            ))

        # Use best method (highest F1) for main results
        best_method_idx = max(range(len(comparison_results)),
                             key=lambda i: comparison_results[i].f1_score)
        best_method = methods[best_method_idx]

        predictions = all_predictions[best_method]
        scores = all_scores[best_method]
        self.predictions = predictions
        self.anomaly_scores = scores
        self.method = best_method

        # Calculate metrics for best method
        metrics = self._calculate_metrics(predictions, scores, y_true)

        # Add comparison metrics
        metrics['method_comparison'] = [comp.model_dump() for comp in comparison_results]

        # Count anomalies
        n_anomalies = int(np.sum(predictions == -1))
        n_normal = int(np.sum(predictions == 1))

        anomaly_info = AnomalyInfo(
            n_anomalies=n_anomalies,
            n_normal=n_normal,
            anomaly_ratio=float(n_anomalies) / len(X),
            method_used=f"Best: {best_method}"
        )

        # Prepare visualization data
        visualization_data = prepare_visualization_data(
            X, predictions, scores, y_true, best_method
        )

        # Add comparison scatter plots for all methods
        comparison_viz = {}
        for method in methods:
            method_viz = prepare_visualization_data(
                X, all_predictions[method], all_scores[method], y_true, method
            )
            comparison_viz[method] = method_viz

        visualization_data['method_comparison'] = comparison_viz

        # Add ROC curves for all methods
        roc_curves = {}
        for method in methods:
            roc_data = self._calculate_roc_curve(y_true, all_scores[method])
            roc_curves[method] = roc_data
        visualization_data['roc_curves'] = roc_curves

        execution_time_ms = (time.time() - start_time) * 1000

        return AnomalyDetectionResponse(
            success=True,
            metrics=metrics,
            anomaly_info=anomaly_info,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            parameters_used=parameters.model_dump(),
            method_comparison=comparison_results
        )

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate anomaly detection metrics.

        Args:
            predictions: Anomaly predictions
            scores: Anomaly scores
            y_true: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        n_anomalies = int(np.sum(predictions == -1))
        n_normal = int(np.sum(predictions == 1))

        metrics = {
            'n_anomalies': n_anomalies,
            'n_normal': n_normal,
            'anomaly_ratio': float(n_anomalies) / len(predictions),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
            'score_min': float(scores.min()),
            'score_max': float(scores.max())
        }

        # Calculate performance metrics if ground truth available
        if y_true is not None:
            try:
                # Confusion matrix
                cm = confusion_matrix(y_true, predictions, labels=[1, -1])
                metrics['confusion_matrix'] = cm.tolist()

                # Classification metrics
                precision = precision_score(y_true, predictions, pos_label=-1, zero_division=0)
                recall = recall_score(y_true, predictions, pos_label=-1, zero_division=0)
                f1 = f1_score(y_true, predictions, pos_label=-1, zero_division=0)

                metrics['precision'] = float(precision)
                metrics['recall'] = float(recall)
                metrics['f1_score'] = float(f1)

                # True/False positives/negatives
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics['true_positives'] = int(tp)
                    metrics['false_positives'] = int(fp)
                    metrics['true_negatives'] = int(tn)
                    metrics['false_negatives'] = int(fn)

                    # Accuracy
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    metrics['accuracy'] = float(accuracy)

            except Exception as e:
                # If metrics calculation fails, continue without them
                pass

        return metrics

    def _calculate_roc_curve(
        self,
        y_true: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate ROC curve data.

        Args:
            y_true: Ground truth labels (1 for normal, -1 for anomaly)
            scores: Anomaly scores

        Returns:
            Dictionary with ROC curve data
        """
        try:
            # Convert labels to binary (0=normal, 1=anomaly)
            y_binary = (y_true == -1).astype(int)

            # Invert scores so higher = more anomalous
            score_inverted = -scores

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_binary, score_inverted)
            roc_auc = auc(fpr, tpr)

            # Sample points for visualization (max 100 points)
            step = max(1, len(fpr) // 100)
            roc_points = [
                {'fpr': float(fpr[i]), 'tpr': float(tpr[i])}
                for i in range(0, len(fpr), step)
            ]

            return {
                'roc_points': roc_points,
                'auc': float(roc_auc),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_decision_boundary(
        self,
        X: np.ndarray,
        parameters: AnomalyDetectionParameters,
        resolution: int = 100
    ) -> Dict[str, Any]:
        """Generate decision boundary data for 2D visualization.

        Args:
            X: Feature array
            parameters: Training parameters
            resolution: Grid resolution

        Returns:
            Dictionary with decision boundary mesh data
        """
        try:
            if self.model is None or X.shape[1] != 2:
                return {}

            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, resolution),
                np.linspace(y_min, y_max, resolution)
            )

            # Predict for grid points
            mesh_points = np.c_[xx.ravel(), yy.ravel()]

            if hasattr(self.model, 'score_samples'):
                # Isolation Forest
                Z = self.model.score_samples(mesh_points)
            elif hasattr(self.model, 'decision_function'):
                # One-Class SVM
                Z = self.model.decision_function(mesh_points)
            else:
                # LOF doesn't support prediction on new data easily
                return {}

            Z = Z.reshape(xx.shape)

            # Convert to list format for JSON serialization
            return {
                'x': xx[0, :].tolist(),
                'y': yy[:, 0].tolist(),
                'z': Z.tolist()
            }

        except Exception as e:
            return {'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies for new data.

        Args:
            X: Feature array

        Returns:
            Array of predictions (1 for normal, -1 for anomaly)

        Raises:
            RuntimeError: If model hasn't been trained
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        if self.method == 'lof':
            raise RuntimeError("LOF does not support prediction on new data")

        return self.model.predict(X)
