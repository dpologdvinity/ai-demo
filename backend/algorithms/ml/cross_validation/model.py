"""
Cross-Validation algorithm implementation.

This module implements various cross-validation strategies for model evaluation,
including K-Fold, Stratified K-Fold, Shuffle Split, Leave-One-Out, and Time Series Split.
"""

import time
from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    LeaveOneOut,
    TimeSeriesSplit,
    cross_val_score,
    cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score

from .data import load_dataset
from .schema import (
    CrossValidationRequest,
    CrossValidationResponse,
    CrossValidationMetrics,
    FoldMetrics,
    VisualizationData,
    ConfusionMatrixData
)


class CrossValidationModel:
    """Cross-Validation model evaluator.

    This class encapsulates various cross-validation strategies for evaluating
    machine learning models, providing comprehensive metrics and visualization data.

    Attributes:
        cv_method: Cross-validation strategy
        n_splits: Number of splits/folds
        model_type: Type of model to evaluate
        scoring: Scoring metric
        cv_splitter: The cross-validation splitter object
        model: The model to evaluate
    """

    def __init__(
        self,
        cv_method: str = 'k_fold',
        n_splits: int = 5,
        model_type: str = 'random_forest',
        scoring: str = 'accuracy',
        shuffle: bool = True,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Initialize Cross-Validation model.

        Args:
            cv_method: CV strategy ('k_fold', 'stratified', 'shuffle_split', 'leave_one_out', 'time_series')
            n_splits: Number of folds/splits
            model_type: Model to evaluate ('random_forest', 'logistic', 'svm', 'knn')
            scoring: Scoring metric ('accuracy', 'f1', 'precision', 'recall', 'roc_auc')
            shuffle: Whether to shuffle data
            test_size: Test size for shuffle_split
            random_state: Random seed
        """
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.model_type = model_type
        self.scoring = scoring
        self.shuffle = shuffle
        self.test_size = test_size
        self.random_state = random_state

        # Create CV splitter
        self.cv_splitter = self._create_cv_splitter()

        # Create model
        self.model = self._create_model()

    def _create_cv_splitter(self):
        """Create the appropriate CV splitter based on method.

        Returns:
            CV splitter object
        """
        if self.cv_method == 'k_fold':
            return KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_method == 'stratified':
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_method == 'shuffle_split':
            return ShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                random_state=self.random_state
            )
        elif self.cv_method == 'leave_one_out':
            return LeaveOneOut()
        elif self.cv_method == 'time_series':
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown CV method: {self.cv_method}")

    def _create_model(self):
        """Create the appropriate model based on model_type.

        Returns:
            Scikit-learn model object
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                random_state=self.random_state
            )
        elif self.model_type == 'knn':
            return KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_scoring_metric(self):
        """Get scoring metric name for sklearn.

        Returns:
            Scoring metric string
        """
        if self.scoring == 'f1':
            return 'f1_weighted'
        elif self.scoring == 'precision':
            return 'precision_weighted'
        elif self.scoring == 'recall':
            return 'recall_weighted'
        else:
            return self.scoring

    def evaluate(self, X: np.ndarray, y: np.ndarray, target_names: List[str]) -> Dict[str, Any]:
        """Perform cross-validation evaluation.

        Args:
            X: Features array
            y: Target array
            target_names: List of target class names

        Returns:
            Dictionary containing fold metrics and aggregate statistics
        """
        scoring_metric = self._get_scoring_metric()

        # Perform cross-validation with both train and test scores
        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=self.cv_splitter,
            scoring=scoring_metric,
            return_train_score=True,
            return_estimator=True
        )

        # Extract scores
        test_scores = cv_results['test_score']
        train_scores = cv_results['train_score']
        estimators = cv_results['estimator']

        # Compute per-fold metrics
        fold_metrics = []
        confusion_matrices = []

        fold_idx = 0
        for train_idx, test_idx in self.cv_splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Get class distribution
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)

            train_dist = {target_names[int(cls)]: int(count) for cls, count in zip(unique_train, counts_train)}
            test_dist = {target_names[int(cls)]: int(count) for cls, count in zip(unique_test, counts_test)}

            # Create fold metrics
            fold_metric = FoldMetrics(
                fold_index=fold_idx,
                train_score=float(train_scores[fold_idx]),
                test_score=float(test_scores[fold_idx]),
                train_size=len(train_idx),
                test_size=len(test_idx),
                class_distribution={
                    'train': train_dist,
                    'test': test_dist
                }
            )
            fold_metrics.append(fold_metric)

            # Get confusion matrix for this fold
            if fold_idx < len(estimators):
                y_pred = estimators[fold_idx].predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                confusion_matrices.append(
                    ConfusionMatrixData(
                        fold_index=fold_idx,
                        matrix=cm.tolist(),
                        class_labels=target_names
                    )
                )

            fold_idx += 1

        # Compute aggregate metrics
        mean_score = float(np.mean(test_scores))
        std_score = float(np.std(test_scores))
        stability_coefficient = float(std_score / mean_score) if mean_score > 0 else 0.0

        metrics = CrossValidationMetrics(
            mean_score=mean_score,
            std_score=std_score,
            min_score=float(np.min(test_scores)),
            max_score=float(np.max(test_scores)),
            mean_train_score=float(np.mean(train_scores)),
            std_train_score=float(np.std(train_scores)),
            stability_coefficient=stability_coefficient,
            scoring_metric=self.scoring
        )

        return {
            'metrics': metrics,
            'fold_metrics': fold_metrics,
            'confusion_matrices': confusion_matrices,
            'test_scores': test_scores,
            'train_scores': train_scores
        }

    def prepare_visualization_data(
        self,
        fold_metrics: List[FoldMetrics],
        confusion_matrices: List[ConfusionMatrixData],
        test_scores: np.ndarray,
        train_scores: np.ndarray
    ) -> VisualizationData:
        """Prepare visualization data for frontend.

        Args:
            fold_metrics: List of per-fold metrics
            confusion_matrices: List of confusion matrices
            test_scores: Array of test scores
            train_scores: Array of train scores

        Returns:
            VisualizationData object
        """
        # Fold performance data (for bar chart)
        fold_performance = []
        for i, metric in enumerate(fold_metrics):
            fold_performance.append({
                'fold': f'Fold {metric.fold_index + 1}',
                'fold_index': metric.fold_index,
                'train_score': metric.train_score,
                'test_score': metric.test_score,
                'train_size': metric.train_size,
                'test_size': metric.test_size
            })

        # Score distribution data (for box plot)
        score_distribution = [
            {
                'type': 'test',
                'scores': test_scores.tolist(),
                'mean': float(np.mean(test_scores)),
                'median': float(np.median(test_scores)),
                'q1': float(np.percentile(test_scores, 25)),
                'q3': float(np.percentile(test_scores, 75)),
                'min': float(np.min(test_scores)),
                'max': float(np.max(test_scores))
            },
            {
                'type': 'train',
                'scores': train_scores.tolist(),
                'mean': float(np.mean(train_scores)),
                'median': float(np.median(train_scores)),
                'q1': float(np.percentile(train_scores, 25)),
                'q3': float(np.percentile(train_scores, 75)),
                'min': float(np.min(train_scores)),
                'max': float(np.max(train_scores))
            }
        ]

        # Train/test splits visualization
        train_test_splits = []
        for metric in fold_metrics:
            train_test_splits.append({
                'fold': metric.fold_index,
                'train_size': metric.train_size,
                'test_size': metric.test_size,
                'train_start': 0,
                'train_end': metric.train_size,
                'test_start': metric.train_size,
                'test_end': metric.train_size + metric.test_size
            })

        # Class distributions per fold
        class_distributions = []
        for metric in fold_metrics:
            class_distributions.append({
                'fold': metric.fold_index,
                'train': metric.class_distribution['train'],
                'test': metric.class_distribution['test']
            })

        return VisualizationData(
            fold_performance=fold_performance,
            score_distribution=score_distribution,
            train_test_splits=train_test_splits,
            confusion_matrices=confusion_matrices,
            class_distributions=class_distributions
        )


def train_cross_validation(request: CrossValidationRequest) -> CrossValidationResponse:
    """Train and evaluate a model using cross-validation.

    Args:
        request: CrossValidationRequest with parameters

    Returns:
        CrossValidationResponse with results

    Raises:
        ValueError: If invalid parameters provided
    """
    try:
        start_time = time.time()

        # Load dataset
        data = load_dataset(request.dataset, normalize=True)
        X = data['X']
        y = data['y']
        target_names = data['target_names']

        # Special handling for leave-one-out
        if request.cv_method == 'leave_one_out':
            n_splits = len(X)
        else:
            n_splits = request.n_splits

        # Create and run cross-validation
        cv_model = CrossValidationModel(
            cv_method=request.cv_method,
            n_splits=n_splits,
            model_type=request.model_type,
            scoring=request.scoring,
            shuffle=request.shuffle,
            test_size=request.test_size,
            random_state=request.random_state
        )

        # Evaluate
        results = cv_model.evaluate(X, y, target_names)

        # Prepare visualization data
        viz_data = cv_model.prepare_visualization_data(
            results['fold_metrics'],
            results['confusion_matrices'],
            results['test_scores'],
            results['train_scores']
        )

        execution_time_ms = (time.time() - start_time) * 1000

        # Build response
        return CrossValidationResponse(
            success=True,
            metrics=results['metrics'],
            fold_metrics=results['fold_metrics'],
            visualization_data=viz_data,
            execution_time_ms=execution_time_ms,
            parameters={
                'cv_method': request.cv_method,
                'n_splits': request.n_splits,
                'model_type': request.model_type,
                'scoring': request.scoring,
                'shuffle': request.shuffle,
                'dataset': request.dataset,
                'test_size': request.test_size,
                'random_state': request.random_state
            },
            message=f"Successfully completed {request.cv_method} cross-validation with {len(results['fold_metrics'])} folds"
        )

    except Exception as e:
        raise ValueError(f"Cross-validation failed: {str(e)}")
