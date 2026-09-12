"""Imbalanced Classification model implementation.

This module implements various techniques for handling class imbalance
including SMOTE, random undersampling, random oversampling, and class weights.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .data import prepare_imbalanced_data, calculate_class_distribution
from .schema import StrategyMetrics, ROCCurveData, PRCurveData


class ImbalancedClassificationModel:
    """Model for handling imbalanced classification tasks.

    This class provides multiple strategies for handling class imbalance:
    1. SMOTE (Synthetic Minority Over-sampling Technique)
    2. Random Undersampling
    3. Random Oversampling
    4. Class Weight Balancing

    Attributes:
        model_type: Type of classifier ('random_forest', 'logistic', 'xgboost')
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        model_type: str = 'random_forest',
        random_state: int = 42
    ):
        """Initialize Imbalanced Classification model.

        Args:
            model_type: Type of classifier to use
            random_state: Random seed for reproducibility
        """
        if not IMBLEARN_AVAILABLE:
            raise ImportError(
                "imbalanced-learn is required for this module. "
                "Install it with: pip install imbalanced-learn"
            )

        self.model_type = model_type
        self.random_state = random_state

    def _create_classifier(
        self,
        class_weight: Optional[Dict[int, float]] = None
    ):
        """Create a classifier instance based on model_type.

        Args:
            class_weight: Optional class weights dictionary

        Returns:
            Classifier instance
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight=class_weight,
                random_state=self.random_state
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                random_state=self.random_state
            )
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not available. Using Random Forest instead.")

            # Calculate scale_pos_weight for XGBoost
            scale_pos_weight = None
            if class_weight:
                scale_pos_weight = class_weight.get(0, 1.0) / class_weight.get(1, 1.0)

            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def train_single_strategy(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        strategy: str,
        **strategy_params
    ) -> Tuple[StrategyMetrics, Dict[str, Any], Dict[str, Any]]:
        """Train model with a single resampling strategy.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            strategy: Strategy name ('smote', 'undersample', 'oversample', 'class_weights', 'original')
            **strategy_params: Additional parameters for the strategy

        Returns:
            Tuple of (metrics, roc_data, pr_data)
        """
        # Apply resampling strategy
        if strategy == 'original':
            X_resampled, y_resampled = X_train, y_train
            class_weight = None
        elif strategy == 'smote':
            X_resampled, y_resampled = self._apply_smote(
                X_train, y_train, **strategy_params
            )
            class_weight = None
        elif strategy == 'undersample':
            X_resampled, y_resampled = self._apply_undersampling(
                X_train, y_train, **strategy_params
            )
            class_weight = None
        elif strategy == 'oversample':
            X_resampled, y_resampled = self._apply_oversampling(
                X_train, y_train, **strategy_params
            )
            class_weight = None
        elif strategy == 'class_weights':
            X_resampled, y_resampled = X_train, y_train
            class_weight = self._compute_class_weights(y_train)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Create and train classifier
        clf = self._create_classifier(class_weight)
        clf.fit(X_resampled, y_resampled)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Get prediction probabilities
        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_proba = clf.decision_function(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)

        # Calculate ROC curve
        roc_data = self._calculate_roc_curve(y_test, y_proba)

        # Calculate Precision-Recall curve
        pr_data = self._calculate_pr_curve(y_test, y_proba)

        return metrics, roc_data, pr_data

    def _apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ratio: float = 0.5,
        k_neighbors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling.

        Args:
            X: Features
            y: Labels
            target_ratio: Target ratio for minority class
            k_neighbors: Number of neighbors for SMOTE

        Returns:
            Resampled (X, y)
        """
        # Determine sampling strategy
        # For binary classification, we want to bring minority to target_ratio
        unique, counts = np.unique(y, return_counts=True)
        majority_count = counts[0] if counts[0] > counts[1] else counts[1]

        sampling_strategy = target_ratio / (1 - target_ratio) if target_ratio < 1.0 else 1.0

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )
        return smote.fit_resample(X, y)

    def _apply_undersampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random undersampling.

        Args:
            X: Features
            y: Labels
            target_ratio: Target ratio for minority class

        Returns:
            Resampled (X, y)
        """
        sampling_strategy = target_ratio / (1 - target_ratio) if target_ratio < 1.0 else 1.0

        undersampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        return undersampler.fit_resample(X, y)

    def _apply_oversampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random oversampling (duplicate minority samples).

        Args:
            X: Features
            y: Labels
            target_ratio: Target ratio for minority class

        Returns:
            Resampled (X, y)
        """
        sampling_strategy = target_ratio / (1 - target_ratio) if target_ratio < 1.0 else 1.0

        oversampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        return oversampler.fit_resample(X, y)

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights.

        Args:
            y: Labels

        Returns:
            Dictionary mapping class to weight
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes.tolist(), weights.tolist()))

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> StrategyMetrics:
        """Calculate comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        Returns:
            StrategyMetrics object
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate metrics for minority class (class 1)
        minority_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        minority_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

        return StrategyMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            roc_auc=float(roc_auc_score(y_true, y_proba)),
            pr_auc=float(average_precision_score(y_true, y_proba)),
            confusion_matrix=cm.tolist(),
            minority_recall=float(minority_recall),
            minority_precision=float(minority_precision)
        )

    def _calculate_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate ROC curve data.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities

        Returns:
            Dictionary with ROC curve data
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        # Subsample for visualization if too many points
        if len(fpr) > 100:
            indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
            fpr = fpr[indices]
            tpr = tpr[indices]
            thresholds = thresholds[indices]

        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(auc)
        }

    def _calculate_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate Precision-Recall curve data.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities

        Returns:
            Dictionary with PR curve data
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        auc = average_precision_score(y_true, y_proba)

        # Subsample for visualization if too many points
        if len(precision) > 100:
            indices = np.linspace(0, len(precision) - 1, 100, dtype=int)
            precision = precision[indices]
            recall = recall[indices]
            # thresholds has one fewer element
            if len(thresholds) > 0:
                threshold_indices = np.linspace(0, len(thresholds) - 1, min(100, len(thresholds)), dtype=int)
                thresholds = thresholds[threshold_indices]

        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist() if len(thresholds) > 0 else [],
            'auc': float(auc)
        }

    def train_compare_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_ratio: float = 0.5,
        k_neighbors: int = 5
    ) -> Dict[str, Any]:
        """Train and compare all strategies.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            target_ratio: Target ratio for minority class
            k_neighbors: Number of neighbors for SMOTE

        Returns:
            Dictionary containing metrics, predictions, and visualization data
        """
        strategies = ['original', 'smote', 'undersample', 'oversample', 'class_weights']

        all_metrics = {}
        all_predictions = {}
        all_roc_curves = {}
        all_pr_curves = {}
        sample_counts = {}

        strategy_params = {
            'target_ratio': target_ratio,
            'k_neighbors': k_neighbors
        }

        for strategy in strategies:
            metrics, roc_data, pr_data = self.train_single_strategy(
                X_train, y_train, X_test, y_test,
                strategy, **strategy_params
            )

            all_metrics[strategy] = metrics
            all_predictions[strategy] = metrics.confusion_matrix
            all_roc_curves[strategy] = roc_data
            all_pr_curves[strategy] = pr_data

            # Calculate sample counts
            if strategy == 'original' or strategy == 'class_weights':
                sample_counts[strategy] = calculate_class_distribution(y_train)
            elif strategy == 'smote':
                X_res, y_res = self._apply_smote(X_train, y_train, **strategy_params)
                sample_counts[strategy] = calculate_class_distribution(y_res)
            elif strategy == 'undersample':
                X_res, y_res = self._apply_undersampling(X_train, y_train, target_ratio)
                sample_counts[strategy] = calculate_class_distribution(y_res)
            elif strategy == 'oversample':
                X_res, y_res = self._apply_oversampling(X_train, y_train, target_ratio)
                sample_counts[strategy] = calculate_class_distribution(y_res)

        return {
            'metrics': all_metrics,
            'predictions': all_predictions,
            'roc_curves': all_roc_curves,
            'pr_curves': all_pr_curves,
            'sample_counts': sample_counts
        }
