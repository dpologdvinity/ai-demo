"""Feature Importance Analysis implementation for ML model interpretability."""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from .data import load_dataset
from .schema import FeatureImportanceRequest, FeatureImportanceResponse


class FeatureImportanceModel:
    """Feature Importance Analysis for ML model interpretability.

    This class implements multiple methods for analyzing feature importance:
    1. Tree-based importance (Mean Decrease in Impurity - MDI)
    2. Permutation importance (model-agnostic)
    3. Optional: SHAP values (if library available)

    Supports multiple models: Random Forest, XGBoost, Gradient Boosting
    Supports both regression and classification tasks
    """

    def __init__(self):
        """Initialize the Feature Importance model."""
        self.model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.is_classification: bool = False

    def train(self, request: FeatureImportanceRequest) -> FeatureImportanceResponse:
        """Train model and compute feature importance using specified methods.

        Args:
            request: FeatureImportanceRequest containing parameters

        Returns:
            FeatureImportanceResponse with importance scores and visualizations

        Raises:
            ValueError: If invalid parameters or unsupported method
        """
        start_time = time.time()

        # Load dataset
        data = load_dataset(
            dataset_name=request.dataset,
            random_state=request.random_state
        )
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        self.feature_names = data['feature_names']
        self.is_classification = data['is_classification']
        dataset_info = data['dataset_info']

        # Initialize model
        self.model = self._create_model(
            model_type=request.model_type,
            n_estimators=request.n_estimators,
            is_classification=self.is_classification,
            random_state=request.random_state
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, self.is_classification)

        # Compute feature importance based on method
        feature_importance = {}
        if request.method in ['tree', 'all']:
            tree_importance = self._compute_tree_importance()
            feature_importance['tree'] = tree_importance

        if request.method in ['permutation', 'all']:
            perm_importance = self._compute_permutation_importance(X_test, y_test)
            feature_importance['permutation'] = perm_importance

        # Compute feature rankings
        feature_rankings = self._compute_rankings(feature_importance, request.top_k)

        # Compute correlation matrix
        correlation_matrix = self._compute_correlation_matrix(X_train)

        # Compute cumulative importance
        cumulative_importance = self._compute_cumulative_importance(feature_importance)

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            feature_importance=feature_importance,
            feature_rankings=feature_rankings,
            correlation_matrix=correlation_matrix,
            top_k=request.top_k
        )

        # Compute statistics
        statistics = self._compute_statistics(feature_importance)

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Prepare model info
        model_info = {
            'model_type': request.model_type,
            'n_estimators': request.n_estimators,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'dataset': dataset_info,
            'methods_used': list(feature_importance.keys()),
            'is_classification': self.is_classification
        }

        return FeatureImportanceResponse(
            metrics=metrics,
            feature_importance=feature_importance,
            feature_rankings=feature_rankings,
            correlation_matrix=correlation_matrix,
            cumulative_importance=cumulative_importance,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            statistics=statistics
        )

    def _create_model(
        self,
        model_type: str,
        n_estimators: int,
        is_classification: bool,
        random_state: int
    ) -> Any:
        """Create and return the appropriate model instance.

        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'gradient_boosting')
            n_estimators: Number of estimators
            is_classification: Whether it's a classification task
            random_state: Random seed

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model type is not supported
        """
        if model_type == 'random_forest':
            if is_classification:
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                return RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1
                )

        elif model_type == 'gradient_boosting':
            if is_classification:
                return GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state
                )
            else:
                return GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state
                )

        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not installed. Install with: pip install xgboost")

            if is_classification:
                return xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1
                )

        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                "Choose from 'random_forest', 'xgboost', or 'gradient_boosting'"
            )

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        is_classification: bool
    ) -> Dict[str, float]:
        """Calculate performance metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            is_classification: Whether it's a classification task

        Returns:
            Dictionary of metrics
        """
        if is_classification:
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
            }
        else:
            return {
                'r2_score': float(r2_score(y_true, y_pred)),
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
            }

    def _compute_tree_importance(self) -> Dict[str, float]:
        """Compute tree-based feature importance (MDI).

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}

        importances = self.model.feature_importances_
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }

    def _compute_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """Compute permutation-based feature importance.

        Args:
            X: Feature array
            y: Target array
            n_repeats: Number of times to permute each feature

        Returns:
            Dictionary mapping feature names to importance scores
        """
        result = permutation_importance(
            self.model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, result.importances_mean)
        }

    def _compute_rankings(
        self,
        feature_importance: Dict[str, Dict[str, float]],
        top_k: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compute feature rankings for each method.

        Args:
            feature_importance: Dictionary of importance scores by method
            top_k: Number of top features to include

        Returns:
            Dictionary of rankings by method
        """
        rankings = {}

        for method, importances in feature_importance.items():
            # Sort features by importance (descending)
            sorted_features = sorted(
                importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Create ranking list
            rankings[method] = [
                {
                    'rank': i + 1,
                    'feature': feature,
                    'importance': importance,
                    'importance_abs': abs(importance),
                    'percentage': (abs(importance) / sum(abs(imp) for _, imp in sorted_features)) * 100
                }
                for i, (feature, importance) in enumerate(sorted_features[:top_k])
            ]

        return rankings

    def _compute_correlation_matrix(self, X: np.ndarray) -> Dict[str, Any]:
        """Compute feature correlation matrix.

        Args:
            X: Feature array

        Returns:
            Dictionary containing correlation matrix data
        """
        correlation = np.corrcoef(X, rowvar=False)

        # Create matrix data for heatmap
        matrix_data = []
        for i, row in enumerate(correlation):
            for j, value in enumerate(row):
                matrix_data.append({
                    'feature_x': self.feature_names[i],
                    'feature_y': self.feature_names[j],
                    'correlation': float(value)
                })

        return {
            'matrix': correlation.tolist(),
            'features': self.feature_names,
            'data': matrix_data
        }

    def _compute_cumulative_importance(
        self,
        feature_importance: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Compute cumulative importance curve.

        Args:
            feature_importance: Dictionary of importance scores by method

        Returns:
            List of cumulative importance data points
        """
        cumulative_data = []

        for method, importances in feature_importance.items():
            # Sort by importance (descending)
            sorted_importances = sorted(
                importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Calculate cumulative sum
            cumulative = 0
            total = sum(abs(imp) for _, imp in sorted_importances)

            for i, (feature, importance) in enumerate(sorted_importances):
                cumulative += abs(importance)
                cumulative_data.append({
                    'method': method,
                    'n_features': i + 1,
                    'feature': feature,
                    'cumulative_importance': cumulative / total if total > 0 else 0
                })

        return cumulative_data

    def _prepare_visualization_data(
        self,
        feature_importance: Dict[str, Dict[str, float]],
        feature_rankings: Dict[str, List[Dict[str, Any]]],
        correlation_matrix: Dict[str, Any],
        top_k: int
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualizations.

        Args:
            feature_importance: Raw importance scores
            feature_rankings: Ranked features
            correlation_matrix: Correlation data
            top_k: Number of top features

        Returns:
            Dictionary of visualization-ready data
        """
        # Bar chart data for feature importance
        bar_chart_data = []
        for method, rankings in feature_rankings.items():
            for item in rankings:
                bar_chart_data.append({
                    'method': method,
                    'feature': item['feature'],
                    'importance': item['importance'],
                    'rank': item['rank']
                })

        # Comparison chart (side-by-side if multiple methods)
        comparison_data = {}
        if len(feature_importance) > 1:
            all_features = set()
            for importances in feature_importance.values():
                all_features.update(importances.keys())

            for feature in all_features:
                comparison_data[feature] = {
                    method: importances.get(feature, 0)
                    for method, importances in feature_importance.items()
                }

        return {
            'bar_chart': bar_chart_data,
            'comparison': comparison_data,
            'correlation_heatmap': correlation_matrix['data'],
            'top_features': {
                method: [item['feature'] for item in rankings]
                for method, rankings in feature_rankings.items()
            }
        }

    def _compute_statistics(
        self,
        feature_importance: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Compute statistical summary of importance scores.

        Args:
            feature_importance: Dictionary of importance scores by method

        Returns:
            Dictionary of statistics by method
        """
        statistics = {}

        for method, importances in feature_importance.items():
            values = list(importances.values())
            if values:
                statistics[method] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'total_features': len(values),
                    'non_zero_features': sum(1 for v in values if abs(v) > 1e-10)
                }

        return statistics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Feature array for prediction

        Returns:
            Predicted values

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Dictionary containing model metadata

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        return {
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_classification': self.is_classification
        }
