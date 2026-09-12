"""Pydantic schemas for Hyperparameter Tuning requests and responses."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator


class HyperparameterTuningRequest(BaseModel):
    """Request schema for Hyperparameter Tuning.

    Attributes:
        method: Search method (grid, random, bayesian, compare).
        model_type: Model to tune (random_forest, svm, xgboost, mlp).
        n_trials: Number of trials for random/bayesian search.
        cv_folds: Number of cross-validation folds.
        scoring: Evaluation metric.
        dataset: Dataset to use.
        random_state: Random seed for reproducibility.
    """

    method: str = Field(
        default="grid",
        description="Search method: grid, random, bayesian, or compare"
    )
    model_type: str = Field(
        default="random_forest",
        description="Model type to tune"
    )
    n_trials: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of trials for random/bayesian search"
    )
    cv_folds: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Number of cross-validation folds"
    )
    scoring: str = Field(
        default="accuracy",
        description="Scoring metric (accuracy, f1, roc_auc, precision)"
    )
    dataset: str = Field(
        default="iris",
        description="Dataset: iris, wine, or breast_cancer"
    )
    random_state: int = Field(
        default=42,
        ge=0,
        le=1000,
        description="Random seed for reproducibility"
    )

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate search method."""
        valid_methods = ['grid', 'random', 'bayesian', 'compare']
        if v not in valid_methods:
            raise ValueError(f'method must be one of {valid_methods}')
        return v

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate model type."""
        valid_models = ['random_forest', 'svm', 'xgboost', 'mlp']
        if v not in valid_models:
            raise ValueError(f'model_type must be one of {valid_models}')
        return v

    @field_validator('scoring')
    @classmethod
    def validate_scoring(cls, v: str) -> str:
        """Validate scoring metric."""
        valid_metrics = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        if v not in valid_metrics:
            raise ValueError(f'scoring must be one of {valid_metrics}')
        return v

    @field_validator('dataset')
    @classmethod
    def validate_dataset(cls, v: str) -> str:
        """Validate dataset name."""
        valid_datasets = ['iris', 'wine', 'breast_cancer']
        if v not in valid_datasets:
            raise ValueError(f'dataset must be one of {valid_datasets}')
        return v


class TrialResult(BaseModel):
    """Individual trial result."""
    trial_number: int
    parameters: Dict[str, Any]
    score: float
    rank: int


class HyperparameterTuningResponse(BaseModel):
    """Response schema for Hyperparameter Tuning results.

    Attributes:
        success: Whether tuning completed successfully.
        best_score: Best cross-validation score achieved.
        best_params: Best hyperparameters found.
        best_estimator_test_score: Test score of best model.
        trials: List of all trial results.
        convergence_data: Convergence plot data.
        heatmap_data: Parameter space heatmap data.
        parameter_importance: Parameter importance scores (Bayesian only).
        comparison_data: Method comparison data (compare mode only).
        visualization_data: Data for frontend visualizations.
        execution_time_ms: Tuning execution time in milliseconds.
        parameters_used: Actual parameters used for tuning.
        error: Error message if tuning failed.
    """

    success: bool
    best_score: float = Field(default=0.0)
    best_params: Dict[str, Any] = Field(default_factory=dict)
    best_estimator_test_score: Optional[float] = None
    trials: List[TrialResult] = Field(default_factory=list)
    convergence_data: List[Dict[str, Any]] = Field(default_factory=list)
    heatmap_data: Optional[Dict[str, Any]] = None
    parameter_importance: Optional[Dict[str, float]] = None
    comparison_data: Optional[List[Dict[str, Any]]] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = Field(default=0.0)
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
