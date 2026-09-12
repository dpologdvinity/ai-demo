"""Pydantic schemas for Gradient Descent API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class GradientDescentRequest(BaseModel):
    """Request schema for gradient descent variants demonstration.

    Attributes:
        optimizer_type: Algorithm to use (sgd, momentum, rmsprop, adam, adagrad)
        learning_rate: Learning rate for optimization
        momentum: Momentum coefficient (for momentum-based optimizers)
        iterations: Number of optimization steps
        compare_all: Whether to run all optimizers for comparison
        test_function: Test function to optimize (rosenbrock, beale, ackley, sphere)
        random_state: Random seed for reproducibility
    """

    optimizer_type: str = Field(
        default='adam',
        description="Optimizer algorithm to use"
    )
    learning_rate: float = Field(
        default=0.01,
        ge=0.001,
        le=0.5,
        description="Learning rate for optimizer"
    )
    momentum: float = Field(
        default=0.9,
        ge=0.0,
        le=0.99,
        description="Momentum coefficient (used by momentum-based optimizers)"
    )
    iterations: int = Field(
        default=100,
        ge=20,
        le=500,
        description="Number of optimization steps"
    )
    compare_all: bool = Field(
        default=True,
        description="Run all optimizers for comparison"
    )
    test_function: str = Field(
        default='rosenbrock',
        description="Test function to optimize"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    @field_validator('optimizer_type')
    @classmethod
    def validate_optimizer_type(cls, v: str) -> str:
        """Validate optimizer choice."""
        allowed = ['sgd', 'momentum', 'rmsprop', 'adam', 'adagrad']
        if v not in allowed:
            raise ValueError(f"optimizer_type must be one of {allowed}")
        return v

    @field_validator('test_function')
    @classmethod
    def validate_test_function(cls, v: str) -> str:
        """Validate test function choice."""
        allowed = ['rosenbrock', 'beale', 'ackley', 'sphere']
        if v not in allowed:
            raise ValueError(f"test_function must be one of {allowed}")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "optimizer_type": "adam",
                "learning_rate": 0.01,
                "momentum": 0.9,
                "iterations": 100,
                "compare_all": True,
                "test_function": "rosenbrock",
                "random_state": 42
            }
        }


class OptimizerResult(BaseModel):
    """Result for a single optimizer run.

    Attributes:
        optimizer_name: Name of the optimizer
        trajectory: List of [x, y] coordinates during optimization
        loss_history: Loss values at each iteration
        final_loss: Final loss value
        iterations_to_converge: Iterations needed to reach convergence threshold
        path_length: Total distance traveled during optimization
    """

    optimizer_name: str = Field(description="Name of the optimizer")
    trajectory: List[List[float]] = Field(description="Optimization trajectory [[x1, y1], [x2, y2], ...]")
    loss_history: List[float] = Field(description="Loss values at each iteration")
    final_loss: float = Field(description="Final loss value achieved")
    iterations_to_converge: Optional[int] = Field(description="Iterations to reach convergence threshold")
    path_length: float = Field(description="Total distance traveled in parameter space")


class GradientDescentResponse(BaseModel):
    """Response schema for gradient descent variants demonstration.

    Attributes:
        success: Whether optimization completed successfully
        results: Results for each optimizer (when compare_all=True)
        single_result: Result for single optimizer (when compare_all=False)
        contour_data: Contour plot data for the test function
        statistics_table: Comparison statistics for all optimizers
        visualization_data: Additional data for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        parameters_used: Parameters used for optimization
    """

    success: bool = Field(description="Whether optimization completed successfully")
    results: Optional[List[OptimizerResult]] = Field(
        default=None,
        description="Results for all optimizers (when compare_all=True)"
    )
    single_result: Optional[OptimizerResult] = Field(
        default=None,
        description="Result for single optimizer (when compare_all=False)"
    )
    contour_data: Dict[str, Any] = Field(
        description="Contour plot data for the test function"
    )
    statistics_table: List[Dict[str, Any]] = Field(
        description="Comparison statistics table"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Additional visualization data"
    )
    execution_time_ms: float = Field(description="Total execution time in milliseconds")
    parameters_used: Dict[str, Any] = Field(description="Parameters used for optimization")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "results": [
                    {
                        "optimizer_name": "Adam",
                        "trajectory": [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
                        "loss_history": [1.0, 0.5, 0.1],
                        "final_loss": 0.1,
                        "iterations_to_converge": 50,
                        "path_length": 1.414
                    }
                ],
                "contour_data": {
                    "x": [-2.0, -1.0, 0.0, 1.0, 2.0],
                    "y": [-2.0, -1.0, 0.0, 1.0, 2.0],
                    "z": [[1.0, 0.5], [0.5, 0.1]]
                },
                "statistics_table": [
                    {
                        "optimizer": "Adam",
                        "final_loss": 0.1,
                        "iterations_to_converge": 50,
                        "path_length": 1.414
                    }
                ],
                "visualization_data": {},
                "execution_time_ms": 123.45,
                "parameters_used": {
                    "optimizer_type": "adam",
                    "learning_rate": 0.01,
                    "iterations": 100
                }
            }
        }
