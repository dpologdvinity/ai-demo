"""Pydantic schemas for Adam Optimizer algorithm."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class AdamOptimizerRequest(BaseModel):
    """Request schema for Adam Optimizer demonstration."""

    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.1,
        description="Base learning rate (alpha)"
    )
    beta1: float = Field(
        default=0.9,
        ge=0.5,
        le=0.99,
        description="Exponential decay rate for first moment estimates"
    )
    beta2: float = Field(
        default=0.999,
        ge=0.9,
        le=0.9999,
        description="Exponential decay rate for second moment estimates"
    )
    epsilon: float = Field(
        default=1e-8,
        ge=1e-10,
        le=1e-6,
        description="Small constant for numerical stability"
    )
    compare_optimizers: bool = Field(
        default=True,
        description="Compare Adam with SGD, Momentum, and RMSprop"
    )
    max_iterations: int = Field(
        default=100,
        ge=50,
        le=500,
        description="Maximum number of optimization iterations"
    )
    function_type: str = Field(
        default="rosenbrock",
        description="Optimization function (rosenbrock, beale, himmelblau)"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class OptimizerTrajectory(BaseModel):
    """Trajectory data for a single optimizer."""

    name: str = Field(description="Optimizer name")
    x_trajectory: List[float] = Field(description="X coordinates of trajectory")
    y_trajectory: List[float] = Field(description="Y coordinates of trajectory")
    loss_trajectory: List[float] = Field(description="Loss values over iterations")
    final_loss: float = Field(description="Final loss value")
    iterations: int = Field(description="Number of iterations until convergence")


class MomentumVisualization(BaseModel):
    """Visualization data for Adam's momentum terms."""

    iterations: List[int] = Field(description="Iteration indices")
    first_moment_x: List[float] = Field(description="First moment estimate for x")
    first_moment_y: List[float] = Field(description="First moment estimate for y")
    second_moment_x: List[float] = Field(description="Second moment estimate for x")
    second_moment_y: List[float] = Field(description="Second moment estimate for y")
    bias_corrected_lr: List[float] = Field(description="Bias-corrected learning rate over time")


class AdamMetrics(BaseModel):
    """Metrics for Adam Optimizer evaluation."""

    adam_final_loss: float = Field(description="Adam final loss")
    adam_iterations: int = Field(description="Adam iterations to convergence")
    sgd_final_loss: Optional[float] = Field(default=None, description="SGD final loss")
    sgd_iterations: Optional[int] = Field(default=None, description="SGD iterations")
    momentum_final_loss: Optional[float] = Field(default=None, description="Momentum final loss")
    momentum_iterations: Optional[int] = Field(default=None, description="Momentum iterations")
    rmsprop_final_loss: Optional[float] = Field(default=None, description="RMSprop final loss")
    rmsprop_iterations: Optional[int] = Field(default=None, description="RMSprop iterations")
    convergence_improvement: Optional[float] = Field(
        default=None,
        description="Percentage improvement over SGD"
    )


class VisualizationData(BaseModel):
    """Visualization data for Adam Optimizer."""

    optimizers: List[OptimizerTrajectory] = Field(
        description="Trajectories for all compared optimizers"
    )
    contour_levels: List[float] = Field(
        description="Contour levels for loss surface visualization"
    )
    x_range: List[float] = Field(description="X-axis range for visualization")
    y_range: List[float] = Field(description="Y-axis range for visualization")
    function_type: str = Field(description="Name of optimization function")
    momentum_data: MomentumVisualization = Field(
        description="Adam momentum visualization data"
    )
    convergence_comparison: Dict[str, List[float]] = Field(
        description="Loss convergence comparison across optimizers"
    )


class ModelInfo(BaseModel):
    """Information about the optimization process."""

    function_name: str = Field(description="Optimization function name")
    function_description: str = Field(description="Function description")
    optimal_value: str = Field(description="Known optimal value")
    starting_point: List[float] = Field(description="Starting coordinates")
    adam_config: Dict[str, float] = Field(description="Adam hyperparameters used")
    total_comparisons: int = Field(description="Number of optimizers compared")


class AdamOptimizerResponse(BaseModel):
    """Response schema for Adam Optimizer demonstration."""

    success: bool = Field(default=True, description="Whether optimization succeeded")
    metrics: AdamMetrics = Field(description="Optimization performance metrics")
    visualization_data: VisualizationData = Field(description="Data for visualization")
    execution_time_ms: float = Field(description="Total execution time in milliseconds")
    model_info: ModelInfo = Field(description="Optimization process information")
    parameters_used: Dict[str, Any] = Field(description="Parameters used for optimization")
    error: Optional[str] = Field(default=None, description="Error message if optimization failed")
