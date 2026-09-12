"""Adam Optimizer implementation and comparison with other optimizers."""

import numpy as np
import time
from typing import List, Tuple, Callable, Dict, Any

from .schema import (
    AdamOptimizerRequest,
    AdamOptimizerResponse,
    AdamMetrics,
    VisualizationData,
    ModelInfo,
    OptimizerTrajectory,
    MomentumVisualization
)
from .data import get_optimization_function, get_dataset_info


class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) Optimizer.

    Combines ideas from Momentum and RMSprop:
    - Maintains exponentially decaying averages of past gradients (first moment)
    - Maintains exponentially decaying averages of past squared gradients (second moment)
    - Uses bias correction to account for initialization at zero
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Step size (alpha)
            beta1: Exponential decay rate for first moment (momentum)
            beta2: Exponential decay rate for second moment (RMSprop)
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # State variables
        self.m = None  # First moment vector (momentum)
        self.v = None  # Second moment vector (squared gradient average)
        self.t = 0     # Time step

        # For visualization
        self.m_history = []
        self.v_history = []
        self.bias_corrected_lr_history = []

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform one optimization step.

        Args:
            x: Current parameters
            gradient: Gradient at current point

        Returns:
            Updated parameters
        """
        # Initialize on first step
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.t += 1

        # Update biased first moment estimate (momentum)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

        # Update biased second moment estimate (squared gradients)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Compute bias-corrected first moment
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second moment
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Compute effective learning rate with bias correction
        effective_lr = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        # Store for visualization
        self.m_history.append(self.m.copy())
        self.v_history.append(self.v.copy())
        self.bias_corrected_lr_history.append(effective_lr)

        # Update parameters
        x_new = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return x_new


class SGDOptimizer:
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return x - self.learning_rate * gradient


class MomentumOptimizer:
    """SGD with Momentum."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(x)

        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        return x + self.velocity


class RMSpropOptimizer:
    """RMSprop optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay_rate: float = 0.9,
        epsilon: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.squared_gradient = None

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.squared_gradient is None:
            self.squared_gradient = np.zeros_like(x)

        self.squared_gradient = (
            self.decay_rate * self.squared_gradient +
            (1 - self.decay_rate) * gradient ** 2
        )

        return x - self.learning_rate * gradient / (np.sqrt(self.squared_gradient) + self.epsilon)


def optimize_function(
    func: Callable,
    grad_func: Callable,
    optimizer: Any,
    start_point: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[List[np.ndarray], List[float], int]:
    """
    Optimize a function using the given optimizer.

    Args:
        func: Objective function to minimize
        grad_func: Gradient function
        optimizer: Optimizer instance
        start_point: Starting point for optimization
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (trajectory, losses, iterations)
    """
    x = start_point.copy()
    trajectory = [x.copy()]
    losses = [func(x)]

    for i in range(max_iterations):
        gradient = grad_func(x)

        # Check for convergence
        if np.linalg.norm(gradient) < tolerance:
            break

        x = optimizer.step(x, gradient)

        trajectory.append(x.copy())
        losses.append(func(x))

        # Divergence check
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            break

    return trajectory, losses, len(trajectory) - 1


class AdamOptimizerModel:
    """Model for Adam Optimizer demonstration."""

    def __init__(self, request: AdamOptimizerRequest):
        """
        Initialize Adam Optimizer model.

        Args:
            request: Request parameters
        """
        self.request = request
        self.func, self.grad_func, self.func_metadata = get_optimization_function(
            request.function_type
        )

    def train(self) -> AdamOptimizerResponse:
        """
        Run optimization comparison.

        Returns:
            AdamOptimizerResponse with results and visualizations
        """
        start_time = time.time()

        try:
            # Set random seed
            np.random.seed(self.request.random_state)

            # Get starting point
            start_point = self.func_metadata['start']

            # Initialize Adam optimizer
            adam = AdamOptimizer(
                learning_rate=self.request.learning_rate,
                beta1=self.request.beta1,
                beta2=self.request.beta2,
                epsilon=self.request.epsilon
            )

            # Optimize with Adam
            adam_trajectory, adam_losses, adam_iterations = optimize_function(
                self.func,
                self.grad_func,
                adam,
                start_point,
                self.request.max_iterations
            )

            # Prepare optimizer trajectories list
            optimizer_trajectories = [
                OptimizerTrajectory(
                    name="Adam",
                    x_trajectory=[p[0] for p in adam_trajectory],
                    y_trajectory=[p[1] for p in adam_trajectory],
                    loss_trajectory=adam_losses,
                    final_loss=adam_losses[-1],
                    iterations=adam_iterations
                )
            ]

            # Initialize metrics
            metrics_dict = {
                'adam_final_loss': float(adam_losses[-1]),
                'adam_iterations': adam_iterations
            }

            # Compare with other optimizers if requested
            if self.request.compare_optimizers:
                # SGD (use smaller learning rate for stability)
                sgd = SGDOptimizer(learning_rate=0.001)
                sgd_trajectory, sgd_losses, sgd_iterations = optimize_function(
                    self.func, self.grad_func, sgd, start_point, self.request.max_iterations
                )
                # Handle NaN/Inf values
                if np.isfinite(sgd_losses[-1]):
                    optimizer_trajectories.append(
                        OptimizerTrajectory(
                            name="SGD",
                            x_trajectory=[p[0] for p in sgd_trajectory],
                            y_trajectory=[p[1] for p in sgd_trajectory],
                            loss_trajectory=sgd_losses,
                            final_loss=sgd_losses[-1],
                            iterations=sgd_iterations
                        )
                    )
                    metrics_dict['sgd_final_loss'] = float(sgd_losses[-1])
                    metrics_dict['sgd_iterations'] = sgd_iterations

                # Momentum (use smaller learning rate)
                momentum = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
                mom_trajectory, mom_losses, mom_iterations = optimize_function(
                    self.func, self.grad_func, momentum, start_point, self.request.max_iterations
                )
                if np.isfinite(mom_losses[-1]):
                    optimizer_trajectories.append(
                        OptimizerTrajectory(
                            name="Momentum",
                            x_trajectory=[p[0] for p in mom_trajectory],
                            y_trajectory=[p[1] for p in mom_trajectory],
                            loss_trajectory=mom_losses,
                            final_loss=mom_losses[-1],
                            iterations=mom_iterations
                        )
                    )
                    metrics_dict['momentum_final_loss'] = float(mom_losses[-1])
                    metrics_dict['momentum_iterations'] = mom_iterations

                # RMSprop
                rmsprop = RMSpropOptimizer(learning_rate=0.001, decay_rate=0.9)
                rms_trajectory, rms_losses, rms_iterations = optimize_function(
                    self.func, self.grad_func, rmsprop, start_point, self.request.max_iterations
                )
                if np.isfinite(rms_losses[-1]):
                    optimizer_trajectories.append(
                        OptimizerTrajectory(
                            name="RMSprop",
                            x_trajectory=[p[0] for p in rms_trajectory],
                            y_trajectory=[p[1] for p in rms_trajectory],
                            loss_trajectory=rms_losses,
                            final_loss=rms_losses[-1],
                            iterations=rms_iterations
                        )
                    )
                    metrics_dict['rmsprop_final_loss'] = float(rms_losses[-1])
                    metrics_dict['rmsprop_iterations'] = rms_iterations

                # Calculate improvement if SGD converged
                if 'sgd_final_loss' in metrics_dict and sgd_losses[-1] > 0 and np.isfinite(sgd_losses[-1]):
                    improvement = ((sgd_losses[-1] - adam_losses[-1]) / sgd_losses[-1]) * 100
                    metrics_dict['convergence_improvement'] = float(improvement)

            # Prepare momentum visualization data
            momentum_viz = MomentumVisualization(
                iterations=list(range(len(adam.m_history))),
                first_moment_x=[float(m[0]) for m in adam.m_history],
                first_moment_y=[float(m[1]) for m in adam.m_history],
                second_moment_x=[float(v[0]) for v in adam.v_history],
                second_moment_y=[float(v[1]) for v in adam.v_history],
                bias_corrected_lr=[float(lr) for lr in adam.bias_corrected_lr_history]
            )

            # Prepare convergence comparison
            convergence_comparison = {
                opt.name: opt.loss_trajectory for opt in optimizer_trajectories
            }

            # Prepare visualization data
            visualization_data = VisualizationData(
                optimizers=optimizer_trajectories,
                contour_levels=self.func_metadata['contour_levels'],
                x_range=self.func_metadata['x_range'],
                y_range=self.func_metadata['y_range'],
                function_type=self.func_metadata['name'],
                momentum_data=momentum_viz,
                convergence_comparison=convergence_comparison
            )

            # Prepare metrics
            metrics = AdamMetrics(**metrics_dict)

            # Prepare model info
            model_info = ModelInfo(
                function_name=self.func_metadata['name'],
                function_description=self.func_metadata['description'],
                optimal_value=self.func_metadata['optimal'],
                starting_point=start_point.tolist(),
                adam_config={
                    'learning_rate': self.request.learning_rate,
                    'beta1': self.request.beta1,
                    'beta2': self.request.beta2,
                    'epsilon': self.request.epsilon
                },
                total_comparisons=len(optimizer_trajectories)
            )

            execution_time_ms = (time.time() - start_time) * 1000

            return AdamOptimizerResponse(
                success=True,
                metrics=metrics,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                model_info=model_info,
                parameters_used=self.request.model_dump()
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return AdamOptimizerResponse(
                success=False,
                metrics=AdamMetrics(
                    adam_final_loss=0.0,
                    adam_iterations=0
                ),
                visualization_data=VisualizationData(
                    optimizers=[],
                    contour_levels=[],
                    x_range=[0, 0],
                    y_range=[0, 0],
                    function_type="",
                    momentum_data=MomentumVisualization(
                        iterations=[],
                        first_moment_x=[],
                        first_moment_y=[],
                        second_moment_x=[],
                        second_moment_y=[],
                        bias_corrected_lr=[]
                    ),
                    convergence_comparison={}
                ),
                execution_time_ms=execution_time_ms,
                model_info=ModelInfo(
                    function_name="",
                    function_description="",
                    optimal_value="",
                    starting_point=[],
                    adam_config={},
                    total_comparisons=0
                ),
                parameters_used=self.request.model_dump(),
                error=str(e)
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'function_name': self.func_metadata['name'],
            'description': self.func_metadata['description'],
            'optimal_value': self.func_metadata['optimal']
        }
