"""Gradient Descent Variants model implementation."""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time

from .data import (
    get_test_function,
    generate_contour_data,
    get_initial_point
)
from .schema import GradientDescentRequest, GradientDescentResponse, OptimizerResult


class BaseOptimizer:
    """Base class for gradient descent optimizers."""

    def __init__(self, learning_rate: float = 0.01):
        """Initialize optimizer.

        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
        self.reset()

    def reset(self):
        """Reset optimizer state."""
        pass

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Perform one optimization step.

        Args:
            params: Current parameters
            grad: Gradient at current parameters

        Returns:
            Updated parameters
        """
        raise NotImplementedError


class SGD(BaseOptimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate: float = 0.01):
        """Initialize SGD optimizer.

        Args:
            learning_rate: Learning rate
        """
        super().__init__(learning_rate)

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Perform SGD update: θ = θ - η * ∇f(θ).

        Args:
            params: Current parameters
            grad: Gradient

        Returns:
            Updated parameters
        """
        return params - self.learning_rate * grad


class Momentum(BaseOptimizer):
    """Gradient Descent with Momentum optimizer."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """Initialize Momentum optimizer.

        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def reset(self):
        """Reset optimizer state."""
        self.velocity = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Perform momentum update.

        v = β * v + η * ∇f(θ)
        θ = θ - v

        Args:
            params: Current parameters
            grad: Gradient

        Returns:
            Updated parameters
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity + self.learning_rate * grad
        return params - self.velocity


class RMSprop(BaseOptimizer):
    """RMSprop optimizer."""

    def __init__(self, learning_rate: float = 0.01, decay: float = 0.9, epsilon: float = 1e-8):
        """Initialize RMSprop optimizer.

        Args:
            learning_rate: Learning rate
            decay: Decay rate for moving average
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.decay = decay
        self.epsilon = epsilon
        self.squared_grad = None

    def reset(self):
        """Reset optimizer state."""
        self.squared_grad = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Perform RMSprop update.

        E[g²] = ρ * E[g²] + (1-ρ) * g²
        θ = θ - η * g / √(E[g²] + ε)

        Args:
            params: Current parameters
            grad: Gradient

        Returns:
            Updated parameters
        """
        if self.squared_grad is None:
            self.squared_grad = np.zeros_like(params)

        self.squared_grad = self.decay * self.squared_grad + (1 - self.decay) * grad**2
        update = self.learning_rate * grad / (np.sqrt(self.squared_grad) + self.epsilon)
        return params - update


class Adam(BaseOptimizer):
    """Adam optimizer (Adaptive Moment Estimation)."""

    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """Initialize Adam optimizer.

        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step

    def reset(self):
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Perform Adam update.

        m = β₁ * m + (1-β₁) * g
        v = β₂ * v + (1-β₂) * g²
        m̂ = m / (1 - β₁ᵗ)
        v̂ = v / (1 - β₂ᵗ)
        θ = θ - η * m̂ / (√v̂ + ε)

        Args:
            params: Current parameters
            grad: Gradient

        Returns:
            Updated parameters
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update parameters
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params - update


class Adagrad(BaseOptimizer):
    """Adagrad optimizer."""

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        """Initialize Adagrad optimizer.

        Args:
            learning_rate: Learning rate
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulated_grad = None

    def reset(self):
        """Reset optimizer state."""
        self.accumulated_grad = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Perform Adagrad update.

        G = G + g²
        θ = θ - η * g / √(G + ε)

        Args:
            params: Current parameters
            grad: Gradient

        Returns:
            Updated parameters
        """
        if self.accumulated_grad is None:
            self.accumulated_grad = np.zeros_like(params)

        self.accumulated_grad += grad**2
        update = self.learning_rate * grad / (np.sqrt(self.accumulated_grad) + self.epsilon)
        return params - update


class GradientDescentModel:
    """Model for demonstrating gradient descent optimization variants."""

    def __init__(self):
        """Initialize gradient descent model."""
        self.optimizers = {
            'sgd': SGD,
            'momentum': Momentum,
            'rmsprop': RMSprop,
            'adam': Adam,
            'adagrad': Adagrad
        }

    def optimize(
        self,
        optimizer_name: str,
        func_name: str,
        learning_rate: float,
        momentum: float,
        iterations: int,
        random_state: int
    ) -> OptimizerResult:
        """Run optimization with a specific optimizer.

        Args:
            optimizer_name: Name of optimizer
            func_name: Name of test function
            learning_rate: Learning rate
            momentum: Momentum coefficient
            iterations: Number of iterations
            random_state: Random seed

        Returns:
            OptimizerResult with trajectory and metrics
        """
        # Get test function and gradient
        func, grad_func, optimal_point, _ = get_test_function(func_name)

        # Initialize optimizer
        if optimizer_name == 'momentum':
            optimizer = self.optimizers[optimizer_name](learning_rate, momentum)
        else:
            optimizer = self.optimizers[optimizer_name](learning_rate)

        # Get initial point
        params = get_initial_point(func_name, random_state)

        # Track trajectory and loss
        trajectory = [params.copy().tolist()]
        loss_history = [float(func(params[0], params[1]))]

        # Convergence threshold
        convergence_threshold = 0.01
        iterations_to_converge = None

        # Optimization loop
        for i in range(iterations):
            # Compute gradient
            grad = grad_func(params[0], params[1])

            # Clip gradient to prevent numerical instability
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 10.0:
                grad = grad * (10.0 / grad_norm)

            # Update parameters
            params = optimizer.step(params, grad)

            # Record trajectory and loss
            trajectory.append(params.copy().tolist())
            loss = float(func(params[0], params[1]))
            loss_history.append(loss)

            # Check convergence
            if iterations_to_converge is None and loss < convergence_threshold:
                iterations_to_converge = i + 1

        # Calculate path length
        path_length = 0.0
        for i in range(len(trajectory) - 1):
            p1 = np.array(trajectory[i])
            p2 = np.array(trajectory[i + 1])
            path_length += float(np.linalg.norm(p2 - p1))

        return OptimizerResult(
            optimizer_name=optimizer_name.upper() if optimizer_name == 'sgd' else optimizer_name.capitalize(),
            trajectory=trajectory,
            loss_history=loss_history,
            final_loss=loss_history[-1],
            iterations_to_converge=iterations_to_converge,
            path_length=path_length
        )

    def run(self, request: GradientDescentRequest) -> GradientDescentResponse:
        """Run gradient descent demonstration.

        Args:
            request: Request parameters

        Returns:
            Response with optimization results
        """
        start_time = time.time()

        # Get test function
        func, _, optimal_point, plot_range = get_test_function(request.test_function)

        # Generate contour data
        contour_data = generate_contour_data(func, plot_range[0], plot_range[1])
        contour_data['optimal_point'] = list(optimal_point)

        # Run optimizations
        if request.compare_all:
            results = []
            for optimizer_name in self.optimizers.keys():
                result = self.optimize(
                    optimizer_name=optimizer_name,
                    func_name=request.test_function,
                    learning_rate=request.learning_rate,
                    momentum=request.momentum,
                    iterations=request.iterations,
                    random_state=request.random_state
                )
                results.append(result)

            single_result = None
        else:
            single_result = self.optimize(
                optimizer_name=request.optimizer_type,
                func_name=request.test_function,
                learning_rate=request.learning_rate,
                momentum=request.momentum,
                iterations=request.iterations,
                random_state=request.random_state
            )
            results = None

        # Create statistics table
        statistics_table = []
        if results:
            for result in results:
                statistics_table.append({
                    'optimizer': result.optimizer_name,
                    'final_loss': round(result.final_loss, 6),
                    'iterations_to_converge': result.iterations_to_converge if result.iterations_to_converge else 'N/A',
                    'path_length': round(result.path_length, 4),
                    'efficiency': round(1.0 / (result.path_length + 1e-6), 4)
                })
        elif single_result:
            statistics_table.append({
                'optimizer': single_result.optimizer_name,
                'final_loss': round(single_result.final_loss, 6),
                'iterations_to_converge': single_result.iterations_to_converge if single_result.iterations_to_converge else 'N/A',
                'path_length': round(single_result.path_length, 4),
                'efficiency': round(1.0 / (single_result.path_length + 1e-6), 4)
            })

        # Prepare visualization data
        visualization_data = {
            'test_function': request.test_function,
            'optimal_point': list(optimal_point),
            'plot_range': {
                'x_range': list(plot_range[0]),
                'y_range': list(plot_range[1])
            },
            'convergence_threshold': 0.01
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return GradientDescentResponse(
            success=True,
            results=results,
            single_result=single_result,
            contour_data=contour_data,
            statistics_table=statistics_table,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            parameters_used={
                'optimizer_type': request.optimizer_type,
                'learning_rate': request.learning_rate,
                'momentum': request.momentum,
                'iterations': request.iterations,
                'compare_all': request.compare_all,
                'test_function': request.test_function
            }
        )
