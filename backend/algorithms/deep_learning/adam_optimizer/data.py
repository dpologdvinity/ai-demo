"""Data generation and optimization functions for Adam Optimizer demonstration."""

import numpy as np
from typing import Callable, Tuple, Dict, Any


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function (Banana function).

    Global minimum at (1, 1) with f(1, 1) = 0

    Args:
        x: Array of shape (2,) containing [x, y]

    Returns:
        Function value at point x
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of Rosenbrock function.

    Args:
        x: Array of shape (2,) containing [x, y]

    Returns:
        Gradient vector at point x
    """
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])


def beale(x: np.ndarray) -> float:
    """
    Beale function.

    Global minimum at (3, 0.5) with f(3, 0.5) = 0

    Args:
        x: Array of shape (2,) containing [x, y]

    Returns:
        Function value at point x
    """
    return ((1.5 - x[0] + x[0]*x[1])**2 +
            (2.25 - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)


def beale_gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of Beale function.

    Args:
        x: Array of shape (2,) containing [x, y]

    Returns:
        Gradient vector at point x
    """
    term1 = 1.5 - x[0] + x[0]*x[1]
    term2 = 2.25 - x[0] + x[0]*x[1]**2
    term3 = 2.625 - x[0] + x[0]*x[1]**3

    dx = (2 * term1 * (x[1] - 1) +
          2 * term2 * (x[1]**2 - 1) +
          2 * term3 * (x[1]**3 - 1))

    dy = (2 * term1 * x[0] +
          2 * term2 * 2 * x[0] * x[1] +
          2 * term3 * 3 * x[0] * x[1]**2)

    return np.array([dx, dy])


def himmelblau(x: np.ndarray) -> float:
    """
    Himmelblau's function.

    Has four identical local minima at:
    - (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    All with f(x) = 0

    Args:
        x: Array of shape (2,) containing [x, y]

    Returns:
        Function value at point x
    """
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def himmelblau_gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of Himmelblau's function.

    Args:
        x: Array of shape (2,) containing [x, y]

    Returns:
        Gradient vector at point x
    """
    term1 = x[0]**2 + x[1] - 11
    term2 = x[0] + x[1]**2 - 7

    dx = 4 * x[0] * term1 + 2 * term2
    dy = 2 * term1 + 4 * x[1] * term2

    return np.array([dx, dy])


def get_optimization_function(
    function_type: str
) -> Tuple[Callable, Callable, Dict[str, Any]]:
    """
    Get optimization function, gradient, and metadata.

    Args:
        function_type: Type of function ('rosenbrock', 'beale', 'himmelblau')

    Returns:
        Tuple of (function, gradient_function, metadata_dict)
    """
    functions = {
        'rosenbrock': {
            'func': rosenbrock,
            'grad': rosenbrock_gradient,
            'name': 'Rosenbrock Function',
            'description': 'Classic banana-shaped valley function',
            'optimal': '(1.0, 1.0) with f = 0',
            'start': np.array([-1.5, 2.5]),
            'x_range': [-2.5, 2.5],
            'y_range': [-1.0, 3.5],
            'contour_levels': [0.1, 1, 5, 20, 50, 100, 200, 500, 1000]
        },
        'beale': {
            'func': beale,
            'grad': beale_gradient,
            'name': 'Beale Function',
            'description': 'Multi-modal function with narrow global minimum',
            'optimal': '(3.0, 0.5) with f = 0',
            'start': np.array([0.5, 0.5]),
            'x_range': [-1.0, 4.5],
            'y_range': [-1.5, 2.5],
            'contour_levels': [0.1, 1, 5, 20, 50, 100, 500, 1000, 5000]
        },
        'himmelblau': {
            'func': himmelblau,
            'grad': himmelblau_gradient,
            'name': "Himmelblau's Function",
            'description': 'Multi-modal function with four identical minima',
            'optimal': 'Four minima at f = 0',
            'start': np.array([0.0, 0.0]),
            'x_range': [-5.0, 5.0],
            'y_range': [-5.0, 5.0],
            'contour_levels': [0.1, 1, 5, 20, 50, 100, 200, 500, 1000]
        }
    }

    if function_type not in functions:
        function_type = 'rosenbrock'

    func_data = functions[function_type]
    return func_data['func'], func_data['grad'], func_data


def generate_contour_grid(
    func: Callable,
    x_range: list,
    y_range: list,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate grid for contour plot visualization.

    Args:
        func: Optimization function
        x_range: [min, max] for x-axis
        y_range: [min, max] for y-axis
        num_points: Number of grid points per dimension

    Returns:
        Tuple of (X_grid, Y_grid, Z_grid) for contour plotting
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    return X, Y, Z


def get_dataset_info() -> Dict[str, Any]:
    """
    Get information about the optimization dataset.

    Returns:
        Dictionary containing dataset metadata
    """
    return {
        "name": "Synthetic Optimization Landscapes",
        "description": "2D non-convex optimization functions for testing optimizers",
        "functions": [
            {
                "name": "Rosenbrock",
                "description": "Banana-shaped valley, difficult to optimize",
                "optimal": "(1, 1)",
                "difficulty": "Hard - narrow curved valley"
            },
            {
                "name": "Beale",
                "description": "Multi-modal with narrow global minimum",
                "optimal": "(3, 0.5)",
                "difficulty": "Medium - multiple local minima"
            },
            {
                "name": "Himmelblau",
                "description": "Four identical local minima",
                "optimal": "Four locations",
                "difficulty": "Medium - symmetric multi-modal"
            }
        ],
        "dimensionality": "2D",
        "use_case": "Optimizer comparison and convergence analysis"
    }
