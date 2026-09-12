"""Data utilities for Gradient Descent Variants demonstration."""

import numpy as np
from typing import Dict, Any, Tuple, Callable


def rosenbrock(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)².

    Global minimum at (1, 1) with f(1, 1) = 0.
    This is a classic optimization test function with a narrow valley.

    Args:
        x: X coordinate(s)
        y: Y coordinate(s)

    Returns:
        Function values
    """
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosenbrock_gradient(x: float, y: float) -> np.ndarray:
    """Gradient of Rosenbrock function.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Gradient [df/dx, df/dy]
    """
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])


def beale(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)².

    Global minimum at (3, 0.5) with f(3, 0.5) = 0.

    Args:
        x: X coordinate(s)
        y: Y coordinate(s)

    Returns:
        Function values
    """
    term1 = (1.5 - x + x * y)**2
    term2 = (2.25 - x + x * y**2)**2
    term3 = (2.625 - x + x * y**3)**2
    return term1 + term2 + term3


def beale_gradient(x: float, y: float) -> np.ndarray:
    """Gradient of Beale function.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Gradient [df/dx, df/dy]
    """
    term1 = 1.5 - x + x * y
    term2 = 2.25 - x + x * y**2
    term3 = 2.625 - x + x * y**3

    dx = 2 * term1 * (y - 1) + 2 * term2 * (y**2 - 1) + 2 * term3 * (y**3 - 1)
    dy = 2 * term1 * x + 2 * term2 * 2 * x * y + 2 * term3 * 3 * x * y**2

    return np.array([dx, dy])


def ackley(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ackley function (simplified 2D version).

    Global minimum at (0, 0) with f(0, 0) = 0.
    This function has many local minima.

    Args:
        x: X coordinate(s)
        y: Y coordinate(s)

    Returns:
        Function values
    """
    a = 20
    b = 0.2
    c = 2 * np.pi

    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))

    return term1 + term2 + a + np.e


def ackley_gradient(x: float, y: float) -> np.ndarray:
    """Gradient of Ackley function.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Gradient [df/dx, df/dy]
    """
    a = 20
    b = 0.2
    c = 2 * np.pi

    r = np.sqrt(x**2 + y**2)
    if r < 1e-10:
        return np.array([0.0, 0.0])

    exp_term = np.exp(-b * r)
    cos_term_x = np.cos(c * x)
    cos_term_y = np.cos(c * y)
    exp_cos = np.exp(0.5 * (cos_term_x + cos_term_y))

    dx = a * b * exp_term * x / r + c * np.sin(c * x) * exp_cos / 2
    dy = a * b * exp_term * y / r + c * np.sin(c * y) * exp_cos / 2

    return np.array([dx, dy])


def sphere(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sphere function: f(x,y) = x² + y².

    Global minimum at (0, 0) with f(0, 0) = 0.
    This is the simplest test function - a simple bowl.

    Args:
        x: X coordinate(s)
        y: Y coordinate(s)

    Returns:
        Function values
    """
    return x**2 + y**2


def sphere_gradient(x: float, y: float) -> np.ndarray:
    """Gradient of Sphere function.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Gradient [df/dx, df/dy]
    """
    return np.array([2 * x, 2 * y])


def get_test_function(name: str) -> Tuple[Callable, Callable, Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Get test function, its gradient, optimal point, and plot range.

    Args:
        name: Name of test function

    Returns:
        Tuple of (function, gradient_function, optimal_point, plot_range)
    """
    functions = {
        'rosenbrock': (
            rosenbrock,
            rosenbrock_gradient,
            (1.0, 1.0),
            ((-2.0, 2.0), (-1.0, 3.0))
        ),
        'beale': (
            beale,
            beale_gradient,
            (3.0, 0.5),
            ((-4.5, 4.5), (-4.5, 4.5))
        ),
        'ackley': (
            ackley,
            ackley_gradient,
            (0.0, 0.0),
            ((-5.0, 5.0), (-5.0, 5.0))
        ),
        'sphere': (
            sphere,
            sphere_gradient,
            (0.0, 0.0),
            ((-5.0, 5.0), (-5.0, 5.0))
        )
    }

    if name not in functions:
        raise ValueError(f"Unknown test function: {name}. Available: {list(functions.keys())}")

    return functions[name]


def generate_contour_data(func: Callable, x_range: Tuple[float, float], y_range: Tuple[float, float], num_points: int = 100) -> Dict[str, Any]:
    """Generate contour plot data for a test function.

    Args:
        func: Test function
        x_range: (min, max) for x axis
        y_range: (min, max) for y axis
        num_points: Number of points per axis

    Returns:
        Dictionary with x, y, z data for contour plot
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    # Apply log scale for better visualization
    Z_log = np.log10(Z + 1e-10)

    return {
        'x': x.tolist(),
        'y': y.tolist(),
        'z': Z_log.tolist(),
        'z_raw': Z.tolist()
    }


def get_initial_point(func_name: str, random_state: int = 42) -> np.ndarray:
    """Get a good starting point for optimization.

    Args:
        func_name: Name of test function
        random_state: Random seed

    Returns:
        Initial point [x, y]
    """
    np.random.seed(random_state)

    initial_points = {
        'rosenbrock': np.array([-1.5, 2.5]),
        'beale': np.array([1.0, 1.0]),
        'ackley': np.array([3.0, 3.0]),
        'sphere': np.array([4.0, 4.0])
    }

    return initial_points.get(func_name, np.array([2.0, 2.0]))


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the test functions dataset.

    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "2D Optimization Test Functions",
        "description": "Classic 2D optimization test functions for comparing gradient descent variants",
        "test_functions": [
            {
                "name": "Rosenbrock",
                "description": "Non-convex function with narrow valley, global minimum at (1, 1)",
                "difficulty": "Hard"
            },
            {
                "name": "Beale",
                "description": "Non-convex function with multiple local minima, global minimum at (3, 0.5)",
                "difficulty": "Hard"
            },
            {
                "name": "Ackley",
                "description": "Function with many local minima, global minimum at (0, 0)",
                "difficulty": "Very Hard"
            },
            {
                "name": "Sphere",
                "description": "Simple convex function, global minimum at (0, 0)",
                "difficulty": "Easy"
            }
        ],
        "dimension": "2D",
        "type": "Synthetic optimization landscapes"
    }
