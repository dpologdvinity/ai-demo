"""Activation Functions model implementation."""

import numpy as np
from typing import Dict, List, Tuple, Any
import time


class ActivationFunctionsModel:
    """Model for computing and demonstrating various activation functions.

    This class implements common neural network activation functions and their
    derivatives for educational visualization purposes.
    """

    def __init__(self, alpha: float = 0.01):
        """Initialize the activation functions model.

        Args:
            alpha: Negative slope for Leaky ReLU and ELU
        """
        self.alpha = alpha

    def relu(self, x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU) activation function.

        Args:
            x: Input array

        Returns:
            Output array with ReLU applied
        """
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation function.

        Args:
            x: Input array

        Returns:
            Derivative array
        """
        return (x > 0).astype(float)

    def leaky_relu(self, x: np.ndarray) -> np.ndarray:
        """Leaky ReLU activation function.

        Args:
            x: Input array

        Returns:
            Output array with Leaky ReLU applied
        """
        return np.where(x > 0, x, self.alpha * x)

    def leaky_relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of Leaky ReLU activation function.

        Args:
            x: Input array

        Returns:
            Derivative array
        """
        return np.where(x > 0, 1.0, self.alpha)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function.

        Args:
            x: Input array

        Returns:
            Output array with sigmoid applied
        """
        # Clip to avoid overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid activation function.

        Args:
            x: Input array

        Returns:
            Derivative array
        """
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function.

        Args:
            x: Input array

        Returns:
            Output array with tanh applied
        """
        return np.tanh(x)

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of tanh activation function.

        Args:
            x: Input array

        Returns:
            Derivative array
        """
        t = np.tanh(x)
        return 1 - t**2

    def elu(self, x: np.ndarray) -> np.ndarray:
        """Exponential Linear Unit (ELU) activation function.

        Args:
            x: Input array

        Returns:
            Output array with ELU applied
        """
        return np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -500, 500)) - 1))

    def elu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ELU activation function.

        Args:
            x: Input array

        Returns:
            Derivative array
        """
        return np.where(x > 0, 1.0, self.alpha * np.exp(np.clip(x, -500, 500)))

    def swish(self, x: np.ndarray) -> np.ndarray:
        """Swish activation function (x * sigmoid(x)).

        Args:
            x: Input array

        Returns:
            Output array with Swish applied
        """
        return x * self.sigmoid(x)

    def swish_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of Swish activation function.

        Args:
            x: Input array

        Returns:
            Derivative array
        """
        s = self.sigmoid(x)
        return s + x * s * (1 - s)

    def compute_function_data(
        self,
        x: np.ndarray,
        function_name: str
    ) -> Dict[str, np.ndarray]:
        """Compute activation function and its derivative for given inputs.

        Args:
            x: Input array
            function_name: Name of the activation function

        Returns:
            Dictionary with 'y' (function values) and 'derivative' arrays
        """
        function_map = {
            'relu': (self.relu, self.relu_derivative),
            'leaky_relu': (self.leaky_relu, self.leaky_relu_derivative),
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'tanh': (self.tanh, self.tanh_derivative),
            'elu': (self.elu, self.elu_derivative),
            'swish': (self.swish, self.swish_derivative)
        }

        if function_name not in function_map:
            raise ValueError(f"Unknown function: {function_name}")

        func, deriv = function_map[function_name]
        y = func(x)
        dy = deriv(x)

        return {'y': y, 'derivative': dy}

    def get_function_properties(self, function_name: str) -> Dict[str, Any]:
        """Get properties of an activation function.

        Args:
            function_name: Name of the activation function

        Returns:
            Dictionary with function properties
        """
        properties = {
            'relu': {
                'function': 'ReLU',
                'formula': 'f(x) = max(0, x)',
                'range': '[0, ∞)',
                'derivative_range': '{0, 1}',
                'monotonic': True,
                'zero_centered': False,
                'gradient_when_negative': 0.0,
                'pros': 'Fast computation, helps with vanishing gradients',
                'cons': 'Dead neurons (zero gradient for x < 0)'
            },
            'leaky_relu': {
                'function': 'Leaky ReLU',
                'formula': f'f(x) = x if x > 0 else {self.alpha}x',
                'range': '(-∞, ∞)',
                'derivative_range': f'{{{self.alpha}, 1}}',
                'monotonic': True,
                'zero_centered': False,
                'gradient_when_negative': self.alpha,
                'pros': 'Fixes dead neuron problem, allows negative values',
                'cons': 'Linearity for negative values'
            },
            'sigmoid': {
                'function': 'Sigmoid',
                'formula': 'f(x) = 1 / (1 + e^(-x))',
                'range': '(0, 1)',
                'derivative_range': '(0, 0.25]',
                'monotonic': True,
                'zero_centered': False,
                'gradient_when_negative': 'varies (e^(-x) / (1 + e^(-x))^2)',
                'pros': 'Smooth gradient, output in (0, 1)',
                'cons': 'Vanishing gradients, not zero-centered'
            },
            'tanh': {
                'function': 'Tanh',
                'formula': 'f(x) = (e^x - e^(-x)) / (e^x + e^(-x))',
                'range': '(-1, 1)',
                'derivative_range': '(0, 1]',
                'monotonic': True,
                'zero_centered': True,
                'gradient_when_negative': 'varies (1 - tanh^2(x))',
                'pros': 'Zero-centered, stronger gradients than sigmoid',
                'cons': 'Still suffers from vanishing gradients'
            },
            'elu': {
                'function': 'ELU',
                'formula': f'f(x) = x if x > 0 else {self.alpha}(e^x - 1)',
                'range': f'({-self.alpha}, ∞)',
                'derivative_range': f'(0, 1]',
                'monotonic': True,
                'zero_centered': False,
                'gradient_when_negative': f'{self.alpha}e^x',
                'pros': 'Smooth, can produce negative values',
                'cons': 'Exponential computation for x < 0'
            },
            'swish': {
                'function': 'Swish',
                'formula': 'f(x) = x * sigmoid(x)',
                'range': '(-∞, ∞)',
                'derivative_range': '(-∞, ∞)',
                'monotonic': False,
                'zero_centered': True,
                'gradient_when_negative': 'varies (smooth, non-zero)',
                'pros': 'Smooth, unbounded above, self-gated',
                'cons': 'More computationally expensive'
            }
        }

        return properties.get(function_name, {})

    def demonstrate_dead_neurons(self, x_negative: float = -5.0) -> Dict[str, Any]:
        """Demonstrate the dead neuron problem with ReLU vs Leaky ReLU.

        Args:
            x_negative: Negative input value for demonstration

        Returns:
            Dictionary with dead neuron demonstration data
        """
        x = np.array([x_negative])

        relu_output = self.relu(x)[0]
        relu_grad = self.relu_derivative(x)[0]

        leaky_relu_output = self.leaky_relu(x)[0]
        leaky_relu_grad = self.leaky_relu_derivative(x)[0]

        return {
            'input_value': float(x_negative),
            'relu': {
                'output': float(relu_output),
                'gradient': float(relu_grad),
                'is_dead': relu_grad == 0.0
            },
            'leaky_relu': {
                'output': float(leaky_relu_output),
                'gradient': float(leaky_relu_grad),
                'is_dead': False
            },
            'explanation': (
                f"For input x={x_negative}: ReLU has zero gradient (dead neuron), "
                f"while Leaky ReLU maintains a small gradient ({self.alpha}), "
                "allowing continued learning."
            )
        }

    def generate_comparison_data(
        self,
        input_range: List[float],
        num_points: int = 200
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
        """Generate data for all activation functions.

        Args:
            input_range: [min, max] range for x values
            num_points: Number of points to generate

        Returns:
            Tuple of (x array, dictionary of function data)
        """
        x = np.linspace(input_range[0], input_range[1], num_points)

        functions = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu', 'swish']
        function_data = {}

        for func_name in functions:
            data = self.compute_function_data(x, func_name)
            function_data[func_name] = {
                'x': x,
                'y': data['y'],
                'derivative': data['derivative']
            }

        return x, function_data

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            'alpha': self.alpha,
            'available_functions': ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu', 'swish'],
            'complexity': {
                'time': 'O(n)',
                'space': 'O(n)'
            }
        }


def compute_activation_functions(
    function_type: str = 'leaky_relu',
    alpha: float = 0.01,
    input_range: List[float] = [-10.0, 10.0],
    compare_all: bool = True,
    num_points: int = 200
) -> Dict[str, Any]:
    """Compute activation functions and their properties.

    Args:
        function_type: Primary activation function to demonstrate
        alpha: Leaky ReLU negative slope
        input_range: Range for x-axis [min, max]
        compare_all: Whether to show all functions together
        num_points: Number of points to generate

    Returns:
        Dictionary with function data, comparison table, and visualizations
    """
    start_time = time.time()

    # Initialize model
    model = ActivationFunctionsModel(alpha=alpha)

    # Generate comparison data
    x, function_data = model.generate_comparison_data(input_range, num_points)

    # Prepare function data for response
    response_function_data = {}
    functions_to_include = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu', 'swish'] if compare_all else [function_type]

    for func_name in functions_to_include:
        response_function_data[func_name] = {
            'x': function_data[func_name]['x'].tolist(),
            'y': function_data[func_name]['y'].tolist(),
            'derivative': function_data[func_name]['derivative'].tolist()
        }

    # Build comparison table
    comparison_table = []
    for func_name in ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu', 'swish']:
        props = model.get_function_properties(func_name)
        comparison_table.append(props)

    # Dead neuron demonstration
    dead_neuron_demo = model.demonstrate_dead_neurons(x_negative=-5.0)

    # Prepare visualization data
    visualization_data = {
        'functions_chart': [
            {
                'name': func_name,
                'data': [
                    {'x': float(response_function_data[func_name]['x'][i]),
                     'y': float(response_function_data[func_name]['y'][i])}
                    for i in range(len(response_function_data[func_name]['x']))
                ]
            }
            for func_name in response_function_data.keys()
        ],
        'derivatives_chart': [
            {
                'name': f"{func_name}_derivative",
                'data': [
                    {'x': float(response_function_data[func_name]['x'][i]),
                     'y': float(response_function_data[func_name]['derivative'][i])}
                    for i in range(len(response_function_data[func_name]['x']))
                ]
            }
            for func_name in response_function_data.keys()
        ],
        'input_range': input_range,
        'num_points': num_points
    }

    # Get model info
    model_info = model.get_model_info()

    execution_time_ms = (time.time() - start_time) * 1000

    return {
        'success': True,
        'function_data': response_function_data,
        'comparison_table': comparison_table,
        'dead_neuron_demo': dead_neuron_demo,
        'visualization_data': visualization_data,
        'execution_time_ms': execution_time_ms,
        'model_info': model_info,
        'parameters_used': {
            'function_type': function_type,
            'alpha': alpha,
            'input_range': input_range,
            'compare_all': compare_all,
            'num_points': num_points
        }
    }
