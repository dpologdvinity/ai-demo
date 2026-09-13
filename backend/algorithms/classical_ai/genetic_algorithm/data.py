"""
Data utilities for Genetic Algorithm.

GA is a general optimization algorithm that doesn't require external data.
"""


def get_ga_info() -> dict:
    """Get information about the GA optimization problem.

    Returns:
        Dictionary with problem information
    """
    return {
        'name': 'Genetic Algorithm',
        'description': 'Optimization using evolutionary principles',
        'objective': 'Maximize f(x) = x * sin(10*pi*x) + 1 over x in [0, 2]',
        'domain': '[0, 2]',
        'function_type': 'multi-modal'
    }
