"""
Data utilities for Simulated Annealing.

SA is a general optimization algorithm that doesn't require external data.
"""


def get_sa_info() -> dict:
    """Get information about the SA optimization problem.

    Returns:
        Dictionary with problem information
    """
    return {
        'name': 'Simulated Annealing',
        'description': 'Probabilistic optimization via simulated annealing',
        'objective': 'Minimize cost(x) = -(x * sin(10*pi*x) + 1) over x in [0, 2]',
        'domain': '[0, 2]',
        'function_type': 'multi-modal'
    }
