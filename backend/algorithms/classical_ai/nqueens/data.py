"""
Data utilities for N-Queens algorithm.

N-Queens is a constraint satisfaction problem that doesn't require
external data loading.
"""


def get_nqueens_info() -> dict:
    """Get information about the N-Queens problem.

    Returns:
        Dictionary with problem information
    """
    return {
        'name': 'N-Queens',
        'description': 'Constraint Satisfaction Problem: Place N queens on an N×N board with no conflicts',
        'min_board_size': 4,
        'max_board_size': 12
    }
