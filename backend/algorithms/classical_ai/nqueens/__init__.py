"""
N-Queens problem solver package.

This package provides a constraint satisfaction problem solver for the
N-Queens problem using backtracking search.
"""

from .model import NQueensSolver
from .schema import NQueensRequest, NQueensResponse
from .data import get_nqueens_info

__all__ = [
    'NQueensSolver',
    'NQueensRequest',
    'NQueensResponse',
    'get_nqueens_info'
]
