"""
Minimax with Alpha-Beta Pruning algorithm package.

This package provides a complete implementation of the Minimax algorithm with
alpha-beta pruning for Tic-Tac-Toe gameplay.
"""

from .model import MinimaxPlayer
from .schema import MinimaxRequest, MinimaxResponse

__all__ = [
    'MinimaxPlayer',
    'MinimaxRequest',
    'MinimaxResponse',
]
