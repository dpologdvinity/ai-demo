"""
A* Pathfinding algorithm package.

This package provides a complete implementation of the A* pathfinding algorithm
for grid-based navigation with obstacle avoidance.
"""

from .model import AStarPathfinder
from .schema import AStarRequest, AStarResponse

__all__ = [
    'AStarPathfinder',
    'AStarRequest',
    'AStarResponse',
]
