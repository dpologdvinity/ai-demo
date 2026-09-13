"""
Pydantic schemas for A* pathfinding algorithm.

This module defines request and response models for the A* pathfinding algorithm,
including parameter validation and response structure.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field


class AStarRequest(BaseModel):
    """Request model for A* pathfinding.

    Attributes:
        grid_size: Size of the square grid (grid_size x grid_size)
        obstacle_density: Fraction of cells that are obstacles (0.0 to 0.6)
        heuristic: Heuristic function to use (manhattan, euclidean, chebyshev)
        random_state: Random seed for reproducibility
    """

    grid_size: int = Field(
        default=15,
        ge=5,
        le=30,
        description="Size of the square grid (grid_size x grid_size)"
    )
    obstacle_density: float = Field(
        default=0.25,
        ge=0.0,
        le=0.6,
        description="Fraction of cells that are obstacles"
    )
    heuristic: Literal['manhattan', 'euclidean', 'chebyshev'] = Field(
        default='manhattan',
        description="Heuristic function for A* search"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class AStarResponse(BaseModel):
    """Response model for A* pathfinding results.

    Attributes:
        success: Whether the algorithm executed successfully
        path_found: Whether a path from start to goal was found
        path: List of [row, col] coordinates from start to goal
        path_length: Length of the path (number of steps)
        nodes_explored: Total number of nodes explored during search
        execution_time_ms: Execution time in milliseconds
        grid: The grid used (0=free, 1=obstacle)
        start: Starting position [row, col]
        goal: Goal position [row, col]
        step_trace: List of exploration steps, each with step number, node, g, h, f, frontier size
    """

    success: bool = Field(description="Whether the algorithm executed successfully")
    path_found: bool = Field(description="Whether a path was found")
    path: List[List[int]] = Field(description="Path from start to goal as list of [row, col]")
    path_length: int = Field(description="Length of the path")
    nodes_explored: int = Field(description="Number of nodes explored")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    grid: List[List[int]] = Field(description="Grid with obstacles (0=free, 1=obstacle)")
    start: List[int] = Field(description="Starting position [row, col]")
    goal: List[int] = Field(description="Goal position [row, col]")
    step_trace: List[Dict[str, Any]] = Field(
        description="Step-by-step trace of exploration (capped at 300 entries)"
    )
