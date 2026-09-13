"""
Pydantic schemas for N-Queens CSP solver.

This module defines request and response models for the N-Queens algorithm.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class NQueensRequest(BaseModel):
    """Request model for N-Queens solver.

    Attributes:
        board_size: Size of the chessboard (N)
        random_state: Random seed for reproducibility
    """

    board_size: int = Field(
        default=8,
        ge=4,
        le=12,
        description="Size of the chessboard (N)"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class NQueensResponse(BaseModel):
    """Response model for N-Queens solver results.

    Attributes:
        success: Whether the solver succeeded
        solved: Whether a solution was found
        solution: Solution as List[int] where solution[col] = row
        backtrack_count: Number of backtracking steps taken
        execution_time_ms: Time taken in milliseconds
        board_size: Size of the board
        step_trace: List of placement attempts (capped at 300 entries)
    """

    success: bool = Field(description="Whether the solver succeeded")
    solved: bool = Field(description="Whether a solution was found")
    solution: List[int] = Field(
        description="Solution: solution[col] = row of queen in that column"
    )
    backtrack_count: int = Field(description="Number of backtracking steps")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    board_size: int = Field(description="Size of the board")
    step_trace: List[Dict[str, Any]] = Field(
        description="Trace of placement attempts and backtracks"
    )
