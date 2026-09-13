"""
Pydantic schemas for Minimax with alpha-beta pruning algorithm.

This module defines request and response models for the Minimax algorithm,
including parameter validation and response structure.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field


class MinimaxRequest(BaseModel):
    """Request model for Minimax with alpha-beta pruning.

    Attributes:
        opponent: Type of opponent ('random' or 'optimal')
        ai_starts: Whether the AI player moves first
        use_alpha_beta: Whether to use alpha-beta pruning optimization
        random_state: Random seed for reproducibility
    """

    opponent: Literal['random', 'optimal'] = Field(
        default='random',
        description="How the non-AI player moves: 'random' or 'optimal'"
    )
    ai_starts: bool = Field(
        default=True,
        description="Whether the AI player moves first"
    )
    use_alpha_beta: bool = Field(
        default=True,
        description="Whether to use alpha-beta pruning for optimization"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class MinimaxResponse(BaseModel):
    """Response model for Minimax game results.

    Attributes:
        success: Whether the game executed successfully
        winner: Result of the game ('ai', 'opponent', or 'draw')
        moves: List of moves played in the game
        total_nodes_evaluated: Total number of nodes evaluated across all AI turns
        execution_time_ms: Execution time in milliseconds
        step_trace: List of AI moves with their evaluated scores and node counts
    """

    success: bool = Field(description="Whether the game executed successfully")
    winner: Literal['ai', 'opponent', 'draw'] = Field(description="Result of the game")
    moves: List[Dict[str, Any]] = Field(description="List of moves played in the game")
    total_nodes_evaluated: int = Field(description="Total nodes evaluated across all AI turns")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    step_trace: List[Dict[str, Any]] = Field(
        description="AI moves with scores and evaluation counts"
    )
