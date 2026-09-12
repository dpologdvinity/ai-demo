"""
Pydantic schemas for Q-Learning API endpoints.

This module defines the request and response models for the Q-Learning
API endpoints, ensuring type safety and validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class QLearningRequest(BaseModel):
    """Request schema for Q-Learning training.

    Attributes:
        learning_rate: Learning rate (alpha) for Q-value updates (0.01-1.0)
        discount_factor: Discount factor (gamma) for future rewards (0.5-0.99)
        epsilon: Exploration rate for epsilon-greedy policy (0.0-1.0)
        episodes: Number of training episodes (100-5000)
        grid_size: Size of the grid world (3-10)
        random_state: Random seed for reproducibility

    Example:
        >>> request = QLearningRequest(
        ...     learning_rate=0.1,
        ...     discount_factor=0.99,
        ...     epsilon=0.1,
        ...     episodes=1000,
        ...     grid_size=5
        ... )
    """

    learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Learning rate (alpha) for Q-value updates"
    )
    discount_factor: float = Field(
        default=0.99,
        ge=0.5,
        lt=1.0,
        description="Discount factor (gamma) for future rewards"
    )
    epsilon: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Exploration rate for epsilon-greedy policy"
    )
    episodes: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Number of training episodes"
    )
    grid_size: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Size of the grid world (grid_size x grid_size)"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class QLearningResponse(BaseModel):
    """Response schema for Q-Learning training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including rewards and success rate
        visualization_data: Grid world, Q-table heatmaps, trajectories, and charts
        execution_time_ms: Training time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed

    Example:
        >>> response = QLearningResponse(
        ...     success=True,
        ...     metrics={'avg_reward_last_100': 8.5, 'success_rate': 0.95},
        ...     visualization_data={...},
        ...     execution_time_ms=125.3,
        ...     parameters_used={'learning_rate': 0.1, 'episodes': 1000}
        ... )
    """

    success: bool
    metrics: Dict[str, float]
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any]
    error: Optional[str] = None


class QLearningInfoResponse(BaseModel):
    """Response schema for Q-Learning algorithm information.

    Attributes:
        metadata: Complete algorithm metadata
        environment_info: Information about the grid world environment

    Example:
        >>> info = QLearningInfoResponse(
        ...     metadata={...},
        ...     environment_info={'type': 'GridWorld', 'actions': 4}
        ... )
    """

    metadata: Dict[str, Any]
    environment_info: Dict[str, Any]
