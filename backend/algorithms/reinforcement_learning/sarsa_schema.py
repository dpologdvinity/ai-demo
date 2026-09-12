"""
Pydantic schemas for SARSA API endpoints.

This module defines the request and response models for the SARSA
API endpoints, ensuring type safety and validation.
"""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field


class SARSARequest(BaseModel):
    """Request schema for SARSA training.

    Attributes:
        environment: Environment name ('CartPole-v1' or 'FrozenLake-v1')
        learning_rate: Learning rate (alpha) for Q-value updates (0.01-1.0)
        discount_factor: Discount factor (gamma) for future rewards (0.8-1.0)
        epsilon: Initial exploration rate for epsilon-greedy policy (0.0-1.0)
        epsilon_decay: Epsilon decay rate per episode (0.9-1.0)
        episodes: Number of training episodes (100-2000)
        random_state: Random seed for reproducibility

    Example:
        >>> request = SARSARequest(
        ...     environment='CartPole-v1',
        ...     learning_rate=0.1,
        ...     discount_factor=0.99,
        ...     epsilon=0.1,
        ...     epsilon_decay=0.995,
        ...     episodes=500
        ... )
    """

    environment: Literal['CartPole-v1', 'FrozenLake-v1'] = Field(
        default='CartPole-v1',
        description="Environment name ('CartPole-v1' or 'FrozenLake-v1')"
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Learning rate (alpha) for Q-value updates"
    )
    discount_factor: float = Field(
        default=0.99,
        ge=0.8,
        le=1.0,
        description="Discount factor (gamma) for future rewards"
    )
    epsilon: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Initial exploration rate for epsilon-greedy policy"
    )
    epsilon_decay: float = Field(
        default=0.995,
        ge=0.9,
        le=1.0,
        description="Epsilon decay rate per episode"
    )
    episodes: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Number of training episodes"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class SARSAResponse(BaseModel):
    """Response schema for SARSA training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including rewards and success rate
        visualization_data: Grid world, Q-table heatmaps, trajectories, and charts
        execution_time_ms: Training time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed

    Example:
        >>> response = SARSAResponse(
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


class SARSAInfoResponse(BaseModel):
    """Response schema for SARSA algorithm information.

    Attributes:
        metadata: Complete algorithm metadata
        environment_info: Information about the grid world environment

    Example:
        >>> info = SARSAInfoResponse(
        ...     metadata={...},
        ...     environment_info={'type': 'GridWorld', 'actions': 4}
        ... )
    """

    metadata: Dict[str, Any]
    environment_info: Dict[str, Any]
