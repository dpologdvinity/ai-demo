"""
Pydantic schemas for DQN API endpoints.

This module defines the request and response models for the DQN
API endpoints, ensuring type safety and validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class DQNRequest(BaseModel):
    """Request schema for DQN training.

    Attributes:
        learning_rate: Learning rate for optimizer (0.0001-0.01)
        gamma: Discount factor for future rewards (0.9-0.999)
        epsilon: Exploration rate for epsilon-greedy policy (0.0-1.0)
        episodes: Number of training episodes (100-2000)
        replay_buffer_size: Experience replay buffer size (1000-50000)
        batch_size: Training batch size (16-128)
        random_state: Random seed for reproducibility

    Example:
        >>> request = DQNRequest(
        ...     learning_rate=0.001,
        ...     gamma=0.99,
        ...     epsilon=0.1,
        ...     episodes=500,
        ...     replay_buffer_size=10000,
        ...     batch_size=32
        ... )
    """

    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Learning rate for optimizer"
    )
    gamma: float = Field(
        default=0.99,
        ge=0.9,
        le=0.999,
        description="Discount factor for future rewards"
    )
    epsilon: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Exploration rate for epsilon-greedy policy"
    )
    episodes: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Number of training episodes"
    )
    replay_buffer_size: int = Field(
        default=10000,
        ge=1000,
        le=50000,
        description="Experience replay buffer size"
    )
    batch_size: int = Field(
        default=32,
        ge=16,
        le=128,
        description="Training batch size"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class DQNResponse(BaseModel):
    """Response schema for DQN training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including rewards, success rate, convergence
        visualization_data: Episode rewards, loss curves, trajectories
        execution_time_ms: Training time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed

    Example:
        >>> response = DQNResponse(
        ...     success=True,
        ...     metrics={'avg_reward_last_100': 195.5, 'success_rate': 0.85},
        ...     visualization_data={...},
        ...     execution_time_ms=45230.5,
        ...     parameters_used={'learning_rate': 0.001, 'episodes': 500}
        ... )
    """

    success: bool
    metrics: Dict[str, Any]
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any]
    error: Optional[str] = None


class DQNInfoResponse(BaseModel):
    """Response schema for DQN algorithm information.

    Attributes:
        metadata: Complete algorithm metadata
        environment_info: Information about the CartPole environment

    Example:
        >>> info = DQNInfoResponse(
        ...     metadata={...},
        ...     environment_info={'name': 'CartPole-v1', 'actions': 2}
        ... )
    """

    metadata: Dict[str, Any]
    environment_info: Dict[str, Any]
