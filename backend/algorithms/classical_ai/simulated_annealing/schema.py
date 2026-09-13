"""
Pydantic schemas for Simulated Annealing optimizer.

This module defines request and response models for the SA optimization.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class SimulatedAnnealingRequest(BaseModel):
    """Request model for Simulated Annealing optimizer.

    Attributes:
        initial_temperature: Starting temperature for annealing
        cooling_rate: Rate at which temperature decreases per iteration
        max_iterations: Maximum number of iterations
        random_state: Random seed for reproducibility
    """

    initial_temperature: float = Field(
        default=10.0,
        ge=0.1,
        le=100.0,
        description="Starting temperature"
    )
    cooling_rate: float = Field(
        default=0.95,
        ge=0.8,
        le=0.999,
        description="Temperature multiplied by this each step (cooling schedule)"
    )
    max_iterations: int = Field(
        default=200,
        ge=20,
        le=1000,
        description="Maximum number of iterations"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class SimulatedAnnealingResponse(BaseModel):
    """Response model for SA optimization results.

    Attributes:
        success: Whether optimization succeeded
        best_solution: Best solution found
        best_cost: Cost of best solution (minimization)
        iterations_run: Number of iterations executed
        execution_time_ms: Time taken in milliseconds
        step_trace: List of iteration statistics
    """

    success: bool = Field(description="Whether optimization succeeded")
    best_solution: float = Field(description="Best solution found")
    best_cost: float = Field(description="Best cost achieved")
    iterations_run: int = Field(description="Number of iterations run")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    step_trace: List[Dict[str, Any]] = Field(
        description="Trace of temperature, cost, and decisions per iteration"
    )
