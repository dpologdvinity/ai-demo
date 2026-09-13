"""
Pydantic schemas for Genetic Algorithm optimizer.

This module defines request and response models for the GA optimization.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class GeneticAlgorithmRequest(BaseModel):
    """Request model for Genetic Algorithm optimizer.

    Attributes:
        population_size: Size of population
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation per individual
        crossover_rate: Probability of crossover between parents
        random_state: Random seed for reproducibility
    """

    population_size: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Size of the population"
    )
    generations: int = Field(
        default=100,
        ge=5,
        le=500,
        description="Number of generations to evolve"
    )
    mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of mutation for each individual"
    )
    crossover_rate: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Probability of crossover between parent pairs"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class GeneticAlgorithmResponse(BaseModel):
    """Response model for GA optimization results.

    Attributes:
        success: Whether optimization succeeded
        best_individual: Best solution found
        best_fitness: Fitness of best solution
        generations_run: Number of generations executed
        execution_time_ms: Time taken in milliseconds
        step_trace: List of generation statistics
    """

    success: bool = Field(description="Whether optimization succeeded")
    best_individual: float = Field(description="Best individual (x value)")
    best_fitness: float = Field(description="Best fitness achieved")
    generations_run: int = Field(description="Number of generations run")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    step_trace: List[Dict[str, Any]] = Field(
        description="Trace of best/avg fitness per generation"
    )
