"""
Genetic Algorithm optimizer implementation.

This module implements a real-valued genetic algorithm to optimize
a multi-modal test function: f(x) = x * sin(10*pi*x) + 1 over x in [0, 2].
"""

import time
import numpy as np
from typing import List, Dict, Any
from .schema import GeneticAlgorithmRequest, GeneticAlgorithmResponse


def fitness_function(x: float) -> float:
    """Evaluate the fitness function.

    Fitness: f(x) = x * sin(10*pi*x) + 1
    Domain: [0, 2]

    Args:
        x: Input value in [0, 2]

    Returns:
        Fitness value
    """
    return x * np.sin(10 * np.pi * x) + 1


class GeneticAlgorithmSolver:
    """Genetic Algorithm optimizer for continuous optimization.

    Solves single-variable optimization using a real-valued GA with:
    - Real-valued representation (individuals are floats in [0, 2])
    - Tournament selection
    - Blend crossover (average of parents)
    - Gaussian mutation
    - Elitism (best individual carries over)

    Attributes:
        population_size: Size of population
        generations: Number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        random_state: Random seed
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        random_state: int = 42
    ):
        """Initialize GA solver.

        Args:
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            random_state: Random seed
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.RandomState(random_state)

    def solve(self, request: GeneticAlgorithmRequest) -> Dict[str, Any]:
        """Run the genetic algorithm optimization.

        Args:
            request: GeneticAlgorithmRequest with parameters

        Returns:
            Dictionary with best_individual, best_fitness, and step_trace
        """
        self.population_size = request.population_size
        self.generations = request.generations
        self.mutation_rate = request.mutation_rate
        self.crossover_rate = request.crossover_rate
        self.rng = np.random.RandomState(request.random_state)

        # Initialize population with random values in [0, 2]
        population = self.rng.uniform(0, 2, self.population_size)
        step_trace = []

        best_individual = None
        best_fitness = -np.inf

        for gen in range(self.generations):
            # Evaluate fitness
            fitness_values = np.array([fitness_function(x) for x in population])

            # Track best in this generation
            gen_best_idx = np.argmax(fitness_values)
            gen_best_fitness = fitness_values[gen_best_idx]
            gen_best_individual = population[gen_best_idx]

            # Update global best
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = gen_best_individual

            # Calculate average fitness
            avg_fitness = np.mean(fitness_values)

            # Sample up to 20 individuals for population_sample
            sample_size = min(20, self.population_size)
            sample_indices = self.rng.choice(
                self.population_size, sample_size, replace=False
            )
            population_sample = population[sample_indices].tolist()

            # Record generation statistics
            step_trace.append({
                'generation': gen,
                'best_fitness': float(gen_best_fitness),
                'avg_fitness': float(avg_fitness),
                'best_individual': float(gen_best_individual),
                'population_sample': population_sample
            })

            # Create next generation
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_individual)

            # Fill rest of population with offspring
            while len(new_population) < self.population_size:
                # Tournament selection (select 2 parents)
                parent1_idx = self._tournament_selection(fitness_values)
                parent2_idx = self._tournament_selection(fitness_values)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                if self.rng.random() < self.crossover_rate:
                    # Blend crossover: offspring is average of parents + perturbation
                    offspring = (parent1 + parent2) / 2
                    # Small random blend factor for diversity
                    blend = self.rng.uniform(-0.1, 0.1)
                    offspring = offspring + blend * (parent2 - parent1)
                else:
                    # No crossover, just copy parent
                    offspring = parent1 if self.rng.random() < 0.5 else parent2

                # Mutation: add gaussian noise
                if self.rng.random() < self.mutation_rate:
                    mutation = self.rng.normal(0, 0.1)
                    offspring = offspring + mutation

                # Clip to valid range [0, 2]
                offspring = np.clip(offspring, 0, 2)
                new_population.append(offspring)

            population = np.array(new_population[:self.population_size])

        return {
            'best_individual': float(best_individual),
            'best_fitness': float(best_fitness),
            'generations_run': self.generations,
            'step_trace': step_trace
        }

    def _tournament_selection(self, fitness_values: np.ndarray, tournament_size: int = 3) -> int:
        """Select individual using tournament selection.

        Args:
            fitness_values: Array of fitness values for population
            tournament_size: Number of individuals in tournament

        Returns:
            Index of selected individual
        """
        tournament_indices = self.rng.choice(
            self.population_size, tournament_size, replace=False
        )
        tournament_fitness = fitness_values[tournament_indices]
        winner_idx_in_tournament = np.argmax(tournament_fitness)
        return tournament_indices[winner_idx_in_tournament]
