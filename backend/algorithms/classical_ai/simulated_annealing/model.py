"""
Simulated Annealing optimizer implementation.

This module implements a simulated annealing algorithm to minimize
a cost function: cost(x) = -(x * sin(10*pi*x) + 1) over x in [0, 2].
(Negated to frame maximization as minimization, since SA is a minimizer.)
"""

import time
import numpy as np
from typing import Dict, Any
from .schema import SimulatedAnnealingRequest, SimulatedAnnealingResponse


def cost_function(x: float) -> float:
    """Evaluate the cost function (negated for minimization).

    Cost: -(x * sin(10*pi*x) + 1)
    This is the negated fitness function, so minimizing cost = maximizing fitness.
    Domain: [0, 2]

    Args:
        x: Input value in [0, 2]

    Returns:
        Cost value
    """
    return -(x * np.sin(10 * np.pi * x) + 1)


class SimulatedAnnealingSolver:
    """Simulated Annealing optimizer for continuous optimization.

    Solves single-variable optimization using simulated annealing with:
    - Metropolis criterion for acceptance
    - Geometric cooling schedule
    - Gaussian perturbation for neighbor generation
    - Tracking of best-ever solution

    Attributes:
        initial_temperature: Starting temperature
        cooling_rate: Temperature decay rate per iteration
        max_iterations: Maximum number of iterations
        random_state: Random seed
    """

    def __init__(
        self,
        initial_temperature: float = 10.0,
        cooling_rate: float = 0.95,
        max_iterations: int = 200,
        random_state: int = 42
    ):
        """Initialize SA solver.

        Args:
            initial_temperature: Starting temperature
            cooling_rate: Temperature decay rate (per iteration)
            max_iterations: Maximum number of iterations
            random_state: Random seed
        """
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.rng = np.random.RandomState(random_state)

    def solve(self, request: SimulatedAnnealingRequest) -> Dict[str, Any]:
        """Run the simulated annealing optimization.

        Args:
            request: SimulatedAnnealingRequest with parameters

        Returns:
            Dictionary with best_solution, best_cost, and step_trace
        """
        self.initial_temperature = request.initial_temperature
        self.cooling_rate = request.cooling_rate
        self.max_iterations = request.max_iterations
        self.rng = np.random.RandomState(request.random_state)

        # Initialize with random solution in [0, 2]
        current_solution = self.rng.uniform(0, 2)
        current_cost = cost_function(current_solution)

        # Track best solution found
        best_solution = current_solution
        best_cost = current_cost

        temperature = self.initial_temperature
        step_trace = []

        for iteration in range(self.max_iterations):
            # Generate neighbor solution via gaussian perturbation
            neighbor_solution = current_solution + self.rng.normal(0, 0.1)
            neighbor_solution = np.clip(neighbor_solution, 0, 2)

            neighbor_cost = cost_function(neighbor_solution)

            # Metropolis criterion: accept if better, or with probability based on temperature
            cost_delta = neighbor_cost - current_cost
            if cost_delta < 0:
                # Neighbor is better
                accepted = True
            else:
                # Accept with probability exp(-delta/T)
                acceptance_prob = np.exp(-cost_delta / temperature)
                accepted = self.rng.random() < acceptance_prob

            # Record step
            step_trace.append({
                'step': iteration,
                'temperature': float(temperature),
                'current_solution': float(current_solution),
                'current_cost': float(current_cost),
                'candidate_solution': float(neighbor_solution),
                'candidate_cost': float(neighbor_cost),
                'accepted': bool(accepted)
            })

            if accepted:
                current_solution = neighbor_solution
                current_cost = neighbor_cost

            # Update best solution if current is better
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution

            # Cool down temperature
            temperature *= self.cooling_rate

        return {
            'best_solution': float(best_solution),
            'best_cost': float(best_cost),
            'iterations_run': self.max_iterations,
            'step_trace': step_trace
        }
