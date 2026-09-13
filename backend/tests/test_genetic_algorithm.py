"""Tests for Genetic Algorithm optimizer implementation."""

import pytest
import numpy as np
from algorithms.classical_ai.genetic_algorithm import (
    GeneticAlgorithmSolver,
    GeneticAlgorithmRequest,
    GeneticAlgorithmResponse
)


class TestGeneticAlgorithmSolver:
    """Test suite for Genetic Algorithm solver."""

    def test_solver_initialization(self):
        """Test that solver initializes correctly."""
        solver = GeneticAlgorithmSolver(
            population_size=50,
            generations=100,
            mutation_rate=0.1,
            crossover_rate=0.7,
            random_state=42
        )
        assert solver.population_size == 50
        assert solver.generations == 100
        assert solver.mutation_rate == 0.1
        assert solver.crossover_rate == 0.7

    def test_solve_with_default_parameters(self):
        """Test solving with default parameters."""
        solver = GeneticAlgorithmSolver()
        request = GeneticAlgorithmRequest()
        result = solver.solve(request)

        assert result['best_individual'] is not None
        assert 0 <= result['best_individual'] <= 2
        assert isinstance(result['best_fitness'], float)
        assert result['generations_run'] == 100
        assert 'step_trace' in result

    def test_solve_with_custom_parameters(self):
        """Test solving with custom parameters."""
        solver = GeneticAlgorithmSolver()
        request = GeneticAlgorithmRequest(
            population_size=30,
            generations=50,
            mutation_rate=0.2,
            crossover_rate=0.8,
            random_state=99
        )
        result = solver.solve(request)

        assert result['generations_run'] == 50
        assert len(result['step_trace']) == 50

    def test_step_trace_structure(self):
        """Test that step_trace has correct structure."""
        solver = GeneticAlgorithmSolver()
        request = GeneticAlgorithmRequest(generations=10)
        result = solver.solve(request)

        assert isinstance(result['step_trace'], list)
        assert len(result['step_trace']) == 10

        for gen_data in result['step_trace']:
            assert 'generation' in gen_data
            assert 'best_fitness' in gen_data
            assert 'avg_fitness' in gen_data
            assert 'best_individual' in gen_data
            assert 'population_sample' in gen_data
            assert isinstance(gen_data['population_sample'], list)
            assert len(gen_data['population_sample']) <= 20

    def test_fitness_improvement_over_generations(self):
        """Test that best fitness improves over generations."""
        solver = GeneticAlgorithmSolver()
        request = GeneticAlgorithmRequest(
            population_size=50,
            generations=100,
            random_state=42
        )
        result = solver.solve(request)

        step_trace = result['step_trace']
        first_gen_fitness = step_trace[0]['best_fitness']
        final_gen_fitness = step_trace[-1]['best_fitness']

        # Final should be >= first (at least not worse)
        assert final_gen_fitness >= first_gen_fitness

    def test_population_size_parameter(self):
        """Test different population sizes."""
        for pop_size in [10, 50, 100]:
            solver = GeneticAlgorithmSolver()
            request = GeneticAlgorithmRequest(
                population_size=pop_size,
                generations=20,
                random_state=42
            )
            result = solver.solve(request)

            assert len(result['step_trace']) == 20
            # Population sample should not exceed population_size
            for gen_data in result['step_trace']:
                assert len(gen_data['population_sample']) <= min(20, pop_size)

    def test_mutation_rate_effect(self):
        """Test effect of mutation rate on convergence."""
        # Low mutation
        solver_low = GeneticAlgorithmSolver()
        request_low = GeneticAlgorithmRequest(
            population_size=50,
            generations=50,
            mutation_rate=0.01,
            random_state=42
        )
        result_low = solver_low.solve(request_low)

        # High mutation
        solver_high = GeneticAlgorithmSolver()
        request_high = GeneticAlgorithmRequest(
            population_size=50,
            generations=50,
            mutation_rate=0.5,
            random_state=42
        )
        result_high = solver_high.solve(request_high)

        # Both should find reasonable solutions
        assert result_low['best_fitness'] > -5
        assert result_high['best_fitness'] > -5

    def test_solution_in_valid_domain(self):
        """Test that best solution is in valid domain [0, 2]."""
        solver = GeneticAlgorithmSolver()
        request = GeneticAlgorithmRequest(generations=100)
        result = solver.solve(request)

        assert 0 <= result['best_individual'] <= 2

    def test_avg_fitness_calculation(self):
        """Test that average fitness is calculated correctly."""
        solver = GeneticAlgorithmSolver()
        request = GeneticAlgorithmRequest(generations=20)
        result = solver.solve(request)

        for gen_data in result['step_trace']:
            avg_fitness = gen_data['avg_fitness']
            best_fitness = gen_data['best_fitness']
            # Average should be <= best
            assert avg_fitness <= best_fitness + 1e-6  # Allow small floating point error

    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        request = GeneticAlgorithmRequest(
            population_size=20,
            generations=10,
            random_state=123
        )

        solver1 = GeneticAlgorithmSolver()
        result1 = solver1.solve(request)

        solver2 = GeneticAlgorithmSolver()
        result2 = solver2.solve(request)

        # Should get same best solution and fitness
        assert abs(result1['best_individual'] - result2['best_individual']) < 1e-6
        assert abs(result1['best_fitness'] - result2['best_fitness']) < 1e-6

    def test_population_sample_within_domain(self):
        """Test that population samples are within [0, 2]."""
        solver = GeneticAlgorithmSolver()
        request = GeneticAlgorithmRequest(generations=20)
        result = solver.solve(request)

        for gen_data in result['step_trace']:
            for individual in gen_data['population_sample']:
                assert 0 <= individual <= 2
