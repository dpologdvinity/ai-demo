"""Tests for Simulated Annealing optimizer implementation."""

import pytest
import numpy as np
from algorithms.classical_ai.simulated_annealing import (
    SimulatedAnnealingSolver,
    SimulatedAnnealingRequest,
    SimulatedAnnealingResponse
)


class TestSimulatedAnnealingSolver:
    """Test suite for Simulated Annealing solver."""

    def test_solver_initialization(self):
        """Test that solver initializes correctly."""
        solver = SimulatedAnnealingSolver(
            initial_temperature=10.0,
            cooling_rate=0.95,
            max_iterations=200,
            random_state=42
        )
        assert solver.initial_temperature == 10.0
        assert solver.cooling_rate == 0.95
        assert solver.max_iterations == 200

    def test_solve_with_default_parameters(self):
        """Test solving with default parameters."""
        solver = SimulatedAnnealingSolver()
        request = SimulatedAnnealingRequest()
        result = solver.solve(request)

        assert result['best_solution'] is not None
        assert 0 <= result['best_solution'] <= 2
        assert isinstance(result['best_cost'], float)
        assert result['iterations_run'] == 200
        assert 'step_trace' in result

    def test_solve_with_custom_parameters(self):
        """Test solving with custom parameters."""
        solver = SimulatedAnnealingSolver()
        request = SimulatedAnnealingRequest(
            initial_temperature=5.0,
            cooling_rate=0.9,
            max_iterations=100,
            random_state=99
        )
        result = solver.solve(request)

        assert result['iterations_run'] == 100
        assert len(result['step_trace']) == 100

    def test_step_trace_structure(self):
        """Test that step_trace has correct structure."""
        solver = SimulatedAnnealingSolver()
        request = SimulatedAnnealingRequest(max_iterations=20)
        result = solver.solve(request)

        assert isinstance(result['step_trace'], list)
        assert len(result['step_trace']) == 20

        for step_data in result['step_trace']:
            assert 'step' in step_data
            assert 'temperature' in step_data
            assert 'current_solution' in step_data
            assert 'current_cost' in step_data
            assert 'candidate_solution' in step_data
            assert 'candidate_cost' in step_data
            assert 'accepted' in step_data
            assert isinstance(step_data['accepted'], bool)

    def test_temperature_decreases(self):
        """Test that temperature decreases over iterations."""
        solver = SimulatedAnnealingSolver()
        request = SimulatedAnnealingRequest(
            initial_temperature=10.0,
            cooling_rate=0.95,
            max_iterations=50,
            random_state=42
        )
        result = solver.solve(request)

        step_trace = result['step_trace']
        first_temp = step_trace[0]['temperature']
        last_temp = step_trace[-1]['temperature']

        assert last_temp < first_temp
        # Check monotonic decrease
        for i in range(len(step_trace) - 1):
            assert step_trace[i + 1]['temperature'] <= step_trace[i]['temperature']

    def test_cooling_rate_effect(self):
        """Test effect of cooling rate on temperature decrease."""
        # Fast cooling
        solver_fast = SimulatedAnnealingSolver()
        request_fast = SimulatedAnnealingRequest(
            initial_temperature=10.0,
            cooling_rate=0.8,
            max_iterations=50,
            random_state=42
        )
        result_fast = solver_fast.solve(request_fast)

        # Slow cooling
        solver_slow = SimulatedAnnealingSolver()
        request_slow = SimulatedAnnealingRequest(
            initial_temperature=10.0,
            cooling_rate=0.99,
            max_iterations=50,
            random_state=42
        )
        result_slow = solver_slow.solve(request_slow)

        # Fast cooling should have lower final temperature
        final_temp_fast = result_fast['step_trace'][-1]['temperature']
        final_temp_slow = result_slow['step_trace'][-1]['temperature']

        assert final_temp_fast < final_temp_slow

    def test_solution_in_valid_domain(self):
        """Test that best solution is in valid domain [0, 2]."""
        solver = SimulatedAnnealingSolver()
        request = SimulatedAnnealingRequest(max_iterations=100)
        result = solver.solve(request)

        assert 0 <= result['best_solution'] <= 2

    def test_candidate_in_valid_domain(self):
        """Test that all candidate solutions are in [0, 2]."""
        solver = SimulatedAnnealingSolver()
        request = SimulatedAnnealingRequest(max_iterations=100)
        result = solver.solve(request)

        for step_data in result['step_trace']:
            assert 0 <= step_data['candidate_solution'] <= 2
            assert 0 <= step_data['current_solution'] <= 2

    def test_acceptance_probability(self):
        """Test that acceptance probability respects Metropolis criterion."""
        solver = SimulatedAnnealingSolver()
        request = SimulatedAnnealingRequest(
            max_iterations=200,
            random_state=42
        )
        result = solver.solve(request)

        for step_data in result['step_trace']:
            candidate_cost = step_data['candidate_cost']
            current_cost = step_data['current_cost']
            accepted = step_data['accepted']

            # If candidate is better, should always be accepted
            if candidate_cost < current_cost:
                assert accepted is True

    def test_best_cost_improves_or_stays_same(self):
        """Test that best cost found never increases."""
        solver = SimulatedAnnealingSolver()
        request = SimulatedAnnealingRequest(
            max_iterations=100,
            random_state=42
        )
        result = solver.solve(request)

        # Best cost should be <= any current cost in trace
        best_cost = result['best_cost']
        for step_data in result['step_trace']:
            assert best_cost <= step_data['current_cost'] + 1e-6

    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        request = SimulatedAnnealingRequest(
            max_iterations=50,
            random_state=123
        )

        solver1 = SimulatedAnnealingSolver()
        result1 = solver1.solve(request)

        solver2 = SimulatedAnnealingSolver()
        result2 = solver2.solve(request)

        # Should get same best solution and cost
        assert abs(result1['best_solution'] - result2['best_solution']) < 1e-6
        assert abs(result1['best_cost'] - result2['best_cost']) < 1e-6

    def test_different_initial_temperatures(self):
        """Test different initial temperatures."""
        for init_temp in [1.0, 5.0, 20.0]:
            solver = SimulatedAnnealingSolver()
            request = SimulatedAnnealingRequest(
                initial_temperature=init_temp,
                max_iterations=50,
                random_state=42
            )
            result = solver.solve(request)

            first_step_temp = result['step_trace'][0]['temperature']
            assert abs(first_step_temp - init_temp) < 1e-6

    def test_step_trace_ordering(self):
        """Test that step trace is in correct order."""
        solver = SimulatedAnnealingSolver()
        request = SimulatedAnnealingRequest(max_iterations=50)
        result = solver.solve(request)

        for i, step_data in enumerate(result['step_trace']):
            assert step_data['step'] == i
