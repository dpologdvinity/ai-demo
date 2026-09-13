"""Tests for A* pathfinding algorithm implementation."""

import pytest
from algorithms.classical_ai.astar import AStarPathfinder, AStarRequest


class TestAStarPathfinder:
    """Test suite for A* pathfinding algorithm."""

    def test_astar_default_parameters(self):
        """Test A* with default parameters."""
        pathfinder = AStarPathfinder()
        request = AStarRequest()

        response = pathfinder.solve(request)

        assert response.success is True
        assert response.start == [0, 0]
        assert response.goal == [14, 14]
        assert response.path_length == len(response.path)
        assert response.nodes_explored > 0
        assert response.execution_time_ms > 0

    def test_astar_with_low_obstacle_density(self):
        """Test that A* always finds a path with low obstacle density."""
        pathfinder = AStarPathfinder()
        request = AStarRequest(grid_size=10, obstacle_density=0.0)

        response = pathfinder.solve(request)

        assert response.success is True
        assert response.path_found is True
        assert len(response.path) > 0
        assert response.path[0] == [0, 0]
        assert response.path[-1] == [9, 9]

    def test_astar_different_heuristics(self):
        """Test A* with different heuristic functions."""
        heuristics = ['manhattan', 'euclidean', 'chebyshev']
        results = []

        for heuristic in heuristics:
            pathfinder = AStarPathfinder()
            request = AStarRequest(grid_size=10, obstacle_density=0.2, heuristic=heuristic)
            response = pathfinder.solve(request)

            assert response.success is True
            results.append((heuristic, response.path_found, response.nodes_explored))

        # Verify all heuristics executed
        assert len(results) == 3
        # All should have same outcome but potentially different nodes explored
        assert all(r[1] == results[0][1] for r in results)

    def test_astar_step_trace_format(self):
        """Test that step trace has correct format."""
        pathfinder = AStarPathfinder()
        request = AStarRequest(grid_size=10, obstacle_density=0.1)

        response = pathfinder.solve(request)

        assert response.success is True
        assert isinstance(response.step_trace, list)

        if len(response.step_trace) > 0:
            step = response.step_trace[0]
            assert 'step' in step
            assert 'node' in step
            assert 'g' in step
            assert 'h' in step
            assert 'f' in step
            assert 'frontier_size' in step

            assert isinstance(step['step'], int)
            assert isinstance(step['node'], list)
            assert len(step['node']) == 2
            assert isinstance(step['g'], float)
            assert isinstance(step['h'], float)
            assert isinstance(step['f'], float)
            assert isinstance(step['frontier_size'], int)

    def test_astar_step_trace_capping(self):
        """Test that step trace is capped at 300 entries."""
        pathfinder = AStarPathfinder()
        request = AStarRequest(grid_size=25, obstacle_density=0.1)

        response = pathfinder.solve(request)

        assert response.success is True
        assert len(response.step_trace) <= 300

    def test_astar_grid_format(self):
        """Test that grid has correct format."""
        pathfinder = AStarPathfinder()
        request = AStarRequest(grid_size=10)

        response = pathfinder.solve(request)

        assert response.success is True
        assert len(response.grid) == 10
        assert all(len(row) == 10 for row in response.grid)
        assert all(cell in [0, 1] for row in response.grid for cell in row)
        # Start and goal should be free
        assert response.grid[0][0] == 0
        assert response.grid[9][9] == 0

    def test_astar_path_validity(self):
        """Test that returned path is valid (adjacent moves, no obstacles)."""
        pathfinder = AStarPathfinder()
        request = AStarRequest(grid_size=10, obstacle_density=0.1)

        response = pathfinder.solve(request)

        if response.path_found and len(response.path) > 1:
            assert response.success is True
            # Check all moves are adjacent (Manhattan distance = 1)
            for i in range(len(response.path) - 1):
                curr = response.path[i]
                next_pos = response.path[i + 1]
                manhattan = abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1])
                assert manhattan == 1, f"Non-adjacent move: {curr} -> {next_pos}"

                # Check no obstacles on path
                assert response.grid[next_pos[0]][next_pos[1]] == 0

    def test_astar_random_state_reproducibility(self):
        """Test that same random_state produces same grid."""
        request = AStarRequest(grid_size=10, random_state=123)

        pathfinder1 = AStarPathfinder()
        response1 = pathfinder1.solve(request)

        pathfinder2 = AStarPathfinder()
        response2 = pathfinder2.solve(request)

        assert response1.grid == response2.grid
        assert response1.path_found == response2.path_found

    def test_astar_with_high_obstacle_density(self):
        """Test A* behavior with high obstacle density (may not find path)."""
        pathfinder = AStarPathfinder()
        request = AStarRequest(grid_size=10, obstacle_density=0.6, random_state=999)

        response = pathfinder.solve(request)

        assert response.success is True
        # May or may not find path with high obstacles
        assert isinstance(response.path_found, bool)

    def test_astar_small_grid(self):
        """Test A* on smallest grid size."""
        pathfinder = AStarPathfinder()
        request = AStarRequest(grid_size=5, obstacle_density=0.0)

        response = pathfinder.solve(request)

        assert response.success is True
        assert response.path_found is True
        assert len(response.path) > 0

    def test_astar_large_grid(self):
        """Test A* on largest grid size."""
        pathfinder = AStarPathfinder()
        request = AStarRequest(grid_size=30, obstacle_density=0.1)

        response = pathfinder.solve(request)

        assert response.success is True
        assert response.start == [0, 0]
        assert response.goal == [29, 29]
