"""
A* pathfinding algorithm implementation.

This module implements the A* search algorithm for grid-based pathfinding,
including support for multiple heuristic functions and step-by-step tracing.
"""

import time
import heapq
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .schema import AStarRequest, AStarResponse


class AStarPathfinder:
    """A* pathfinding algorithm implementation.

    This class implements the A* search algorithm for grid-based pathfinding,
    using a priority queue and supporting multiple heuristic functions
    (Manhattan, Euclidean, Chebyshev).
    """

    def __init__(self):
        """Initialize the A* pathfinder."""
        self.grid = None
        self.grid_size = 0
        self.start = (0, 0)
        self.goal = None
        self.open_set = []
        self.closed_set = set()
        self.came_from = {}
        self.g_score = {}
        self.f_score = {}
        self.nodes_explored = 0
        self.step_trace = []

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int], heuristic_type: str) -> float:
        """Calculate heuristic distance from pos to goal.

        Args:
            pos: Current position (row, col)
            goal: Goal position (row, col)
            heuristic_type: Type of heuristic ('manhattan', 'euclidean', 'chebyshev')

        Returns:
            Heuristic distance estimate
        """
        row, col = pos
        goal_row, goal_col = goal
        dr = abs(row - goal_row)
        dc = abs(col - goal_col)

        if heuristic_type == 'manhattan':
            return float(dr + dc)
        elif heuristic_type == 'euclidean':
            return math.sqrt(dr * dr + dc * dc)
        elif heuristic_type == 'chebyshev':
            return float(max(dr, dc))
        else:
            return float(dr + dc)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (4-directional movement).

        Args:
            pos: Current position (row, col)

        Returns:
            List of valid neighbor positions
        """
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.grid_size and
                0 <= new_col < self.grid_size and
                self.grid[new_row][new_col] == 0):
                neighbors.append((new_row, new_col))
        return neighbors

    def _generate_grid(self, grid_size: int, obstacle_density: float, random_state: int) -> List[List[int]]:
        """Generate a grid with random obstacles.

        Start and goal are always (0,0) and (grid_size-1, grid_size-1).
        Regenerates if start/goal land on obstacles or no path exists.

        Args:
            grid_size: Size of the square grid
            obstacle_density: Fraction of cells that should be obstacles
            random_state: Random seed

        Returns:
            Grid with obstacles (0=free, 1=obstacle)
        """
        rng = np.random.RandomState(random_state)
        max_retries = 10
        attempt = 0

        while attempt < max_retries:
            grid = rng.choice([0, 1], size=(grid_size, grid_size), p=[1 - obstacle_density, obstacle_density])

            # Ensure start and goal are free
            grid[0, 0] = 0
            grid[grid_size - 1, grid_size - 1] = 0

            # Quick check: is there a path? (via simple BFS)
            if self._can_reach_goal(grid, grid_size):
                return grid.tolist()

            attempt += 1

        # If no valid path found after retries, return grid anyway with success=True, path_found=False
        grid[0, 0] = 0
        grid[grid_size - 1, grid_size - 1] = 0
        return grid.tolist()

    def _can_reach_goal(self, grid: np.ndarray, grid_size: int) -> bool:
        """Quick BFS check to see if goal is reachable from start.

        Args:
            grid: The grid (0=free, 1=obstacle)
            grid_size: Size of the grid

        Returns:
            True if goal is reachable, False otherwise
        """
        from collections import deque

        queue = deque([(0, 0)])
        visited = {(0, 0)}

        while queue:
            row, col = queue.popleft()
            if row == grid_size - 1 and col == grid_size - 1:
                return True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < grid_size and
                    0 <= new_col < grid_size and
                    grid[new_row, new_col] == 0 and
                    (new_row, new_col) not in visited):
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))

        return False

    def solve(self, request: AStarRequest) -> Dict[str, Any]:
        """Solve A* pathfinding with the given request parameters.

        Args:
            request: AStarRequest with grid parameters and heuristic type

        Returns:
            Dictionary with path, metrics, and step trace

        Raises:
            Exception: If pathfinding fails
        """
        start_time = time.time()

        try:
            # Generate grid
            self.grid_size = request.grid_size
            self.grid = self._generate_grid(
                request.grid_size,
                request.obstacle_density,
                request.random_state
            )
            self.start = (0, 0)
            self.goal = (request.grid_size - 1, request.grid_size - 1)

            # Initialize A* data structures
            self.open_set = [(0, self.start)]  # (f_score, position)
            self.closed_set = set()
            self.came_from = {}
            self.g_score = {self.start: 0}
            self.f_score = {self.start: self._heuristic(self.start, self.goal, request.heuristic)}
            self.nodes_explored = 0
            self.step_trace = []

            # A* main loop
            while self.open_set:
                # Get node with lowest f_score
                _, current = heapq.heappop(self.open_set)

                if current in self.closed_set:
                    continue

                self.closed_set.add(current)
                self.nodes_explored += 1

                # Record step in trace
                g = self.g_score[current]
                h = self._heuristic(current, self.goal, request.heuristic)
                f = self.f_score[current]
                frontier_size = len(self.open_set)

                self.step_trace.append({
                    'step': self.nodes_explored,
                    'node': list(current),
                    'g': float(g),
                    'h': float(h),
                    'f': float(f),
                    'frontier_size': frontier_size
                })

                # Check if goal reached
                if current == self.goal:
                    # Reconstruct path
                    path = self._reconstruct_path(self.came_from, current)
                    execution_time_ms = (time.time() - start_time) * 1000

                    # Cap step trace at 300 entries
                    trace = self._cap_step_trace(self.step_trace, 300)

                    return AStarResponse(
                        success=True,
                        path_found=True,
                        path=path,
                        path_length=len(path),
                        nodes_explored=self.nodes_explored,
                        execution_time_ms=execution_time_ms,
                        grid=self.grid,
                        start=list(self.start),
                        goal=list(self.goal),
                        step_trace=trace
                    )

                # Explore neighbors
                for neighbor in self._get_neighbors(current):
                    if neighbor in self.closed_set:
                        continue

                    tentative_g = self.g_score[current] + 1

                    if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                        self.came_from[neighbor] = current
                        self.g_score[neighbor] = tentative_g
                        h = self._heuristic(neighbor, self.goal, request.heuristic)
                        self.f_score[neighbor] = tentative_g + h
                        heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))

            # No path found
            execution_time_ms = (time.time() - start_time) * 1000
            trace = self._cap_step_trace(self.step_trace, 300)

            return AStarResponse(
                success=True,
                path_found=False,
                path=[],
                path_length=0,
                nodes_explored=self.nodes_explored,
                execution_time_ms=execution_time_ms,
                grid=self.grid,
                start=list(self.start),
                goal=list(self.goal),
                step_trace=trace
            )

        except Exception as e:
            raise Exception(f"A* pathfinding failed: {str(e)}")

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[List[int]]:
        """Reconstruct the path from start to goal.

        Args:
            came_from: Dictionary mapping positions to their predecessors
            current: Current position (goal)

        Returns:
            List of [row, col] positions from start to goal
        """
        path = [list(current)]
        while current in came_from:
            current = came_from[current]
            path.append(list(current))
        path.reverse()
        return path

    def _cap_step_trace(self, trace: List[Dict[str, Any]], max_entries: int) -> List[Dict[str, Any]]:
        """Cap step trace at max_entries, keeping evenly spaced samples plus last few steps.

        Args:
            trace: Full step trace
            max_entries: Maximum number of entries to keep

        Returns:
            Capped step trace
        """
        if len(trace) <= max_entries:
            return trace

        # Keep first, last few, and evenly spaced samples
        kept_indices = set()

        # Always keep first and last
        kept_indices.add(0)
        kept_indices.add(len(trace) - 1)

        # Keep last 10 steps
        for i in range(max(1, len(trace) - 10), len(trace)):
            kept_indices.add(i)

        # Keep evenly spaced samples across the rest
        remaining_budget = max_entries - len(kept_indices)
        if remaining_budget > 0:
            step = max(1, (len(trace) - 11) // remaining_budget)
            for i in range(0, len(trace) - 11, step):
                if i not in kept_indices:
                    kept_indices.add(i)
                    if len(kept_indices) >= max_entries:
                        break

        kept_indices = sorted(list(kept_indices))[:max_entries]
        return [trace[i] for i in kept_indices]
