"""
N-Queens problem solver using backtracking search.

This module implements a constraint satisfaction problem (CSP) solver
for the N-Queens problem using backtracking.
"""

import time
from typing import List, Dict, Any
from .schema import NQueensRequest, NQueensResponse


class NQueensSolver:
    """Solver for the N-Queens constraint satisfaction problem.

    Uses backtracking search to find a valid placement of N queens on an
    N×N chessboard such that no two queens threaten each other.

    Attributes:
        board_size: Size of the chessboard
        board: Current board state (row index for each column, -1 if unfilled)
        backtrack_count: Number of times backtracking occurred
        step_trace: List of placement attempts and backtracks
    """

    def __init__(self, board_size: int = 8):
        """Initialize the N-Queens solver.

        Args:
            board_size: Size of the chessboard (4 to 12)
        """
        self.board_size = board_size
        self.board = [-1] * board_size
        self.backtrack_count = 0
        self.step_trace = []
        self.step_count = 0

    def is_safe(self, col: int, row: int) -> bool:
        """Check if placing a queen at (row, col) is safe.

        Args:
            col: Column index
            row: Row index

        Returns:
            True if the placement is safe, False otherwise
        """
        # Check if any queen is on the same row
        for c in range(col):
            if self.board[c] == row:
                return False

        # Check upper-left diagonal
        for c in range(col):
            if abs(self.board[c] - row) == abs(c - col):
                return False

        return True

    def solve(self, request: NQueensRequest) -> Dict[str, Any]:
        """Solve the N-Queens problem using backtracking search.

        Args:
            request: NQueensRequest containing board_size and random_state

        Returns:
            Dictionary with solution, backtrack_count, and step_trace
        """
        self.board_size = request.board_size
        self.board = [-1] * self.board_size
        self.backtrack_count = 0
        self.step_trace = []
        self.step_count = 0

        def backtrack(col: int) -> bool:
            """Recursively place queens using backtracking.

            Args:
                col: Current column to fill

            Returns:
                True if a solution is found, False otherwise
            """
            if col == self.board_size:
                return True

            for row in range(self.board_size):
                self.step_count += 1
                if self.is_safe(col, row):
                    # Place queen
                    self.board[col] = row
                    self._record_step(col, row, True)

                    if backtrack(col + 1):
                        return True

                    # Backtrack
                    self.board[col] = -1
                    self.backtrack_count += 1
                else:
                    # Record failed attempt
                    self._record_step(col, row, False)

            return False

        # Solve the problem
        solved = backtrack(0)

        # Cap step_trace at 300 entries, evenly sampled
        if len(self.step_trace) > 300:
            step_indices = self._sample_steps(len(self.step_trace), 300)
            self.step_trace = [self.step_trace[i] for i in step_indices]

        return {
            'solved': solved,
            'solution': self.board if solved else [],
            'backtrack_count': self.backtrack_count,
            'step_trace': self.step_trace
        }

    def _record_step(self, col: int, row: int, accepted: bool) -> None:
        """Record a placement attempt in the trace.

        Args:
            col: Column index
            row: Row index
            accepted: Whether the placement was accepted
        """
        board_state = self.board.copy()
        self.step_trace.append({
            'step': self.step_count,
            'column': col,
            'row_tried': row,
            'accepted': accepted,
            'board_state': board_state
        })

    def _sample_steps(self, total: int, target: int) -> List[int]:
        """Sample evenly across total steps, including the last few.

        Args:
            total: Total number of steps
            target: Target number of samples

        Returns:
            List of indices to sample
        """
        if total <= target:
            return list(range(total))

        # Keep last 20 steps always, sample the rest evenly
        keep_last = min(20, target // 4)
        sample_count = target - keep_last

        indices = []
        for i in range(sample_count):
            idx = int(i * (total - keep_last) / sample_count)
            indices.append(idx)

        # Add last keep_last steps
        for i in range(keep_last):
            indices.append(total - keep_last + i)

        return sorted(set(indices))
