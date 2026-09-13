"""Tests for N-Queens CSP solver implementation."""

import pytest
from algorithms.classical_ai.nqueens import NQueensSolver, NQueensRequest, NQueensResponse


class TestNQueensSolver:
    """Test suite for N-Queens solver."""

    def test_solver_initialization(self):
        """Test that solver initializes correctly."""
        solver = NQueensSolver(board_size=8)
        assert solver.board_size == 8
        assert len(solver.board) == 8
        assert all(x == -1 for x in solver.board)

    def test_is_safe_single_queen(self):
        """Test safety check with single queen placement."""
        solver = NQueensSolver(board_size=4)
        solver.board = [1, -1, -1, -1]

        # Should be safe at (row=3, col=1) - no row/diagonal conflict with queen at (row=1, col=0)
        assert solver.is_safe(1, 3)

        # Should not be safe at (row=1, col=1) - same row as queen at col 0
        assert not solver.is_safe(1, 1)

        # Should not be safe at (row=0, col=1) - diagonal conflict with queen at (row=1, col=0)
        assert not solver.is_safe(1, 0)

    def test_solve_board_size_4(self):
        """Test solving N-Queens for board_size=4."""
        solver = NQueensSolver()
        request = NQueensRequest(board_size=4, random_state=42)
        result = solver.solve(request)

        assert result['solved'] is True
        assert len(result['solution']) == 4
        assert result['backtrack_count'] >= 0

        # Verify solution is valid
        solution = result['solution']
        assert len(set(solution)) == 4  # All queens in different rows

        # Check no diagonal conflicts
        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(solution[i] - solution[j]) != abs(i - j)

    def test_solve_board_size_8(self):
        """Test solving N-Queens for board_size=8 (standard)."""
        solver = NQueensSolver()
        request = NQueensRequest(board_size=8, random_state=42)
        result = solver.solve(request)

        assert result['solved'] is True
        assert len(result['solution']) == 8

        # Verify solution is valid
        solution = result['solution']
        assert len(set(solution)) == 8  # All queens in different rows

    def test_step_trace_structure(self):
        """Test that step_trace has correct structure."""
        solver = NQueensSolver()
        request = NQueensRequest(board_size=4, random_state=42)
        result = solver.solve(request)

        assert 'step_trace' in result
        assert isinstance(result['step_trace'], list)

        if len(result['step_trace']) > 0:
            step = result['step_trace'][0]
            assert 'step' in step
            assert 'column' in step
            assert 'row_tried' in step
            assert 'accepted' in step
            assert 'board_state' in step

    def test_step_trace_capped_at_300(self):
        """Test that step_trace is capped at 300 entries."""
        solver = NQueensSolver()
        request = NQueensRequest(board_size=12, random_state=42)
        result = solver.solve(request)

        assert len(result['step_trace']) <= 300

    def test_different_board_sizes(self):
        """Test solver with different board sizes."""
        for board_size in [4, 5, 6, 8]:
            solver = NQueensSolver()
            request = NQueensRequest(board_size=board_size, random_state=42)
            result = solver.solve(request)

            assert result['solved'] is True
            assert len(result['solution']) == board_size

    def test_backtrack_count(self):
        """Test that backtrack_count is tracked."""
        solver = NQueensSolver()
        request = NQueensRequest(board_size=6, random_state=42)
        result = solver.solve(request)

        assert result['backtrack_count'] >= 0
        # For board_size=6, backtracking should occur
        assert result['backtrack_count'] > 0

    def test_solution_validity(self):
        """Test that returned solution is valid (no queens attacking)."""
        solver = NQueensSolver()
        request = NQueensRequest(board_size=8, random_state=42)
        result = solver.solve(request)

        assert result['solved'] is True
        solution = result['solution']

        # Check all rows are different
        assert len(set(solution)) == len(solution)

        # Check no diagonals conflict
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                row_i, row_j = solution[i], solution[j]
                col_i, col_j = i, j
                assert abs(row_i - row_j) != abs(col_i - col_j)
