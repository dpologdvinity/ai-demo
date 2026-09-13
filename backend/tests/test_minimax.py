"""Tests for Minimax with alpha-beta pruning algorithm implementation."""

import pytest
from algorithms.classical_ai.minimax import MinimaxPlayer, MinimaxRequest


class TestMinimaxPlayer:
    """Test suite for Minimax algorithm."""

    def test_minimax_default_parameters(self):
        """Test Minimax with default parameters."""
        player = MinimaxPlayer()
        request = MinimaxRequest()

        response = player.solve(request)

        assert response.success is True
        assert response.winner in ['ai', 'opponent', 'draw']
        assert len(response.moves) > 0
        assert response.total_nodes_evaluated > 0
        assert response.execution_time_ms > 0

    def test_minimax_ai_starts(self):
        """Test Minimax when AI starts."""
        player = MinimaxPlayer()
        request = MinimaxRequest(ai_starts=True, opponent='random')

        response = player.solve(request)

        assert response.success is True
        # First move should be by AI
        assert response.moves[0]['player'] == 'ai'

    def test_minimax_opponent_starts(self):
        """Test Minimax when opponent starts."""
        player = MinimaxPlayer()
        request = MinimaxRequest(ai_starts=False, opponent='random')

        response = player.solve(request)

        assert response.success is True
        # First move should be by opponent
        assert response.moves[0]['player'] == 'opponent'

    def test_minimax_with_optimal_opponent(self):
        """Test Minimax against optimal opponent."""
        player = MinimaxPlayer()
        request = MinimaxRequest(opponent='optimal', ai_starts=True)

        response = player.solve(request)

        assert response.success is True
        # Optimal vs Optimal should draw
        assert response.winner == 'draw'

    def test_minimax_with_random_opponent(self):
        """Test Minimax against random opponent."""
        player = MinimaxPlayer()
        request = MinimaxRequest(opponent='random', ai_starts=True)

        response = player.solve(request)

        assert response.success is True
        # AI should win or draw against random opponent
        assert response.winner in ['ai', 'draw']

    def test_minimax_alpha_beta_vs_no_pruning(self):
        """Test that alpha-beta pruning and no pruning produce same winner."""
        request = MinimaxRequest(opponent='random', ai_starts=True, random_state=42)

        player_with_pruning = MinimaxPlayer(use_alpha_beta=True, random_state=42)
        response_with_pruning = player_with_pruning.solve(request)

        player_without_pruning = MinimaxPlayer(use_alpha_beta=False, random_state=42)
        response_without_pruning = player_without_pruning.solve(request)

        assert response_with_pruning.success is True
        assert response_without_pruning.success is True
        # Same opponent and random seed should produce same winner
        assert response_with_pruning.winner == response_without_pruning.winner

    def test_minimax_moves_format(self):
        """Test that moves have correct format."""
        player = MinimaxPlayer()
        request = MinimaxRequest()

        response = player.solve(request)

        assert response.success is True
        assert isinstance(response.moves, list)

        for move in response.moves:
            assert 'move_number' in move
            assert 'player' in move
            assert 'position' in move
            assert 'board_after' in move

            assert isinstance(move['move_number'], int)
            assert move['player'] in ['ai', 'opponent']
            assert isinstance(move['position'], list)
            assert len(move['position']) == 2
            assert isinstance(move['board_after'], list)
            assert len(move['board_after']) == 3
            assert all(len(row) == 3 for row in move['board_after'])

    def test_minimax_board_state_progression(self):
        """Test that board state progresses correctly through moves."""
        player = MinimaxPlayer()
        request = MinimaxRequest()

        response = player.solve(request)

        assert response.success is True

        for move in response.moves:
            board_after = move['board_after']
            # All cells should be '', 'X', or 'O'
            for row in board_after:
                for cell in row:
                    assert cell in ['', 'X', 'O']

    def test_minimax_step_trace_format(self):
        """Test that step trace has correct format."""
        player = MinimaxPlayer()
        request = MinimaxRequest(ai_starts=True)

        response = player.solve(request)

        assert response.success is True
        assert isinstance(response.step_trace, list)

        for step in response.step_trace:
            assert 'move_number' in step
            assert 'position' in step
            assert 'score' in step
            assert 'nodes_evaluated' in step

            assert isinstance(step['move_number'], int)
            assert isinstance(step['position'], list)
            assert len(step['position']) == 2
            assert isinstance(step['score'], int)
            assert step['score'] in [-1, 0, 1]
            assert isinstance(step['nodes_evaluated'], int)
            assert step['nodes_evaluated'] > 0

    def test_minimax_game_completion(self):
        """Test that games always complete."""
        for _ in range(3):
            player = MinimaxPlayer()
            request = MinimaxRequest(opponent='random')

            response = player.solve(request)

            assert response.success is True
            # Game should end (max 9 moves)
            assert len(response.moves) > 0
            assert len(response.moves) <= 9

    def test_minimax_random_state_reproducibility(self):
        """Test that same random_state produces same game outcome."""
        request = MinimaxRequest(opponent='random', random_state=123)

        player1 = MinimaxPlayer(random_state=123)
        response1 = player1.solve(request)

        player2 = MinimaxPlayer(random_state=123)
        response2 = player2.solve(request)

        assert response1.success is True
        assert response2.success is True
        # With same seed, opponent moves should be identical
        assert response1.winner == response2.winner

    def test_minimax_nodes_evaluated_count(self):
        """Test that nodes evaluated count is tracked correctly."""
        player = MinimaxPlayer()
        request = MinimaxRequest(ai_starts=True)

        response = player.solve(request)

        assert response.success is True
        assert response.total_nodes_evaluated > 0

        # Sum of nodes in step_trace should match total
        trace_sum = sum(step['nodes_evaluated'] for step in response.step_trace)
        assert trace_sum == response.total_nodes_evaluated

    def test_minimax_ai_plays_optimally_against_random(self):
        """Test that AI with optimal opponent setting doesn't lose."""
        player = MinimaxPlayer()
        request = MinimaxRequest(opponent='random', ai_starts=True, random_state=42)

        response = player.solve(request)

        assert response.success is True
        # AI should not lose to random
        assert response.winner in ['ai', 'draw']

    def test_minimax_positions_valid(self):
        """Test that all moves are on valid board positions."""
        player = MinimaxPlayer()
        request = MinimaxRequest()

        response = player.solve(request)

        assert response.success is True

        for move in response.moves:
            row, col = move['position']
            assert 0 <= row < 3
            assert 0 <= col < 3

    def test_minimax_no_duplicate_positions(self):
        """Test that the same position is not played twice in a game."""
        player = MinimaxPlayer()
        request = MinimaxRequest()

        response = player.solve(request)

        assert response.success is True

        played_positions = []
        for move in response.moves:
            pos = tuple(move['position'])
            assert pos not in played_positions, f"Duplicate position {pos}"
            played_positions.append(pos)
