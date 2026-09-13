"""
Minimax algorithm with alpha-beta pruning for Tic-Tac-Toe.

This module implements the Minimax algorithm with optional alpha-beta pruning
for playing optimal or near-optimal Tic-Tac-Toe games.
"""

import time
from typing import Dict, Any, List, Literal, Tuple, Optional
import numpy as np

from .schema import MinimaxRequest, MinimaxResponse


class MinimaxPlayer:
    """Minimax algorithm player for Tic-Tac-Toe.

    Implements the Minimax algorithm with optional alpha-beta pruning for
    playing Tic-Tac-Toe. AI is always 'X', and opponent is 'O'.
    """

    def __init__(self, use_alpha_beta: bool = True, random_state: int = 42):
        """Initialize the Minimax player.

        Args:
            use_alpha_beta: Whether to use alpha-beta pruning
            random_state: Random seed for reproducibility
        """
        self.use_alpha_beta = use_alpha_beta
        self.rng = np.random.RandomState(random_state)
        self.board = ['' for _ in range(9)]  # Flat 3x3 board (0-8)
        self.nodes_evaluated = 0
        self.total_nodes_evaluated = 0
        self.moves_record = []
        self.step_trace = []

    def _board_to_grid(self, board: List[str]) -> List[List[str]]:
        """Convert flat board list to 3x3 grid.

        Args:
            board: Flat list of 9 elements ('', 'X', 'O')

        Returns:
            3x3 grid representation
        """
        grid = []
        for i in range(3):
            grid.append(board[i*3:(i+1)*3])
        return grid

    def _check_winner(self, board: List[str]) -> Optional[str]:
        """Check if there's a winner on the board.

        Args:
            board: Flat list of 9 elements

        Returns:
            'X' if X wins, 'O' if O wins, None if no winner
        """
        # All possible winning combinations
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]

        for combo in winning_combos:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] != '':
                return board[combo[0]]

        return None

    def _is_terminal(self, board: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if the board is in a terminal state.

        Args:
            board: Flat list of 9 elements

        Returns:
            (is_terminal, winner) - winner is None for draw
        """
        winner = self._check_winner(board)
        if winner:
            return True, winner

        # Check for draw (no empty spaces)
        if all(cell != '' for cell in board):
            return True, None

        return False, None

    def _get_available_moves(self, board: List[str]) -> List[int]:
        """Get list of available move positions.

        Args:
            board: Flat list of 9 elements

        Returns:
            List of indices where moves can be made
        """
        return [i for i in range(9) if board[i] == '']

    def _minimax(self, board: List[str], depth: int, is_maximizing: bool,
                 alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[int, int]:
        """Minimax algorithm with optional alpha-beta pruning.

        Args:
            board: Current board state
            depth: Current depth in the game tree
            is_maximizing: True if we're maximizing (AI's turn), False if minimizing (opponent's turn)
            alpha: Alpha value for alpha-beta pruning
            beta: Beta value for alpha-beta pruning

        Returns:
            (best_score, node_count) where node_count is nodes evaluated in this subtree
        """
        self.nodes_evaluated += 1
        nodes = 1

        # Check terminal state
        is_terminal, winner = self._is_terminal(board)
        if is_terminal:
            if winner == 'X':
                return (1, nodes)  # AI wins
            elif winner == 'O':
                return (-1, nodes)  # Opponent wins
            else:
                return (0, nodes)  # Draw

        available_moves = self._get_available_moves(board)

        if is_maximizing:
            max_eval = float('-inf')
            for move in available_moves:
                board[move] = 'X'
                eval_score, subtree_nodes = self._minimax(board, depth + 1, False, alpha, beta)
                nodes += subtree_nodes
                board[move] = ''

                max_eval = max(max_eval, eval_score)

                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff

            return (max_eval, nodes)
        else:
            min_eval = float('inf')
            for move in available_moves:
                board[move] = 'O'
                eval_score, subtree_nodes = self._minimax(board, depth + 1, True, alpha, beta)
                nodes += subtree_nodes
                board[move] = ''

                min_eval = min(min_eval, eval_score)

                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff

            return (min_eval, nodes)

    def _ai_move(self, board: List[str]) -> int:
        """Determine the best move for the AI using Minimax.

        Args:
            board: Current board state

        Returns:
            Index of the best move
        """
        available_moves = self._get_available_moves(board)
        best_move = available_moves[0]
        best_score = float('-inf')
        best_nodes = 0

        self.nodes_evaluated = 0

        for move in available_moves:
            board[move] = 'X'
            score, nodes = self._minimax(board, 0, False)
            board[move] = ''

            if score > best_score:
                best_score = score
                best_move = move
                best_nodes = nodes

        self.total_nodes_evaluated += self.nodes_evaluated

        # Record step
        move_pos = [best_move // 3, best_move % 3]
        self.step_trace.append({
            'move_number': len(self.moves_record) + 1,
            'position': move_pos,
            'score': int(best_score),
            'nodes_evaluated': self.nodes_evaluated
        })

        return best_move

    def _opponent_move(self, board: List[str], opponent_type: Literal['random', 'optimal']) -> int:
        """Determine opponent's move.

        Args:
            board: Current board state
            opponent_type: Type of opponent ('random' or 'optimal')

        Returns:
            Index of the opponent's move
        """
        available_moves = self._get_available_moves(board)

        if opponent_type == 'random':
            return int(self.rng.choice(available_moves))
        else:  # optimal
            # Use minimax to find best move for opponent
            best_move = available_moves[0]
            best_score = float('inf')

            self.nodes_evaluated = 0

            for move in available_moves:
                board[move] = 'O'
                score, _ = self._minimax(board, 0, True)
                board[move] = ''

                if score < best_score:
                    best_score = score
                    best_move = move

            return best_move

    def solve(self, request: MinimaxRequest) -> Dict[str, Any]:
        """Play a complete Tic-Tac-Toe game using Minimax.

        Args:
            request: MinimaxRequest with game parameters

        Returns:
            Dictionary with game results, moves, and evaluation info

        Raises:
            Exception: If game fails
        """
        start_time = time.time()

        try:
            self.board = ['' for _ in range(9)]
            self.moves_record = []
            self.step_trace = []
            self.total_nodes_evaluated = 0

            current_player = 'X' if request.ai_starts else 'O'
            move_number = 0

            # Play the game
            while True:
                is_terminal, winner = self._is_terminal(self.board)
                if is_terminal:
                    break

                move_number += 1

                if current_player == 'X':  # AI's turn
                    move = self._ai_move(self.board)
                    self.board[move] = 'X'
                    grid_after = self._board_to_grid(self.board)
                    self.moves_record.append({
                        'move_number': move_number,
                        'player': 'ai',
                        'position': [move // 3, move % 3],
                        'board_after': grid_after
                    })
                    current_player = 'O'
                else:  # Opponent's turn
                    move = self._opponent_move(self.board, request.opponent)
                    self.board[move] = 'O'
                    grid_after = self._board_to_grid(self.board)
                    self.moves_record.append({
                        'move_number': move_number,
                        'player': 'opponent',
                        'position': [move // 3, move % 3],
                        'board_after': grid_after
                    })
                    current_player = 'X'

            # Determine winner
            _, winner = self._is_terminal(self.board)
            if winner == 'X':
                result_winner = 'ai'
            elif winner == 'O':
                result_winner = 'opponent'
            else:
                result_winner = 'draw'

            execution_time_ms = (time.time() - start_time) * 1000

            return MinimaxResponse(
                success=True,
                winner=result_winner,
                moves=self.moves_record,
                total_nodes_evaluated=self.total_nodes_evaluated,
                execution_time_ms=execution_time_ms,
                step_trace=self.step_trace
            )

        except Exception as e:
            raise Exception(f"Minimax game failed: {str(e)}")
