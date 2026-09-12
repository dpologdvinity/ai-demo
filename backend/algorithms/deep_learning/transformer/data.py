"""Data generation and handling for Transformer algorithm.

This module provides functions for generating sequence-to-sequence datasets
for training and evaluating Transformer models.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence-to-sequence tasks.

    This dataset handles pairs of input and target sequences for training
    sequence-to-sequence models like Transformers.

    Attributes:
        src_sequences: Source (input) sequences
        tgt_sequences: Target (output) sequences
    """

    def __init__(self, src_sequences: torch.Tensor, tgt_sequences: torch.Tensor):
        """Initialize the dataset.

        Args:
            src_sequences: Source sequences tensor of shape (n_samples, seq_len)
            tgt_sequences: Target sequences tensor of shape (n_samples, seq_len)
        """
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.src_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (source_sequence, target_sequence)
        """
        return self.src_sequences[idx], self.tgt_sequences[idx]


def generate_reversal_data(
    n_samples: int = 1000,
    seq_length: int = 10,
    vocab_size: int = 20,
    train_split: float = 0.8,
    random_state: int = 42
) -> Dict[str, Any]:
    """Generate sequence reversal dataset.

    Creates a dataset where the task is to reverse the input sequence.
    Example: [1, 2, 3, 4, 5] -> [5, 4, 3, 2, 1]

    Args:
        n_samples: Number of sequence pairs to generate
        seq_length: Length of each sequence
        vocab_size: Size of vocabulary (max token value)
        train_split: Fraction of data to use for training
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - src_sequences: Source sequences (n_samples, seq_length)
            - tgt_sequences: Target sequences (reversed)
            - vocab_size: Vocabulary size
            - seq_length: Sequence length
            - train_indices: Indices for training split
            - test_indices: Indices for test split
            - n_train: Number of training samples
            - n_test: Number of test samples
    """
    np.random.seed(random_state)

    # Generate random sequences (excluding 0 which we'll use for padding if needed)
    src_sequences = np.random.randint(1, vocab_size + 1, size=(n_samples, seq_length))

    # Create target sequences by reversing the source
    tgt_sequences = np.flip(src_sequences, axis=1).copy()

    # Create train/test split
    n_train = int(n_samples * train_split)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    return {
        'src_sequences': src_sequences,
        'tgt_sequences': tgt_sequences,
        'vocab_size': vocab_size,
        'seq_length': seq_length,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'n_train': len(train_indices),
        'n_test': len(test_indices)
    }


def generate_addition_data(
    n_samples: int = 1000,
    max_num: int = 99,
    train_split: float = 0.8,
    random_state: int = 42
) -> Dict[str, Any]:
    """Generate integer addition dataset.

    Creates a dataset where the task is to add two numbers.
    Input format: [num1_tens, num1_ones, num2_tens, num2_ones]
    Output format: [sum_hundreds, sum_tens, sum_ones]

    Args:
        n_samples: Number of samples to generate
        max_num: Maximum value for each number (e.g., 99 for 2-digit)
        train_split: Fraction of data to use for training
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing source sequences, target sequences, and split info
    """
    np.random.seed(random_state)

    src_sequences = []
    tgt_sequences = []

    for _ in range(n_samples):
        # Generate two random numbers
        num1 = np.random.randint(0, max_num + 1)
        num2 = np.random.randint(0, max_num + 1)
        total = num1 + num2

        # Convert to digit sequences
        # Input: [num1_tens, num1_ones, num2_tens, num2_ones]
        src = [num1 // 10, num1 % 10, num2 // 10, num2 % 10]

        # Output: [sum_hundreds, sum_tens, sum_ones]
        tgt = [total // 100, (total % 100) // 10, total % 10]

        src_sequences.append(src)
        tgt_sequences.append(tgt)

    src_sequences = np.array(src_sequences)
    tgt_sequences = np.array(tgt_sequences)

    # Create train/test split
    n_train = int(n_samples * train_split)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    return {
        'src_sequences': src_sequences,
        'tgt_sequences': tgt_sequences,
        'vocab_size': 10,  # Digits 0-9
        'src_length': 4,
        'tgt_length': 3,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'n_train': len(train_indices),
        'n_test': len(test_indices)
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the Transformer dataset.

    Returns:
        Dictionary containing dataset metadata
    """
    return {
        "name": "Sequence Reversal",
        "description": "A simple sequence-to-sequence task where the model learns to reverse sequences",
        "task_type": "sequence_reversal",
        "input_format": "Integer sequences [1, 2, 3, 4, 5]",
        "output_format": "Reversed sequences [5, 4, 3, 2, 1]",
        "default_params": {
            "n_samples": 1000,
            "seq_length": 10,
            "vocab_size": 20,
            "train_split": 0.8
        },
        "alternatives": [
            {
                "name": "Integer Addition",
                "description": "Learn to add two numbers represented as digit sequences"
            },
            {
                "name": "Copy Task",
                "description": "Learn to copy input sequences to output"
            }
        ]
    }
