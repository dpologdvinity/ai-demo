"""Transformer model implementation using PyTorch.

This module provides a Transformer class for sequence-to-sequence tasks using
PyTorch's nn.Transformer module.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math


class PositionalEncoding(nn.Module):
    """Positional encoding module for Transformer.

    Adds positional information to the input embeddings using sine and cosine
    functions of different frequencies.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """Transformer model for sequence-to-sequence tasks.

    This model uses PyTorch's nn.Transformer for sequence-to-sequence learning.
    It includes embeddings, positional encoding, and output projection.

    Attributes:
        embedding: Token embedding layer
        pos_encoder: Positional encoding layer
        transformer: PyTorch Transformer
        fc_out: Output projection layer
        d_model: Model dimension
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        """Initialize the Transformer model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(TransformerSeq2Seq, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding (shared for source and target)
        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for padding token

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size + 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder.

        Args:
            sz: Size of the mask

        Returns:
            Causal mask tensor
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the Transformer.

        Args:
            src: Source sequences (batch_size, src_seq_len)
            tgt: Target sequences (batch_size, tgt_seq_len)
            src_mask: Source mask (optional)
            tgt_mask: Target mask (optional)

        Returns:
            Output logits (batch_size, tgt_seq_len, vocab_size)
        """
        # Embed and add positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Generate target mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Transformer forward pass
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

        # Project to vocabulary
        output = self.fc_out(output)

        return output

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerModel:
    """Wrapper class for training and evaluating Transformer models.

    This class provides a high-level interface for training Transformer models
    on sequence-to-sequence tasks.

    Attributes:
        model: PyTorch Transformer model
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU or GPU)
        training_history: List of training losses per epoch
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        learning_rate: float = 0.001,
        dropout: float = 0.1,
        random_state: int = 42
    ):
        """Initialize the Transformer model wrapper.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder/decoder layers
            dim_feedforward: Feedforward dimension
            learning_rate: Learning rate
            dropout: Dropout rate
            random_state: Random seed
        """
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size

        # Initialize model
        self.model = TransformerSeq2Seq(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)

        # Loss function (ignore padding token 0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training history
        self.training_history = []

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """Train the Transformer model.

        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            test_loader: DataLoader for test data (optional)

        Returns:
            Dictionary containing training results
        """
        self.model.train()
        epoch_losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for src, tgt in train_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                # Create target input (shift right) and target output
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(src, tgt_input)

                # Compute loss
                output_flat = output.reshape(-1, self.vocab_size + 1)
                tgt_flat = tgt_output.reshape(-1)
                loss = self.criterion(output_flat, tgt_flat)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)
            self.training_history.append(avg_loss)

        # Evaluate on test set if provided
        test_metrics = {}
        if test_loader is not None:
            test_metrics = self.evaluate(test_loader)

        return {
            'training_history': epoch_losses,
            'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
            'test_metrics': test_metrics
        }

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dictionary containing test metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for src, tgt in test_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                # Create target input and output
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # Forward pass
                output = self.model(src, tgt_input)

                # Compute loss
                output_flat = output.reshape(-1, self.vocab_size + 1)
                tgt_flat = tgt_output.reshape(-1)
                loss = self.criterion(output_flat, tgt_flat)
                total_loss += loss.item()

                # Compute accuracy
                predictions = output.argmax(dim=-1)
                correct = (predictions == tgt_output).sum().item()
                total_correct += correct
                total_tokens += tgt_output.numel()

        avg_loss = total_loss / len(test_loader)
        accuracy = total_correct / total_tokens

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        }

    def predict(
        self,
        src_sequences: torch.Tensor,
        max_length: int = 20,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate predictions for source sequences.

        Args:
            src_sequences: Source sequences (batch_size, seq_len)
            max_length: Maximum length for generated sequences
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (predictions, attention_weights)
        """
        self.model.eval()
        src_sequences = src_sequences.to(self.device)
        batch_size = src_sequences.size(0)

        # Start with a start token (using 1 as start token)
        tgt_sequences = torch.ones((batch_size, 1), dtype=torch.long).to(self.device)

        with torch.no_grad():
            for _ in range(max_length - 1):
                output = self.model(src_sequences, tgt_sequences)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt_sequences = torch.cat([tgt_sequences, next_token], dim=1)

        attention_weights = None
        if return_attention:
            # Extract attention weights from the last prediction
            # This is a simplified version - full implementation would extract from transformer layers
            attention_weights = self._extract_attention_weights(src_sequences, tgt_sequences)

        return tgt_sequences, attention_weights

    def _extract_attention_weights(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """Extract attention weights from the model.

        Args:
            src: Source sequences
            tgt: Target sequences

        Returns:
            Attention weights tensor
        """
        # Simplified attention weight extraction
        # In practice, you'd hook into the transformer layers
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)

        # Create dummy attention weights (uniform distribution)
        # Real implementation would extract from model layers
        attention = torch.ones(batch_size, tgt_len, src_len) / src_len

        return attention

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture.

        Returns:
            Dictionary containing model information
        """
        return {
            'total_parameters': self.model.count_parameters(),
            'd_model': self.model.d_model,
            'vocab_size': self.vocab_size,
            'architecture': 'Transformer',
            'device': str(self.device)
        }
