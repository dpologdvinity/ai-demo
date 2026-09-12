"""Seq2Seq model implementation with attention mechanism."""

import time
import random
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .schema import (
    Seq2SeqParameters,
    Seq2SeqResponse,
    TranslationPair,
    TrainingHistory
)
from .data import get_dataset_by_task

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class Vocabulary:
    """Vocabulary builder for sequence data."""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_count = {}
        self.n_words = 4  # Count PAD, SOS, EOS, UNK

    def add_sentence(self, sentence: str):
        """Add all words in a sentence to vocabulary."""
        for word in sentence.lower().split():
            self.add_word(word)

    def add_word(self, word: str):
        """Add a single word to vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1


class Seq2SeqDataset(Dataset):
    """Dataset for Seq2Seq training."""

    def __init__(self, pairs: List[Tuple[str, str]], input_vocab: Vocabulary, target_vocab: Vocabulary):
        self.pairs = pairs
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target_seq = self.pairs[idx]
        input_indices = self.sentence_to_indices(input_seq, self.input_vocab)
        target_indices = self.sentence_to_indices(target_seq, self.target_vocab)
        return input_indices, target_indices

    @staticmethod
    def sentence_to_indices(sentence: str, vocab: Vocabulary) -> List[int]:
        """Convert sentence to list of word indices."""
        indices = [vocab.word2idx.get(word.lower(), vocab.word2idx["<UNK>"]) for word in sentence.split()]
        indices.append(vocab.word2idx["<EOS>"])
        return indices


class Encoder(nn.Module):
    """LSTM-based encoder."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, hidden=None):
        """Forward pass through encoder.

        Args:
            input_seq: Input sequence tensor [batch_size, seq_len]
            hidden: Initial hidden state (optional)

        Returns:
            outputs: Encoder outputs [batch_size, seq_len, hidden_size]
            hidden: Final hidden state
        """
        embedded = self.dropout(self.embedding(input_seq))
        outputs, hidden = self.lstm(embedded, hidden)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism."""

    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """Calculate attention weights.

        Args:
            decoder_hidden: Decoder hidden state [batch_size, hidden_size]
            encoder_outputs: Encoder outputs [batch_size, seq_len, hidden_size]

        Returns:
            attention_weights: Attention weights [batch_size, seq_len]
            context: Context vector [batch_size, hidden_size]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Repeat decoder hidden state for each encoder output
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate and compute attention scores
        energy = torch.tanh(self.attn(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
        attention_scores = self.v(energy).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return attention_weights, context


class Decoder(nn.Module):
    """LSTM-based decoder with optional attention."""

    def __init__(self, output_size: int, hidden_size: int, num_layers: int, dropout: float, use_attention: bool):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size * 2 if use_attention else hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)

        if use_attention:
            self.attention = BahdanauAttention(hidden_size)

    def forward(self, input_token, hidden, encoder_outputs=None):
        """Forward pass through decoder.

        Args:
            input_token: Input token [batch_size, 1]
            hidden: Hidden state from previous time step
            encoder_outputs: Encoder outputs for attention (if enabled)

        Returns:
            output: Output logits [batch_size, output_size]
            hidden: Updated hidden state
            attention_weights: Attention weights (if attention enabled)
        """
        embedded = self.dropout(self.embedding(input_token))

        attention_weights = None
        if self.use_attention and encoder_outputs is not None:
            # Extract LSTM hidden state for attention (take last layer)
            if isinstance(hidden, tuple):
                decoder_hidden = hidden[0][-1]  # [batch_size, hidden_size]
            else:
                decoder_hidden = hidden[-1]

            attention_weights, context = self.attention(decoder_hidden, encoder_outputs)

            # Concatenate embedded input with context vector
            lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        else:
            lstm_input = embedded

        output, hidden = self.lstm(lstm_input, hidden)
        output = self.out(output.squeeze(1))

        return output, hidden, attention_weights


class Seq2SeqModel:
    """Seq2Seq model with encoder-decoder architecture."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def train(self, params: Seq2SeqParameters) -> Seq2SeqResponse:
        """Train Seq2Seq model.

        Args:
            params: Training parameters

        Returns:
            Seq2SeqResponse with training results
        """
        start_time = time.time()

        try:
            # Load dataset
            if params.use_custom_pairs and params.custom_input_sequences:
                pairs = list(zip(params.custom_input_sequences, params.custom_target_sequences))
            else:
                pairs = get_dataset_by_task(params.task, params.language_pair)

            logger.info(f"Loaded {len(pairs)} training pairs for task: {params.task}")

            # Build vocabularies
            input_vocab = Vocabulary()
            target_vocab = Vocabulary()

            for input_seq, target_seq in pairs:
                input_vocab.add_sentence(input_seq)
                target_vocab.add_sentence(target_seq)

            logger.info(f"Input vocabulary size: {input_vocab.n_words}")
            logger.info(f"Target vocabulary size: {target_vocab.n_words}")

            # Split data into train and validation
            random.shuffle(pairs)
            split_idx = int(0.8 * len(pairs))
            train_pairs = pairs[:split_idx]
            val_pairs = pairs[split_idx:]

            # Create models
            encoder = Encoder(
                input_vocab.n_words,
                params.hidden_size,
                params.num_layers,
                params.dropout
            ).to(self.device)

            decoder = Decoder(
                target_vocab.n_words,
                params.hidden_size,
                params.num_layers,
                params.dropout,
                params.attention
            ).to(self.device)

            # Training setup
            criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=params.learning_rate)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=params.learning_rate)

            # Training loop
            training_history = []

            for epoch in range(params.epochs):
                epoch_loss = 0
                encoder.train()
                decoder.train()

                for input_seq, target_seq in train_pairs:
                    loss = self._train_step(
                        input_seq, target_seq, encoder, decoder,
                        encoder_optimizer, decoder_optimizer, criterion,
                        input_vocab, target_vocab, params.teacher_forcing_ratio
                    )
                    epoch_loss += loss

                avg_loss = epoch_loss / len(train_pairs)

                # Validation BLEU score
                val_bleu = self._evaluate_bleu(
                    val_pairs[:min(10, len(val_pairs))],
                    encoder, decoder, input_vocab, target_vocab
                )

                training_history.append(
                    TrainingHistory(epoch=epoch + 1, loss=avg_loss, bleu=val_bleu)
                )

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{params.epochs} - Loss: {avg_loss:.4f}, BLEU: {val_bleu:.4f}")

            # Generate predictions on validation set
            sample_predictions = self._generate_predictions(
                val_pairs[:10], encoder, decoder, input_vocab, target_vocab, params.attention
            )

            # Get attention heatmap for first sample
            attention_heatmap = None
            if params.attention and len(sample_predictions) > 0:
                attention_heatmap = self._create_attention_heatmap(
                    sample_predictions[0], encoder, decoder, input_vocab, target_vocab
                )

            execution_time_ms = (time.time() - start_time) * 1000

            # Calculate metrics
            avg_bleu = np.mean([pred.bleu_score for pred in sample_predictions])

            return Seq2SeqResponse(
                success=True,
                task=params.task,
                sample_predictions=sample_predictions,
                training_history=training_history,
                attention_heatmap=attention_heatmap,
                metrics={
                    "final_loss": training_history[-1].loss if training_history else 0,
                    "final_bleu": training_history[-1].bleu if training_history else 0,
                    "avg_bleu": avg_bleu,
                    "total_pairs": len(pairs),
                    "train_pairs": len(train_pairs),
                    "val_pairs": len(val_pairs)
                },
                architecture_info={
                    "encoder_layers": params.num_layers,
                    "decoder_layers": params.num_layers,
                    "hidden_size": params.hidden_size,
                    "attention": params.attention,
                    "attention_type": "Bahdanau" if params.attention else None,
                    "dropout": params.dropout
                },
                vocabulary_info={
                    "input_vocab_size": input_vocab.n_words,
                    "target_vocab_size": target_vocab.n_words,
                    "special_tokens": ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
                },
                visualization_data={
                    "loss_curve": [{"epoch": h.epoch, "loss": h.loss} for h in training_history],
                    "bleu_curve": [{"epoch": h.epoch, "bleu": h.bleu} for h in training_history]
                },
                execution_time_ms=execution_time_ms,
                parameters_used=params.model_dump()
            )

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return Seq2SeqResponse(
                success=False,
                task=params.task,
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def _train_step(self, input_seq: str, target_seq: str, encoder: Encoder, decoder: Decoder,
                    encoder_optimizer: optim.Optimizer, decoder_optimizer: optim.Optimizer,
                    criterion: nn.Module, input_vocab: Vocabulary, target_vocab: Vocabulary,
                    teacher_forcing_ratio: float) -> float:
        """Single training step."""
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Convert sequences to tensors
        input_indices = Seq2SeqDataset.sentence_to_indices(input_seq, input_vocab)
        target_indices = Seq2SeqDataset.sentence_to_indices(target_seq, target_vocab)

        input_tensor = torch.LongTensor([input_indices]).to(self.device)
        target_tensor = torch.LongTensor(target_indices).to(self.device)

        # Encode
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # Decode
        decoder_hidden = encoder_hidden
        decoder_input = torch.LongTensor([[target_vocab.word2idx["<SOS>"]]]).to(self.device)

        loss = 0
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        for di in range(len(target_indices)):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))

            if use_teacher_forcing:
                decoder_input = target_tensor[di].unsqueeze(0).unsqueeze(0)
            else:
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / len(target_indices)

    def _evaluate_bleu(self, pairs: List[Tuple[str, str]], encoder: Encoder, decoder: Decoder,
                      input_vocab: Vocabulary, target_vocab: Vocabulary) -> float:
        """Calculate BLEU score on validation pairs."""
        encoder.eval()
        decoder.eval()

        total_bleu = 0
        with torch.no_grad():
            for input_seq, target_seq in pairs:
                prediction = self._translate(input_seq, encoder, decoder, input_vocab, target_vocab)
                bleu = self._calculate_bleu(prediction, target_seq)
                total_bleu += bleu

        return total_bleu / len(pairs) if pairs else 0

    def _translate(self, input_seq: str, encoder: Encoder, decoder: Decoder,
                   input_vocab: Vocabulary, target_vocab: Vocabulary, max_length: int = 50) -> str:
        """Translate input sequence to output sequence."""
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            # Convert input to tensor
            input_indices = Seq2SeqDataset.sentence_to_indices(input_seq, input_vocab)
            input_tensor = torch.LongTensor([input_indices]).to(self.device)

            # Encode
            encoder_outputs, encoder_hidden = encoder(input_tensor)

            # Decode
            decoder_hidden = encoder_hidden
            decoder_input = torch.LongTensor([[target_vocab.word2idx["<SOS>"]]]).to(self.device)

            decoded_words = []
            for _ in range(max_length):
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                topv, topi = decoder_output.topk(1)
                token_idx = topi.item()

                if token_idx == target_vocab.word2idx["<EOS>"]:
                    break

                word = target_vocab.idx2word.get(token_idx, "<UNK>")
                decoded_words.append(word)

                decoder_input = topi.detach()

        return ' '.join(decoded_words)

    def _calculate_bleu(self, prediction: str, target: str) -> float:
        """Calculate simple BLEU score (unigram precision)."""
        pred_words = set(prediction.lower().split())
        target_words = set(target.lower().split())

        if len(pred_words) == 0:
            return 0.0

        matches = len(pred_words & target_words)
        precision = matches / len(pred_words)

        # Add brevity penalty
        bp = min(1.0, len(pred_words) / max(len(target_words), 1))

        return precision * bp

    def _generate_predictions(self, pairs: List[Tuple[str, str]], encoder: Encoder, decoder: Decoder,
                             input_vocab: Vocabulary, target_vocab: Vocabulary,
                             use_attention: bool) -> List[TranslationPair]:
        """Generate predictions for sample pairs."""
        predictions = []

        for input_seq, target_seq in pairs:
            prediction = self._translate(input_seq, encoder, decoder, input_vocab, target_vocab)
            bleu_score = self._calculate_bleu(prediction, target_seq)

            predictions.append(TranslationPair(
                input=input_seq,
                target=target_seq,
                prediction=prediction,
                bleu_score=bleu_score,
                attention_weights=None  # Filled in by attention heatmap function if needed
            ))

        return predictions

    def _create_attention_heatmap(self, sample: TranslationPair, encoder: Encoder, decoder: Decoder,
                                 input_vocab: Vocabulary, target_vocab: Vocabulary) -> Dict[str, Any]:
        """Create attention heatmap for visualization."""
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            input_indices = Seq2SeqDataset.sentence_to_indices(sample.input, input_vocab)
            input_tensor = torch.LongTensor([input_indices]).to(self.device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)

            decoder_hidden = encoder_hidden
            decoder_input = torch.LongTensor([[target_vocab.word2idx["<SOS>"]]]).to(self.device)

            attention_matrix = []
            decoded_words = []

            for _ in range(len(sample.target.split()) + 1):
                decoder_output, decoder_hidden, attention_weights = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                if attention_weights is not None:
                    attention_matrix.append(attention_weights.cpu().numpy()[0].tolist())

                topv, topi = decoder_output.topk(1)
                token_idx = topi.item()

                if token_idx == target_vocab.word2idx["<EOS>"]:
                    decoded_words.append("<EOS>")
                    break

                word = target_vocab.idx2word.get(token_idx, "<UNK>")
                decoded_words.append(word)
                decoder_input = topi.detach()

        return {
            "input_words": sample.input.split(),
            "output_words": decoded_words,
            "attention_weights": attention_matrix
        }
