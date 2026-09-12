"""BERT Fine-tuning algorithm implementation."""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .schema import (
    BERTFinetuningParameters,
    BERTFinetuningResponse,
    TrainingHistory,
    PredictionResult,
    AttentionVisualization,
    ConfusionMatrix
)
from .data import (
    get_default_dataset,
    prepare_custom_dataset,
    split_dataset,
    get_sample_texts_for_inference
)


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        """Initialize dataset.

        Args:
            texts: List of text samples
            labels: List of labels
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTFinetuningModel:
    """BERT Fine-tuning algorithm implementation.

    This class implements BERT fine-tuning for text classification using
    Hugging Face transformers library. Supports both bert-base-uncased and
    distilbert-base-uncased models.

    Attributes:
        model: BERT model for sequence classification
        tokenizer: BERT tokenizer
        parameters: Training parameters
        device: Training device (cpu or cuda)
        class_names: Names of classification classes

    Example:
        >>> params = BERTFinetuningParameters(epochs=3, batch_size=16)
        >>> model = BERTFinetuningModel()
        >>> response = model.train(params)
        >>> print(f"Training completed in {response.execution_time_ms:.2f}ms")
    """

    def __init__(self):
        """Initialize the BERT Fine-tuning model."""
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.parameters: Optional[BERTFinetuningParameters] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ["Negative", "Neutral", "Positive"]
        self.num_classes = 3

    def train(self, parameters: BERTFinetuningParameters) -> BERTFinetuningResponse:
        """Fine-tune BERT model for text classification.

        Args:
            parameters: Training parameters

        Returns:
            BERTFinetuningResponse containing training results and predictions

        Example:
            >>> params = BERTFinetuningParameters(learning_rate=2e-5)
            >>> model = BERTFinetuningModel()
            >>> result = model.train(params)
            >>> if result.success:
            ...     print(f"Final accuracy: {result.metrics['final_accuracy']}")
        """
        try:
            start_time = time.time()
            self.parameters = parameters

            # Load dataset
            if parameters.use_custom_dataset and parameters.custom_texts and parameters.custom_labels:
                texts, labels = prepare_custom_dataset(
                    parameters.custom_texts,
                    parameters.custom_labels
                )
            else:
                texts, labels, self.class_names = get_default_dataset()

            # Split dataset
            train_texts, train_labels, val_texts, val_labels = split_dataset(texts, labels)

            # Initialize model and tokenizer
            self._initialize_model(parameters.model_name)

            # Create datasets and dataloaders
            train_dataset = TextClassificationDataset(
                train_texts, train_labels, self.tokenizer, parameters.max_length
            )
            val_dataset = TextClassificationDataset(
                val_texts, val_labels, self.tokenizer, parameters.max_length
            )

            train_loader = DataLoader(
                train_dataset, batch_size=parameters.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=parameters.batch_size
            )

            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=parameters.learning_rate)
            total_steps = len(train_loader) * parameters.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=total_steps // 10,
                num_training_steps=total_steps
            )

            # Training loop
            training_history = []
            for epoch in range(parameters.epochs):
                train_loss, train_acc = self._train_epoch(
                    train_loader, optimizer, scheduler
                )
                val_loss, val_acc = self._validate_epoch(val_loader)

                training_history.append(
                    TrainingHistory(
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        train_accuracy=train_acc,
                        val_loss=val_loss,
                        val_accuracy=val_acc
                    )
                )

            # Generate predictions on validation set
            predictions = self._generate_predictions(val_texts, val_labels)

            # Compute confusion matrix
            confusion_mat = self._compute_confusion_matrix(val_labels, predictions)

            # Generate attention visualizations for sample texts
            sample_texts = get_sample_texts_for_inference()[:3]
            attention_viz = self._generate_attention_visualizations(sample_texts)

            # Calculate metrics
            metrics = self._calculate_metrics(training_history, val_labels, predictions)

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(
                training_history, predictions, confusion_mat, attention_viz
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return BERTFinetuningResponse(
                success=True,
                training_history=training_history,
                predictions=predictions[:20],  # Return top 20 predictions
                confusion_matrix=confusion_mat,
                attention_visualizations=attention_viz,
                metrics=metrics,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                model_info=self._get_model_info()
            )

        except Exception as e:
            return BERTFinetuningResponse(
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def _initialize_model(self, model_name: str):
        """Initialize BERT model and tokenizer.

        Args:
            model_name: Name of the BERT model variant
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_classes,
            output_attentions=True,
            output_hidden_states=False
        )
        self.model.to(self.device)

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler
    ) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def _generate_predictions(
        self,
        texts: List[str],
        true_labels: List[int]
    ) -> List[PredictionResult]:
        """Generate predictions for given texts.

        Args:
            texts: List of text samples
            true_labels: True labels

        Returns:
            List of prediction results
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for text, true_label in zip(texts, true_labels):
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.parameters.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                predicted_label = int(np.argmax(probs))
                confidence = float(probs[predicted_label])

                predictions.append(
                    PredictionResult(
                        text=text,
                        predicted_label=predicted_label,
                        predicted_class=self.class_names[predicted_label],
                        confidence=confidence,
                        probabilities=probs.tolist(),
                        true_label=true_label
                    )
                )

        return predictions

    def _compute_confusion_matrix(
        self,
        true_labels: List[int],
        predictions: List[PredictionResult]
    ) -> ConfusionMatrix:
        """Compute confusion matrix.

        Args:
            true_labels: True labels
            predictions: Prediction results

        Returns:
            Confusion matrix data
        """
        pred_labels = [p.predicted_label for p in predictions]

        cm = confusion_matrix(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, zero_division=0
        )

        accuracy = sum(1 for tl, pl in zip(true_labels, pred_labels) if tl == pl) / len(true_labels)

        return ConfusionMatrix(
            matrix=cm.tolist(),
            labels=self.class_names,
            accuracy=accuracy,
            precision=precision.tolist(),
            recall=recall.tolist(),
            f1_score=f1.tolist()
        )

    def _generate_attention_visualizations(
        self,
        texts: List[str]
    ) -> List[AttentionVisualization]:
        """Generate attention visualizations for sample texts.

        Args:
            texts: List of sample texts

        Returns:
            List of attention visualizations
        """
        self.model.eval()
        visualizations = []

        with torch.no_grad():
            for idx, text in enumerate(texts):
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.parameters.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Get attention weights from last layer, average across heads
                attentions = outputs.attentions[-1]  # Last layer
                avg_attention = attentions.mean(dim=1).squeeze(0).cpu().numpy()

                # Get tokens
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())

                # Find actual sequence length (before padding)
                seq_length = attention_mask.sum().item()

                # Trim to actual sequence length
                tokens = tokens[:seq_length]
                avg_attention = avg_attention[:seq_length, :seq_length]

                visualizations.append(
                    AttentionVisualization(
                        text=text,
                        tokens=tokens,
                        attention_weights=avg_attention.tolist(),
                        sample_index=idx
                    )
                )

        return visualizations

    def _calculate_metrics(
        self,
        training_history: List[TrainingHistory],
        true_labels: List[int],
        predictions: List[PredictionResult]
    ) -> Dict[str, Any]:
        """Calculate model metrics.

        Args:
            training_history: Training history
            true_labels: True labels
            predictions: Predictions

        Returns:
            Dictionary of metrics
        """
        final_epoch = training_history[-1]
        pred_labels = [p.predicted_label for p in predictions]

        correct = sum(1 for tl, pl in zip(true_labels, pred_labels) if tl == pl)
        accuracy = correct / len(true_labels)

        avg_confidence = np.mean([p.confidence for p in predictions])

        return {
            "final_train_loss": final_epoch.train_loss,
            "final_train_accuracy": final_epoch.train_accuracy,
            "final_val_loss": final_epoch.val_loss,
            "final_val_accuracy": final_epoch.val_accuracy,
            "final_accuracy": accuracy,
            "average_confidence": avg_confidence,
            "total_epochs": len(training_history),
            "num_predictions": len(predictions),
            "num_classes": self.num_classes
        }

    def _prepare_visualization_data(
        self,
        training_history: List[TrainingHistory],
        predictions: List[PredictionResult],
        confusion_matrix: ConfusionMatrix,
        attention_viz: List[AttentionVisualization]
    ) -> Dict[str, Any]:
        """Prepare data for visualization.

        Args:
            training_history: Training history
            predictions: Predictions
            confusion_matrix: Confusion matrix
            attention_viz: Attention visualizations

        Returns:
            Visualization data dictionary
        """
        # Training curves
        epochs = [h.epoch for h in training_history]
        train_losses = [h.train_loss for h in training_history]
        train_accuracies = [h.train_accuracy for h in training_history]
        val_losses = [h.val_loss for h in training_history]
        val_accuracies = [h.val_accuracy for h in training_history]

        # Class distribution in predictions
        class_counts = {name: 0 for name in self.class_names}
        for pred in predictions:
            class_counts[pred.predicted_class] += 1

        return {
            "training_curves": {
                "epochs": epochs,
                "train_loss": train_losses,
                "train_accuracy": train_accuracies,
                "val_loss": val_losses,
                "val_accuracy": val_accuracies
            },
            "prediction_distribution": {
                "labels": list(class_counts.keys()),
                "counts": list(class_counts.values())
            },
            "confusion_matrix": {
                "matrix": confusion_matrix.matrix,
                "labels": confusion_matrix.labels,
                "accuracy": confusion_matrix.accuracy
            },
            "sample_predictions": [
                {
                    "text": p.text[:100] + "..." if len(p.text) > 100 else p.text,
                    "predicted": p.predicted_class,
                    "confidence": p.confidence,
                    "correct": p.predicted_label == p.true_label if p.true_label is not None else None
                }
                for p in predictions[:10]
            ],
            "attention_heatmaps": [
                {
                    "text": viz.text,
                    "tokens": viz.tokens,
                    "weights": viz.attention_weights
                }
                for viz in attention_viz
            ]
        }

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary containing model information
        """
        num_parameters = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        config = self.model.config

        return {
            "model_name": self.parameters.model_name,
            "num_parameters": num_parameters,
            "num_trainable_parameters": num_trainable,
            "num_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "device": str(self.device)
        }
