"""Text Classification algorithm implementation using scikit-learn."""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from .schema import (
    TextClassificationParameters,
    TextClassificationResponse,
    TextPrediction,
    ClassMetrics,
    TopFeature
)
from .data import (
    get_default_dataset,
    prepare_custom_dataset,
    get_dataset_info,
    get_category_descriptions
)


class TextClassificationModel:
    """Text Classification algorithm implementation.

    This class implements text classification using TF-IDF vectorization
    and various classifiers including Naive Bayes, Logistic Regression,
    and Support Vector Machines.

    Attributes:
        vectorizer: TF-IDF vectorizer
        classifier: Trained classifier model
        parameters: Classification parameters used
        class_names: List of class names

    Example:
        >>> params = TextClassificationParameters(classifier_type='naive_bayes')
        >>> model = TextClassificationModel()
        >>> response = model.train(params)
        >>> print(f"Accuracy: {response.overall_metrics['accuracy']:.2f}")
    """

    def __init__(self):
        """Initialize the Text Classification model."""
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier = None
        self.parameters: Optional[TextClassificationParameters] = None
        self.class_names: List[str] = []

    def train(
        self,
        parameters: TextClassificationParameters
    ) -> TextClassificationResponse:
        """Train text classification model and evaluate on test set.

        Args:
            parameters: Classification parameters including model type and features

        Returns:
            TextClassificationResponse containing predictions, metrics, and visualizations

        Example:
            >>> params = TextClassificationParameters(max_features=500)
            >>> model = TextClassificationModel()
            >>> result = model.train(params)
            >>> if result.success:
            ...     print(f"Training completed in {result.execution_time_ms:.2f}ms")
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
                texts, labels = get_default_dataset()

            # Get dataset info
            dataset_info = get_dataset_info(texts, labels)
            self.class_names = sorted(set(labels))

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts,
                labels,
                test_size=parameters.test_size,
                random_state=42,
                stratify=labels
            )

            # Vectorize text using TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=parameters.max_features,
                ngram_range=parameters.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )

            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)

            # Train classifier
            self.classifier = self._get_classifier(parameters.classifier_type)
            self.classifier.fit(X_train_tfidf, y_train)

            # Make predictions
            y_pred = self.classifier.predict(X_test_tfidf)
            y_pred_proba = self._get_probabilities(X_test_tfidf)

            # Generate predictions list
            predictions = self._create_predictions(
                X_test, y_test, y_pred, y_pred_proba
            )

            # Calculate confusion matrix
            conf_matrix = confusion_matrix(
                y_test, y_pred, labels=self.class_names
            )

            # Calculate metrics
            class_metrics = self._calculate_class_metrics(y_test, y_pred)
            overall_metrics = self._calculate_overall_metrics(y_test, y_pred)

            # Get top features per class
            top_features = self._get_top_features_per_class(n_features=10)

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(
                conf_matrix,
                predictions,
                dataset_info
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return TextClassificationResponse(
                success=True,
                predictions=predictions,
                confusion_matrix=conf_matrix.tolist(),
                class_names=self.class_names,
                class_metrics=class_metrics,
                overall_metrics=overall_metrics,
                top_features_per_class=top_features,
                sample_predictions=predictions[:10],
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                model_info={
                    "classifier_type": parameters.classifier_type,
                    "vectorizer": "TF-IDF",
                    "vocabulary_size": len(self.vectorizer.vocabulary_),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "num_classes": len(self.class_names),
                    "class_names": self.class_names
                }
            )

        except Exception as e:
            return TextClassificationResponse(
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def _get_classifier(self, classifier_type: str):
        """Get classifier instance based on type.

        Args:
            classifier_type: Type of classifier to use

        Returns:
            Instantiated classifier
        """
        if classifier_type == "naive_bayes":
            return MultinomialNB(alpha=1.0)
        elif classifier_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            )
        elif classifier_type == "svm":
            return LinearSVC(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def _get_probabilities(self, X_tfidf) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X_tfidf: TF-IDF transformed features

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X_tfidf)
        elif hasattr(self.classifier, 'decision_function'):
            # For SVM, convert decision function to probabilities
            decision = self.classifier.decision_function(X_tfidf)
            if len(self.class_names) == 2:
                # Binary classification
                proba = np.exp(decision) / (1 + np.exp(decision))
                return np.vstack([1 - proba, proba]).T
            else:
                # Multi-class: use softmax
                exp_scores = np.exp(decision)
                return exp_scores / exp_scores.sum(axis=1, keepdims=True)
        else:
            # Fallback: return uniform probabilities
            n_samples = X_tfidf.shape[0]
            n_classes = len(self.class_names)
            return np.ones((n_samples, n_classes)) / n_classes

    def _create_predictions(
        self,
        texts: List[str],
        y_true: List[str],
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> List[TextPrediction]:
        """Create prediction objects.

        Args:
            texts: Input texts
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        Returns:
            List of TextPrediction objects
        """
        predictions = []

        for i, text in enumerate(texts):
            # Get probabilities for this sample
            proba_dict = {
                class_name: float(y_proba[i][j])
                for j, class_name in enumerate(self.class_names)
            }

            # Get confidence (probability of predicted class)
            pred_idx = self.class_names.index(y_pred[i])
            confidence = float(y_proba[i][pred_idx])

            prediction = TextPrediction(
                text=text,
                true_label=y_true[i],
                predicted_label=y_pred[i],
                confidence=confidence,
                probabilities=proba_dict
            )
            predictions.append(prediction)

        return predictions

    def _calculate_class_metrics(
        self,
        y_true: List[str],
        y_pred: np.ndarray
    ) -> List[ClassMetrics]:
        """Calculate per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            List of ClassMetrics objects
        """
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.class_names, zero_division=0
        )

        class_metrics = []
        for i, class_name in enumerate(self.class_names):
            metrics = ClassMetrics(
                class_name=class_name,
                precision=round(float(precision[i]), 4),
                recall=round(float(recall[i]), 4),
                f1_score=round(float(f1[i]), 4),
                support=int(support[i])
            )
            class_metrics.append(metrics)

        return class_metrics

    def _calculate_overall_metrics(
        self,
        y_true: List[str],
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate overall metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of overall metrics
        """
        from sklearn.metrics import precision_recall_fscore_support

        accuracy = accuracy_score(y_true, y_pred)

        # Calculate macro and weighted averages
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=self.class_names, zero_division=0
        )

        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        # Weighted averages
        _, _, _, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.class_names, zero_division=0
        )
        total = np.sum(support)

        weighted_precision = np.sum(precision * support) / total if total > 0 else 0
        weighted_recall = np.sum(recall * support) / total if total > 0 else 0
        weighted_f1 = np.sum(f1 * support) / total if total > 0 else 0

        return {
            "accuracy": round(float(accuracy), 4),
            "macro_precision": round(float(macro_precision), 4),
            "macro_recall": round(float(macro_recall), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_precision), 4),
            "weighted_recall": round(float(weighted_recall), 4),
            "weighted_f1": round(float(weighted_f1), 4)
        }

    def _get_top_features_per_class(self, n_features: int = 10) -> Dict[str, List[TopFeature]]:
        """Get top discriminative features for each class.

        Args:
            n_features: Number of top features to return per class

        Returns:
            Dictionary mapping class names to top features
        """
        top_features = {}
        feature_names = self.vectorizer.get_feature_names_out()

        if hasattr(self.classifier, 'coef_'):
            # For logistic regression and SVM
            coef = self.classifier.coef_
            for i, class_name in enumerate(self.class_names):
                if len(self.class_names) == 2 and coef.shape[0] == 1:
                    # Binary classification
                    class_coef = coef[0] if i == 1 else -coef[0]
                else:
                    class_coef = coef[i]

                # Get top feature indices
                top_indices = np.argsort(class_coef)[-n_features:][::-1]

                features = [
                    TopFeature(
                        feature=feature_names[idx],
                        weight=round(float(class_coef[idx]), 4)
                    )
                    for idx in top_indices
                ]
                top_features[class_name] = features

        elif hasattr(self.classifier, 'feature_log_prob_'):
            # For Naive Bayes
            feature_log_prob = self.classifier.feature_log_prob_
            for i, class_name in enumerate(self.class_names):
                # Get top feature indices
                top_indices = np.argsort(feature_log_prob[i])[-n_features:][::-1]

                features = [
                    TopFeature(
                        feature=feature_names[idx],
                        weight=round(float(feature_log_prob[i][idx]), 4)
                    )
                    for idx in top_indices
                ]
                top_features[class_name] = features

        return top_features

    def _prepare_visualization_data(
        self,
        conf_matrix: np.ndarray,
        predictions: List[TextPrediction],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for visualization.

        Args:
            conf_matrix: Confusion matrix
            predictions: List of predictions
            dataset_info: Dataset information

        Returns:
            Dictionary containing formatted visualization data
        """
        # Normalize confusion matrix for heatmap
        conf_matrix_normalized = (
            conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        )

        # Confusion matrix data
        confusion_data = {
            "matrix": conf_matrix.tolist(),
            "matrix_normalized": conf_matrix_normalized.tolist(),
            "class_names": self.class_names
        }

        # Confidence distribution
        confidences = [p.confidence for p in predictions]
        confidence_data = {
            "values": confidences,
            "avg_confidence": round(np.mean(confidences), 4),
            "min_confidence": round(np.min(confidences), 4),
            "max_confidence": round(np.max(confidences), 4)
        }

        # Predictions table data (sample)
        sample_predictions = [
            {
                "text": p.text[:100] + "..." if len(p.text) > 100 else p.text,
                "true_label": p.true_label,
                "predicted_label": p.predicted_label,
                "confidence": round(p.confidence, 4),
                "correct": p.true_label == p.predicted_label
            }
            for p in predictions[:20]
        ]

        # Category descriptions
        category_info = get_category_descriptions()

        return {
            "confusion_matrix": confusion_data,
            "confidence_distribution": confidence_data,
            "sample_predictions": sample_predictions,
            "dataset_info": dataset_info,
            "category_descriptions": category_info
        }
