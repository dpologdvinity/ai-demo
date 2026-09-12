"""Topic Modeling (LDA) algorithm implementation using scikit-learn."""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

from .schema import (
    TopicModelingParameters,
    TopicModelingResponse,
    Topic,
    TopicWord,
    DocumentTopic
)
from .data import (
    get_default_documents,
    prepare_custom_documents,
    get_dataset_info
)


class TopicModelingModel:
    """Topic Modeling (LDA) algorithm implementation.

    This class implements Latent Dirichlet Allocation (LDA) for discovering
    abstract topics in document collections using probabilistic modeling.

    Attributes:
        parameters: Model parameters
        documents: Input document corpus
        vectorizer: CountVectorizer for text preprocessing
        lda_model: Trained LDA model
        doc_term_matrix: Document-term matrix
        feature_names: Vocabulary words

    Example:
        >>> params = TopicModelingParameters(n_topics=5)
        >>> model = TopicModelingModel()
        >>> response = model.train(params)
        >>> print(f"Discovered {len(response.topics)} topics")
    """

    def __init__(self):
        """Initialize the Topic Modeling model."""
        self.parameters: Optional[TopicModelingParameters] = None
        self.documents: Optional[List[str]] = None
        self.vectorizer: Optional[CountVectorizer] = None
        self.lda_model: Optional[LatentDirichletAllocation] = None
        self.doc_term_matrix: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def train(self, parameters: TopicModelingParameters) -> TopicModelingResponse:
        """Train LDA model to discover topics in document corpus.

        Args:
            parameters: Model parameters including n_topics, iterations, etc.

        Returns:
            TopicModelingResponse containing topics, distributions, and metrics

        Example:
            >>> params = TopicModelingParameters(n_topics=5, max_iterations=100)
            >>> model = TopicModelingModel()
            >>> result = model.train(params)
            >>> if result.success:
            ...     print(f"Coherence: {result.coherence_score:.4f}")
        """
        try:
            start_time = time.time()
            self.parameters = parameters

            # Load documents
            if parameters.use_custom_documents and parameters.custom_documents:
                self.documents = prepare_custom_documents(parameters.custom_documents)
            else:
                self.documents = get_default_documents()

            # Preprocess documents
            self._preprocess_documents()

            # Create document-term matrix
            self._create_document_term_matrix()

            # Train LDA model
            self._train_lda_model()

            # Extract topics
            topics = self._extract_topics()

            # Get document-topic distributions
            document_topics = self._get_document_topic_distributions()

            # Get topic-word matrix for heatmap
            topic_word_matrix = self._get_topic_word_matrix()

            # Calculate metrics
            metrics = self._calculate_metrics()

            # Calculate coherence score (simplified version)
            coherence_score = self._calculate_coherence_score()

            # Get perplexity
            perplexity = self._calculate_perplexity()

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(
                topics, document_topics, topic_word_matrix
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return TopicModelingResponse(
                success=True,
                topics=topics,
                document_topics=document_topics,
                topic_word_matrix=topic_word_matrix,
                vocabulary=self.feature_names[:50],  # Top 50 words for display
                coherence_score=coherence_score,
                perplexity=perplexity,
                metrics=metrics,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            return TopicModelingResponse(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def _preprocess_documents(self) -> None:
        """Preprocess documents by cleaning and normalizing text."""
        cleaned_docs = []
        for doc in self.documents:
            # Convert to lowercase
            doc = doc.lower()
            # Remove special characters and extra whitespace
            doc = re.sub(r'[^\w\s]', ' ', doc)
            doc = re.sub(r'\s+', ' ', doc).strip()
            cleaned_docs.append(doc)
        self.documents = cleaned_docs

    def _create_document_term_matrix(self) -> None:
        """Create document-term matrix using CountVectorizer."""
        # Convert alpha/beta parameters
        alpha = self.parameters.alpha
        beta = self.parameters.beta

        # Create vectorizer with stop words removal
        self.vectorizer = CountVectorizer(
            max_df=self.parameters.max_df,
            min_df=self.parameters.min_df,
            stop_words='english',
            lowercase=True,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'  # Words with at least 2 letters
        )

        # Fit and transform documents
        self.doc_term_matrix = self.vectorizer.fit_transform(self.documents)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

    def _train_lda_model(self) -> None:
        """Train LDA model on document-term matrix."""
        # Convert alpha/beta to numeric values
        doc_topic_prior = None if self.parameters.alpha == 'auto' else float(self.parameters.alpha)
        topic_word_prior = None if self.parameters.beta == 'auto' else float(self.parameters.beta)

        # Create and train LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.parameters.n_topics,
            max_iter=self.parameters.max_iterations,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )

        self.lda_model.fit(self.doc_term_matrix)

    def _extract_topics(self, n_top_words: int = 10) -> List[Topic]:
        """Extract topics with their top words.

        Args:
            n_top_words: Number of top words per topic

        Returns:
            List of Topic objects
        """
        topics = []

        for topic_idx, topic in enumerate(self.lda_model.components_):
            # Get indices of top words
            top_indices = topic.argsort()[-n_top_words:][::-1]

            # Get words and weights
            top_words = []
            for idx in top_indices:
                word = self.feature_names[idx]
                weight = float(topic[idx])
                top_words.append(TopicWord(word=word, weight=weight))

            # Create keywords string
            keywords = ', '.join([tw.word for tw in top_words[:5]])

            topics.append(Topic(
                topic_id=topic_idx,
                top_words=top_words,
                keywords=keywords
            ))

        return topics

    def _get_document_topic_distributions(self) -> List[DocumentTopic]:
        """Get document-topic distribution for each document.

        Returns:
            List of DocumentTopic objects
        """
        # Transform documents to get topic distributions
        doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)

        document_topics = []
        for doc_idx, dist in enumerate(doc_topic_dist):
            # Get dominant topic
            dominant_topic = int(np.argmax(dist))

            # Create document preview (first 100 chars)
            preview = self.documents[doc_idx][:100]
            if len(self.documents[doc_idx]) > 100:
                preview += "..."

            document_topics.append(DocumentTopic(
                document_id=doc_idx,
                document_preview=preview,
                dominant_topic=dominant_topic,
                topic_distribution=dist.tolist()
            ))

        return document_topics

    def _get_topic_word_matrix(self, n_words: int = 20) -> List[List[float]]:
        """Get topic-word distribution matrix for heatmap visualization.

        Args:
            n_words: Number of top words to include

        Returns:
            Matrix of shape (n_topics, n_words) with normalized probabilities
        """
        matrix = []

        for topic in self.lda_model.components_:
            # Get top word indices
            top_indices = topic.argsort()[-n_words:][::-1]

            # Get normalized probabilities for top words
            top_probs = topic[top_indices]
            # Normalize to 0-1 range for better visualization
            normalized = (top_probs / top_probs.max()).tolist()

            matrix.append(normalized)

        return matrix

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate model metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "n_topics": self.parameters.n_topics,
            "n_documents": len(self.documents),
            "vocabulary_size": len(self.feature_names),
            "n_iterations": self.lda_model.n_iter_,
            "min_df": self.parameters.min_df,
            "max_df": self.parameters.max_df,
            "alpha": str(self.parameters.alpha),
            "beta": str(self.parameters.beta)
        }

    def _calculate_coherence_score(self) -> float:
        """Calculate simplified coherence score.

        This is a simplified version. For production, use gensim's CoherenceModel.

        Returns:
            Coherence score (higher is better)
        """
        try:
            # Simplified coherence based on topic distinctiveness
            # Get top words for each topic
            n_top_words = 10
            topic_words_sets = []

            for topic in self.lda_model.components_:
                top_indices = topic.argsort()[-n_top_words:][::-1]
                words = set([self.feature_names[idx] for idx in top_indices])
                topic_words_sets.append(words)

            # Calculate average pairwise distinctiveness
            total_overlap = 0
            n_pairs = 0

            for i in range(len(topic_words_sets)):
                for j in range(i + 1, len(topic_words_sets)):
                    overlap = len(topic_words_sets[i] & topic_words_sets[j])
                    total_overlap += overlap
                    n_pairs += 1

            if n_pairs == 0:
                return 0.0

            avg_overlap = total_overlap / n_pairs
            # Convert to coherence score (less overlap = higher coherence)
            coherence = 1.0 - (avg_overlap / n_top_words)

            return round(coherence, 4)

        except Exception:
            return 0.0

    def _calculate_perplexity(self) -> float:
        """Calculate model perplexity.

        Returns:
            Perplexity score (lower is better)
        """
        try:
            perplexity = self.lda_model.perplexity(self.doc_term_matrix)
            return round(perplexity, 2)
        except Exception:
            return 0.0

    def _prepare_visualization_data(
        self,
        topics: List[Topic],
        document_topics: List[DocumentTopic],
        topic_word_matrix: List[List[float]]
    ) -> Dict[str, Any]:
        """Prepare visualization data for frontend.

        Args:
            topics: Extracted topics
            document_topics: Document-topic distributions
            topic_word_matrix: Topic-word matrix

        Returns:
            Dictionary with visualization data
        """
        # Topic word clouds data
        word_clouds = []
        for topic in topics:
            word_cloud_data = [
                {"text": tw.word, "value": tw.weight}
                for tw in topic.top_words
            ]
            word_clouds.append({
                "topic_id": topic.topic_id,
                "data": word_cloud_data
            })

        # Top keywords per topic (bar chart)
        topic_keywords = []
        for topic in topics:
            keywords = [
                {"word": tw.word, "weight": tw.weight}
                for tw in topic.top_words[:10]
            ]
            topic_keywords.append({
                "topic_id": topic.topic_id,
                "keywords": keywords
            })

        # Documents by dominant topic
        docs_by_topic = {}
        for doc_topic in document_topics:
            topic_id = doc_topic.dominant_topic
            if topic_id not in docs_by_topic:
                docs_by_topic[topic_id] = []
            docs_by_topic[topic_id].append({
                "document_id": doc_topic.document_id,
                "preview": doc_topic.document_preview
            })

        # Heatmap data
        heatmap_words = []
        if self.feature_names and topic_word_matrix:
            n_words = len(topic_word_matrix[0]) if topic_word_matrix else 0
            for topic in self.lda_model.components_:
                top_indices = topic.argsort()[-n_words:][::-1]
                heatmap_words = [self.feature_names[idx] for idx in top_indices]
                break  # Just need one set of words

        return {
            "word_clouds": word_clouds,
            "topic_keywords": topic_keywords,
            "documents_by_topic": docs_by_topic,
            "heatmap": {
                "matrix": topic_word_matrix,
                "words": heatmap_words[:20],  # Limit for display
                "topics": [f"Topic {i}" for i in range(self.parameters.n_topics)]
            },
            "topic_distribution": {
                "labels": [f"Topic {i}" for i in range(self.parameters.n_topics)],
                "sample_docs": [
                    {
                        "doc_id": dt.document_id,
                        "preview": dt.document_preview,
                        "distribution": dt.topic_distribution
                    }
                    for dt in document_topics[:10]  # First 10 documents
                ]
            }
        }
