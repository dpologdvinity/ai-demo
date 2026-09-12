"""Bag of Words model implementation using scikit-learn."""

import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import CountVectorizer
import logging

from .schema import BagOfWordsRequest, BagOfWordsResponse, DocumentBow
from .data import get_sample_corpus, validate_custom_corpus

logger = logging.getLogger(__name__)


class BagOfWordsModel:
    """Bag of Words text vectorization model using CountVectorizer."""

    def __init__(self):
        """Initialize the Bag of Words model."""
        self.vectorizer = None
        self.feature_names = None
        self.document_term_matrix = None

    def vectorize(self, request: BagOfWordsRequest) -> BagOfWordsResponse:
        """
        Create Bag of Words representation from documents.

        Args:
            request: BagOfWordsRequest with vectorization parameters

        Returns:
            BagOfWordsResponse with BoW matrix, vocabulary, and statistics

        Raises:
            ValueError: If parameters or corpus are invalid
        """
        start_time = time.time()

        try:
            # Get corpus
            if request.custom_documents:
                validate_custom_corpus(request.custom_documents)
                corpus = request.custom_documents
                logger.info(f"Using custom corpus with {len(corpus)} documents")
            else:
                corpus = get_sample_corpus()
                logger.info(f"Using default corpus with {len(corpus)} documents")

            # Create CountVectorizer with specified parameters
            self.vectorizer = CountVectorizer(
                max_features=request.max_features,
                ngram_range=request.ngram_range,
                min_df=request.min_df,
                max_df=request.max_df,
                binary=request.binary,
                lowercase=True,
                stop_words='english'
            )

            # Fit and transform documents
            self.document_term_matrix = self.vectorizer.fit_transform(corpus)
            self.feature_names = self.vectorizer.get_feature_names_out().tolist()

            # Convert sparse matrix to dense for visualization
            dtm_dense = self.document_term_matrix.toarray()

            # Calculate metrics
            metrics = self._calculate_metrics(dtm_dense, corpus)

            # Get top terms per document
            top_terms_per_doc = self._get_top_terms_per_document(
                dtm_dense, corpus, top_n=10
            )

            # Get most and least frequent terms
            most_frequent = self._get_most_frequent_terms(dtm_dense, top_n=20)
            least_frequent = self._get_least_frequent_terms(dtm_dense, top_n=10)

            # Prepare visualization data
            heatmap_data = self._prepare_heatmap_data(dtm_dense, corpus)
            word_freq_data = self._prepare_word_frequency_data(dtm_dense, top_n=30)

            # Get vocabulary statistics
            vocab_stats = self._get_vocabulary_stats(dtm_dense)

            # Document previews
            doc_previews = [doc[:100] + "..." if len(doc) > 100 else doc
                          for doc in corpus]

            execution_time = (time.time() - start_time) * 1000

            return BagOfWordsResponse(
                success=True,
                metrics=metrics,
                vocabulary=self.feature_names,
                document_term_matrix=dtm_dense.tolist(),
                top_terms_per_doc=top_terms_per_doc,
                most_frequent_terms=most_frequent,
                least_frequent_terms=least_frequent,
                document_previews=doc_previews,
                heatmap_data=heatmap_data,
                word_frequency_data=word_freq_data,
                vocabulary_stats=vocab_stats,
                execution_time_ms=execution_time,
                parameters_used={
                    "max_features": request.max_features,
                    "ngram_range": request.ngram_range,
                    "min_df": request.min_df,
                    "max_df": request.max_df,
                    "binary": request.binary,
                    "stop_words": "english"
                },
                model_info={
                    "vectorizer_type": "CountVectorizer",
                    "sklearn_version": "1.5.2",
                    "corpus_source": "custom" if request.custom_documents else "default"
                }
            )

        except Exception as e:
            logger.error(f"Vectorization error: {str(e)}", exc_info=True)
            return BagOfWordsResponse(
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def _calculate_metrics(
        self,
        dtm: np.ndarray,
        corpus: List[str]
    ) -> Dict[str, Any]:
        """Calculate metrics about the BoW representation.

        Args:
            dtm: Document-term matrix
            corpus: List of documents

        Returns:
            Dictionary of metrics
        """
        n_docs, vocab_size = dtm.shape
        total_terms = dtm.sum()
        non_zero_elements = np.count_nonzero(dtm)
        total_elements = dtm.size
        sparsity = 1 - (non_zero_elements / total_elements)

        return {
            "vocabulary_size": vocab_size,
            "document_count": n_docs,
            "total_terms": int(total_terms),
            "avg_terms_per_doc": float(dtm.sum(axis=1).mean()),
            "avg_unique_terms_per_doc": float((dtm > 0).sum(axis=1).mean()),
            "sparsity": float(sparsity),
            "density": float(1 - sparsity),
            "max_term_frequency": int(dtm.max()),
            "min_term_frequency": int(dtm[dtm > 0].min()) if non_zero_elements > 0 else 0
        }

    def _get_top_terms_per_document(
        self,
        dtm: np.ndarray,
        corpus: List[str],
        top_n: int = 10
    ) -> List[DocumentBow]:
        """Get top terms for each document.

        Args:
            dtm: Document-term matrix
            corpus: List of documents
            top_n: Number of top terms to retrieve

        Returns:
            List of DocumentBow objects
        """
        results = []

        for doc_idx in range(dtm.shape[0]):
            doc_vector = dtm[doc_idx]

            # Get top term indices
            top_indices = np.argsort(doc_vector)[::-1][:top_n]

            # Filter out zero counts
            top_indices = [idx for idx in top_indices if doc_vector[idx] > 0]

            top_terms = [
                {
                    "term": self.feature_names[idx],
                    "frequency": int(doc_vector[idx])
                }
                for idx in top_indices
            ]

            doc_preview = corpus[doc_idx][:100] + "..." if len(corpus[doc_idx]) > 100 else corpus[doc_idx]

            results.append(DocumentBow(
                doc_id=doc_idx,
                doc_preview=doc_preview,
                top_terms=top_terms,
                total_terms=int(doc_vector.sum()),
                unique_terms=int((doc_vector > 0).sum())
            ))

        return results

    def _get_most_frequent_terms(
        self,
        dtm: np.ndarray,
        top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """Get most frequent terms across all documents.

        Args:
            dtm: Document-term matrix
            top_n: Number of top terms to retrieve

        Returns:
            List of term dictionaries with frequencies
        """
        # Sum across all documents
        term_frequencies = dtm.sum(axis=0)

        # Get top indices
        top_indices = np.argsort(term_frequencies)[::-1][:top_n]

        return [
            {
                "term": self.feature_names[idx],
                "frequency": int(term_frequencies[idx]),
                "document_frequency": int((dtm[:, idx] > 0).sum())
            }
            for idx in top_indices
        ]

    def _get_least_frequent_terms(
        self,
        dtm: np.ndarray,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get least frequent terms in vocabulary.

        Args:
            dtm: Document-term matrix
            top_n: Number of least frequent terms to retrieve

        Returns:
            List of term dictionaries with frequencies
        """
        # Sum across all documents
        term_frequencies = dtm.sum(axis=0)

        # Get least frequent indices (but non-zero)
        non_zero_mask = term_frequencies > 0
        non_zero_indices = np.where(non_zero_mask)[0]

        if len(non_zero_indices) == 0:
            return []

        # Sort by frequency
        sorted_indices = non_zero_indices[np.argsort(term_frequencies[non_zero_indices])]
        least_indices = sorted_indices[:top_n]

        return [
            {
                "term": self.feature_names[idx],
                "frequency": int(term_frequencies[idx]),
                "document_frequency": int((dtm[:, idx] > 0).sum())
            }
            for idx in least_indices
        ]

    def _prepare_heatmap_data(
        self,
        dtm: np.ndarray,
        corpus: List[str]
    ) -> Dict[str, Any]:
        """Prepare data for document-term matrix heatmap visualization.

        Args:
            dtm: Document-term matrix
            corpus: List of documents

        Returns:
            Dictionary with heatmap configuration
        """
        # Limit to top terms for visualization clarity
        n_terms_display = min(50, len(self.feature_names))

        # Get most frequent terms for display
        term_frequencies = dtm.sum(axis=0)
        top_term_indices = np.argsort(term_frequencies)[::-1][:n_terms_display]

        # Extract subset of matrix
        dtm_subset = dtm[:, top_term_indices]
        terms_subset = [self.feature_names[idx] for idx in top_term_indices]

        # Document labels
        doc_labels = [f"Doc {i+1}" for i in range(dtm.shape[0])]

        return {
            "matrix": dtm_subset.tolist(),
            "x_labels": terms_subset,
            "y_labels": doc_labels,
            "title": "Document-Term Matrix Heatmap",
            "x_axis_label": "Terms",
            "y_axis_label": "Documents",
            "color_scale": "Blues"
        }

    def _prepare_word_frequency_data(
        self,
        dtm: np.ndarray,
        top_n: int = 30
    ) -> Dict[str, Any]:
        """Prepare data for word frequency bar chart.

        Args:
            dtm: Document-term matrix
            top_n: Number of top words to display

        Returns:
            Dictionary with bar chart configuration
        """
        term_frequencies = dtm.sum(axis=0)
        top_indices = np.argsort(term_frequencies)[::-1][:top_n]

        return {
            "terms": [self.feature_names[idx] for idx in top_indices],
            "frequencies": [int(term_frequencies[idx]) for idx in top_indices],
            "title": f"Top {top_n} Most Frequent Words",
            "x_axis_label": "Frequency",
            "y_axis_label": "Terms"
        }

    def _get_vocabulary_stats(self, dtm: np.ndarray) -> Dict[str, Any]:
        """Get detailed vocabulary statistics.

        Args:
            dtm: Document-term matrix

        Returns:
            Dictionary with vocabulary statistics
        """
        term_frequencies = dtm.sum(axis=0)
        doc_frequencies = (dtm > 0).sum(axis=0)

        return {
            "total_unique_terms": len(self.feature_names),
            "avg_term_frequency": float(term_frequencies.mean()),
            "max_term_frequency": int(term_frequencies.max()),
            "min_term_frequency": int(term_frequencies[term_frequencies > 0].min()) if (term_frequencies > 0).any() else 0,
            "avg_document_frequency": float(doc_frequencies.mean()),
            "terms_in_all_docs": int((doc_frequencies == dtm.shape[0]).sum()),
            "terms_in_one_doc": int((doc_frequencies == 1).sum())
        }
