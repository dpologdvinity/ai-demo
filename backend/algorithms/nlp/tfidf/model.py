"""TF-IDF (Term Frequency-Inverse Document Frequency) model implementation."""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

from .schema import TFIDFRequest, TFIDFResponse, DocumentTerms
from .data import get_sample_corpus, validate_custom_corpus


class TFIDFModel:
    """TF-IDF vectorizer for text analysis and feature extraction.

    This class implements TF-IDF (Term Frequency-Inverse Document Frequency),
    a statistical measure used to evaluate the importance of words in documents.
    """

    def __init__(
        self,
        max_features: int = 100,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
        use_idf: bool = True,
        normalize: bool = True
    ):
        """Initialize TF-IDF vectorizer.

        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-gram sizes (min_n, max_n)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_idf: Whether to use IDF weighting
            normalize: Whether to normalize vectors
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.normalize = 'l2' if normalize else None

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.documents: List[str] = []

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit the vectorizer and transform documents.

        Args:
            documents: List of text documents

        Returns:
            TF-IDF matrix as numpy array

        Raises:
            ValueError: If documents list is invalid
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self.documents = documents

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            use_idf=self.use_idf,
            norm=self.normalize,
            stop_words='english',
            lowercase=True
        )

        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(documents).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

        return self.tfidf_matrix

    def get_top_terms_per_document(
        self,
        n_terms: int = 10,
        preview_length: int = 100
    ) -> List[DocumentTerms]:
        """Get top N terms for each document.

        Args:
            n_terms: Number of top terms to extract per document
            preview_length: Length of document preview text

        Returns:
            List of DocumentTerms objects
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model has not been fitted yet")

        results = []

        for doc_idx in range(len(self.documents)):
            doc_scores = self.tfidf_matrix[doc_idx]

            # Get indices of top terms
            top_indices = np.argsort(doc_scores)[::-1][:n_terms]

            # Create term-score pairs
            top_terms = [
                {
                    "term": self.feature_names[idx],
                    "score": float(doc_scores[idx]),
                    "rank": rank + 1
                }
                for rank, idx in enumerate(top_indices)
                if doc_scores[idx] > 0
            ]

            # Create document preview
            doc_preview = self.documents[doc_idx][:preview_length]
            if len(self.documents[doc_idx]) > preview_length:
                doc_preview += "..."

            results.append(DocumentTerms(
                doc_id=doc_idx,
                doc_preview=doc_preview,
                top_terms=top_terms
            ))

        return results

    def get_top_terms_global(self, n_terms: int = 20) -> List[Dict[str, Any]]:
        """Get most important terms across all documents.

        Args:
            n_terms: Number of top terms to return

        Returns:
            List of dictionaries with term statistics
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model has not been fitted yet")

        # Calculate mean TF-IDF score across all documents for each term
        mean_scores = np.mean(self.tfidf_matrix, axis=0)

        # Calculate max TF-IDF score for each term
        max_scores = np.max(self.tfidf_matrix, axis=0)

        # Calculate document frequency (how many docs contain each term)
        doc_freq = np.sum(self.tfidf_matrix > 0, axis=0)

        # Get top term indices
        top_indices = np.argsort(mean_scores)[::-1][:n_terms]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "term": self.feature_names[idx],
                "mean_tfidf": float(mean_scores[idx]),
                "max_tfidf": float(max_scores[idx]),
                "doc_frequency": int(doc_freq[idx]),
                "rank": rank + 1
            })

        return results

    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get statistics about the vocabulary.

        Returns:
            Dictionary with vocabulary statistics
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model has not been fitted yet")

        # Calculate statistics
        vocab_size = len(self.feature_names)
        total_docs = len(self.documents)

        # Terms per document
        terms_per_doc = np.sum(self.tfidf_matrix > 0, axis=1)

        # Document frequency stats
        doc_freq = np.sum(self.tfidf_matrix > 0, axis=0)

        return {
            "vocabulary_size": vocab_size,
            "total_documents": total_docs,
            "avg_terms_per_doc": float(np.mean(terms_per_doc)),
            "min_terms_per_doc": int(np.min(terms_per_doc)),
            "max_terms_per_doc": int(np.max(terms_per_doc)),
            "avg_doc_frequency": float(np.mean(doc_freq)),
            "sparsity": float(1.0 - np.count_nonzero(self.tfidf_matrix) / self.tfidf_matrix.size)
        }

    def get_heatmap_data(self, max_terms: int = 30, max_docs: int = 20) -> Dict[str, Any]:
        """Prepare data for heatmap visualization.

        Args:
            max_terms: Maximum number of terms to include
            max_docs: Maximum number of documents to include

        Returns:
            Dictionary with heatmap data
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model has not been fitted yet")

        # Limit matrix size for visualization
        n_docs = min(len(self.documents), max_docs)
        n_terms = min(len(self.feature_names), max_terms)

        # Select top terms by mean TF-IDF score
        mean_scores = np.mean(self.tfidf_matrix, axis=0)
        top_term_indices = np.argsort(mean_scores)[::-1][:n_terms]

        # Extract submatrix
        heatmap_matrix = self.tfidf_matrix[:n_docs, top_term_indices]

        # Get selected feature names
        selected_features = [self.feature_names[idx] for idx in top_term_indices]

        # Create document labels
        doc_labels = [f"Doc {i+1}" for i in range(n_docs)]

        return {
            "matrix": heatmap_matrix.tolist(),
            "terms": selected_features,
            "documents": doc_labels,
            "n_docs": n_docs,
            "n_terms": n_terms
        }

    def search_term(self, term: str) -> Dict[str, Any]:
        """Search for a specific term across documents.

        Args:
            term: Term to search for

        Returns:
            Dictionary with term statistics across documents
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model has not been fitted yet")

        term_lower = term.lower()

        if term_lower not in self.feature_names:
            return {
                "term": term,
                "found": False,
                "message": f"Term '{term}' not in vocabulary"
            }

        # Get term index
        term_idx = self.feature_names.index(term_lower)

        # Get scores for this term across all documents
        term_scores = self.tfidf_matrix[:, term_idx]

        # Find documents containing this term
        doc_indices = np.where(term_scores > 0)[0]

        return {
            "term": term_lower,
            "found": True,
            "document_count": len(doc_indices),
            "mean_score": float(np.mean(term_scores[term_scores > 0])) if len(doc_indices) > 0 else 0.0,
            "max_score": float(np.max(term_scores)),
            "documents": [
                {
                    "doc_id": int(idx),
                    "score": float(term_scores[idx])
                }
                for idx in doc_indices
            ]
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Dictionary with model configuration and statistics
        """
        return {
            "algorithm": "TF-IDF (Term Frequency-Inverse Document Frequency)",
            "library": "scikit-learn TfidfVectorizer",
            "parameters": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "use_idf": self.use_idf,
                "normalize": self.normalize is not None,
                "stop_words": "english"
            },
            "statistics": self.get_vocabulary_stats() if self.tfidf_matrix is not None else {}
        }


def train_tfidf(request: TFIDFRequest) -> TFIDFResponse:
    """Train TF-IDF vectorizer and return results.

    Args:
        request: TF-IDF training request with parameters

    Returns:
        TFIDFResponse with results and visualizations
    """
    try:
        start_time = time.time()

        # Get documents
        if request.custom_documents:
            validate_custom_corpus(request.custom_documents)
            documents = request.custom_documents
        else:
            documents = get_sample_corpus()

        # Initialize model
        model = TFIDFModel(
            max_features=request.max_features,
            ngram_range=request.ngram_range,
            min_df=request.min_df,
            max_df=request.max_df,
            use_idf=request.use_idf,
            normalize=request.normalize
        )

        # Fit and transform
        tfidf_matrix = model.fit_transform(documents)

        # Get top terms per document
        top_terms_per_doc = model.get_top_terms_per_document(n_terms=10)

        # Get global top terms
        top_terms_global = model.get_top_terms_global(n_terms=20)

        # Get heatmap data
        heatmap_data = model.get_heatmap_data(max_terms=30, max_docs=20)

        # Get vocabulary statistics
        vocab_stats = model.get_vocabulary_stats()

        # Prepare document previews
        document_previews = [doc[:150] + "..." if len(doc) > 150 else doc for doc in documents]

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Prepare visualization data
        visualization_data = {
            "heatmap": heatmap_data,
            "top_terms_chart": {
                "labels": [item["term"] for item in top_terms_global[:15]],
                "values": [item["mean_tfidf"] for item in top_terms_global[:15]]
            },
            "document_stats": {
                "doc_ids": list(range(len(documents))),
                "term_counts": [sum(1 for score in tfidf_matrix[i] if score > 0)
                               for i in range(len(documents))]
            }
        }

        return TFIDFResponse(
            success=True,
            metrics={
                "vocabulary_size": vocab_stats["vocabulary_size"],
                "total_documents": vocab_stats["total_documents"],
                "avg_terms_per_doc": vocab_stats["avg_terms_per_doc"],
                "sparsity": vocab_stats["sparsity"],
                "ngram_type": f"{request.ngram_range[0]}-{request.ngram_range[1]} grams"
            },
            feature_names=model.feature_names,
            tfidf_matrix=tfidf_matrix.tolist(),
            top_terms_per_doc=top_terms_per_doc,
            top_terms_global=top_terms_global,
            document_previews=document_previews,
            heatmap_data=heatmap_data,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            parameters_used={
                "max_features": request.max_features,
                "ngram_range": request.ngram_range,
                "min_df": request.min_df,
                "max_df": request.max_df,
                "use_idf": request.use_idf,
                "normalize": request.normalize
            },
            model_info=model.get_model_info()
        )

    except Exception as e:
        return TFIDFResponse(
            success=False,
            execution_time_ms=0.0,
            error=str(e)
        )
