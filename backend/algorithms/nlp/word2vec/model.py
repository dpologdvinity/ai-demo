"""Word2Vec algorithm implementation.

Note: This is a simplified implementation using co-occurrence matrix and SVD
for demonstration. For production use, consider using gensim.Word2Vec which
provides the full Skip-gram and CBOW implementations with negative sampling.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import lil_matrix

from .schema import (
    Word2VecParameters,
    Word2VecResponse,
    SimilarWord,
    AnalogyResult
)
from .data import (
    get_default_corpus,
    prepare_custom_corpus,
    get_corpus_info,
    get_sample_queries
)


class Word2VecModel:
    """Word2Vec algorithm implementation.

    This class implements a simplified Word2Vec algorithm for demonstration
    using co-occurrence matrix factorization. While this approach approximates
    Word2Vec's behavior, the actual Word2Vec uses Skip-gram or CBOW with
    negative sampling for more efficient and effective learning.

    Word2Vec learns distributed representations of words where semantically
    similar words are mapped to nearby points in the embedding space.

    Attributes:
        word_vectors: Dictionary mapping words to their vector representations
        parameters: Training parameters used
        corpus: Training corpus (list of tokenized sentences)
        vocabulary: List of words in the vocabulary
        word_to_idx: Mapping from words to indices
        idx_to_word: Mapping from indices to words

    Example:
        >>> params = Word2VecParameters(vector_size=100, window=5)
        >>> w2v = Word2VecModel()
        >>> response = w2v.train(params)
        >>> print(f"Vocabulary size: {response.metrics['vocab_size']}")
    """

    def __init__(self):
        """Initialize the Word2Vec model."""
        self.word_vectors: Optional[Dict[str, np.ndarray]] = None
        self.parameters: Optional[Word2VecParameters] = None
        self.corpus: Optional[List[List[str]]] = None
        self.vocabulary: Optional[List[str]] = None
        self.word_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_word: Optional[Dict[int, str]] = None

    def train(self, parameters: Word2VecParameters) -> Word2VecResponse:
        """Train Word2Vec model on text corpus.

        Args:
            parameters: Training parameters including vector_size, window, etc.

        Returns:
            Word2VecResponse containing training results, embeddings, and examples

        Example:
            >>> params = Word2VecParameters(vector_size=100, epochs=10)
            >>> model = Word2VecModel()
            >>> result = model.train(params)
            >>> if result.success:
            ...     print(f"Training completed in {result.execution_time_ms:.2f}ms")
        """
        try:
            start_time = time.time()
            self.parameters = parameters

            # Load or prepare corpus
            if parameters.use_custom_corpus and parameters.custom_corpus:
                self.corpus = prepare_custom_corpus(parameters.custom_corpus)
            else:
                self.corpus = get_default_corpus()

            # Get corpus info
            corpus_info = get_corpus_info(self.corpus)

            # Build vocabulary
            self._build_vocabulary()

            # Build co-occurrence matrix
            cooccurrence_matrix = self._build_cooccurrence_matrix()

            # Apply dimensionality reduction (SVD) to get word vectors
            self._compute_embeddings(cooccurrence_matrix)

            # Calculate metrics
            metrics = self._calculate_metrics(corpus_info)

            # Generate embeddings visualization (2D projection)
            embeddings_2d = self._generate_embeddings_2d()

            # Get similar words examples
            similar_words = self._get_similar_words_examples()

            # Get word analogies examples
            analogies = self._get_analogy_examples()

            # Get vocabulary sample
            vocabulary_sample = self._get_vocabulary_sample()

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(embeddings_2d)

            # Get model info
            model_info = self._get_model_info()

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return Word2VecResponse(
                success=True,
                metrics=metrics,
                embeddings_2d=embeddings_2d,
                similar_words=similar_words,
                analogies=analogies,
                vocabulary_sample=vocabulary_sample,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                model_info=model_info
            )

        except ValueError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return Word2VecResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Invalid parameters: {str(e)}"
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return Word2VecResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Training failed: {str(e)}"
            )

    def _build_vocabulary(self):
        """Build vocabulary from corpus."""
        # Count word frequencies
        word_counts = Counter()
        for sentence in self.corpus:
            word_counts.update(sentence)

        # Filter by min_count
        self.vocabulary = [
            word for word, count in word_counts.items()
            if count >= self.parameters.min_count
        ]
        self.vocabulary.sort()  # Sort for consistency

        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def _build_cooccurrence_matrix(self) -> lil_matrix:
        """Build word co-occurrence matrix.

        Returns:
            Sparse co-occurrence matrix
        """
        vocab_size = len(self.vocabulary)
        cooccurrence = lil_matrix((vocab_size, vocab_size), dtype=np.float32)

        # Build co-occurrence matrix based on window
        for sentence in self.corpus:
            # Filter sentence to only include vocabulary words
            sentence = [word for word in sentence if word in self.word_to_idx]

            for i, target_word in enumerate(sentence):
                target_idx = self.word_to_idx[target_word]

                # Look at context words within window
                start = max(0, i - self.parameters.window)
                end = min(len(sentence), i + self.parameters.window + 1)

                for j in range(start, end):
                    if i != j:
                        context_word = sentence[j]
                        context_idx = self.word_to_idx[context_word]

                        # Distance-based weighting (closer words have higher weight)
                        distance = abs(i - j)
                        weight = 1.0 / distance

                        cooccurrence[target_idx, context_idx] += weight

        return cooccurrence

    def _compute_embeddings(self, cooccurrence_matrix: lil_matrix):
        """Compute word embeddings using SVD.

        Args:
            cooccurrence_matrix: Sparse co-occurrence matrix
        """
        # Apply positive pointwise mutual information (PPMI)
        # Convert to CSR for efficient operations
        cooccurrence_csr = cooccurrence_matrix.tocsr()

        # Calculate word and context totals
        word_counts = np.array(cooccurrence_csr.sum(axis=1)).flatten()
        context_counts = np.array(cooccurrence_csr.sum(axis=0)).flatten()
        total_count = cooccurrence_csr.sum()

        # Compute PPMI
        cooccurrence_csr.data = np.log(
            (cooccurrence_csr.data * total_count) /
            (word_counts[cooccurrence_csr.nonzero()[0]] *
             context_counts[cooccurrence_csr.nonzero()[1]])
        )
        # Set negative values to 0 (Positive PMI)
        cooccurrence_csr.data = np.maximum(cooccurrence_csr.data, 0)

        # Apply SVD for dimensionality reduction
        n_components = min(self.parameters.vector_size, len(self.vocabulary) - 1)
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=self.parameters.random_state
        )
        embeddings = svd.fit_transform(cooccurrence_csr)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms

        # Store embeddings
        self.word_vectors = {
            word: embeddings[idx]
            for word, idx in self.word_to_idx.items()
        }

    def _calculate_metrics(self, corpus_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate training metrics.

        Args:
            corpus_info: Information about the training corpus

        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            'vocab_size': len(self.vocabulary),
            'vector_dimensions': self.parameters.vector_size,
            'total_sentences': corpus_info['total_sentences'],
            'total_words': corpus_info['total_words'],
            'unique_words': corpus_info['unique_words'],
            'avg_sentence_length': corpus_info['avg_sentence_length'],
            'training_algorithm': 'Skip-gram' if self.parameters.sg == 1 else 'CBOW',
            'window_size': self.parameters.window,
            'min_count': self.parameters.min_count
        }

        return metrics

    def _generate_embeddings_2d(self, n_words: int = 50) -> List[Dict[str, Any]]:
        """Generate 2D projections of word embeddings using t-SNE.

        Args:
            n_words: Number of most frequent words to visualize

        Returns:
            List of dictionaries with word and 2D coordinates
        """
        # Get most frequent words (first n from vocabulary)
        n_words = min(n_words, len(self.vocabulary))
        words = self.vocabulary[:n_words]

        # Get their embeddings
        embeddings = np.array([self.word_vectors[word] for word in words])

        # Use t-SNE for dimensionality reduction to 2D
        tsne = TSNE(
            n_components=2,
            random_state=self.parameters.random_state,
            perplexity=min(30, len(words) - 1)  # Ensure perplexity < n_samples
        )
        embeddings_2d = tsne.fit_transform(embeddings)

        # Format for visualization
        result = []
        for i, word in enumerate(words):
            result.append({
                'word': word,
                'x': float(embeddings_2d[i, 0]),
                'y': float(embeddings_2d[i, 1])
            })

        return result

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _get_similar_words_examples(self, n_queries: int = 5, topn: int = 5) -> Dict[str, List[SimilarWord]]:
        """Get similar words for sample queries.

        Args:
            n_queries: Number of query words to sample
            topn: Number of similar words to return per query

        Returns:
            Dictionary mapping query words to their similar words
        """
        sample_queries = get_sample_queries()['similarity_queries']
        similar_words = {}

        # Filter queries to only include words in vocabulary
        valid_queries = [word for word in sample_queries if word in self.word_vectors][:n_queries]

        for query_word in valid_queries:
            try:
                query_vec = self.word_vectors[query_word]
                similarities = []

                # Calculate similarity with all other words
                for word in self.vocabulary:
                    if word != query_word:
                        similarity = self._cosine_similarity(query_vec, self.word_vectors[word])
                        similarities.append((word, similarity))

                # Sort by similarity and get top N
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similar = similarities[:topn]

                similar_words[query_word] = [
                    SimilarWord(word=word, similarity=float(score))
                    for word, score in top_similar
                ]
            except Exception:
                # Word not in vocabulary or other error
                continue

        return similar_words

    def _get_analogy_examples(self, n_examples: int = 3) -> List[AnalogyResult]:
        """Get word analogy examples.

        Computes analogies of the form: word1 - word2 + word3 ≈ ?

        Args:
            n_examples: Number of analogy examples to generate

        Returns:
            List of analogy results
        """
        sample_queries = get_sample_queries()['analogy_queries']
        analogies = []

        for positive1, positive2, negative1 in sample_queries[:n_examples]:
            # Check if all words are in vocabulary
            if all(word in self.word_vectors for word in [positive1, positive2, negative1]):
                try:
                    # Compute analogy: positive1 - negative1 + positive2
                    result_vec = (
                        self.word_vectors[positive1] -
                        self.word_vectors[negative1] +
                        self.word_vectors[positive2]
                    )

                    # Find most similar word
                    best_word = None
                    best_similarity = -1

                    for word in self.vocabulary:
                        # Exclude input words
                        if word not in [positive1, positive2, negative1]:
                            similarity = self._cosine_similarity(result_vec, self.word_vectors[word])
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_word = word

                    if best_word:
                        query = f"{positive1} - {negative1} + {positive2}"
                        analogies.append(
                            AnalogyResult(
                                query=query,
                                result=best_word,
                                similarity=float(best_similarity)
                            )
                        )
                except Exception:
                    # Analogy computation failed
                    continue

        return analogies

    def _get_vocabulary_sample(self, n_words: int = 20) -> List[str]:
        """Get sample of vocabulary words.

        Args:
            n_words: Number of words to sample

        Returns:
            List of vocabulary words
        """
        return self.vocabulary[:min(n_words, len(self.vocabulary))]

    def _prepare_visualization_data(self, embeddings_2d: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            embeddings_2d: 2D embeddings with words

        Returns:
            Dictionary with visualization data
        """
        return {
            'embeddings': embeddings_2d,
            'method': 't-SNE',
            'dimensions': 2
        }

    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Dictionary with model configuration and statistics
        """
        return {
            'vector_size': self.parameters.vector_size,
            'window': self.parameters.window,
            'min_count': self.parameters.min_count,
            'sg': self.parameters.sg,
            'algorithm': 'Skip-gram' if self.parameters.sg == 1 else 'CBOW',
            'epochs': self.parameters.epochs,
            'vocabulary_size': len(self.vocabulary),
            'implementation': 'Co-occurrence matrix with SVD (simplified)'
        }

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector representation for a word.

        Args:
            word: Word to get vector for

        Returns:
            Word vector as numpy array, or None if word not in vocabulary
        """
        if self.word_vectors is None:
            raise RuntimeError("Model must be trained before getting word vectors")

        return self.word_vectors.get(word)

    def find_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words to a given word.

        Args:
            word: Query word
            topn: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples

        Raises:
            RuntimeError: If model hasn't been trained
            KeyError: If word not in vocabulary
        """
        if self.word_vectors is None:
            raise RuntimeError("Model must be trained before finding similar words")

        if word not in self.word_vectors:
            raise KeyError(f"Word '{word}' not in vocabulary")

        query_vec = self.word_vectors[word]
        similarities = []

        for other_word in self.vocabulary:
            if other_word != word:
                similarity = self._cosine_similarity(query_vec, self.word_vectors[other_word])
                similarities.append((other_word, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def compute_analogy(
        self,
        positive: List[str],
        negative: List[str],
        topn: int = 1
    ) -> List[Tuple[str, float]]:
        """Compute word analogy.

        Example: king - man + woman ≈ queen
        positive = ['king', 'woman'], negative = ['man']

        Args:
            positive: Words to add
            negative: Words to subtract
            topn: Number of results to return

        Returns:
            List of (word, similarity_score) tuples

        Raises:
            RuntimeError: If model hasn't been trained
        """
        if self.word_vectors is None:
            raise RuntimeError("Model must be trained before computing analogies")

        # Compute result vector
        result_vec = np.zeros(self.parameters.vector_size)
        for word in positive:
            if word in self.word_vectors:
                result_vec += self.word_vectors[word]
        for word in negative:
            if word in self.word_vectors:
                result_vec -= self.word_vectors[word]

        # Find most similar words
        similarities = []
        exclude_words = set(positive + negative)

        for word in self.vocabulary:
            if word not in exclude_words:
                similarity = self._cosine_similarity(result_vec, self.word_vectors[word])
                similarities.append((word, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
