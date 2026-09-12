"""GloVe (Global Vectors) word embeddings implementation.

GloVe is an unsupervised learning algorithm for obtaining vector representations
for words. Training is performed on aggregated global word-word co-occurrence
statistics from a corpus, and the resulting representations showcase interesting
linear substructures of the word vector space.

Reference: Pennington et al., 2014 - "GloVe: Global Vectors for Word Representation"
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.manifold import TSNE

from .schema import (
    GloVeParameters,
    GloVeResponse,
    SimilarWord,
    AnalogyResult
)
from .data import (
    get_demo_glove_embeddings,
    get_vocabulary_stats,
    get_sample_queries,
    get_sample_analogies,
    get_embeddings_info,
    find_similar_words,
    solve_analogy,
    cosine_similarity
)


class GloVeModel:
    """GloVe word embeddings model.

    This class provides functionality for loading pre-trained GloVe embeddings
    and performing various operations like finding similar words, solving
    word analogies, and computing similarity matrices.

    For demonstration purposes, this uses a curated subset of words with
    synthetic embeddings. In production, you would load actual pre-trained
    GloVe vectors from files like glove.6B.100d.txt.

    Attributes:
        embeddings: Dictionary mapping words to their vector representations
        parameters: Parameters used for queries
        vocabulary: List of words in the vocabulary

    Example:
        >>> params = GloVeParameters(embedding_dim=100, query_word="king")
        >>> model = GloVeModel()
        >>> response = model.query(params)
        >>> print(f"Similar words to 'king': {response.similar_words}")
    """

    def __init__(self):
        """Initialize the GloVe model."""
        self.embeddings: Optional[Dict[str, np.ndarray]] = None
        self.parameters: Optional[GloVeParameters] = None
        self.vocabulary: Optional[List[str]] = None

    def query(self, parameters: GloVeParameters) -> GloVeResponse:
        """Query GloVe embeddings for similar words and analogies.

        Args:
            parameters: Query parameters including embedding dimension and query words

        Returns:
            GloVeResponse containing similar words, analogies, and visualizations

        Example:
            >>> params = GloVeParameters(
            ...     embedding_dim=100,
            ...     query_word="king",
            ...     top_k=10
            ... )
            >>> model = GloVeModel()
            >>> result = model.query(params)
            >>> if result.success:
            ...     print(f"Found {len(result.similar_words)} similar words")
        """
        try:
            start_time = time.time()
            self.parameters = parameters

            # Load pre-trained embeddings
            self.embeddings = get_demo_glove_embeddings(parameters.embedding_dim)
            self.vocabulary = list(self.embeddings.keys())

            # Calculate metrics
            metrics = self._calculate_metrics()

            # Find similar words for query word
            similar_words = self._get_similar_words()

            # Solve word analogy
            analogy_result = self._solve_analogy()

            # Generate 2D embeddings visualization
            embeddings_2d = self._generate_embeddings_2d()

            # Create cosine similarity matrix for selected words
            cosine_matrix = self._create_cosine_similarity_matrix()

            # Get vocabulary sample
            vocabulary_sample = self._get_vocabulary_sample()

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(embeddings_2d)

            # Get model info
            model_info = self._get_model_info()

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return GloVeResponse(
                success=True,
                metrics=metrics,
                embeddings_2d=embeddings_2d,
                similar_words=similar_words,
                analogy_result=analogy_result,
                cosine_similarity_matrix=cosine_matrix,
                vocabulary_sample=vocabulary_sample,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                model_info=model_info
            )

        except ValueError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return GloVeResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Invalid parameters: {str(e)}"
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return GloVeResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump() if parameters else {},
                error=f"Query failed: {str(e)}"
            )

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate metrics about the embeddings.

        Returns:
            Dictionary containing vocabulary and embedding statistics
        """
        stats = get_vocabulary_stats(self.embeddings)

        return {
            "vocab_size": stats["vocab_size"],
            "embedding_dim": stats["embedding_dim"],
            "total_parameters": stats["total_parameters"],
            "memory_mb": round(stats["memory_mb"], 2),
            "available_queries": len(get_sample_queries())
        }

    def _get_similar_words(self) -> List[SimilarWord]:
        """Find similar words to the query word.

        Returns:
            List of SimilarWord objects with similarity scores
        """
        query_word = self.parameters.query_word.lower()

        # Check if query word exists in vocabulary
        if query_word not in self.embeddings:
            # Return empty list with a note that word is not in vocabulary
            return []

        # Find similar words
        similar = find_similar_words(
            query_word,
            self.embeddings,
            top_k=self.parameters.top_k
        )

        return [
            SimilarWord(word=word, similarity=round(sim, 4))
            for word, sim in similar
        ]

    def _solve_analogy(self) -> Optional[AnalogyResult]:
        """Solve word analogy: A - B + C = ?

        Returns:
            AnalogyResult with the solution or None if words not in vocabulary
        """
        word_a = self.parameters.analogy_word_a.lower()
        word_b = self.parameters.analogy_word_b.lower()
        word_c = self.parameters.analogy_word_c.lower()

        # Check if all words exist in vocabulary
        missing_words = []
        for word in [word_a, word_b, word_c]:
            if word not in self.embeddings:
                missing_words.append(word)

        if missing_words:
            # Return result indicating missing words
            return AnalogyResult(
                query=f"{word_a} - {word_b} + {word_c}",
                result_word=f"N/A (missing: {', '.join(missing_words)})",
                similarity=0.0,
                top_results=[]
            )

        # Solve analogy
        results = solve_analogy(
            word_a,
            word_b,
            word_c,
            self.embeddings,
            top_k=5
        )

        if not results:
            return None

        # Format top results
        top_results = [
            SimilarWord(word=word, similarity=round(sim, 4))
            for word, sim in results
        ]

        return AnalogyResult(
            query=f"{word_a} - {word_b} + {word_c}",
            result_word=results[0][0],
            similarity=round(results[0][1], 4),
            top_results=top_results
        )

    def _generate_embeddings_2d(self, n_words: int = 50) -> List[Dict[str, Any]]:
        """Generate 2D projections of embeddings using t-SNE.

        Args:
            n_words: Number of words to visualize

        Returns:
            List of dictionaries with word, x, y coordinates, and category
        """
        # Select a diverse subset of words
        sample_words = self._select_diverse_words(n_words)

        # Get vectors for selected words
        vectors = np.array([self.embeddings[word] for word in sample_words])

        # Apply t-SNE for dimensionality reduction
        # Use a smaller perplexity for small datasets
        perplexity = min(30, len(sample_words) - 1)
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            max_iter=1000
        )
        embeddings_2d = tsne.fit_transform(vectors)

        # Assign categories for color coding
        categories = self._assign_word_categories(sample_words)

        # Format results
        results = []
        for i, word in enumerate(sample_words):
            results.append({
                "word": word,
                "x": float(embeddings_2d[i, 0]),
                "y": float(embeddings_2d[i, 1]),
                "category": categories.get(word, "other")
            })

        return results

    def _select_diverse_words(self, n_words: int) -> List[str]:
        """Select a diverse subset of words for visualization.

        Args:
            n_words: Number of words to select

        Returns:
            List of selected words
        """
        # Priority words to include
        priority_words = [
            self.parameters.query_word.lower(),
            self.parameters.analogy_word_a.lower(),
            self.parameters.analogy_word_b.lower(),
            self.parameters.analogy_word_c.lower(),
            "king", "queen", "man", "woman",
            "paris", "france", "london", "england",
            "computer", "technology", "science", "research"
        ]

        # Filter priority words that exist in vocabulary
        selected = [w for w in priority_words if w in self.embeddings]

        # Add remaining words from vocabulary
        remaining = [w for w in self.vocabulary if w not in selected]
        np.random.seed(42)
        np.random.shuffle(remaining)

        selected.extend(remaining[:n_words - len(selected)])

        return selected[:n_words]

    def _assign_word_categories(self, words: List[str]) -> Dict[str, str]:
        """Assign semantic categories to words for visualization.

        Args:
            words: List of words to categorize

        Returns:
            Dictionary mapping words to categories
        """
        categories = {}

        category_keywords = {
            "royalty": ["king", "queen", "prince", "princess", "monarch", "royal"],
            "gender": ["man", "woman", "boy", "girl", "male", "female", "he", "she"],
            "geography": ["paris", "london", "berlin", "france", "england", "germany", "city", "capital"],
            "technology": ["computer", "software", "algorithm", "programming", "data", "digital"],
            "science": ["physics", "chemistry", "biology", "science", "research", "theory"],
            "animals": ["dog", "cat", "bird", "fish", "lion", "tiger", "animal"],
            "emotions": ["happy", "sad", "love", "hate", "joy", "fear", "emotion"],
        }

        for word in words:
            assigned = False
            for category, keywords in category_keywords.items():
                if word in keywords:
                    categories[word] = category
                    assigned = True
                    break
            if not assigned:
                categories[word] = "other"

        return categories

    def _create_cosine_similarity_matrix(self) -> Dict[str, Any]:
        """Create cosine similarity matrix for selected words.

        Returns:
            Dictionary with matrix data for heatmap visualization
        """
        # Select interesting words for the matrix
        matrix_words = [
            self.parameters.query_word.lower(),
            "king", "queen", "man", "woman",
            "paris", "london", "france", "england",
            "computer", "technology"
        ]

        # Filter words that exist in vocabulary
        matrix_words = [w for w in matrix_words if w in self.embeddings]

        # Remove duplicates while preserving order
        seen = set()
        matrix_words = [w for w in matrix_words if not (w in seen or seen.add(w))]

        # Limit to max 15 words for readability
        matrix_words = matrix_words[:15]

        # Compute similarity matrix
        n = len(matrix_words)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = cosine_similarity(
                        self.embeddings[matrix_words[i]],
                        self.embeddings[matrix_words[j]]
                    )

        return {
            "words": matrix_words,
            "matrix": similarity_matrix.tolist(),
            "shape": [n, n]
        }

    def _get_vocabulary_sample(self, n: int = 30) -> List[str]:
        """Get a sample of vocabulary words.

        Args:
            n: Number of words to sample

        Returns:
            List of vocabulary words
        """
        if len(self.vocabulary) <= n:
            return sorted(self.vocabulary)

        # Include query words
        sample = [
            self.parameters.query_word.lower(),
            self.parameters.analogy_word_a.lower(),
            self.parameters.analogy_word_b.lower(),
            self.parameters.analogy_word_c.lower(),
        ]

        # Filter existing words
        sample = [w for w in sample if w in self.embeddings]

        # Add random words
        remaining = [w for w in self.vocabulary if w not in sample]
        np.random.seed(42)
        np.random.shuffle(remaining)

        sample.extend(remaining[:n - len(sample)])

        return sorted(list(set(sample)))[:n]

    def _prepare_visualization_data(self, embeddings_2d: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            embeddings_2d: 2D embedding coordinates

        Returns:
            Dictionary containing formatted visualization data
        """
        return {
            "type": "scatter_2d",
            "data": embeddings_2d,
            "config": {
                "color_by": "category",
                "show_labels": True,
                "point_size": 8
            }
        }

    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the GloVe model.

        Returns:
            Dictionary containing model information
        """
        embeddings_info = get_embeddings_info(self.parameters.embedding_dim)

        return {
            "algorithm": "GloVe (Global Vectors for Word Representation)",
            "source": embeddings_info["source"],
            "dimension": embeddings_info["dimension"],
            "description": embeddings_info["description"],
            "original_paper": embeddings_info["original_paper"],
            "vocabulary_size": len(self.embeddings),
            "sample_analogies": get_sample_analogies(),
            "sample_queries": get_sample_queries()
        }
