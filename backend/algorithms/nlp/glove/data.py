"""GloVe embeddings data loader and utilities.

This module provides functions for loading pre-trained GloVe embeddings.
For demonstration purposes, we use a subset of common words with simulated
embeddings based on semantic patterns.
"""

from typing import Dict, List, Tuple
import numpy as np


def get_demo_glove_embeddings(dim: int = 100) -> Dict[str, np.ndarray]:
    """Get demonstration GloVe embeddings for common words.

    For a production system, you would load actual pre-trained GloVe vectors
    from files like glove.6B.100d.txt. This demo uses a curated subset of
    semantically related words with synthetic embeddings that preserve
    realistic relationships.

    Args:
        dim: Embedding dimension (50, 100, 200, or 300)

    Returns:
        Dictionary mapping words to their embedding vectors
    """
    # Seed for reproducibility
    np.random.seed(42)

    # Define semantic groups for realistic embeddings
    word_groups = {
        # Royalty
        "royalty": ["king", "queen", "prince", "princess", "monarch", "royal", "throne", "crown"],
        # Gender
        "male": ["man", "boy", "father", "son", "brother", "uncle", "grandfather", "he", "him", "his"],
        "female": ["woman", "girl", "mother", "daughter", "sister", "aunt", "grandmother", "she", "her"],
        # Family
        "family": ["family", "parent", "child", "sibling", "relative", "household"],
        # Geography
        "geography": ["paris", "france", "london", "england", "berlin", "germany", "rome", "italy",
                     "madrid", "spain", "city", "capital", "country", "nation"],
        # Technology
        "technology": ["computer", "software", "hardware", "algorithm", "programming", "code",
                      "data", "network", "internet", "digital", "technology", "system"],
        # Science
        "science": ["physics", "chemistry", "biology", "mathematics", "theory", "experiment",
                   "research", "scientist", "laboratory", "discovery"],
        # Animals
        "animals": ["dog", "cat", "bird", "fish", "lion", "tiger", "elephant", "horse", "animal", "pet"],
        # Food
        "food": ["food", "bread", "water", "coffee", "tea", "restaurant", "kitchen", "meal", "eat", "drink"],
        # Time
        "time": ["time", "day", "night", "morning", "evening", "hour", "minute", "year", "month", "week"],
        # Emotions
        "emotions": ["happy", "sad", "angry", "fear", "love", "hate", "joy", "surprise", "emotion", "feeling"],
        # Numbers
        "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "number"],
        # Colors
        "colors": ["red", "blue", "green", "yellow", "black", "white", "color", "bright", "dark"],
        # Adjectives
        "size": ["big", "small", "large", "tiny", "huge", "little", "great"],
        "quality": ["good", "bad", "best", "worst", "better", "worse", "excellent", "poor"],
        # Verbs
        "motion": ["go", "come", "run", "walk", "move", "travel", "fly", "swim"],
        "communication": ["say", "tell", "speak", "talk", "write", "read", "listen", "hear"],
        # Common words
        "common": ["the", "be", "to", "of", "and", "a", "in", "that", "have", "it"]
    }

    embeddings = {}

    # Generate embeddings for each word group
    for group_name, words in word_groups.items():
        # Create a base vector for this semantic group
        base_vector = np.random.randn(dim)
        base_vector = base_vector / np.linalg.norm(base_vector)

        for word in words:
            # Add some noise to create individual word vectors
            noise = np.random.randn(dim) * 0.3
            word_vector = base_vector + noise
            # Normalize to unit vector
            word_vector = word_vector / np.linalg.norm(word_vector)
            embeddings[word] = word_vector

    # Add special relationships for analogies
    # king - man + woman ≈ queen
    if "king" in embeddings and "man" in embeddings and "woman" in embeddings and "queen" in embeddings:
        # Adjust vectors to support this analogy
        gender_diff = embeddings["woman"] - embeddings["man"]
        embeddings["queen"] = embeddings["king"] + gender_diff
        embeddings["queen"] = embeddings["queen"] / np.linalg.norm(embeddings["queen"])

    # paris - france + germany ≈ berlin
    if "paris" in embeddings and "france" in embeddings and "germany" in embeddings and "berlin" in embeddings:
        country_diff = embeddings["germany"] - embeddings["france"]
        embeddings["berlin"] = embeddings["paris"] + country_diff
        embeddings["berlin"] = embeddings["berlin"] / np.linalg.norm(embeddings["berlin"])

    return embeddings


def get_vocabulary_stats(embeddings: Dict[str, np.ndarray]) -> Dict[str, any]:
    """Get statistics about the vocabulary.

    Args:
        embeddings: Dictionary of word embeddings

    Returns:
        Dictionary containing vocabulary statistics
    """
    if not embeddings:
        return {
            "vocab_size": 0,
            "embedding_dim": 0,
            "total_parameters": 0
        }

    vocab_size = len(embeddings)
    embedding_dim = len(next(iter(embeddings.values())))

    return {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "total_parameters": vocab_size * embedding_dim,
        "memory_mb": (vocab_size * embedding_dim * 4) / (1024 * 1024)  # float32
    }


def get_sample_queries() -> Dict[str, List[str]]:
    """Get sample query words for demonstration.

    Returns:
        Dictionary with query categories and example words
    """
    return {
        "royalty": ["king", "queen", "prince"],
        "geography": ["paris", "london", "berlin"],
        "technology": ["computer", "software", "algorithm"],
        "animals": ["dog", "cat", "lion"],
        "emotions": ["happy", "love", "joy"]
    }


def get_sample_analogies() -> List[Tuple[str, str, str, str]]:
    """Get sample word analogies for demonstration.

    Returns:
        List of tuples (word_a, word_b, word_c, expected_result)
        representing analogies: word_a - word_b + word_c ≈ expected_result
    """
    return [
        ("king", "man", "woman", "queen"),
        ("paris", "france", "germany", "berlin"),
        ("brother", "man", "woman", "sister"),
        ("uncle", "man", "woman", "aunt"),
        ("father", "man", "woman", "mother"),
    ]


def get_embeddings_info(dim: int) -> Dict[str, any]:
    """Get information about GloVe embeddings.

    Args:
        dim: Embedding dimension

    Returns:
        Dictionary containing information about the embeddings
    """
    return {
        "source": "Demo GloVe Embeddings",
        "dimension": dim,
        "training_corpus": "Simulated from semantic word groups",
        "description": (
            "These are demonstration embeddings created to preserve semantic relationships "
            "between words. In a production system, you would load actual pre-trained GloVe "
            "vectors trained on large corpora like Wikipedia + Gigaword (6B tokens)."
        ),
        "available_dimensions": [50, 100, 200, 300],
        "original_paper": "Pennington et al., 2014 - GloVe: Global Vectors for Word Representation"
    }


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if norm_product == 0:
        return 0.0

    return float(dot_product / norm_product)


def find_similar_words(
    query_word: str,
    embeddings: Dict[str, np.ndarray],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """Find most similar words to a query word.

    Args:
        query_word: Word to find similar words for
        embeddings: Dictionary of word embeddings
        top_k: Number of similar words to return

    Returns:
        List of (word, similarity) tuples sorted by similarity
    """
    if query_word not in embeddings:
        return []

    query_vector = embeddings[query_word]
    similarities = []

    for word, vector in embeddings.items():
        if word != query_word:
            similarity = cosine_similarity(query_vector, vector)
            similarities.append((word, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def solve_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    embeddings: Dict[str, np.ndarray],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Solve word analogy: A is to B as C is to ?

    Computes: vector(A) - vector(B) + vector(C) and finds nearest words.

    Args:
        word_a: First word (A)
        word_b: Second word (B)
        word_c: Third word (C)
        embeddings: Dictionary of word embeddings
        top_k: Number of results to return

    Returns:
        List of (word, similarity) tuples
    """
    # Check if all words exist in vocabulary
    if word_a not in embeddings or word_b not in embeddings or word_c not in embeddings:
        return []

    # Compute analogy vector: A - B + C
    analogy_vector = (
        embeddings[word_a] - embeddings[word_b] + embeddings[word_c]
    )

    # Normalize
    analogy_vector = analogy_vector / np.linalg.norm(analogy_vector)

    # Find most similar words (excluding input words)
    similarities = []
    exclude_words = {word_a, word_b, word_c}

    for word, vector in embeddings.items():
        if word not in exclude_words:
            similarity = cosine_similarity(analogy_vector, vector)
            similarities.append((word, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]
