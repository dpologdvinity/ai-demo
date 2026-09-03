"""
Algorithm metadata utilities for standardized algorithm information.

This module provides data structures and a registry system for managing
algorithm metadata, parameters, and configuration across the application.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DifficultyLevel(str, Enum):
    """Algorithm difficulty levels for user guidance."""

    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class AlgorithmCategory(str, Enum):
    """Categories for organizing algorithms."""

    ML = "ml"
    DEEP_LEARNING = "deep_learning"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class AlgorithmParameter(BaseModel):
    """Configuration parameter for an algorithm.

    Attributes:
        name: Internal parameter name (e.g., 'learning_rate')
        label: Display label for UI (e.g., 'Learning Rate')
        type: Parameter input type ('number', 'select', 'boolean', 'range')
        default: Default value for the parameter
        min: Minimum value (for numeric types)
        max: Maximum value (for numeric types)
        step: Step increment (for numeric types)
        options: Available options (for select types)
        description: User-friendly description of the parameter
    """

    name: str
    label: str
    type: str  # 'number', 'select', 'boolean', 'range'
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Dict[str, Any]]] = None
    description: str


class AlgorithmComplexity(BaseModel):
    """Time and space complexity information.

    Attributes:
        time: Time complexity in Big-O notation (e.g., "O(n^2)")
        space: Space complexity in Big-O notation (e.g., "O(n)")
    """

    time: str  # e.g., "O(n^2)", "O(n log n)"
    space: str  # e.g., "O(n)", "O(1)"


class AlgorithmMetadata(BaseModel):
    """Complete metadata for an algorithm.

    This model contains all information needed to display, configure,
    and execute an algorithm, including parameters, complexity analysis,
    and educational content.

    Attributes:
        id: Unique identifier for the algorithm
        name: Display name of the algorithm
        slug: URL-safe identifier (e.g., 'linear-regression')
        category: Algorithm category
        description: Brief description of the algorithm
        difficulty: Difficulty level for users
        tags: Searchable tags
        use_cases: Common use cases and applications
        complexity: Time and space complexity
        parameters: Configurable parameters
        dataset_name: Name of dataset from DatasetManager
        visualization_type: Type of visualization ('line_chart', 'scatter_plot', etc.)
        theory: Detailed explanation of algorithm theory
        pros: Advantages of this algorithm
        cons: Limitations and disadvantages
        related_algorithms: Related algorithm slugs
    """

    id: str
    name: str
    slug: str
    category: AlgorithmCategory
    description: str
    difficulty: DifficultyLevel
    tags: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)
    complexity: AlgorithmComplexity
    parameters: List[AlgorithmParameter] = Field(default_factory=list)
    dataset_name: str  # Which dataset from DatasetManager to use
    visualization_type: str  # 'line_chart', 'scatter_plot', 'confusion_matrix', etc.
    theory: str = ""  # Algorithm theory/explanation
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    related_algorithms: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class AlgorithmRegistry:
    """Central registry for all algorithms.

    This singleton-like class maintains a registry of all available algorithms
    and provides methods for registration, retrieval, and search.

    Example:
        >>> metadata = AlgorithmMetadata(...)
        >>> AlgorithmRegistry.register(metadata)
        >>> algo = AlgorithmRegistry.get('linear-regression')
        >>> all_ml = AlgorithmRegistry.get_by_category(AlgorithmCategory.ML)
    """

    _registry: Dict[str, AlgorithmMetadata] = {}

    @classmethod
    def register(cls, metadata: AlgorithmMetadata) -> None:
        """Register an algorithm in the registry.

        Args:
            metadata: Algorithm metadata to register
        """
        cls._registry[metadata.slug] = metadata

    @classmethod
    def get(cls, slug: str) -> Optional[AlgorithmMetadata]:
        """Get algorithm metadata by slug.

        Args:
            slug: URL-safe algorithm identifier

        Returns:
            Algorithm metadata if found, None otherwise
        """
        return cls._registry.get(slug)

    @classmethod
    def get_by_category(cls, category: AlgorithmCategory) -> List[AlgorithmMetadata]:
        """Get all algorithms in a specific category.

        Args:
            category: Algorithm category to filter by

        Returns:
            List of algorithm metadata in the specified category
        """
        return [
            algo for algo in cls._registry.values()
            if algo.category == category
        ]

    @classmethod
    def get_all(cls) -> List[AlgorithmMetadata]:
        """Get all registered algorithms.

        Returns:
            List of all registered algorithm metadata
        """
        return list(cls._registry.values())

    @classmethod
    def search(cls, query: str) -> List[AlgorithmMetadata]:
        """Search algorithms by name, tags, or description.

        Performs case-insensitive search across algorithm names,
        descriptions, and tags.

        Args:
            query: Search query string

        Returns:
            List of matching algorithm metadata
        """
        query_lower = query.lower()
        return [
            algo for algo in cls._registry.values()
            if query_lower in algo.name.lower()
            or query_lower in algo.description.lower()
            or any(query_lower in tag.lower() for tag in algo.tags)
        ]

    @classmethod
    def clear(cls) -> None:
        """Clear the registry.

        Primarily used for testing to ensure clean state between tests.
        """
        cls._registry.clear()


def create_algorithm_metadata(
    name: str,
    slug: str,
    category: str,
    description: str,
    difficulty: str,
    **kwargs: Any
) -> AlgorithmMetadata:
    """Helper function to create algorithm metadata with defaults.

    This convenience function simplifies metadata creation by handling
    enum conversions and providing sensible defaults.

    Args:
        name: Display name of the algorithm
        slug: URL-safe identifier
        category: Category name (will be converted to AlgorithmCategory)
        description: Brief description
        difficulty: Difficulty level (will be converted to DifficultyLevel)
        **kwargs: Additional metadata fields

    Returns:
        Configured AlgorithmMetadata instance

    Example:
        >>> metadata = create_algorithm_metadata(
        ...     name="Linear Regression",
        ...     slug="linear-regression",
        ...     category="ml",
        ...     description="A simple regression algorithm",
        ...     difficulty="Beginner",
        ...     tags=["regression", "supervised"],
        ...     complexity=AlgorithmComplexity(time="O(n)", space="O(1)")
        ... )
    """
    return AlgorithmMetadata(
        id=slug,
        name=name,
        slug=slug,
        category=AlgorithmCategory(category),
        description=description,
        difficulty=DifficultyLevel(difficulty),
        **kwargs
    )
