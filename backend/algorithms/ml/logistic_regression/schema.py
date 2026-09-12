"""Metadata and schema definitions for Logistic Regression.

This module defines the algorithm metadata including parameters,
complexity analysis, and educational content.
"""

from utils.algorithm_metadata import (
    AlgorithmMetadata,
    AlgorithmParameter,
    AlgorithmComplexity,
    AlgorithmCategory,
    DifficultyLevel,
    AlgorithmRegistry
)


def get_metadata() -> AlgorithmMetadata:
    """Get complete metadata for Logistic Regression algorithm.

    Returns:
        AlgorithmMetadata instance with full algorithm information
    """
    metadata = AlgorithmMetadata(
        id='logistic-regression',
        name='Logistic Regression',
        slug='logistic-regression',
        category=AlgorithmCategory.ML,
        description='Supervised learning algorithm for binary and multiclass classification problems',
        difficulty=DifficultyLevel.BEGINNER,
        tags=['supervised', 'classification', 'linear-models', 'probabilistic'],
        use_cases=[
            'Email spam detection',
            'Disease diagnosis',
            'Customer churn prediction',
            'Credit risk assessment',
            'Sentiment analysis'
        ],
        complexity=AlgorithmComplexity(
            time='O(n²)',
            space='O(n)'
        ),
        parameters=[
            AlgorithmParameter(
                name='C',
                label='Regularization Strength (C)',
                type='range',
                default=1.0,
                min=0.01,
                max=10.0,
                step=0.1,
                description='Inverse of regularization strength. Smaller values specify stronger regularization.'
            ),
            AlgorithmParameter(
                name='penalty',
                label='Penalty Type',
                type='select',
                default='l2',
                options=[
                    {'value': 'l1', 'label': 'L1 (Lasso)'},
                    {'value': 'l2', 'label': 'L2 (Ridge)'}
                ],
                description='Type of regularization to apply. L1 promotes sparsity, L2 promotes smaller weights.'
            ),
            AlgorithmParameter(
                name='max_iter',
                label='Maximum Iterations',
                type='number',
                default=100,
                min=50,
                max=500,
                step=10,
                description='Maximum number of iterations for the solver to converge.'
            ),
            AlgorithmParameter(
                name='solver',
                label='Solver Algorithm',
                type='select',
                default='lbfgs',
                options=[
                    {'value': 'lbfgs', 'label': 'LBFGS (L2 only)'},
                    {'value': 'liblinear', 'label': 'Liblinear (L1/L2, small datasets)'},
                    {'value': 'saga', 'label': 'SAGA (L1/L2, large datasets)'}
                ],
                description='Optimization algorithm to use. LBFGS is good for small datasets, SAGA for large ones.'
            )
        ],
        dataset_name='iris',
        visualization_type='scatter_plot_confusion_matrix',
        theory=(
            'Logistic Regression is a statistical model that uses a logistic function to model '
            'a binary or multiclass dependent variable. Despite its name, it is used for classification, '
            'not regression.\n\n'
            'The model estimates the probability that an instance belongs to a particular class using '
            'the sigmoid (logistic) function:\n\n'
            'P(y=1|x) = 1 / (1 + e^(-z))\n\n'
            'where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ\n\n'
            'For multiclass problems, logistic regression extends to multinomial (softmax) regression, '
            'which uses the softmax function to compute probabilities for each class.'
        ),
        pros=[
            'Simple and interpretable model',
            'Outputs calibrated probabilities',
            'Fast training and prediction',
            'Works well with linearly separable data',
            'Less prone to overfitting with regularization',
            'No hyperparameter tuning required for basic use'
        ],
        cons=[
            'Assumes linear decision boundary',
            'Poor performance on non-linear problems',
            'Sensitive to feature scaling',
            'May underfit with high-dimensional data',
            'Requires feature engineering for complex patterns'
        ],
        related_algorithms=[
            'linear-regression',
            'support-vector-machine',
            'naive-bayes',
            'decision-tree'
        ]
    )

    return metadata


def register_algorithm():
    """Register Logistic Regression in the global algorithm registry."""
    metadata = get_metadata()
    AlgorithmRegistry.register(metadata)
