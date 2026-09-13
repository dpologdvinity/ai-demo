"""
Genetic Algorithm optimizer package.

This package provides a real-valued genetic algorithm for continuous
optimization of multi-modal functions.
"""

from .model import GeneticAlgorithmSolver
from .schema import GeneticAlgorithmRequest, GeneticAlgorithmResponse
from .data import get_ga_info

__all__ = [
    'GeneticAlgorithmSolver',
    'GeneticAlgorithmRequest',
    'GeneticAlgorithmResponse',
    'get_ga_info'
]
