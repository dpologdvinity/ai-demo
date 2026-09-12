"""Part-of-Speech Tagging model implementation using spaCy and NLTK."""

import time
import logging
from typing import List, Dict, Any, Optional
from collections import Counter
import spacy
from spacy.language import Language

from .schema import (
    POSTaggingParameters,
    POSTaggingResponse,
    TaggedWord,
    DependencyEdge
)
from .data import get_sample_sentences, get_pos_tag_descriptions, get_pos_tag_colors

logger = logging.getLogger(__name__)


class POSTaggingModel:
    """Part-of-Speech Tagging model using spaCy.

    Performs POS tagging with support for fine-grained (Penn Treebank) and
    coarse-grained (Universal) tag schemes, along with dependency parsing.
    """

    def __init__(self):
        """Initialize POS tagging model."""
        self._nlp: Optional[Language] = None
        self._load_spacy_model()

    def _load_spacy_model(self) -> None:
        """Load the spaCy model for POS tagging."""
        try:
            self._nlp = spacy.load('en_core_web_sm')
            logger.info("Loaded spaCy model for POS tagging: en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model not found. "
                "It will be loaded on first use or raise an error."
            )

    def _ensure_model_loaded(self) -> Language:
        """Ensure spaCy model is loaded.

        Returns:
            Loaded spaCy model

        Raises:
            ValueError: If model cannot be loaded
        """
        if self._nlp is None:
            try:
                logger.info("Loading spaCy model: en_core_web_sm")
                self._nlp = spacy.load('en_core_web_sm')
            except OSError as e:
                raise ValueError(
                    "Could not load spaCy model 'en_core_web_sm'. "
                    "Please ensure it is installed using: "
                    "python -m spacy download en_core_web_sm"
                )
        return self._nlp

    def tag_text(
        self,
        text: str,
        show_fine_grained: bool = True,
        show_dependencies: bool = True,
        tag_scheme: str = 'penn'
    ) -> Dict[str, Any]:
        """Perform POS tagging on text.

        Args:
            text: Text to tag
            show_fine_grained: Whether to show fine-grained tags
            show_dependencies: Whether to include dependency parsing
            tag_scheme: Tag scheme to use ('penn' or 'universal')

        Returns:
            Dictionary with tagged words and dependency information
        """
        nlp = self._ensure_model_loaded()
        doc = nlp(text)

        # Get tag descriptions
        tag_descriptions = get_pos_tag_descriptions()

        # Tag each word
        tagged_words = []
        for i, token in enumerate(doc):
            # Determine which tag to display based on settings
            if tag_scheme == 'universal' or not show_fine_grained:
                primary_tag = token.pos_
            else:
                primary_tag = token.tag_

            # Get description
            description = tag_descriptions.get(
                primary_tag,
                {'full': primary_tag, 'description': 'Unknown tag'}
            )

            tagged_word = TaggedWord(
                text=token.text,
                pos_coarse=token.pos_,  # Universal POS
                pos_fine=token.tag_,     # Penn Treebank tag
                tag=primary_tag,
                description=f"{description['full']}: {description['description']}",
                index=i,
                lemma=token.lemma_,
                is_stop=token.is_stop
            )

            # Add dependency information if requested
            if show_dependencies:
                tagged_word.dependency = token.dep_
                tagged_word.head_text = token.head.text
                tagged_word.head_index = token.head.i

            tagged_words.append(tagged_word)

        # Extract dependency edges for visualization
        dependency_edges = []
        if show_dependencies:
            for token in doc:
                if token.dep_ != 'ROOT':  # Skip root node
                    dependency_edges.append(DependencyEdge(
                        source=token.head.i,
                        target=token.i,
                        label=token.dep_,
                        source_text=token.head.text,
                        target_text=token.text
                    ))

        return {
            'tagged_words': tagged_words,
            'dependency_edges': dependency_edges,
            'doc': doc
        }

    def process(self, params: POSTaggingParameters) -> POSTaggingResponse:
        """Process POS tagging request.

        Args:
            params: POS tagging parameters

        Returns:
            POSTaggingResponse with tagging results
        """
        start_time = time.time()

        try:
            # Get text to analyze
            if params.use_custom_text and params.custom_text:
                text = params.custom_text
                text_source = "custom"
            else:
                # Use sample sentence
                sentences = get_sample_sentences()
                if params.text_index >= len(sentences):
                    raise ValueError(
                        f"text_index {params.text_index} out of range. "
                        f"Must be between 0 and {len(sentences) - 1}"
                    )
                sentence_data = sentences[params.text_index]
                text = sentence_data['text']
                text_source = f"sample_{params.text_index}"

            # Perform POS tagging
            result = self.tag_text(
                text=text,
                show_fine_grained=params.show_fine_grained,
                show_dependencies=params.show_dependencies,
                tag_scheme=params.tag_scheme
            )

            tagged_words = result['tagged_words']
            dependency_edges = result['dependency_edges']

            # Calculate POS distribution (using primary tags)
            pos_counts = Counter([word.tag for word in tagged_words])
            pos_distribution = dict(pos_counts)

            # Get top tag frequencies
            tag_frequencies = [
                {'tag': tag, 'count': count, 'percentage': (count / len(tagged_words)) * 100}
                for tag, count in pos_counts.most_common(10)
            ]

            # Calculate metrics
            unique_tags = len(pos_counts)
            total_words = len(tagged_words)
            avg_word_length = sum(len(word.text) for word in tagged_words) / total_words

            metrics = {
                'total_words': total_words,
                'unique_pos_tags': unique_tags,
                'avg_word_length': round(avg_word_length, 2),
                'dependency_edges': len(dependency_edges),
                'text_source': text_source
            }

            # Create visualization data
            pos_colors = get_pos_tag_colors()

            # Prepare pie chart data for POS distribution
            pie_data = {
                'labels': list(pos_distribution.keys()),
                'values': list(pos_distribution.values()),
                'colors': [pos_colors.get(tag, '#CCCCCC') for tag in pos_distribution.keys()]
            }

            # Prepare bar chart data for tag frequencies
            bar_data = {
                'labels': [item['tag'] for item in tag_frequencies],
                'values': [item['count'] for item in tag_frequencies],
                'colors': [pos_colors.get(item['tag'], '#CCCCCC') for item in tag_frequencies]
            }

            # Prepare dependency tree data
            dependency_tree = {
                'nodes': [
                    {
                        'id': word.index,
                        'label': word.text,
                        'tag': word.tag,
                        'color': pos_colors.get(word.pos_coarse, '#CCCCCC')
                    }
                    for word in tagged_words
                ],
                'edges': [
                    {
                        'source': edge.source,
                        'target': edge.target,
                        'label': edge.label
                    }
                    for edge in dependency_edges
                ]
            }

            visualization_data = {
                'pos_distribution': pie_data,
                'tag_frequencies': bar_data,
                'dependency_tree': dependency_tree,
                'pos_colors': pos_colors
            }

            # Execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Build response
            return POSTaggingResponse(
                success=True,
                metrics=metrics,
                tagged_words=tagged_words,
                pos_distribution=pos_distribution,
                tag_frequencies=tag_frequencies,
                dependency_edges=dependency_edges,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used={
                    'tagger': params.tagger,
                    'text_index': params.text_index,
                    'show_fine_grained': params.show_fine_grained,
                    'show_dependencies': params.show_dependencies,
                    'tag_scheme': params.tag_scheme,
                    'use_custom_text': params.use_custom_text
                },
                model_info={
                    'library': 'spaCy',
                    'model': 'en_core_web_sm',
                    'tag_scheme': params.tag_scheme,
                    'supports_dependencies': True
                },
                text_analyzed=text
            )

        except ValueError as e:
            logger.error(f"Validation error in POS tagging: {str(e)}")
            return POSTaggingResponse(
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Error in POS tagging: {str(e)}", exc_info=True)
            return POSTaggingResponse(
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error=f"POS tagging failed: {str(e)}"
            )
