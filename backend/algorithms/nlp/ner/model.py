"""Named Entity Recognition model implementation using spaCy."""

import time
import logging
from typing import List, Dict, Any, Optional
import spacy
from spacy.language import Language

from .schema import NERParameters, NERResponse, Entity, AnnotatedSpan
from .data import get_default_texts, get_entity_type_info

logger = logging.getLogger(__name__)


class NERModel:
    """Named Entity Recognition model using spaCy.

    Uses pre-trained spaCy models to extract named entities from text.
    Supports filtering by entity type and confidence threshold.
    """

    def __init__(self):
        """Initialize NER model."""
        self._models: Dict[str, Language] = {}
        self._load_default_model()

    def _load_default_model(self) -> None:
        """Load the default spaCy model."""
        try:
            self._models['en_core_web_sm'] = spacy.load('en_core_web_sm')
            logger.info("Loaded default spaCy model: en_core_web_sm")
        except OSError:
            logger.warning(
                "Default spaCy model not found. "
                "It will be loaded on first use."
            )

    def _get_model(self, model_name: str) -> Language:
        """Get or load a spaCy model.

        Args:
            model_name: Name of the spaCy model

        Returns:
            Loaded spaCy model

        Raises:
            ValueError: If model cannot be loaded
        """
        if model_name not in self._models:
            try:
                logger.info(f"Loading spaCy model: {model_name}")
                self._models[model_name] = spacy.load(model_name)
            except OSError as e:
                raise ValueError(
                    f"Could not load spaCy model '{model_name}'. "
                    f"Please ensure it is installed. Error: {str(e)}"
                )

        return self._models[model_name]

    def extract_entities(
        self,
        text: str,
        model_name: str = 'en_core_web_sm',
        entity_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.0,
        merge_entities: bool = True
    ) -> List[Entity]:
        """Extract named entities from text.

        Args:
            text: Text to extract entities from
            model_name: spaCy model to use
            entity_types: List of entity types to extract (None = all types)
            confidence_threshold: Minimum confidence score
            merge_entities: Whether to merge adjacent entities of same type

        Returns:
            List of extracted entities
        """
        nlp = self._get_model(model_name)
        doc = nlp(text)

        entities = []
        for ent in doc.ents:
            # Filter by entity type if specified
            if entity_types and ent.label_ not in entity_types:
                continue

            # Filter by confidence threshold (spaCy doesn't provide confidence scores
            # by default, so we use 1.0 for all entities from pre-trained models)
            confidence = 1.0

            if confidence >= confidence_threshold:
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=confidence
                ))

        # Merge adjacent entities of the same type if requested
        if merge_entities and len(entities) > 1:
            entities = self._merge_adjacent_entities(entities, text)

        return entities

    def _merge_adjacent_entities(
        self,
        entities: List[Entity],
        text: str
    ) -> List[Entity]:
        """Merge adjacent entities of the same type.

        Args:
            entities: List of entities to merge
            text: Original text

        Returns:
            List of entities with adjacent same-type entities merged
        """
        if not entities:
            return entities

        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: e.start)
        merged = []
        current = sorted_entities[0]

        for next_entity in sorted_entities[1:]:
            # Check if entities are adjacent (within 2 chars, allowing for space/punctuation)
            # and have the same label
            if (current.label == next_entity.label and
                next_entity.start - current.end <= 2):
                # Merge entities
                current = Entity(
                    text=text[current.start:next_entity.end],
                    label=current.label,
                    start=current.start,
                    end=next_entity.end,
                    confidence=(current.confidence + next_entity.confidence) / 2
                )
            else:
                # Entities not adjacent or different types, add current and move to next
                merged.append(current)
                current = next_entity

        # Add the last entity
        merged.append(current)

        return merged

    def create_annotated_spans(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[AnnotatedSpan]:
        """Create annotated spans for visualization.

        Breaks text into spans, marking entity spans with their labels.

        Args:
            text: Original text
            entities: List of extracted entities

        Returns:
            List of annotated spans
        """
        if not entities:
            # Return entire text as single non-entity span
            return [AnnotatedSpan(
                text=text,
                label=None,
                start=0,
                end=len(text)
            )]

        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda e: e.start)

        spans = []
        last_end = 0

        for entity in sorted_entities:
            # Add non-entity span before this entity
            if entity.start > last_end:
                spans.append(AnnotatedSpan(
                    text=text[last_end:entity.start],
                    label=None,
                    start=last_end,
                    end=entity.start
                ))

            # Add entity span
            spans.append(AnnotatedSpan(
                text=entity.text,
                label=entity.label,
                start=entity.start,
                end=entity.end
            ))

            last_end = entity.end

        # Add remaining non-entity text
        if last_end < len(text):
            spans.append(AnnotatedSpan(
                text=text[last_end:],
                label=None,
                start=last_end,
                end=len(text)
            ))

        return spans

    def count_entities_by_type(self, entities: List[Entity]) -> Dict[str, int]:
        """Count entities by type.

        Args:
            entities: List of extracted entities

        Returns:
            Dictionary mapping entity types to counts
        """
        counts: Dict[str, int] = {}
        for entity in entities:
            counts[entity.label] = counts.get(entity.label, 0) + 1

        return counts

    def process(self, params: NERParameters) -> NERResponse:
        """Process text with Named Entity Recognition.

        Args:
            params: NER parameters

        Returns:
            NER response with extracted entities and visualization data
        """
        start_time = time.time()

        try:
            # Get text to process
            if params.use_custom_text and params.custom_text:
                text = params.custom_text
                text_title = "Custom Text"
            else:
                # Use sample text based on text_index
                all_texts = get_default_texts()
                text_idx = min(params.text_index, len(all_texts) - 1)
                sample = all_texts[text_idx]
                text = sample['text']
                text_title = sample['title']

            # Extract entities
            entities = self.extract_entities(
                text=text,
                model_name=params.model_name,
                entity_types=params.entity_types,
                confidence_threshold=params.confidence_threshold,
                merge_entities=params.merge_entities
            )

            # Create annotated spans for visualization
            annotated_spans = self.create_annotated_spans(text, entities)

            # Count entities by type
            entity_distribution = self.count_entities_by_type(entities)

            # Calculate metrics
            execution_time = (time.time() - start_time) * 1000

            # Prepare visualization data
            entity_type_info = get_entity_type_info()

            # Count entity frequency (how many times each entity text appears)
            entity_text_frequency: Dict[str, int] = {}
            for entity in entities:
                entity_text_frequency[entity.text] = entity_text_frequency.get(entity.text, 0) + 1

            # Get top entities by frequency
            top_entities = sorted(
                entity_text_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 entities

            visualization_data = {
                'text_title': text_title,
                'total_entities': len(entities),
                'entity_types_found': len(entity_distribution),
                'entity_type_info': entity_type_info,
                'distribution_chart': [
                    {
                        'type': entity_type,
                        'count': count,
                        'color': entity_type_info.get(entity_type, {}).get('color', '#gray'),
                        'percentage': round(count / len(entities) * 100, 1) if entities else 0
                    }
                    for entity_type, count in sorted(
                        entity_distribution.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                ],
                'top_entities': [
                    {
                        'text': entity_text,
                        'count': count,
                        'label': next((e.label for e in entities if e.text == entity_text), 'UNKNOWN')
                    }
                    for entity_text, count in top_entities
                ]
            }

            return NERResponse(
                success=True,
                metrics={
                    'total_entities': len(entities),
                    'entity_types': len(entity_distribution),
                    'text_length': len(text),
                    'entities_per_100_chars': round(len(entities) / len(text) * 100, 2) if text else 0
                },
                entities=entities,
                entity_distribution=entity_distribution,
                annotated_text=annotated_spans,
                visualization_data=visualization_data,
                execution_time_ms=execution_time,
                parameters_used={
                    'model_name': params.model_name,
                    'entity_types': params.entity_types,
                    'confidence_threshold': params.confidence_threshold,
                    'text_index': params.text_index,
                    'merge_entities': params.merge_entities
                },
                model_info={
                    'model_name': params.model_name,
                    'spacy_version': spacy.__version__,
                    'supports_confidence': False  # Standard models don't provide confidence scores
                }
            )

        except Exception as e:
            logger.error(f"NER processing error: {str(e)}", exc_info=True)
            execution_time = (time.time() - start_time) * 1000

            return NERResponse(
                success=False,
                execution_time_ms=execution_time,
                error=f"NER processing failed: {str(e)}"
            )
