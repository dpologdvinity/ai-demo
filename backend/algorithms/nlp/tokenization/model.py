"""Tokenization model implementation with multiple tokenization strategies."""

import time
import re
import string
from typing import List, Dict, Any, Optional
from collections import Counter
import logging

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# spaCy for advanced tokenization
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Transformers for BPE and WordPiece
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .schema import TokenizationRequest, TokenizationResponse, TokenizerResult, TokenInfo
from .data import get_sample_texts, validate_custom_text

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TokenizationModel:
    """Multi-strategy tokenization model."""

    def __init__(self):
        """Initialize the tokenization model."""
        self.stop_words = set(stopwords.words('english'))
        self.spacy_nlp = None
        self.bert_tokenizer = None
        self.gpt_tokenizer = None

        # Load spaCy model if available
        if SPACY_AVAILABLE:
            try:
                self.spacy_nlp = spacy.load('en_core_web_sm')
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")

        # Load transformer tokenizers if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
            except Exception as e:
                logger.warning(f"Failed to load transformer tokenizers: {e}")

    def tokenize(self, request: TokenizationRequest) -> TokenizationResponse:
        """
        Tokenize text using specified strategy or compare all strategies.

        Args:
            request: TokenizationRequest with tokenization parameters

        Returns:
            TokenizationResponse with tokenization results

        Raises:
            ValueError: If parameters or text are invalid
        """
        start_time = time.time()

        try:
            # Get text to tokenize
            if request.custom_text:
                validate_custom_text(request.custom_text)
                text = request.custom_text
                logger.info("Using custom text for tokenization")
            else:
                texts = get_sample_texts()
                text = texts[request.text_index]
                logger.info(f"Using sample text {request.text_index}")

            # Preprocess text
            processed_text = self._preprocess_text(
                text,
                lowercase=request.lowercase,
                remove_punctuation=request.remove_punctuation
            )

            if request.compare_mode:
                # Compare all tokenizers
                comparison_results = self._compare_all_tokenizers(
                    processed_text,
                    request
                )

                # Calculate overall statistics
                statistics = self._calculate_comparison_statistics(comparison_results)

                execution_time = (time.time() - start_time) * 1000

                return TokenizationResponse(
                    success=True,
                    original_text=text,
                    text_preview=text[:200] + "..." if len(text) > 200 else text,
                    comparison_results=comparison_results,
                    statistics=statistics,
                    execution_time_ms=execution_time,
                    parameters_used={
                        "tokenizer_type": "all (comparison mode)",
                        "lowercase": request.lowercase,
                        "remove_punctuation": request.remove_punctuation,
                        "remove_stopwords": request.remove_stopwords,
                        "max_tokens": request.max_tokens
                    }
                )
            else:
                # Single tokenizer mode
                result = self._tokenize_with_strategy(
                    processed_text,
                    request.tokenizer_type,
                    request
                )

                # Calculate frequency distribution
                freq_dist = self._prepare_frequency_distribution(result)

                execution_time = (time.time() - start_time) * 1000

                return TokenizationResponse(
                    success=True,
                    original_text=text,
                    text_preview=text[:200] + "..." if len(text) > 200 else text,
                    main_result=result,
                    statistics={
                        "token_count": result.token_count,
                        "unique_tokens": result.unique_token_count,
                        "avg_token_length": result.avg_token_length,
                        "vocabulary_size": result.vocabulary_size
                    },
                    frequency_distribution=freq_dist,
                    execution_time_ms=execution_time,
                    parameters_used={
                        "tokenizer_type": request.tokenizer_type,
                        "lowercase": request.lowercase,
                        "remove_punctuation": request.remove_punctuation,
                        "remove_stopwords": request.remove_stopwords,
                        "max_tokens": request.max_tokens
                    }
                )

        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}", exc_info=True)
            return TokenizationResponse(
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def _preprocess_text(
        self,
        text: str,
        lowercase: bool,
        remove_punctuation: bool
    ) -> str:
        """Preprocess text before tokenization.

        Args:
            text: Input text
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation

        Returns:
            Preprocessed text
        """
        if lowercase:
            text = text.lower()

        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        return text

    def _tokenize_with_strategy(
        self,
        text: str,
        strategy: str,
        request: TokenizationRequest
    ) -> TokenizerResult:
        """Tokenize text with a specific strategy.

        Args:
            text: Text to tokenize
            strategy: Tokenization strategy
            request: Original request with parameters

        Returns:
            TokenizerResult with tokens and statistics
        """
        if strategy == 'whitespace':
            return self._whitespace_tokenize(text, request)
        elif strategy == 'word':
            return self._word_tokenize(text, request)
        elif strategy == 'sentence':
            return self._sentence_tokenize(text, request)
        elif strategy == 'character':
            return self._character_tokenize(text, request)
        elif strategy == 'wordpiece':
            return self._wordpiece_tokenize(text, request)
        elif strategy == 'bpe':
            return self._bpe_tokenize(text, request)
        elif strategy == 'spacy':
            return self._spacy_tokenize(text, request)
        else:
            raise ValueError(f"Unknown tokenizer strategy: {strategy}")

    def _whitespace_tokenize(
        self,
        text: str,
        request: TokenizationRequest
    ) -> TokenizerResult:
        """Simple whitespace tokenization.

        Args:
            text: Text to tokenize
            request: Request parameters

        Returns:
            TokenizerResult
        """
        tokens_raw = text.split()

        # Apply filters
        if request.remove_stopwords:
            tokens_raw = [t for t in tokens_raw if t.lower() not in self.stop_words]

        # Limit tokens
        tokens_raw = tokens_raw[:request.max_tokens]

        # Create token info objects
        tokens = []
        char_pos = 0
        for i, token_text in enumerate(tokens_raw):
            # Find position in original text
            start = text.find(token_text, char_pos)
            end = start + len(token_text) if start != -1 else None
            char_pos = end if end else char_pos

            tokens.append(TokenInfo(
                text=token_text,
                index=i,
                start_char=start if start != -1 else None,
                end_char=end,
                is_stopword=token_text.lower() in self.stop_words
            ))

        # Calculate statistics
        token_texts = [t.text for t in tokens]
        return self._create_result(
            'Whitespace',
            tokens,
            token_texts
        )

    def _word_tokenize(
        self,
        text: str,
        request: TokenizationRequest
    ) -> TokenizerResult:
        """NLTK word tokenization.

        Args:
            text: Text to tokenize
            request: Request parameters

        Returns:
            TokenizerResult
        """
        tokens_raw = word_tokenize(text)

        # Apply filters
        if request.remove_stopwords:
            tokens_raw = [t for t in tokens_raw if t.lower() not in self.stop_words]

        # Limit tokens
        tokens_raw = tokens_raw[:request.max_tokens]

        # Create token info objects
        tokens = []
        for i, token_text in enumerate(tokens_raw):
            tokens.append(TokenInfo(
                text=token_text,
                index=i,
                is_stopword=token_text.lower() in self.stop_words
            ))

        token_texts = [t.text for t in tokens]
        return self._create_result(
            'Word (NLTK)',
            tokens,
            token_texts
        )

    def _sentence_tokenize(
        self,
        text: str,
        request: TokenizationRequest
    ) -> TokenizerResult:
        """NLTK sentence tokenization.

        Args:
            text: Text to tokenize
            request: Request parameters

        Returns:
            TokenizerResult
        """
        sentences = sent_tokenize(text)
        sentences = sentences[:request.max_tokens]

        tokens = []
        for i, sentence in enumerate(sentences):
            tokens.append(TokenInfo(
                text=sentence,
                index=i
            ))

        token_texts = [t.text for t in tokens]
        return self._create_result(
            'Sentence (NLTK)',
            tokens,
            token_texts
        )

    def _character_tokenize(
        self,
        text: str,
        request: TokenizationRequest
    ) -> TokenizerResult:
        """Character-level tokenization.

        Args:
            text: Text to tokenize
            request: Request parameters

        Returns:
            TokenizerResult
        """
        chars = list(text)[:request.max_tokens]

        tokens = []
        for i, char in enumerate(chars):
            tokens.append(TokenInfo(
                text=char,
                index=i,
                start_char=i,
                end_char=i + 1
            ))

        token_texts = [t.text for t in tokens]
        return self._create_result(
            'Character',
            tokens,
            token_texts
        )

    def _wordpiece_tokenize(
        self,
        text: str,
        request: TokenizationRequest
    ) -> TokenizerResult:
        """WordPiece tokenization using BERT tokenizer.

        Args:
            text: Text to tokenize
            request: Request parameters

        Returns:
            TokenizerResult
        """
        if not TRANSFORMERS_AVAILABLE or self.bert_tokenizer is None:
            # Fallback to word tokenization
            logger.warning("BERT tokenizer not available, using word tokenization")
            return self._word_tokenize(text, request)

        # Tokenize with BERT
        encoded = self.bert_tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )

        tokens_raw = self.bert_tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        offsets = encoded.get('offset_mapping', [])

        # Limit tokens
        tokens_raw = tokens_raw[:request.max_tokens]
        offsets = offsets[:request.max_tokens]

        tokens = []
        for i, (token_text, offset) in enumerate(zip(tokens_raw, offsets)):
            start_char, end_char = offset if offset else (None, None)
            tokens.append(TokenInfo(
                text=token_text,
                index=i,
                start_char=start_char,
                end_char=end_char
            ))

        token_texts = [t.text for t in tokens]
        return self._create_result(
            'WordPiece (BERT)',
            tokens,
            token_texts
        )

    def _bpe_tokenize(
        self,
        text: str,
        request: TokenizationRequest
    ) -> TokenizerResult:
        """Byte Pair Encoding (BPE) tokenization using GPT-2 tokenizer.

        Args:
            text: Text to tokenize
            request: Request parameters

        Returns:
            TokenizerResult
        """
        if not TRANSFORMERS_AVAILABLE or self.gpt_tokenizer is None:
            # Fallback to word tokenization
            logger.warning("GPT tokenizer not available, using word tokenization")
            return self._word_tokenize(text, request)

        # Tokenize with GPT-2
        encoded = self.gpt_tokenizer(
            text,
            add_special_tokens=False
        )

        tokens_raw = self.gpt_tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        tokens_raw = tokens_raw[:request.max_tokens]

        tokens = []
        for i, token_text in enumerate(tokens_raw):
            tokens.append(TokenInfo(
                text=token_text,
                index=i
            ))

        token_texts = [t.text for t in tokens]
        return self._create_result(
            'BPE (GPT-2)',
            tokens,
            token_texts
        )

    def _spacy_tokenize(
        self,
        text: str,
        request: TokenizationRequest
    ) -> TokenizerResult:
        """spaCy tokenization with linguistic features.

        Args:
            text: Text to tokenize
            request: Request parameters

        Returns:
            TokenizerResult
        """
        if not SPACY_AVAILABLE or self.spacy_nlp is None:
            # Fallback to word tokenization
            logger.warning("spaCy not available, using word tokenization")
            return self._word_tokenize(text, request)

        doc = self.spacy_nlp(text)

        tokens_raw = list(doc)

        # Apply filters
        if request.remove_stopwords:
            tokens_raw = [t for t in tokens_raw if not t.is_stop]

        # Limit tokens
        tokens_raw = tokens_raw[:request.max_tokens]

        tokens = []
        for i, token in enumerate(tokens_raw):
            tokens.append(TokenInfo(
                text=token.text,
                index=i,
                start_char=token.idx,
                end_char=token.idx + len(token.text),
                is_stopword=token.is_stop,
                pos_tag=token.pos_
            ))

        token_texts = [t.text for t in tokens]
        return self._create_result(
            'spaCy',
            tokens,
            token_texts
        )

    def _create_result(
        self,
        tokenizer_name: str,
        tokens: List[TokenInfo],
        token_texts: List[str]
    ) -> TokenizerResult:
        """Create a TokenizerResult from tokens.

        Args:
            tokenizer_name: Name of the tokenizer
            tokens: List of TokenInfo objects
            token_texts: List of token text strings

        Returns:
            TokenizerResult
        """
        # Calculate statistics
        token_count = len(tokens)
        unique_tokens = len(set(token_texts))
        avg_length = sum(len(t) for t in token_texts) / token_count if token_count > 0 else 0

        # Calculate frequency distribution (top 20)
        freq_counter = Counter(token_texts)
        token_frequency = [
            {"token": token, "frequency": count}
            for token, count in freq_counter.most_common(20)
        ]

        return TokenizerResult(
            tokenizer_name=tokenizer_name,
            tokens=tokens,
            token_count=token_count,
            unique_token_count=unique_tokens,
            avg_token_length=round(avg_length, 2),
            vocabulary_size=unique_tokens,
            token_frequency=token_frequency
        )

    def _compare_all_tokenizers(
        self,
        text: str,
        request: TokenizationRequest
    ) -> List[TokenizerResult]:
        """Compare all available tokenizers.

        Args:
            text: Text to tokenize
            request: Request parameters

        Returns:
            List of TokenizerResult for each tokenizer
        """
        strategies = ['whitespace', 'word', 'sentence', 'character', 'wordpiece', 'bpe', 'spacy']
        results = []

        for strategy in strategies:
            try:
                result = self._tokenize_with_strategy(text, strategy, request)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to tokenize with {strategy}: {e}")

        return results

    def _calculate_comparison_statistics(
        self,
        results: List[TokenizerResult]
    ) -> Dict[str, Any]:
        """Calculate statistics across all tokenizers.

        Args:
            results: List of TokenizerResult objects

        Returns:
            Dictionary with comparison statistics
        """
        if not results:
            return {}

        return {
            "tokenizer_count": len(results),
            "token_counts": {
                result.tokenizer_name: result.token_count
                for result in results
            },
            "unique_token_counts": {
                result.tokenizer_name: result.unique_token_count
                for result in results
            },
            "avg_token_lengths": {
                result.tokenizer_name: result.avg_token_length
                for result in results
            },
            "vocabulary_sizes": {
                result.tokenizer_name: result.vocabulary_size
                for result in results
            }
        }

    def _prepare_frequency_distribution(
        self,
        result: TokenizerResult
    ) -> Dict[str, Any]:
        """Prepare frequency distribution data for visualization.

        Args:
            result: TokenizerResult

        Returns:
            Dictionary with chart data
        """
        freq_data = result.token_frequency[:20]  # Top 20

        return {
            "tokens": [item["token"] for item in freq_data],
            "frequencies": [item["frequency"] for item in freq_data],
            "title": f"Top 20 Token Frequencies ({result.tokenizer_name})",
            "x_axis_label": "Frequency",
            "y_axis_label": "Token"
        }
