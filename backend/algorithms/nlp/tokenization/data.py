"""Sample texts and data for Tokenization demonstration."""

from typing import List, Dict, Any


def get_sample_texts() -> List[str]:
    """Get a diverse collection of sample texts for tokenization demonstration.

    Returns a collection of 20+ texts including simple sentences, complex paragraphs,
    code snippets, multilingual examples, and texts with special characters to
    showcase different tokenization strategies.

    Returns:
        List of sample text strings
    """
    return [
        # Simple sentences
        "The quick brown fox jumps over the lazy dog.",

        "Hello, world! How are you doing today?",

        "Machine learning is transforming the world of technology.",

        # Complex sentences
        "Despite the challenging circumstances, she persevered and achieved remarkable success "
        "through determination, hard work, and unwavering commitment to her goals.",

        "The scientist's groundbreaking research on artificial intelligence, published in Nature "
        "last month, has revolutionized our understanding of neural networks and deep learning.",

        # Technical/code-like text
        "def tokenize(text): return text.split() # Simple tokenization function",

        "SELECT * FROM users WHERE email='user@example.com' AND status='active';",

        "import numpy as np\nfrom sklearn.feature_extraction.text import TfidfVectorizer",

        # Text with punctuation and special characters
        "Wait... what?! I can't believe it's already 2025! Time flies so fast!!!",

        "She said, 'It's amazing!' Then he replied: 'Indeed, it is remarkable.'",

        "Email me at john.doe@example.com or call (555) 123-4567 for more info.",

        # Social media / informal text
        "OMG this is sooooo cool!! 😊 Can't wait to try it out #NLP #MachineLearning",

        "Just finished reading an AMAZING book 10/10 would recommend to everyone!!!",

        # Multilingual examples
        "Hello, Bonjour, Hola, Ciao, Hallo, Привет, こんにちは, 你好!",

        "café, naïve, résumé, façade - words with accents and diacritics.",

        # Numbers and mixed content
        "The company's revenue increased by 25.7% in Q4 2024, reaching $1.5 billion.",

        "There are 365 days in a year, except for leap years which have 366 days.",

        # Long, complex paragraph
        "Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses "
        "on enabling computers to understand, interpret, and generate human language. It combines "
        "computational linguistics—rule-based modeling of human language—with statistical, machine "
        "learning, and deep learning models. These technologies enable computers to process human "
        "language in the form of text or voice data and to 'understand' its full meaning, complete "
        "with the speaker's or writer's intent and sentiment.",

        # Scientific text
        "The mitochondria, often referred to as the 'powerhouse of the cell,' generate adenosine "
        "triphosphate (ATP) through oxidative phosphorylation, a process critical for cellular "
        "energy production and metabolic regulation.",

        # Literary/poetic text
        "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles.",

        # News headline style
        "BREAKING: Tech Giant Announces Revolutionary AI Breakthrough - Stock Prices Surge 15% "
        "After Hours Following Surprise Product Launch Event in Silicon Valley.",
    ]


def get_text_info(text_index: int) -> Dict[str, Any]:
    """Get information about a specific sample text.

    Args:
        text_index: Index of the sample text (0-19)

    Returns:
        Dictionary containing text metadata
    """
    texts = get_sample_texts()

    if not 0 <= text_index < len(texts):
        raise ValueError(f"text_index must be between 0 and {len(texts) - 1}")

    text = texts[text_index]

    # Categorize text type
    categories = [
        "Simple sentence",
        "Greeting",
        "Statement",
        "Complex sentence",
        "Scientific text",
        "Code snippet",
        "SQL query",
        "Python import",
        "Expressive text",
        "Quoted dialogue",
        "Contact information",
        "Social media post",
        "Review/recommendation",
        "Multilingual text",
        "Accented words",
        "Financial data",
        "Factual information",
        "Technical paragraph",
        "Scientific description",
        "Literary quote",
        "News headline"
    ]

    category = categories[text_index] if text_index < len(categories) else "General text"

    return {
        "index": text_index,
        "category": category,
        "length": len(text),
        "word_count_estimate": len(text.split()),
        "preview": text[:100] + "..." if len(text) > 100 else text
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get comprehensive dataset information for the API.

    Returns:
        Dictionary with dataset details and statistics
    """
    texts = get_sample_texts()

    total_chars = sum(len(text) for text in texts)
    total_words = sum(len(text.split()) for text in texts)

    return {
        "name": "Diverse Tokenization Sample Texts",
        "description": "Collection of 20+ texts showcasing various tokenization challenges",
        "size": len(texts),
        "total_characters": total_chars,
        "total_words_estimate": total_words,
        "avg_text_length": round(total_chars / len(texts), 1),
        "text_categories": [
            "Simple sentences",
            "Complex paragraphs",
            "Code snippets",
            "SQL queries",
            "Punctuation-heavy text",
            "Social media posts",
            "Multilingual examples",
            "Scientific text",
            "Literary quotes",
            "News headlines"
        ],
        "features": [
            "Diverse vocabulary and writing styles",
            "Special characters and punctuation",
            "Code and technical content",
            "Multilingual and accented characters",
            "Various text lengths and complexities"
        ],
        "use_cases": [
            "NLP preprocessing",
            "Search indexing",
            "Language modeling",
            "Machine translation",
            "Text analysis"
        ]
    }


def validate_custom_text(text: str) -> None:
    """Validate custom text input.

    Args:
        text: Custom text string

    Raises:
        ValueError: If text is invalid
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if len(text) < 3:
        raise ValueError("Text must contain at least 3 characters")

    if len(text) > 10000:
        raise ValueError("Text cannot exceed 10,000 characters")
