"""Data loading and preprocessing utilities for Word2Vec algorithm."""

from typing import List, Dict, Any
import re


def get_default_corpus() -> List[List[str]]:
    """Get default text corpus for Word2Vec training.

    Returns a sample corpus containing various topics including technology,
    science, business, and general knowledge. Each sentence is tokenized
    into a list of words.

    Returns:
        List of tokenized sentences (each sentence is a list of words)
    """
    # Sample corpus with diverse topics
    raw_corpus = [
        # Technology
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Neural networks are computational models inspired by the human brain structure.",
        "Deep learning uses multiple layers of neural networks to process information.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and analyze visual information from images.",
        "Algorithms are step by step instructions for solving computational problems.",
        "Data science combines statistics programming and domain expertise to extract insights.",
        "Cloud computing provides on demand access to computing resources over the internet.",
        "Software engineering involves designing developing and maintaining software systems.",
        "Artificial intelligence systems can perform tasks that typically require human intelligence.",

        # Science
        "Physics studies matter energy and the fundamental forces of nature.",
        "Chemistry explores the composition structure and properties of substances.",
        "Biology examines living organisms and their interactions with the environment.",
        "Mathematics provides the language and tools for quantitative reasoning.",
        "Astronomy investigates celestial objects and phenomena in the universe.",
        "Genetics studies heredity and variation in living organisms through DNA.",
        "Evolution explains how species change over time through natural selection.",
        "Quantum mechanics describes the behavior of matter at atomic scales.",
        "Thermodynamics studies energy transfer and the laws governing heat and work.",
        "Ecology examines relationships between organisms and their environment.",

        # Business
        "Marketing involves promoting and selling products or services to customers.",
        "Finance manages money investments and financial planning for organizations.",
        "Management coordinates resources and activities to achieve organizational goals.",
        "Economics studies production distribution and consumption of goods and services.",
        "Entrepreneurship involves creating and running new business ventures.",
        "Strategy defines long term goals and the approach to achieve competitive advantage.",
        "Innovation drives the development of new products processes and business models.",
        "Leadership inspires and guides teams toward achieving common objectives.",
        "Operations manages the production and delivery of goods and services.",
        "Analytics uses data analysis to support business decision making.",

        # General Knowledge
        "Education provides knowledge skills and values to individuals and society.",
        "Culture encompasses the beliefs customs and practices of social groups.",
        "History records and interprets past events and human civilizations.",
        "Art expresses creativity and emotions through various forms and media.",
        "Music combines sounds and rhythms to create aesthetic experiences.",
        "Literature includes written works of artistic and intellectual value.",
        "Philosophy examines fundamental questions about existence knowledge and ethics.",
        "Psychology studies mental processes and human behavior patterns.",
        "Sociology analyzes social relationships institutions and structures.",
        "Communication involves exchanging information ideas and messages between people.",

        # Additional context for better embeddings
        "Scientists use experiments and observations to test hypotheses and theories.",
        "Engineers design build and optimize systems machines and infrastructure.",
        "Developers write code to create software applications and programs.",
        "Researchers investigate questions and contribute to knowledge advancement.",
        "Teachers educate students and facilitate learning in various subjects.",
        "Doctors diagnose treat and prevent diseases and medical conditions.",
        "Artists create visual auditory and performing works of expression.",
        "Athletes compete in sports and physical activities requiring skill and training.",
        "Writers compose books articles and other written content for readers.",
        "Musicians perform compose and produce music using instruments and vocals.",

        # Technology relationships
        "Python is a programming language widely used in data science and machine learning.",
        "JavaScript enables interactive web applications and dynamic content in browsers.",
        "Databases store organize and manage large amounts of structured information.",
        "APIs allow different software systems to communicate and exchange data.",
        "Cybersecurity protects computer systems and networks from digital attacks.",
        "Blockchain technology enables secure decentralized transaction recording.",
        "Internet of things connects physical devices to networks for data exchange.",
        "Virtual reality creates immersive simulated environments for users.",
        "Augmented reality overlays digital information onto the physical world.",
        "Robotics combines mechanical engineering and artificial intelligence for automation.",
    ]

    # Tokenize each sentence
    tokenized_corpus = [preprocess_text(sentence) for sentence in raw_corpus]

    return tokenized_corpus


def preprocess_text(text: str) -> List[str]:
    """Preprocess and tokenize text.

    Converts text to lowercase, removes punctuation, and splits into words.

    Args:
        text: Raw text string

    Returns:
        List of preprocessed tokens
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Split into words
    words = text.split()

    return words


def prepare_custom_corpus(text: str) -> List[List[str]]:
    """Prepare custom corpus text for Word2Vec training.

    Splits text into sentences and tokenizes each sentence.

    Args:
        text: Raw text string containing multiple sentences

    Returns:
        List of tokenized sentences
    """
    # Split into sentences (simple split by period, exclamation, question mark)
    sentences = re.split(r'[.!?]+', text)

    # Preprocess each sentence
    tokenized_sentences = []
    for sentence in sentences:
        tokens = preprocess_text(sentence)
        if len(tokens) > 0:  # Only include non-empty sentences
            tokenized_sentences.append(tokens)

    return tokenized_sentences


def get_corpus_info(corpus: List[List[str]]) -> Dict[str, Any]:
    """Get information about the corpus.

    Args:
        corpus: Tokenized corpus

    Returns:
        Dictionary with corpus statistics
    """
    total_sentences = len(corpus)
    total_words = sum(len(sentence) for sentence in corpus)

    # Count unique words
    unique_words = set()
    for sentence in corpus:
        unique_words.update(sentence)

    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0

    return {
        'total_sentences': total_sentences,
        'total_words': total_words,
        'unique_words': len(unique_words),
        'avg_sentence_length': round(avg_sentence_length, 2)
    }


def get_sample_queries() -> Dict[str, List[str]]:
    """Get sample words and analogies for demonstration.

    Returns:
        Dictionary containing sample words for similarity queries
        and analogy triplets
    """
    return {
        'similarity_queries': [
            'machine',
            'learning',
            'data',
            'computer',
            'science',
            'language',
            'programming'
        ],
        'analogy_queries': [
            # Format: [positive1, positive2, negative1]
            # Represents: positive1 - negative1 + positive2
            ['machine', 'learning', 'computer'],  # machine is to computer as learning is to ?
            ['data', 'science', 'information'],   # data is to information as science is to ?
            ['programming', 'code', 'language'],  # programming is to language as code is to ?
            ['neural', 'network', 'brain'],       # neural is to brain as network is to ?
        ]
    }
