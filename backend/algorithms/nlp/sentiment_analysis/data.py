"""Sample data and utilities for Sentiment Analysis."""

from typing import List


def get_default_texts() -> List[str]:
    """Get default sample texts for sentiment analysis.

    Returns a diverse collection of texts with positive, negative, and neutral sentiments
    including movie reviews, product reviews, tweets, and general comments.

    Returns:
        List of sample text strings for sentiment analysis
    """
    return [
        # Positive sentiments
        "This movie was absolutely amazing! The acting was superb and the story kept me engaged throughout.",
        "I love this product! It exceeded all my expectations and works perfectly.",
        "What a fantastic experience! The customer service was outstanding and very helpful.",
        "Best purchase I've ever made. Highly recommend to anyone looking for quality.",
        "The food was delicious and the ambiance was perfect. Will definitely come back!",
        "Incredible performance! This is exactly what I was looking for.",
        "Five stars all the way! This is a game changer for me.",
        "Really enjoyed this book. The author's writing style is captivating and thoughtful.",
        "Awesome results! I'm so happy I decided to try this.",
        "Brilliant work! The attention to detail is impressive.",
        "This app has made my life so much easier. Love all the features!",
        "The service was exceptional and the staff were very friendly.",

        # Negative sentiments
        "This was the worst movie I've ever seen. Complete waste of time.",
        "Terrible product. Broke after one day of use. Very disappointed.",
        "Horrible experience. The staff was rude and unhelpful.",
        "Do not buy this! Total scam and poor quality.",
        "The food was cold and tasteless. Never going back to this place.",
        "Awful performance. Nothing worked as advertised.",
        "One star is too generous. This is completely useless.",
        "I regret buying this. It's poorly made and overpriced.",
        "Disappointing results. Did not meet any of my expectations.",
        "Waste of money. The worst purchase I've made this year.",
        "This app crashes constantly and has too many bugs.",
        "Very poor customer service. They didn't help at all.",

        # Neutral sentiments
        "The product arrived on time as expected.",
        "It's okay. Nothing special but it does what it's supposed to do.",
        "The movie was average. Some parts were good, others not so much.",
        "Standard quality. About what you'd expect for the price.",
        "The experience was neither good nor bad, just ordinary.",
        "It works as described. No complaints but nothing to rave about either.",
        "The food was acceptable but nothing memorable.",
        "This is a basic product that meets minimum requirements.",
        "The service was adequate. They did their job.",
        "It's fine for occasional use. Not particularly impressive.",
    ]


def get_sample_reviews() -> dict:
    """Get categorized sample reviews for testing.

    Returns:
        Dictionary with 'positive', 'negative', and 'neutral' categories,
        each containing sample review texts
    """
    return {
        "positive": [
            "Outstanding quality and exceptional value for money!",
            "This exceeded my wildest expectations. Absolutely love it!",
            "Perfect in every way. Couldn't be happier with this purchase.",
        ],
        "negative": [
            "Completely dissatisfied. This is not what was advertised.",
            "Poor quality and terrible customer support. Very frustrating.",
            "Don't waste your money on this. Total disappointment.",
        ],
        "neutral": [
            "It's okay. Does the job but nothing special.",
            "Average product. Some pros and cons balance out.",
            "Neither impressed nor disappointed. Just what I expected.",
        ]
    }


def prepare_custom_texts(texts: List[str]) -> List[str]:
    """Prepare custom texts for analysis.

    Validates and cleans custom text inputs.

    Args:
        texts: List of text strings to analyze

    Returns:
        Cleaned and validated list of texts

    Raises:
        ValueError: If texts list is empty or contains invalid entries
    """
    if not texts or len(texts) == 0:
        raise ValueError("Custom texts list cannot be empty")

    # Filter out empty or whitespace-only texts
    cleaned_texts = [text.strip() for text in texts if text and text.strip()]

    if not cleaned_texts:
        raise ValueError("All custom texts are empty after cleaning")

    # Limit text length to prevent processing issues
    max_length = 1000
    cleaned_texts = [
        text[:max_length] if len(text) > max_length else text
        for text in cleaned_texts
    ]

    return cleaned_texts


def get_texts_info(texts: List[str]) -> dict:
    """Get information about the text corpus.

    Args:
        texts: List of text strings

    Returns:
        Dictionary containing corpus statistics
    """
    if not texts:
        return {
            "total_texts": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "total_characters": 0
        }

    lengths = [len(text) for text in texts]

    return {
        "total_texts": len(texts),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_characters": sum(lengths)
    }
