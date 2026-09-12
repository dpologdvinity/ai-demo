"""Sample data and utilities for BERT Fine-tuning."""

from typing import List, Tuple, Dict, Any


def get_default_dataset() -> Tuple[List[str], List[int], List[str]]:
    """Get default sample dataset for BERT fine-tuning.

    Returns a small dataset with 45 samples across 3 classes:
    - Class 0: Negative sentiment (movie/product reviews)
    - Class 1: Neutral sentiment
    - Class 2: Positive sentiment

    Returns:
        Tuple of (texts, labels, class_names)
    """
    texts = [
        # Negative sentiment (Class 0) - 15 samples
        "This movie was a complete waste of time. The plot was terrible and the acting was awful.",
        "Horrible product! Broke after just one day of use. Complete disappointment.",
        "Worst experience ever. The service was terrible and staff were rude.",
        "Do not recommend this at all. Poor quality and overpriced.",
        "Terrible book. The story was boring and characters were poorly developed.",
        "The app crashes constantly and is full of bugs. Very frustrating to use.",
        "Disappointing meal. The food was cold and tasteless.",
        "Awful hotel experience. Dirty room and noisy environment.",
        "This gadget is useless. Doesn't work as advertised at all.",
        "Very poor customer support. They were unhelpful and dismissive.",
        "The concert was terrible. Sound quality was awful and venue was poorly organized.",
        "Regret buying this product. Total waste of money.",
        "The software is buggy and unreliable. Lost my work multiple times.",
        "Bad purchase decision. The item arrived damaged and defective.",
        "Horrible travel experience. Delayed flights and lost luggage.",

        # Neutral sentiment (Class 1) - 15 samples
        "The product arrived on time and matches the description.",
        "It's an average movie. Some good parts, some not so good.",
        "The service was adequate. Nothing special but got the job done.",
        "Standard quality for the price. About what you'd expect.",
        "The book has its pros and cons. Decent read overall.",
        "The app works as described. Basic functionality is there.",
        "The restaurant is okay. Food is acceptable, nothing memorable.",
        "The hotel was clean and functional. Standard amenities provided.",
        "This tool does what it says. No complaints, no praise.",
        "Customer service was professional. They addressed my concerns.",
        "The course covered the basics. Good for beginners, nothing advanced.",
        "The device functions properly. Meets minimum requirements.",
        "The show was entertaining enough. Not amazing, but watchable.",
        "The store has a reasonable selection. Prices are fair.",
        "The experience was neither good nor bad, just average.",

        # Positive sentiment (Class 2) - 15 samples
        "Absolutely amazing movie! The story was captivating and acting was superb.",
        "Love this product! Exceeded all my expectations. Highly recommend!",
        "Outstanding service! The staff were friendly and incredibly helpful.",
        "Best purchase I've made this year. Quality is exceptional!",
        "Brilliant book! The author's writing is beautiful and thought-provoking.",
        "This app is fantastic! Easy to use and packed with great features.",
        "Delicious food and wonderful atmosphere. Will definitely return!",
        "Perfect hotel stay. Clean, comfortable, and excellent location.",
        "This gadget is a game-changer! Works flawlessly and saves so much time.",
        "Excellent customer support! They went above and beyond to help.",
        "The concert was incredible! Amazing performance and great venue.",
        "So happy with this purchase! Great value for money.",
        "The software is powerful and intuitive. Makes my work much easier.",
        "Fantastic quality! The item is well-made and looks beautiful.",
        "Wonderful travel experience! Everything went smoothly and exceeded expectations.",
    ]

    labels = [0] * 15 + [1] * 15 + [2] * 15

    class_names = ["Negative", "Neutral", "Positive"]

    return texts, labels, class_names


def get_sample_texts_for_inference() -> List[str]:
    """Get sample texts for inference/prediction demonstrations.

    Returns:
        List of sample texts spanning different sentiments
    """
    return [
        "This is absolutely terrible! I hate it so much.",
        "It's okay, nothing special but it works fine.",
        "Amazing! This is exactly what I needed. Love it!",
        "Not impressed. Expected better quality for this price.",
        "The product is decent. Does the job as advertised.",
        "Incredible experience! Highly recommend to everyone!"
    ]


def prepare_custom_dataset(
    texts: List[str],
    labels: List[int]
) -> Tuple[List[str], List[int]]:
    """Prepare custom dataset for training.

    Validates and cleans custom dataset inputs.

    Args:
        texts: List of text strings
        labels: List of corresponding labels

    Returns:
        Tuple of (cleaned_texts, labels)

    Raises:
        ValueError: If inputs are invalid
    """
    if not texts or len(texts) == 0:
        raise ValueError("Texts list cannot be empty")

    if not labels or len(labels) == 0:
        raise ValueError("Labels list cannot be empty")

    if len(texts) != len(labels):
        raise ValueError("Texts and labels must have the same length")

    if len(texts) < 6:
        raise ValueError("Need at least 6 samples for training (minimum for train/val split)")

    # Validate labels are 0, 1, or 2
    unique_labels = set(labels)
    if not unique_labels.issubset({0, 1, 2}):
        raise ValueError("All labels must be 0, 1, or 2 for 3-class classification")

    if len(unique_labels) < 2:
        raise ValueError("Dataset must have at least 2 different classes")

    # Clean texts
    cleaned_texts = [text.strip() for text in texts if text.strip()]

    if len(cleaned_texts) != len(texts):
        raise ValueError("Some texts are empty or contain only whitespace")

    return cleaned_texts, labels


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the default dataset.

    Returns:
        Dictionary containing dataset metadata
    """
    texts, labels, class_names = get_default_dataset()

    return {
        "name": "Sentiment Classification Dataset",
        "description": "Small text classification dataset for sentiment analysis",
        "total_samples": len(texts),
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_distribution": {
            class_names[i]: labels.count(i) for i in range(len(class_names))
        },
        "sample_texts": [
            texts[0],  # Negative example
            texts[15],  # Neutral example
            texts[30]  # Positive example
        ],
        "supports_custom": True,
        "min_samples_required": 6
    }


def split_dataset(
    texts: List[str],
    labels: List[int],
    train_ratio: float = 0.7
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Split dataset into train and validation sets.

    Args:
        texts: List of text samples
        labels: List of labels
        train_ratio: Ratio of training samples (default: 0.7)

    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels)
    """
    # Simple split (in production, use stratified split)
    split_idx = int(len(texts) * train_ratio)

    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    val_texts = texts[split_idx:]
    val_labels = labels[split_idx:]

    return train_texts, train_labels, val_texts, val_labels
