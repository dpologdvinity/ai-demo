"""Sample data and corpus for Bag of Words demonstration."""

from typing import List, Dict, Any


def get_sample_corpus() -> List[str]:
    """Get a sample corpus of documents for Bag of Words demonstration.

    Returns a diverse collection of documents including reviews, tweets, and
    short articles to showcase BoW's text representation capabilities.

    Returns:
        List of document strings
    """
    return [
        # Product reviews
        "This smartphone has an amazing camera and battery life. The display is "
        "crystal clear and the performance is lightning fast. Highly recommend "
        "this phone to anyone looking for a premium device.",

        "Terrible customer service and poor quality product. The item arrived "
        "damaged and getting a refund was a nightmare. Would not recommend "
        "buying from this seller. Very disappointed with the experience.",

        "Good value for money. The laptop works well for basic tasks like web "
        "browsing and document editing. Not suitable for gaming or heavy video "
        "editing but perfect for students and casual users.",

        # Social media posts / tweets
        "Just finished an incredible workout at the gym. Feeling energized and "
        "ready to tackle the day. Remember to stay hydrated and take care of "
        "your body. Fitness is a journey not a destination.",

        "Breaking news: Major technology company announces revolutionary new "
        "product. Stock prices surge as investors show confidence. Analysts "
        "predict strong sales in the upcoming quarter.",

        "Beautiful sunset at the beach today. The colors were absolutely stunning. "
        "Nature never fails to amaze me. Taking a moment to appreciate the simple "
        "things in life. Grateful for these peaceful moments.",

        # News headlines / articles
        "Scientists discover breakthrough in renewable energy technology. New solar "
        "panels achieve record efficiency levels. This advancement could accelerate "
        "the global transition to clean energy sources.",

        "Local community rallies to support families affected by recent flooding. "
        "Volunteers donate supplies and time to help with recovery efforts. "
        "Fundraising campaign exceeds expectations showing strong community spirit.",

        "Stock market experiences significant volatility amid economic uncertainty. "
        "Investors remain cautious as inflation concerns persist. Financial experts "
        "advise diversification and long-term investment strategies.",

        # Food reviews
        "The restaurant exceeded all expectations. The pasta was perfectly cooked "
        "and the sauce was delicious. Service was attentive and the ambiance was "
        "romantic. Will definitely be returning for another meal.",

        "Average coffee shop with mediocre drinks. The espresso was bitter and the "
        "pastries were stale. Staff seemed rushed and unfriendly. Many better "
        "options available in the neighborhood.",

        "Best pizza in the city hands down. The crust is thin and crispy, toppings "
        "are fresh and generous. Authentic Italian flavors at reasonable prices. "
        "A must-visit for pizza lovers.",

        # Movie/entertainment reviews
        "This movie is a masterpiece of modern cinema. The cinematography is "
        "breathtaking and the performances are phenomenal. The plot keeps you "
        "engaged from start to finish. Deserves all the awards and recognition.",

        "Boring and predictable storyline with weak character development. The "
        "special effects were good but cannot save a poorly written script. "
        "Would not recommend wasting your time on this film.",

        "Entertaining summer blockbuster with great action sequences. Not much "
        "depth but perfect for a fun night out. Good popcorn movie that delivers "
        "on excitement and visual spectacle.",

        # Travel experiences
        "Paris is magical in the spring. The architecture is stunning and the food "
        "is incredible. Visited the Eiffel Tower, Louvre Museum, and charming cafes. "
        "A dream destination that lives up to the hype.",

        "Relaxing beach vacation with beautiful tropical scenery. The resort was "
        "comfortable with excellent amenities. Snorkeling and water sports were "
        "highlights. Perfect place to unwind and escape daily stress.",

        "Challenging but rewarding mountain hiking adventure. The trails were steep "
        "but the summit views were worth every step. Fresh air and exercise in "
        "nature provide great mental clarity and peace.",

        # Technology discussions
        "Artificial intelligence is transforming industries at an unprecedented pace. "
        "Machine learning algorithms are improving efficiency and enabling new "
        "capabilities. The future of technology looks incredibly promising.",

        "Cybersecurity threats continue to evolve and become more sophisticated. "
        "Companies must invest in robust security measures to protect sensitive data. "
        "Employee training and awareness are critical defense layers.",
    ]


def get_corpus_info() -> Dict[str, Any]:
    """Get information about the sample corpus.

    Returns:
        Dictionary containing corpus metadata
    """
    corpus = get_sample_corpus()

    return {
        "name": "Mixed Text Corpus",
        "description": "Diverse collection of reviews, social media posts, news articles, "
                      "and discussions covering various topics",
        "document_count": len(corpus),
        "avg_doc_length": sum(len(doc.split()) for doc in corpus) / len(corpus),
        "document_types": [
            "Product Reviews",
            "Social Media Posts",
            "News Articles",
            "Food Reviews",
            "Entertainment Reviews",
            "Travel Experiences",
            "Technology Discussions"
        ],
        "use_cases": [
            "Document classification",
            "Text clustering",
            "Information retrieval",
            "Spam detection",
            "Sentiment analysis preprocessing",
            "Topic modeling"
        ]
    }


def validate_custom_corpus(documents: List[str]) -> None:
    """Validate custom document corpus.

    Args:
        documents: List of document strings

    Raises:
        ValueError: If corpus is invalid
    """
    if not documents:
        raise ValueError("Document corpus cannot be empty")

    if len(documents) < 2:
        raise ValueError("Corpus must contain at least 2 documents")

    if len(documents) > 100:
        raise ValueError("Corpus cannot exceed 100 documents")

    for i, doc in enumerate(documents):
        if not doc or not doc.strip():
            raise ValueError(f"Document {i} is empty")

        if len(doc.split()) < 3:
            raise ValueError(f"Document {i} must contain at least 3 words")


def get_dataset_info() -> Dict[str, Any]:
    """Get comprehensive dataset information for the API.

    Returns:
        Dictionary with dataset details and statistics
    """
    corpus = get_sample_corpus()

    total_words = sum(len(doc.split()) for doc in corpus)
    unique_words = len(set(word.lower() for doc in corpus for word in doc.split()))

    return {
        "name": "Sample Mixed Corpus",
        "description": "Multi-domain text collection for Bag of Words demonstration",
        "size": len(corpus),
        "total_words": total_words,
        "unique_words": unique_words,
        "avg_doc_length": round(total_words / len(corpus), 1),
        "document_types": 7,
        "features": [
            "Diverse topics and writing styles",
            "Real-world text examples",
            "Varied vocabulary and sentiment",
            "Short to medium length documents"
        ]
    }
