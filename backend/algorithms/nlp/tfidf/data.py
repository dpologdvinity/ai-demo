"""Sample data and corpus for TF-IDF demonstration."""

from typing import List, Dict, Any


def get_sample_corpus() -> List[str]:
    """Get a sample corpus of documents for TF-IDF demonstration.

    Returns a diverse collection of news articles and reviews covering
    different topics to showcase TF-IDF's ability to identify important terms.

    Returns:
        List of document strings
    """
    return [
        # Technology articles
        "Artificial intelligence and machine learning are revolutionizing the tech industry. "
        "Deep learning models are achieving unprecedented accuracy in image recognition and "
        "natural language processing tasks. Neural networks continue to advance rapidly.",

        "Cloud computing services have transformed how businesses operate. Companies are "
        "migrating their infrastructure to AWS, Azure, and Google Cloud platforms. Scalability "
        "and cost efficiency are driving this digital transformation.",

        "Quantum computing promises to solve complex problems that classical computers cannot. "
        "Researchers are making breakthroughs in quantum algorithms and error correction. "
        "Major tech companies are investing heavily in quantum hardware development.",

        # Healthcare articles
        "Medical breakthroughs in cancer research are offering new hope to patients. "
        "Immunotherapy and targeted treatments are showing remarkable results. Clinical trials "
        "are advancing precision medicine and personalized healthcare approaches.",

        "Telemedicine and digital health platforms are expanding access to healthcare services. "
        "Remote patient monitoring and virtual consultations have become essential tools. "
        "Healthcare providers are adopting telehealth technologies at unprecedented rates.",

        "Vaccine development has accelerated with modern biotechnology techniques. mRNA vaccines "
        "represent a revolutionary approach to preventing infectious diseases. Public health "
        "initiatives are essential for global vaccination programs.",

        # Finance articles
        "Cryptocurrency markets continue to experience significant volatility. Bitcoin and "
        "Ethereum are leading the digital currency revolution. Blockchain technology is "
        "disrupting traditional financial systems and enabling decentralized finance.",

        "Stock market investors are navigating economic uncertainty and inflation concerns. "
        "Portfolio diversification remains crucial for managing investment risk. Financial "
        "advisors recommend balanced strategies for long-term wealth building.",

        "Central banks are adjusting monetary policy to address inflation pressures. Interest "
        "rate decisions impact borrowing costs and economic growth. Fiscal policy coordination "
        "is critical for macroeconomic stability.",

        # Environment articles
        "Climate change is driving extreme weather events worldwide. Rising temperatures are "
        "melting polar ice caps and causing sea level rise. International cooperation is "
        "essential for reducing carbon emissions and achieving sustainability goals.",

        "Renewable energy sources like solar and wind power are becoming more cost-effective. "
        "Clean energy transition is accelerating as countries commit to net-zero targets. "
        "Battery storage technology is improving the reliability of renewable power grids.",

        "Biodiversity loss threatens ecosystem stability and species survival. Conservation "
        "efforts are protecting endangered habitats and wildlife populations. Sustainable "
        "agriculture practices can help preserve natural resources for future generations.",

        # Sports articles
        "Olympic athletes are breaking world records and pushing human performance limits. "
        "Training techniques and sports science are advancing athletic capabilities. "
        "International competitions showcase the dedication and talent of elite competitors.",

        "Professional basketball leagues are expanding their global reach and fan engagement. "
        "Star players are dominating the court with exceptional skills and athleticism. "
        "Team strategies and coaching innovations are evolving the game.",

        "Football season brings excitement as teams compete for championship titles. "
        "Quarterback performances and defensive strategies determine game outcomes. "
        "Fantasy football has become increasingly popular among sports enthusiasts.",

        # Entertainment articles
        "Streaming platforms are revolutionizing how audiences consume entertainment content. "
        "Original series and movies are attracting millions of subscribers worldwide. "
        "Traditional television networks are adapting to changing viewing habits.",

        "Music industry artists are leveraging social media to connect with fans directly. "
        "Streaming services have transformed music distribution and revenue models. "
        "Live concerts and festivals remain essential for artist-fan engagement.",

        "Film production has rebounded with blockbuster releases drawing audiences to theaters. "
        "Special effects and cinematography continue to push creative boundaries. "
        "Independent filmmakers are gaining recognition through film festival circuits.",

        # Education articles
        "Online learning platforms are democratizing access to quality education worldwide. "
        "Students can take courses from top universities without leaving home. "
        "Digital literacy and remote learning skills are becoming increasingly important.",

        "STEM education initiatives are preparing students for technology-driven careers. "
        "Coding bootcamps and technical training programs are meeting industry demand. "
        "Educational institutions are updating curricula to align with workforce needs.",
    ]


def get_corpus_info() -> Dict[str, Any]:
    """Get information about the sample corpus.

    Returns:
        Dictionary containing corpus metadata
    """
    corpus = get_sample_corpus()

    return {
        "name": "Sample News & Reviews Corpus",
        "description": "Diverse collection of articles covering technology, healthcare, "
                      "finance, environment, sports, entertainment, and education topics",
        "document_count": len(corpus),
        "avg_doc_length": sum(len(doc.split()) for doc in corpus) / len(corpus),
        "topics": [
            "Technology & AI",
            "Healthcare & Medicine",
            "Finance & Economics",
            "Environment & Climate",
            "Sports & Athletics",
            "Entertainment & Media",
            "Education & Learning"
        ],
        "use_cases": [
            "Document classification",
            "Information retrieval",
            "Keyword extraction",
            "Topic modeling",
            "Search engine ranking"
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

        if len(doc.split()) < 5:
            raise ValueError(f"Document {i} must contain at least 5 words")


def get_dataset_info() -> Dict[str, Any]:
    """Get comprehensive dataset information for the API.

    Returns:
        Dictionary with dataset details and statistics
    """
    corpus = get_sample_corpus()

    total_words = sum(len(doc.split()) for doc in corpus)
    unique_words = len(set(word.lower() for doc in corpus for word in doc.split()))

    return {
        "name": "Sample News Corpus",
        "description": "Multi-topic document collection for TF-IDF demonstration",
        "size": len(corpus),
        "total_words": total_words,
        "unique_words": unique_words,
        "avg_doc_length": round(total_words / len(corpus), 1),
        "topics": 7,
        "features": [
            "Diverse topics for clear term distinction",
            "Professional writing style",
            "Varied vocabulary and terminology",
            "Real-world content examples"
        ]
    }
