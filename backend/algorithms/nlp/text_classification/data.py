"""Sample data and utilities for Text Classification."""

from typing import List, Tuple, Dict, Any


def get_default_dataset() -> Tuple[List[str], List[str]]:
    """Get default sample dataset for text classification.

    Returns a diverse collection of texts with categories:
    - technology: Tech news and product reviews
    - sports: Sports news and commentary
    - politics: Political news and analysis
    - entertainment: Movie/TV reviews and entertainment news
    - business: Business news and financial reports

    Returns:
        Tuple of (texts, labels) for classification
    """
    texts = [
        # Technology (20 samples)
        "Apple unveils new iPhone with groundbreaking AI features and improved camera system.",
        "Google's latest algorithm update aims to improve search result quality and relevance.",
        "Microsoft announces major updates to Windows operating system with enhanced security.",
        "Tesla's new electric vehicle breaks range records with innovative battery technology.",
        "Amazon launches new cloud computing service for machine learning applications.",
        "Samsung's foldable smartphone receives positive reviews for innovative design.",
        "Intel releases new processor chip with significant performance improvements.",
        "Facebook parent Meta invests billions in virtual reality and metaverse development.",
        "Netflix develops new streaming technology to reduce bandwidth usage.",
        "IBM's quantum computer achieves breakthrough in computational power.",
        "The latest smartphone features an impressive camera and lightning-fast processor.",
        "Cybersecurity experts warn of new vulnerability in popular software.",
        "Artificial intelligence system beats human champions in complex strategy game.",
        "New programming language promises to revolutionize software development.",
        "Tech startup raises $100 million for innovative blockchain solution.",
        "Scientists develop advanced neural network for medical diagnosis.",
        "Major data breach affects millions of users on popular social platform.",
        "Revolutionary battery technology could extend electric vehicle range dramatically.",
        "Software engineers unveil open-source framework for building AI applications.",
        "Cloud computing giant expands data center infrastructure globally.",

        # Sports (20 samples)
        "Lakers win championship with dominant performance in Game 7 finals.",
        "Olympic athlete breaks world record in 100-meter sprint with incredible time.",
        "Manchester United signs star midfielder in record-breaking transfer deal.",
        "Tennis champion retires after winning historic 20th Grand Slam title.",
        "NFL team makes dramatic comeback to win Super Bowl in overtime thriller.",
        "Baseball player hits 700th career home run, joining elite club of legends.",
        "Soccer World Cup final draws record television audience worldwide.",
        "NBA rookie sensation leads team to unexpected playoff appearance.",
        "Formula 1 driver wins consecutive races with flawless racing strategy.",
        "Boxing champion defends title in unanimous decision victory.",
        "The team scored a decisive victory in yesterday's intense playoff game.",
        "Marathon runner completes race despite challenging weather conditions.",
        "Hockey team advances to finals after dramatic penalty shootout win.",
        "Gymnast performs perfect routine to claim gold medal at championships.",
        "Cricket match ends in thrilling tie with last-ball drama.",
        "Swimmer breaks Olympic record with stunning performance in final.",
        "Basketball coach announces retirement after 30 years of excellence.",
        "Golf tournament sees unexpected winner in dramatic finish.",
        "Rugby team dominates rivals in one-sided championship match.",
        "Track and field athlete qualifies for Olympics with personal best time.",

        # Politics (20 samples)
        "Congress passes landmark legislation on climate change with bipartisan support.",
        "Presidential candidate leads in polls ahead of crucial primary elections.",
        "Senate debates controversial bill on healthcare reform for third consecutive day.",
        "Supreme Court delivers historic ruling on voting rights legislation.",
        "International summit addresses global trade policies and tariff negotiations.",
        "Governor announces new infrastructure plan worth billions for state modernization.",
        "Political scandal rocks administration as investigation reveals corruption.",
        "United Nations votes on resolution addressing humanitarian crisis.",
        "Election results show surprising shift in voter preferences nationwide.",
        "Lawmakers propose constitutional amendment on term limits for officials.",
        "The senator delivered a passionate speech about education reform policy.",
        "Foreign policy experts debate implications of recent diplomatic agreement.",
        "Protest movement gains momentum calling for government accountability.",
        "Presidential debate focuses on economic policy and job creation.",
        "New legislation aims to strengthen campaign finance regulations.",
        "Mayor announces controversial plan to address homelessness in city.",
        "Political parties negotiate coalition government after close election.",
        "Referendum results indicate strong public support for constitutional change.",
        "Ambassador discusses peace negotiations in conflict-affected region.",
        "Parliament votes to implement stricter environmental regulations.",

        # Entertainment (20 samples)
        "Blockbuster movie breaks box office records with opening weekend success.",
        "Emmy Awards ceremony celebrates outstanding achievements in television.",
        "Popular streaming series renewed for fifth season after record viewership.",
        "Music festival announces star-studded lineup for summer concert series.",
        "Actor wins Oscar for powerful performance in critically acclaimed drama.",
        "Broadway musical receives standing ovations and rave reviews from critics.",
        "Documentary film explores environmental issues with stunning cinematography.",
        "Celebrity couple announces engagement after years of dating rumors.",
        "Animation studio releases heartwarming family film to widespread praise.",
        "Rock band embarks on world tour celebrating 30th anniversary.",
        "The new comedy special had audiences laughing throughout the performance.",
        "Film director's latest work premieres at prestigious international festival.",
        "Television network cancels long-running show after declining ratings.",
        "Pop star's new album debuts at number one on charts worldwide.",
        "Theater production reimagines classic play with modern interpretation.",
        "Gaming industry celebrates record-breaking year with innovative releases.",
        "Fashion show features designs from emerging and established designers.",
        "Reality TV show finale draws millions of viewers for dramatic conclusion.",
        "Author's bestselling novel adapted into major motion picture.",
        "Concert tour sells out venues in minutes due to overwhelming demand.",

        # Business (20 samples)
        "Stock market reaches all-time high amid strong corporate earnings reports.",
        "Major retail chain files for bankruptcy protection after years of losses.",
        "Tech company valuation exceeds one trillion dollars in historic milestone.",
        "Federal Reserve announces interest rate decision affecting global markets.",
        "Merger between industry giants creates powerful new business entity.",
        "Startup company disrupts traditional industry with innovative business model.",
        "Oil prices fluctuate amid geopolitical tensions and supply concerns.",
        "Cryptocurrency market experiences volatility following regulatory announcements.",
        "Manufacturing sector reports strong growth in quarterly economic data.",
        "Trade agreement between nations promises to boost bilateral commerce.",
        "The company reported record profits in its quarterly earnings statement.",
        "Real estate market shows signs of cooling after years of growth.",
        "Corporate CEO resigns following board dispute over company direction.",
        "E-commerce platform expands operations to new international markets.",
        "Banking sector faces scrutiny over lending practices and regulations.",
        "Labor union negotiates new contract with major employer.",
        "Investment fund launches new strategy focused on sustainable companies.",
        "Retail sales exceed expectations during holiday shopping season.",
        "Unemployment rate drops to lowest level in decades.",
        "Supply chain disruptions impact manufacturing and shipping industries.",
    ]

    labels = (
        ["technology"] * 20 +
        ["sports"] * 20 +
        ["politics"] * 20 +
        ["entertainment"] * 20 +
        ["business"] * 20
    )

    return texts, labels


def get_category_descriptions() -> Dict[str, str]:
    """Get descriptions for each category.

    Returns:
        Dictionary mapping category names to descriptions
    """
    return {
        "technology": "News and information about technology, gadgets, software, and innovation",
        "sports": "Sports news, game results, athlete profiles, and sporting events",
        "politics": "Political news, government policy, elections, and legislative updates",
        "entertainment": "Movies, TV shows, music, celebrities, and entertainment industry news",
        "business": "Business news, financial markets, corporate earnings, and economic data"
    }


def prepare_custom_dataset(
    texts: List[str],
    labels: List[str]
) -> Tuple[List[str], List[str]]:
    """Prepare custom dataset for classification.

    Validates and cleans custom text and label inputs.

    Args:
        texts: List of text strings
        labels: List of corresponding labels

    Returns:
        Tuple of (cleaned_texts, cleaned_labels)

    Raises:
        ValueError: If inputs are invalid or mismatched
    """
    if not texts or not labels:
        raise ValueError("Texts and labels cannot be empty")

    if len(texts) != len(labels):
        raise ValueError(
            f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})"
        )

    # Filter out empty texts
    cleaned_data = [
        (text.strip(), label.strip())
        for text, label in zip(texts, labels)
        if text and text.strip() and label and label.strip()
    ]

    if not cleaned_data:
        raise ValueError("All texts or labels are empty after cleaning")

    if len(cleaned_data) < 10:
        raise ValueError("Dataset must contain at least 10 samples for classification")

    # Check that we have at least 2 classes
    unique_labels = set(label for _, label in cleaned_data)
    if len(unique_labels) < 2:
        raise ValueError("Dataset must contain at least 2 different classes")

    cleaned_texts, cleaned_labels = zip(*cleaned_data)

    # Limit text length
    max_length = 1000
    cleaned_texts = [
        text[:max_length] if len(text) > max_length else text
        for text in cleaned_texts
    ]

    return list(cleaned_texts), list(cleaned_labels)


def get_dataset_info(texts: List[str], labels: List[str]) -> Dict[str, Any]:
    """Get information about the dataset.

    Args:
        texts: List of text strings
        labels: List of labels

    Returns:
        Dictionary containing dataset statistics
    """
    if not texts or not labels:
        return {
            "total_samples": 0,
            "num_classes": 0,
            "class_distribution": {},
            "avg_text_length": 0
        }

    from collections import Counter

    class_counts = Counter(labels)
    avg_length = sum(len(text) for text in texts) / len(texts)

    return {
        "total_samples": len(texts),
        "num_classes": len(class_counts),
        "class_distribution": dict(class_counts),
        "avg_text_length": round(avg_length, 2),
        "class_names": sorted(class_counts.keys())
    }
