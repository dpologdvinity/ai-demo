"""Sample documents and data utilities for Topic Modeling (LDA)."""

from typing import List, Dict, Any


def get_default_documents() -> List[str]:
    """Get default sample documents across diverse topics.

    Returns:
        List of 50+ sample documents covering various topics
    """
    documents = [
        # Technology (10 docs)
        "Artificial intelligence and machine learning are transforming the technology industry. Deep learning models can now recognize images and understand natural language with remarkable accuracy.",
        "Cloud computing platforms like AWS, Azure, and Google Cloud provide scalable infrastructure for modern applications. Containers and Kubernetes enable efficient deployment and orchestration.",
        "Cybersecurity threats continue to evolve with sophisticated ransomware attacks and data breaches. Organizations must implement robust security measures and employee training programs.",
        "The latest smartphones feature advanced cameras, 5G connectivity, and powerful processors. Mobile apps dominate software development with billions of downloads annually.",
        "Quantum computing promises to revolutionize cryptography and complex simulations. Major tech companies are investing heavily in quantum hardware and algorithms.",
        "Blockchain technology enables decentralized applications and cryptocurrencies. Smart contracts automate transactions without intermediaries on distributed ledgers.",
        "Virtual reality and augmented reality are creating immersive experiences for gaming and education. VR headsets are becoming more affordable and accessible.",
        "Open source software drives innovation with collaborative development. Linux, Python, and JavaScript frameworks power countless applications worldwide.",
        "Data science combines statistics, programming, and domain expertise to extract insights from large datasets. Visualization tools help communicate findings effectively.",
        "Internet of Things devices connect homes, cities, and industries. Smart sensors and automation improve efficiency and enable predictive maintenance.",

        # Sports (10 docs)
        "The World Cup brings together nations in the world's most popular sporting event. Soccer fans celebrate their teams with passionate support and colorful displays.",
        "Basketball playoffs feature intense competition between elite teams. Star players showcase incredible athleticism with dunks, three-pointers, and defensive plays.",
        "Tennis grand slam tournaments attract the world's top players competing for prestigious titles. Wimbledon, the US Open, and French Open captivate audiences globally.",
        "Olympic athletes train for years to compete in track and field, swimming, and gymnastics events. The games promote international friendship and sportsmanship.",
        "Professional football leagues entertain millions with exciting games every season. Quarterbacks, running backs, and receivers execute complex offensive strategies.",
        "Baseball's World Series determines the champion of America's pastime. Home runs, strikeouts, and dramatic ninth-inning rallies keep fans on the edge of their seats.",
        "Marathon runners demonstrate incredible endurance covering 26.2 miles. Major city marathons like Boston and New York attract elite and amateur runners alike.",
        "Golf tournaments challenge players with precision shots and strategic course management. The Masters, PGA Championship, and British Open are prestigious major championships.",
        "Formula One racing combines speed, technology, and driver skill in high-stakes competition. Teams invest millions in aerodynamics and engine performance.",
        "Extreme sports like skateboarding, surfing, and snowboarding showcase athletic creativity. Athletes push boundaries with innovative tricks and daring stunts.",

        # Politics (10 docs)
        "Democratic elections allow citizens to choose their representatives and shape government policy. Voter turnout and campaign financing remain important issues.",
        "International diplomacy requires negotiation and compromise to resolve conflicts peacefully. United Nations assemblies bring world leaders together for dialogue.",
        "Political campaigns utilize social media and data analytics to reach voters. Debates and town halls provide opportunities for candidates to present their platforms.",
        "Government budgets allocate taxpayer funds to defense, education, healthcare, and infrastructure. Deficit spending and debt levels spark economic policy debates.",
        "Legislative bodies debate and pass laws on issues ranging from taxation to civil rights. Committee hearings examine proposed legislation in detail.",
        "Supreme Court rulings interpret constitutional law and set legal precedents. Judicial appointments can influence policy for decades.",
        "Political parties organize around shared ideologies and policy priorities. Primary elections determine party nominees for general elections.",
        "Foreign policy shapes international relations through trade agreements and military alliances. Sanctions and diplomatic pressure address human rights concerns.",
        "Campaign finance reform aims to reduce money's influence in politics. Disclosure requirements and contribution limits seek to promote transparency.",
        "Grassroots activism mobilizes citizens around social and political causes. Protests, petitions, and community organizing can influence public policy.",

        # Science (10 docs)
        "Climate scientists study global warming through ice core samples and atmospheric measurements. Rising temperatures threaten ecosystems and coastal communities.",
        "Genetic research explores DNA sequences to understand heredity and disease. CRISPR technology enables precise gene editing with medical applications.",
        "Space exploration missions send probes to Mars, Jupiter, and beyond. Telescopes detect exoplanets orbiting distant stars in the search for life.",
        "Particle physics experiments at CERN investigate fundamental forces and subatomic particles. The Large Hadron Collider confirmed the existence of the Higgs boson.",
        "Neuroscience research maps brain circuits and studies consciousness. Imaging techniques reveal how neurons communicate and process information.",
        "Medical researchers develop new treatments for cancer, Alzheimer's, and infectious diseases. Clinical trials test drug safety and efficacy.",
        "Environmental science examines ecosystems, biodiversity, and conservation. Deforestation and pollution threaten species with extinction.",
        "Chemistry advances create new materials with remarkable properties. Nanotechnology manipulates matter at the atomic scale.",
        "Astronomy discoveries expand our understanding of the universe's origins and structure. Black holes, dark matter, and cosmic microwave background radiation puzzle scientists.",
        "Renewable energy research develops solar panels, wind turbines, and battery storage. Sustainable solutions address climate change and energy security.",

        # Entertainment (10 docs)
        "Hollywood blockbusters dominate box office revenues with spectacular visual effects and star-studded casts. Superhero franchises continue to attract massive audiences.",
        "Streaming services have revolutionized television with on-demand content and original series. Binge-watching has become a popular cultural phenomenon.",
        "Music festivals bring together artists and fans for multi-day celebrations. Live performances create memorable experiences with amazing stage productions.",
        "Video games have evolved into a massive entertainment industry with competitive esports tournaments. Multiplayer online games connect players worldwide.",
        "Theater productions on Broadway showcase talented performers in musicals and dramatic plays. Classic shows and new productions entertain diverse audiences.",
        "Animation studios create beloved films using computer graphics and traditional techniques. Pixar and Disney movies appeal to children and adults alike.",
        "Reality television shows feature competitions, makeovers, and unscripted drama. Contestants compete for prizes while audiences vote for favorites.",
        "Stand-up comedy specials stream on platforms featuring comedians from diverse backgrounds. Humor addresses social issues and everyday life experiences.",
        "Book clubs and literary festivals celebrate authors and reading culture. Bestselling novels span genres from mystery to science fiction.",
        "Concert tours by famous musicians sell out arenas and stadiums. Pop, rock, country, and hip-hop artists perform for enthusiastic crowds.",

        # Business (10 docs)
        "Startup companies seek venture capital funding to scale operations and develop products. Entrepreneurs pitch innovative ideas to angel investors and VC firms.",
        "Stock markets reflect investor confidence with daily price fluctuations. Trading algorithms and institutional investors influence market dynamics.",
        "Corporate mergers and acquisitions reshape industries through consolidation. Due diligence and regulatory approval processes ensure fair deals.",
        "Marketing strategies leverage social media influencers and targeted advertising. Brand awareness campaigns reach consumers through multiple channels.",
        "Supply chain management optimizes logistics and inventory across global networks. Just-in-time delivery reduces costs and improves efficiency.",
        "Human resources departments recruit talent and manage employee benefits. Workplace culture and professional development programs retain skilled workers.",
        "Financial statements report revenue, expenses, and profitability to shareholders. Quarterly earnings calls discuss performance and future guidance.",
        "Real estate markets fluctuate with interest rates and economic conditions. Commercial and residential properties attract diverse investors.",
        "Retail businesses adapt to e-commerce competition with omnichannel strategies. Customer experience and fast shipping drive online sales growth.",
        "Management consulting firms advise companies on strategy and operations. Business transformation projects implement new technologies and processes.",
    ]

    return documents


def prepare_custom_documents(custom_docs: List[str]) -> List[str]:
    """Prepare and validate custom documents.

    Args:
        custom_docs: List of custom documents

    Returns:
        Cleaned list of documents

    Raises:
        ValueError: If documents are invalid
    """
    if not custom_docs:
        raise ValueError("Custom documents cannot be empty")

    if len(custom_docs) < 5:
        raise ValueError("At least 5 documents required for topic modeling")

    # Filter out empty documents
    cleaned = [doc.strip() for doc in custom_docs if doc and doc.strip()]

    if len(cleaned) < 5:
        raise ValueError("At least 5 non-empty documents required")

    return cleaned


def get_dataset_info(documents: List[str] = None) -> Dict[str, Any]:
    """Get information about the document corpus.

    Args:
        documents: Optional list of documents (uses default if not provided)

    Returns:
        Dictionary with corpus statistics
    """
    if documents is None:
        documents = get_default_documents()

    total_words = sum(len(doc.split()) for doc in documents)
    avg_doc_length = total_words / len(documents) if documents else 0

    return {
        "name": "Multi-topic Document Corpus",
        "description": "Diverse collection of documents across technology, sports, politics, science, entertainment, and business",
        "num_documents": len(documents),
        "total_words": total_words,
        "avg_document_length": round(avg_doc_length, 1),
        "topics_covered": [
            "Technology & Computing",
            "Sports & Athletics",
            "Politics & Government",
            "Science & Research",
            "Entertainment & Media",
            "Business & Finance"
        ],
        "supports_custom": True
    }
