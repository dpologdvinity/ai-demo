"""Data loading and sample texts for Named Entity Recognition algorithm."""

from typing import List, Dict, Any


def get_default_texts() -> List[Dict[str, str]]:
    """Get default sample texts for NER demonstration.

    Returns a diverse set of texts containing various entity types including
    persons, organizations, locations, dates, and more.

    Returns:
        List of dictionaries with 'id', 'title', and 'text' keys
    """
    return [
        {
            'id': 'tech_news',
            'title': 'Tech Industry News',
            'text': 'Apple CEO Tim Cook announced the new iPhone 15 at the Steve Jobs Theater in Cupertino, California on September 12, 2023. The event was attended by tech journalists from The Verge, TechCrunch, and CNET.'
        },
        {
            'id': 'business',
            'title': 'Business Announcement',
            'text': 'Microsoft Corporation completed its acquisition of Activision Blizzard for $69 billion in October 2023. CEO Satya Nadella stated that the deal would strengthen Microsoft\'s position in the gaming industry.'
        },
        {
            'id': 'politics',
            'title': 'Political Summit',
            'text': 'President Joe Biden met with Prime Minister Rishi Sunak at the White House on Thursday, June 8, 2023. They discussed economic cooperation between the United States and the United Kingdom.'
        },
        {
            'id': 'sports',
            'title': 'Sports Championship',
            'text': 'Lionel Messi led Argentina to victory in the FIFA World Cup final against France in Doha, Qatar on December 18, 2022. The match was held at Lusail Stadium and watched by billions worldwide.'
        },
        {
            'id': 'science',
            'title': 'Scientific Discovery',
            'text': 'Dr. Jennifer Doudna and Dr. Emmanuelle Charpentier won the Nobel Prize in Chemistry in 2020 for their work on CRISPR-Cas9 gene editing technology at the University of California, Berkeley and the Max Planck Institute.'
        },
        {
            'id': 'entertainment',
            'title': 'Movie Release',
            'text': 'Director Christopher Nolan\'s film Oppenheimer premiered in July 2023 starring Cillian Murphy as J. Robert Oppenheimer. The biographical drama was produced by Universal Pictures and filmed in Los Alamos, New Mexico.'
        },
        {
            'id': 'finance',
            'title': 'Market Update',
            'text': 'Goldman Sachs and JPMorgan Chase reported strong earnings for Q2 2023. Analysts at Morgan Stanley predicted continued growth in the financial sector through the end of the year.'
        },
        {
            'id': 'education',
            'title': 'University Research',
            'text': 'Professor Sarah Johnson from Stanford University published groundbreaking research on artificial intelligence in Nature magazine on March 15, 2023. The study was funded by the National Science Foundation.'
        },
        {
            'id': 'health',
            'title': 'Medical Breakthrough',
            'text': 'The Mayo Clinic and Johns Hopkins Hospital announced a collaborative study on immunotherapy treatments. Dr. Michael Chen and Dr. Lisa Patel will lead the research starting in January 2024.'
        },
        {
            'id': 'environment',
            'title': 'Climate Conference',
            'text': 'The United Nations Climate Change Conference (COP28) took place in Dubai, United Arab Emirates in November 2023. Representatives from over 190 countries attended, including UN Secretary-General António Guterres.'
        },
        {
            'id': 'space',
            'title': 'Space Exploration',
            'text': 'NASA\'s James Webb Space Telescope captured stunning images of distant galaxies in July 2023. The mission is managed by the Space Telescope Science Institute in Baltimore, Maryland under direction of Administrator Bill Nelson.'
        },
        {
            'id': 'automotive',
            'title': 'Electric Vehicle Launch',
            'text': 'Tesla CEO Elon Musk unveiled the Cybertruck at the company\'s Gigafactory in Austin, Texas on November 30, 2023. The electric pickup truck will compete with Ford\'s F-150 Lightning and Rivian\'s R1T.'
        },
        {
            'id': 'social_media',
            'title': 'Tech Platform Update',
            'text': 'Meta Platforms, formerly Facebook, launched its Twitter competitor called Threads on July 5, 2023. CEO Mark Zuckerberg announced that the app gained 100 million users in its first week.'
        },
        {
            'id': 'legal',
            'title': 'Court Decision',
            'text': 'The Supreme Court of the United States ruled on the landmark case on June 30, 2023. Chief Justice John Roberts delivered the majority opinion, with Justice Sonia Sotomayor writing a dissenting opinion.'
        },
        {
            'id': 'travel',
            'title': 'Tourism Record',
            'text': 'Paris, France welcomed over 30 million tourists in 2023, making it the most visited city in Europe. The Louvre Museum and Eiffel Tower remained the top attractions, according to the French Ministry of Tourism.'
        },
        {
            'id': 'energy',
            'title': 'Renewable Energy Project',
            'text': 'NextEra Energy completed construction of the world\'s largest solar farm in Riverside County, California on August 22, 2023. The $2.5 billion project will generate 690 megawatts of clean energy for Southern California Edison customers.'
        },
        {
            'id': 'retail',
            'title': 'E-commerce Expansion',
            'text': 'Amazon opened three new fulfillment centers in Texas, Ohio, and Georgia in September 2023. The company plans to hire 5,000 employees across these facilities by the end of the year.'
        },
        {
            'id': 'aerospace',
            'title': 'Aircraft Development',
            'text': 'Boeing delivered the first 777X aircraft to Emirates Airlines at their facility in Everett, Washington. The delivery was delayed from 2020 but finally completed in May 2023 after extensive testing by the Federal Aviation Administration.'
        },
        {
            'id': 'cryptocurrency',
            'title': 'Digital Currency Regulation',
            'text': 'The Securities and Exchange Commission approved the first Bitcoin ETF on January 10, 2024. Commissioner Gary Gensler emphasized that the approval doesn\'t indicate endorsement of cryptocurrency as an investment.'
        },
        {
            'id': 'pharma',
            'title': 'Drug Approval',
            'text': 'Pfizer and BioNTech received FDA approval for their updated COVID-19 vaccine on September 11, 2023. Dr. Peter Marks, director of the FDA\'s Center for Biologics Evaluation and Research, announced the approval.'
        },
        {
            'id': 'fashion',
            'title': 'Fashion Week',
            'text': 'Paris Fashion Week showcased collections from Louis Vuitton, Chanel, and Dior from September 25 to October 3, 2023. Designer Virginie Viard presented Chanel\'s Spring/Summer 2024 collection at the Grand Palais.'
        },
        {
            'id': 'food',
            'title': 'Restaurant Opening',
            'text': 'Chef Gordon Ramsay opened his new restaurant, Savoy Grill 2.0, in London on November 15, 2023. The Michelin-starred establishment features British cuisine and was reviewed by The Times and The Guardian.'
        },
        {
            'id': 'gaming',
            'title': 'Video Game Release',
            'text': 'Sony Interactive Entertainment released The Last of Us Part III on PlayStation 5 on December 1, 2023. Game director Neil Druckmann and actress Ashley Johnson attended the launch event in Los Angeles, California.'
        },
        {
            'id': 'telecom',
            'title': '5G Network Expansion',
            'text': 'Verizon Communications and AT&T completed their nationwide 5G rollout covering 98% of the US population by October 2023. The companies invested $50 billion combined in infrastructure improvements.'
        },
        {
            'id': 'real_estate',
            'title': 'Commercial Development',
            'text': 'Brookfield Properties announced the development of a 1,000-foot skyscraper in downtown Chicago, Illinois. The $800 million project, designed by architecture firm Skidmore, Owings & Merrill, will be completed by 2026.'
        },
        {
            'id': 'literature',
            'title': 'Book Release',
            'text': 'Author Margaret Atwood released her new novel The Testament on September 10, 2023, published by Doubleday. The book tour included stops at Barnes & Noble locations in New York City, Boston, and San Francisco.'
        },
        {
            'id': 'music',
            'title': 'Album Launch',
            'text': 'Taylor Swift released her album 1989 (Taylor\'s Version) on October 27, 2023, through Republic Records. The album topped the Billboard 200 chart and broke streaming records on Spotify and Apple Music.'
        },
        {
            'id': 'insurance',
            'title': 'Corporate Merger',
            'text': 'Berkshire Hathaway acquired Alleghany Corporation for $11.6 billion in October 2022. Warren Buffett stated the acquisition strengthens Berkshire\'s insurance operations and property-casualty business.'
        },
        {
            'id': 'hospitality',
            'title': 'Hotel Chain Expansion',
            'text': 'Marriott International opened 500 new hotels across Asia, Europe, and North America in 2023. CEO Anthony Capuano announced plans for 200 additional properties in the Middle East by 2025.'
        },
        {
            'id': 'agriculture',
            'title': 'Farming Technology',
            'text': 'John Deere launched autonomous tractors with AI-powered guidance systems at the Consumer Electronics Show in Las Vegas, Nevada on January 5, 2023. The technology was developed in partnership with NVIDIA Corporation.'
        },
        {
            'id': 'maritime',
            'title': 'Shipping Industry',
            'text': 'Maersk and MSC Mediterranean Shipping Company announced a joint venture for green shipping routes between Rotterdam, Netherlands and Singapore. The initiative aims to reduce carbon emissions by 40% by 2030.'
        },
        {
            'id': 'defense',
            'title': 'Military Contract',
            'text': 'Lockheed Martin received a $8.3 billion contract from the Department of Defense to produce F-35 Lightning II fighter jets. The contract was announced by Secretary of Defense Lloyd Austin on July 28, 2023.'
        }
    ]


def get_entity_type_info() -> Dict[str, Dict[str, str]]:
    """Get information about entity types supported by spaCy.

    Returns:
        Dictionary mapping entity type codes to descriptions and colors
    """
    return {
        'PERSON': {
            'description': 'People, including fictional characters',
            'color': '#3b82f6',  # blue
            'examples': 'Tim Cook, Jennifer Doudna, Lionel Messi'
        },
        'ORG': {
            'description': 'Companies, agencies, institutions',
            'color': '#10b981',  # green
            'examples': 'Apple, Microsoft, United Nations'
        },
        'GPE': {
            'description': 'Geopolitical entities (countries, cities, states)',
            'color': '#f59e0b',  # orange
            'examples': 'California, Paris, United States'
        },
        'DATE': {
            'description': 'Absolute or relative dates or periods',
            'color': '#8b5cf6',  # purple
            'examples': 'June 8, 2023, Thursday, 2020'
        },
        'TIME': {
            'description': 'Times smaller than a day',
            'color': '#ec4899',  # pink
            'examples': '3:00 PM, morning, evening'
        },
        'MONEY': {
            'description': 'Monetary values, including currency',
            'color': '#14b8a6',  # teal
            'examples': '$69 billion, 100 euros'
        },
        'PERCENT': {
            'description': 'Percentage values',
            'color': '#f97316',  # orange-red
            'examples': '50%, three percent'
        },
        'PRODUCT': {
            'description': 'Objects, vehicles, foods, etc.',
            'color': '#06b6d4',  # cyan
            'examples': 'iPhone 15, Cybertruck'
        },
        'EVENT': {
            'description': 'Named hurricanes, battles, wars, sports events',
            'color': '#84cc16',  # lime
            'examples': 'World Cup, COP28'
        },
        'LOC': {
            'description': 'Non-GPE locations, mountain ranges, bodies of water',
            'color': '#eab308',  # yellow
            'examples': 'Lusail Stadium, White House'
        },
        'FAC': {
            'description': 'Buildings, airports, highways, bridges',
            'color': '#64748b',  # slate
            'examples': 'Gigafactory, Louvre Museum'
        },
        'NORP': {
            'description': 'Nationalities, religious or political groups',
            'color': '#a855f7',  # purple-alt
            'examples': 'American, Democratic, Catholic'
        },
        'WORK_OF_ART': {
            'description': 'Titles of books, songs, movies, etc.',
            'color': '#ec4899',  # pink-alt
            'examples': 'Oppenheimer, Nature magazine'
        },
        'LAW': {
            'description': 'Named laws, acts, or legal documents',
            'color': '#78716c',  # stone
            'examples': 'Constitution, Civil Rights Act'
        },
        'LANGUAGE': {
            'description': 'Named languages',
            'color': '#06b6d4',  # cyan-alt
            'examples': 'English, Spanish, Mandarin'
        }
    }


def get_all_entity_types() -> List[str]:
    """Get list of all supported entity types.

    Returns:
        List of entity type codes
    """
    return list(get_entity_type_info().keys())


def format_text_with_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    """Format text with entity annotations for display.

    Args:
        text: Original text
        entities: List of entity dictionaries with 'start', 'end', 'label', 'text'

    Returns:
        Formatted text with entity markers
    """
    if not entities:
        return text

    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])

    formatted_parts = []
    last_end = 0

    for entity in sorted_entities:
        # Add text before entity
        if entity['start'] > last_end:
            formatted_parts.append(text[last_end:entity['start']])

        # Add entity with label
        formatted_parts.append(f"[{entity['text']}]({entity['label']})")

        last_end = entity['end']

    # Add remaining text
    if last_end < len(text):
        formatted_parts.append(text[last_end:])

    return ''.join(formatted_parts)


def get_sample_text_by_id(text_id: str) -> Dict[str, str]:
    """Get a specific sample text by ID.

    Args:
        text_id: ID of the sample text

    Returns:
        Dictionary with text information, or first text if ID not found
    """
    texts = get_default_texts()
    for text in texts:
        if text['id'] == text_id:
            return text

    # Return first text if ID not found
    return texts[0]
