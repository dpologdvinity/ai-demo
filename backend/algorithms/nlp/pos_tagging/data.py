"""Data loading and sample sentences for Part-of-Speech Tagging algorithm."""

from typing import List, Dict, Any


def get_sample_sentences() -> List[Dict[str, str]]:
    """Get sample sentences for POS tagging demonstration.

    Returns a diverse set of sentences with various grammatical structures,
    tenses, voices, and syntactic patterns.

    Returns:
        List of dictionaries with 'id', 'title', 'text', and 'description' keys
    """
    return [
        {
            'id': 'simple_present',
            'title': 'Simple Present',
            'text': 'The cat sits on the mat.',
            'description': 'Simple declarative sentence with present tense'
        },
        {
            'id': 'simple_past',
            'title': 'Simple Past',
            'text': 'She walked to the store yesterday.',
            'description': 'Simple past tense with temporal adverb'
        },
        {
            'id': 'simple_future',
            'title': 'Simple Future',
            'text': 'They will travel to Paris next summer.',
            'description': 'Future tense with modal auxiliary'
        },
        {
            'id': 'present_continuous',
            'title': 'Present Continuous',
            'text': 'The children are playing in the garden.',
            'description': 'Present continuous tense'
        },
        {
            'id': 'past_continuous',
            'title': 'Past Continuous',
            'text': 'I was reading a book when the phone rang.',
            'description': 'Past continuous with subordinate clause'
        },
        {
            'id': 'present_perfect',
            'title': 'Present Perfect',
            'text': 'We have seen that movie three times.',
            'description': 'Present perfect tense'
        },
        {
            'id': 'past_perfect',
            'title': 'Past Perfect',
            'text': 'He had finished his homework before dinner.',
            'description': 'Past perfect with temporal clause'
        },
        {
            'id': 'question',
            'title': 'Question',
            'text': 'Where did you find that beautiful painting?',
            'description': 'Interrogative sentence with adjective modifier'
        },
        {
            'id': 'imperative',
            'title': 'Imperative',
            'text': 'Please close the door quietly.',
            'description': 'Imperative sentence with adverb'
        },
        {
            'id': 'passive_voice',
            'title': 'Passive Voice',
            'text': 'The letter was written by my grandmother.',
            'description': 'Passive voice construction'
        },
        {
            'id': 'passive_present',
            'title': 'Passive Present',
            'text': 'English is spoken in many countries around the world.',
            'description': 'Present tense passive voice'
        },
        {
            'id': 'complex_sentence',
            'title': 'Complex Sentence',
            'text': 'Although it was raining heavily, we decided to go hiking.',
            'description': 'Complex sentence with subordinating conjunction'
        },
        {
            'id': 'compound_sentence',
            'title': 'Compound Sentence',
            'text': 'The sun was shining brightly, but the wind was cold.',
            'description': 'Compound sentence with coordinating conjunction'
        },
        {
            'id': 'relative_clause',
            'title': 'Relative Clause',
            'text': 'The scientist who discovered the vaccine received the Nobel Prize.',
            'description': 'Sentence with relative clause'
        },
        {
            'id': 'adjectives',
            'title': 'Multiple Adjectives',
            'text': 'She wore a beautiful, elegant, and expensive silk dress.',
            'description': 'Multiple adjectives modifying a noun'
        },
        {
            'id': 'adverbs',
            'title': 'Multiple Adverbs',
            'text': 'He quickly and carefully examined the ancient manuscript thoroughly.',
            'description': 'Multiple adverbs modifying verbs'
        },
        {
            'id': 'prepositions',
            'title': 'Prepositional Phrases',
            'text': 'The book on the table near the window belongs to my sister.',
            'description': 'Multiple prepositional phrases'
        },
        {
            'id': 'conjunctions',
            'title': 'Coordinating Conjunctions',
            'text': 'Neither the manager nor the employees knew about the changes.',
            'description': 'Correlative conjunctions'
        },
        {
            'id': 'modal_verbs',
            'title': 'Modal Verbs',
            'text': 'You should definitely consider this opportunity carefully.',
            'description': 'Modal auxiliary verbs'
        },
        {
            'id': 'phrasal_verbs',
            'title': 'Phrasal Verbs',
            'text': 'She looked up the information and wrote down the important details.',
            'description': 'Phrasal verbs with particles'
        },
        {
            'id': 'gerunds',
            'title': 'Gerunds',
            'text': 'Swimming in the ocean is my favorite summer activity.',
            'description': 'Gerund as subject'
        },
        {
            'id': 'infinitives',
            'title': 'Infinitives',
            'text': 'To succeed in life requires dedication and perseverance.',
            'description': 'Infinitive as subject'
        },
        {
            'id': 'participles',
            'title': 'Participles',
            'text': 'Exhausted from the long journey, the travelers rested at the inn.',
            'description': 'Participial phrase'
        },
        {
            'id': 'comparatives',
            'title': 'Comparatives & Superlatives',
            'text': 'This solution is more efficient than the previous one and the most cost-effective option.',
            'description': 'Comparative and superlative adjectives'
        },
        {
            'id': 'complex_syntax',
            'title': 'Complex Syntax',
            'text': 'Having carefully reviewed all the evidence, the judge concluded that justice had been served.',
            'description': 'Complex sentence with perfect participle and passive subordinate clause'
        }
    ]


def get_pos_tag_descriptions() -> Dict[str, Dict[str, str]]:
    """Get descriptions for common POS tags.

    Returns:
        Dictionary mapping tag codes to their descriptions
    """
    return {
        # Universal POS tags
        'ADJ': {'full': 'Adjective', 'description': 'Describes or modifies nouns'},
        'ADP': {'full': 'Adposition', 'description': 'Preposition or postposition'},
        'ADV': {'full': 'Adverb', 'description': 'Modifies verbs, adjectives, or other adverbs'},
        'AUX': {'full': 'Auxiliary', 'description': 'Auxiliary verb (be, have, will, etc.)'},
        'CONJ': {'full': 'Conjunction', 'description': 'Connects words, phrases, or clauses'},
        'CCONJ': {'full': 'Coordinating Conjunction', 'description': 'Coordinates equal elements (and, or, but)'},
        'DET': {'full': 'Determiner', 'description': 'Determines or quantifies nouns (the, a, some)'},
        'INTJ': {'full': 'Interjection', 'description': 'Exclamation (oh, wow, hey)'},
        'NOUN': {'full': 'Noun', 'description': 'Person, place, thing, or idea'},
        'NUM': {'full': 'Numeral', 'description': 'Number or quantity'},
        'PART': {'full': 'Particle', 'description': 'Function word (to, not)'},
        'PRON': {'full': 'Pronoun', 'description': 'Replaces or refers to nouns'},
        'PROPN': {'full': 'Proper Noun', 'description': 'Specific name of person, place, or thing'},
        'PUNCT': {'full': 'Punctuation', 'description': 'Punctuation marks'},
        'SCONJ': {'full': 'Subordinating Conjunction', 'description': 'Introduces subordinate clauses (because, if, when)'},
        'SYM': {'full': 'Symbol', 'description': 'Mathematical or other symbols'},
        'VERB': {'full': 'Verb', 'description': 'Action or state of being'},
        'X': {'full': 'Other', 'description': 'Other/unknown category'},
        'SPACE': {'full': 'Space', 'description': 'Whitespace'},

        # Penn Treebank tags (fine-grained)
        'CC': {'full': 'Coordinating conjunction', 'description': 'and, or, but'},
        'CD': {'full': 'Cardinal number', 'description': 'one, two, three'},
        'DT': {'full': 'Determiner', 'description': 'the, a, an'},
        'EX': {'full': 'Existential there', 'description': 'there is, there are'},
        'FW': {'full': 'Foreign word', 'description': 'Foreign words'},
        'IN': {'full': 'Preposition/subordinating conjunction', 'description': 'in, of, like, after'},
        'JJ': {'full': 'Adjective', 'description': 'big, old, green'},
        'JJR': {'full': 'Adjective, comparative', 'description': 'bigger, older, greener'},
        'JJS': {'full': 'Adjective, superlative', 'description': 'biggest, oldest, greenest'},
        'LS': {'full': 'List item marker', 'description': '1, 2, a, b'},
        'MD': {'full': 'Modal', 'description': 'can, could, will, would'},
        'NN': {'full': 'Noun, singular', 'description': 'cat, tree, person'},
        'NNS': {'full': 'Noun, plural', 'description': 'cats, trees, people'},
        'NNP': {'full': 'Proper noun, singular', 'description': 'John, London, NASA'},
        'NNPS': {'full': 'Proper noun, plural', 'description': 'Americans, Beatles'},
        'PDT': {'full': 'Predeterminer', 'description': 'all, both, half'},
        'POS': {'full': 'Possessive ending', 'description': "'s"},
        'PRP': {'full': 'Personal pronoun', 'description': 'I, you, he, she'},
        'PRP$': {'full': 'Possessive pronoun', 'description': 'my, your, his, her'},
        'RB': {'full': 'Adverb', 'description': 'quickly, very, well'},
        'RBR': {'full': 'Adverb, comparative', 'description': 'faster, better'},
        'RBS': {'full': 'Adverb, superlative', 'description': 'fastest, best'},
        'RP': {'full': 'Particle', 'description': 'up, off, out'},
        'SYM': {'full': 'Symbol', 'description': '+, %, &'},
        'TO': {'full': 'to', 'description': 'to (infinitive marker)'},
        'UH': {'full': 'Interjection', 'description': 'oh, wow, oops'},
        'VB': {'full': 'Verb, base form', 'description': 'be, have, go'},
        'VBD': {'full': 'Verb, past tense', 'description': 'was, had, went'},
        'VBG': {'full': 'Verb, gerund/present participle', 'description': 'being, having, going'},
        'VBN': {'full': 'Verb, past participle', 'description': 'been, had, gone'},
        'VBP': {'full': 'Verb, non-3rd person singular present', 'description': 'am, are, have'},
        'VBZ': {'full': 'Verb, 3rd person singular present', 'description': 'is, has, goes'},
        'WDT': {'full': 'Wh-determiner', 'description': 'which, that, whatever'},
        'WP': {'full': 'Wh-pronoun', 'description': 'who, what, whom'},
        'WP$': {'full': 'Possessive wh-pronoun', 'description': 'whose'},
        'WRB': {'full': 'Wh-adverb', 'description': 'when, where, why, how'},
    }


def get_pos_tag_colors() -> Dict[str, str]:
    """Get color mappings for POS tags for visualization.

    Returns:
        Dictionary mapping POS tag categories to color codes
    """
    return {
        # Universal tags
        'NOUN': '#4285F4',      # Blue
        'PROPN': '#185ABC',     # Dark Blue
        'VERB': '#EA4335',      # Red
        'AUX': '#F66',          # Light Red
        'ADJ': '#FBBC04',       # Yellow
        'ADV': '#FF9800',       # Orange
        'PRON': '#34A853',      # Green
        'DET': '#0F9D58',       # Dark Green
        'ADP': '#9C27B0',       # Purple
        'CONJ': '#673AB7',      # Deep Purple
        'CCONJ': '#673AB7',     # Deep Purple
        'SCONJ': '#7B1FA2',     # Dark Purple
        'PART': '#E91E63',      # Pink
        'NUM': '#00BCD4',       # Cyan
        'INTJ': '#FF5722',      # Deep Orange
        'PUNCT': '#9E9E9E',     # Grey
        'SYM': '#607D8B',       # Blue Grey
        'X': '#BDBDBD',         # Light Grey
        'SPACE': '#F5F5F5',     # Very Light Grey
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the POS tagging dataset.

    Returns:
        Dictionary with dataset metadata
    """
    sentences = get_sample_sentences()

    return {
        'name': 'Diverse Grammatical Structures Dataset',
        'description': (
            'Collection of 25+ sentences demonstrating various grammatical structures, '
            'tenses, voices, and syntactic patterns for POS tagging demonstration'
        ),
        'size': len(sentences),
        'features': [
            'Simple and complex sentences',
            'Various verb tenses (present, past, future, perfect)',
            'Active and passive voice',
            'Questions and imperatives',
            'Subordinate and relative clauses',
            'Multiple parts of speech',
            'Phrasal verbs and participles',
            'Comparative and superlative forms'
        ],
        'categories': [
            'Simple declarative',
            'Questions',
            'Complex sentences',
            'Passive voice',
            'Multiple clauses',
            'Various tenses',
            'Participles and gerunds'
        ],
        'samples': sentences[:5]  # First 5 samples as preview
    }
