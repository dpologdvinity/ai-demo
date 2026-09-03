from fastapi import APIRouter, HTTPException
import time
from typing import Dict, Any
import logging

from algorithms.nlp.word2vec import (
    Word2VecModel,
    Word2VecParameters,
    Word2VecResponse
)
from algorithms.nlp.ner import (
    NERModel,
    NERParameters,
    NERResponse,
    get_default_texts,
    get_entity_type_info,
    get_all_entity_types
)
from algorithms.nlp.sentiment_analysis import (
    SentimentAnalysisModel,
    SentimentAnalysisParameters,
    SentimentAnalysisResponse
)
from algorithms.nlp.tfidf import (
    TFIDFModel,
    train_tfidf,
    TFIDFRequest,
    TFIDFResponse,
    get_dataset_info as get_tfidf_dataset_info
)
from algorithms.nlp.bag_of_words import (
    BagOfWordsModel,
    BagOfWordsRequest,
    BagOfWordsResponse,
    get_dataset_info as get_bow_dataset_info
)
from algorithms.nlp.bert_finetuning import (
    BERTFinetuningModel,
    BERTFinetuningParameters,
    BERTFinetuningResponse,
    get_dataset_info as get_bert_dataset_info
)
from algorithms.nlp.glove import (
    GloVeModel,
    GloVeParameters,
    GloVeResponse
)
from algorithms.nlp.text_classification import (
    TextClassificationModel,
    TextClassificationParameters,
    TextClassificationResponse,
    get_dataset_info as get_text_classification_dataset_info,
    get_category_descriptions
)
from algorithms.nlp.tokenization import (
    TokenizationModel,
    TokenizationRequest,
    TokenizationResponse,
    get_dataset_info as get_tokenization_dataset_info,
    get_sample_texts,
    get_text_info
)
from algorithms.nlp.topic_modeling import (
    TopicModelingModel,
    TopicModelingParameters,
    TopicModelingResponse,
    get_dataset_info as get_topic_modeling_dataset_info
)
from algorithms.nlp.seq2seq import (
    Seq2SeqModel,
    Seq2SeqParameters,
    Seq2SeqResponse,
    get_dataset_info as get_seq2seq_dataset_info
)
from algorithms.nlp.pos_tagging import (
    POSTaggingModel,
    POSTaggingParameters,
    POSTaggingResponse,
    get_sample_sentences,
    get_pos_tag_descriptions,
    get_dataset_info as get_pos_tagging_dataset_info
)
from utils.algorithm_metadata import (
    AlgorithmMetadata,
    AlgorithmParameter,
    AlgorithmComplexity,
    AlgorithmCategory,
    DifficultyLevel,
    AlgorithmRegistry,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/nlp", tags=["Natural Language Processing"])


# Register Word2Vec metadata
word2vec_metadata = AlgorithmMetadata(
    id="word2vec",
    name="Word2Vec",
    slug="word2vec",
    category=AlgorithmCategory.NLP,
    description="Neural network model for learning word embeddings from text",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["nlp", "embeddings", "word-vectors", "unsupervised"],
    use_cases=[
        "Text similarity",
        "Document clustering",
        "Feature extraction for NLP",
        "Semantic search"
    ],
    complexity=AlgorithmComplexity(
        time="O(corpus_size × window × vector_dim)",
        space="O(vocab_size × vector_dim)"
    ),
    parameters=[
        AlgorithmParameter(
            name="vector_size",
            label="Embedding Dimension",
            type="range",
            default=100,
            min=50,
            max=300,
            step=10,
            description="Dimensionality of word embeddings"
        ),
        AlgorithmParameter(
            name="window",
            label="Context Window Size",
            type="range",
            default=5,
            min=2,
            max=10,
            step=1,
            description="Maximum distance between current and predicted word"
        ),
        AlgorithmParameter(
            name="min_count",
            label="Minimum Word Frequency",
            type="range",
            default=5,
            min=1,
            max=20,
            step=1,
            description="Ignores words with frequency lower than this"
        ),
        AlgorithmParameter(
            name="sg",
            label="Training Algorithm",
            type="select",
            default=0,
            options=[
                {"label": "CBOW (Continuous Bag of Words)", "value": 0},
                {"label": "Skip-gram", "value": 1}
            ],
            description="Training algorithm: CBOW predicts target word from context, Skip-gram predicts context from target"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=10,
            min=5,
            max=50,
            step=5,
            description="Number of training iterations over the corpus"
        )
    ],
    dataset_name="sample_corpus",
    visualization_type="scatter_2d",
    theory=(
        "Word2Vec learns vector representations (embeddings) of words from large text corpora. "
        "The key insight is that words appearing in similar contexts tend to have similar meanings. "
        "Word2Vec uses shallow neural networks to learn these representations through two architectures: "
        "CBOW (Continuous Bag of Words) predicts a target word from its surrounding context words, while "
        "Skip-gram predicts context words from a target word. The resulting word vectors capture semantic "
        "and syntactic relationships, enabling operations like 'king - man + woman ≈ queen'. These embeddings "
        "can be used as features for downstream NLP tasks, significantly improving performance over traditional "
        "bag-of-words representations."
    ),
    pros=[
        "Captures semantic and syntactic word relationships",
        "Produces dense, low-dimensional representations",
        "Enables word similarity and analogy queries",
        "Fast training on large corpora",
        "Pre-trained models available for many languages"
    ],
    cons=[
        "Cannot handle out-of-vocabulary words",
        "Single vector per word (no context-dependent embeddings)",
        "Requires large corpus for quality embeddings",
        "Fixed vocabulary after training",
        "Does not capture word order within context window"
    ],
    related_algorithms=["glove", "fasttext", "bert"]
)

AlgorithmRegistry.register(word2vec_metadata)


# Register GloVe metadata
glove_metadata = AlgorithmMetadata(
    id="glove",
    name="GloVe Word Embeddings",
    slug="glove",
    category=AlgorithmCategory.NLP,
    description="Word representation learning using global word co-occurrence statistics",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["nlp", "embeddings", "word-vectors", "unsupervised"],
    use_cases=[
        "Text similarity",
        "Document clustering",
        "Machine translation",
        "Named entity recognition",
        "Sentiment analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(corpus_size) for training, O(1) for lookup",
        space="O(vocab_size × embedding_dim)"
    ),
    parameters=[
        AlgorithmParameter(
            name="embedding_dim",
            label="Embedding Dimension",
            type="select",
            default=100,
            options=[
                {"label": "50 dimensions", "value": 50},
                {"label": "100 dimensions", "value": 100},
                {"label": "200 dimensions", "value": 200},
                {"label": "300 dimensions", "value": 300}
            ],
            description="Vector dimension for word embeddings"
        ),
        AlgorithmParameter(
            name="top_k",
            label="Similar Words Count",
            type="range",
            default=10,
            min=5,
            max=20,
            step=1,
            description="Number of similar words to show"
        ),
        AlgorithmParameter(
            name="query_word",
            label="Query Word",
            type="text",
            default="king",
            description="Word to find similar words for"
        ),
        AlgorithmParameter(
            name="analogy_word_a",
            label="Analogy Word A",
            type="text",
            default="king",
            description="First word in analogy (A - B + C)"
        ),
        AlgorithmParameter(
            name="analogy_word_b",
            label="Analogy Word B",
            type="text",
            default="man",
            description="Second word in analogy (A - B + C)"
        ),
        AlgorithmParameter(
            name="analogy_word_c",
            label="Analogy Word C",
            type="text",
            default="woman",
            description="Third word in analogy (A - B + C)"
        )
    ],
    dataset_name="pre_trained_glove",
    visualization_type="scatter_2d",
    theory=(
        "GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm "
        "for obtaining vector representations of words. Unlike Word2Vec which uses local context "
        "windows, GloVe is trained on global word-word co-occurrence statistics from a corpus. "
        "The key insight is that ratios of word-word co-occurrence probabilities encode meaning. "
        "For example, the ratio of probabilities P(k|ice)/P(k|steam) will be large for k=solid, "
        "small for k=gas, and near 1 for k=water or k=fashion. GloVe learns word vectors such that "
        "their dot product equals the logarithm of the words' probability of co-occurrence. "
        "This results in a word vector space where the dimensions have meaningful interpretations "
        "and linear substructures emerge (e.g., vec(king) - vec(man) + vec(woman) ≈ vec(queen)). "
        "GloVe embeddings are widely used as pre-trained features for various NLP tasks."
    ),
    pros=[
        "Captures both global corpus statistics and local context",
        "Efficient training through matrix factorization",
        "Pre-trained vectors available for multiple languages",
        "Linear substructures enable word analogies",
        "Better performance than count-based methods on word similarity tasks"
    ],
    cons=[
        "Requires large corpus for quality embeddings",
        "Cannot handle out-of-vocabulary words",
        "Single static embedding per word (no context-dependent meanings)",
        "Training requires significant memory for co-occurrence matrix",
        "Fixed vocabulary after training"
    ],
    related_algorithms=["word2vec", "fasttext", "bert"]
)

AlgorithmRegistry.register(glove_metadata)


# Register Sentiment Analysis metadata
sentiment_analysis_metadata = AlgorithmMetadata(
    id="sentiment-analysis",
    name="Sentiment Analysis",
    slug="sentiment-analysis",
    category=AlgorithmCategory.NLP,
    description="Classify text sentiment as positive, negative, or neutral",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["nlp", "classification", "text-analysis", "sentiment"],
    use_cases=[
        "Social media monitoring",
        "Customer feedback analysis",
        "Brand reputation tracking",
        "Market research",
        "Product review analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(n × m) where n=number of texts, m=average text length",
        space="O(n) for storing predictions"
    ),
    parameters=[
        AlgorithmParameter(
            name="model_type",
            label="Model Type",
            type="select",
            default="vader",
            options=[
                {"label": "VADER (Rule-based)", "value": "vader"},
                {"label": "TextBlob", "value": "textblob"},
                {"label": "Transformers", "value": "transformers"}
            ],
            description="Sentiment analysis model to use"
        ),
        AlgorithmParameter(
            name="confidence_threshold",
            label="Confidence Threshold",
            type="range",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Minimum confidence score threshold"
        ),
        AlgorithmParameter(
            name="neutral_threshold",
            label="Neutral Threshold",
            type="range",
            default=0.05,
            min=0.0,
            max=0.5,
            step=0.01,
            description="Threshold for neutral classification (VADER compound score range)"
        )
    ],
    dataset_name="sample_reviews",
    visualization_type="pie_chart",
    theory=(
        "Sentiment Analysis is a natural language processing technique used to determine whether text "
        "expresses positive, negative, or neutral sentiment. VADER (Valence Aware Dictionary and sEntiment "
        "Reasoner) is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments "
        "expressed in social media. It uses a combination of sentiment lexicons (words associated with "
        "sentiment intensity) and grammatical rules to analyze text. VADER considers word intensity, "
        "punctuation, capitalization, degree modifiers, and negation to produce a compound sentiment score "
        "ranging from -1 (most negative) to +1 (most positive). The algorithm excels at handling social "
        "media text, including emoticons, slang, and acronyms, making it ideal for analyzing tweets, "
        "reviews, and informal text."
    ),
    pros=[
        "Fast and efficient - no training required",
        "Works well on social media text and reviews",
        "Handles emoticons, slang, and informal language",
        "Provides detailed sentiment scores",
        "No need for labeled training data"
    ],
    cons=[
        "Rule-based approach may miss complex context",
        "Limited to English language",
        "Cannot learn domain-specific sentiment patterns",
        "May struggle with sarcasm and irony",
        "Fixed lexicon cannot adapt to new terms"
    ],
    related_algorithms=["text-classification", "tfidf", "word2vec"]
)

AlgorithmRegistry.register(sentiment_analysis_metadata)


# Register TF-IDF metadata
tfidf_metadata = AlgorithmMetadata(
    id="tfidf",
    name="TF-IDF",
    slug="tfidf",
    category=AlgorithmCategory.NLP,
    description="Statistical measure for evaluating word importance in documents",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["nlp", "feature-extraction", "text-mining", "information-retrieval"],
    use_cases=[
        "Document classification",
        "Information retrieval",
        "Search engines",
        "Text mining",
        "Keyword extraction"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*m)",
        space="O(n*m)"
    ),
    parameters=[
        AlgorithmParameter(
            name="max_features",
            label="Maximum Features",
            type="range",
            default=100,
            min=10,
            max=500,
            step=10,
            description="Maximum number of features (terms) to extract"
        ),
        AlgorithmParameter(
            name="ngram_range",
            label="N-gram Range",
            type="select",
            default=(1, 1),
            options=[
                {"label": "Unigrams (1,1)", "value": (1, 1)},
                {"label": "Unigrams + Bigrams (1,2)", "value": (1, 2)},
                {"label": "Unigrams + Bigrams + Trigrams (1,3)", "value": (1, 3)}
            ],
            description="Range of n-grams to extract"
        ),
        AlgorithmParameter(
            name="min_df",
            label="Min Document Frequency",
            type="range",
            default=1,
            min=1,
            max=10,
            step=1,
            description="Minimum number of documents a term must appear in"
        ),
        AlgorithmParameter(
            name="max_df",
            label="Max Document Frequency",
            type="range",
            default=1.0,
            min=0.5,
            max=1.0,
            step=0.1,
            description="Maximum document frequency (as proportion)"
        ),
        AlgorithmParameter(
            name="use_idf",
            label="Use IDF",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Enable inverse document frequency weighting"
        ),
    ],
    dataset_name="news_corpus",
    visualization_type="heatmap",
    theory=(
        "TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that "
        "reflects how important a word is to a document in a collection of documents. "
        "It is composed of two parts:\n\n"
        "1. Term Frequency (TF): Measures how frequently a term appears in a document. "
        "TF(t,d) = (Number of times term t appears in document d) / (Total terms in document d)\n\n"
        "2. Inverse Document Frequency (IDF): Measures how important a term is across all documents. "
        "Rare terms have high IDF scores, while common terms have low scores. "
        "IDF(t) = log(Total documents / Documents containing term t)\n\n"
        "TF-IDF(t,d) = TF(t,d) × IDF(t)\n\n"
        "This weighting scheme penalizes common words (like 'the', 'is') and emphasizes "
        "distinctive words that carry semantic meaning. TF-IDF is widely used in information "
        "retrieval, text mining, and document similarity calculations."
    ),
    pros=[
        "Simple and interpretable statistical measure",
        "Effectively identifies important terms in documents",
        "Handles vocabulary of any size",
        "Fast computation and low memory requirements",
        "Works well for sparse text data"
    ],
    cons=[
        "Doesn't capture semantic relationships between words",
        "Ignores word order and context",
        "Sensitive to document length",
        "May struggle with synonyms and polysemy",
        "Requires large corpus for best results"
    ],
    related_algorithms=["word2vec", "bag-of-words", "lsa"]
)

AlgorithmRegistry.register(tfidf_metadata)


# Register Bag of Words metadata
bag_of_words_metadata = AlgorithmMetadata(
    id="bag-of-words",
    name="Bag of Words (BoW)",
    slug="bag-of-words",
    category=AlgorithmCategory.NLP,
    description="Text representation as word frequency vectors",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["nlp", "feature-extraction", "text-representation", "vectorization"],
    use_cases=[
        "Document classification",
        "Text clustering",
        "Information retrieval",
        "Spam detection",
        "Topic modeling preprocessing"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*m) where n=docs, m=avg words",
        space="O(vocab_size*n_docs)"
    ),
    parameters=[
        AlgorithmParameter(
            name="max_features",
            label="Maximum Vocabulary Size",
            type="range",
            default=100,
            min=10,
            max=1000,
            step=10,
            description="Maximum number of terms in the vocabulary"
        ),
        AlgorithmParameter(
            name="ngram_range",
            label="N-gram Range",
            type="select",
            default=(1, 1),
            options=[
                {"label": "Unigrams (1,1)", "value": (1, 1)},
                {"label": "Unigrams + Bigrams (1,2)", "value": (1, 2)},
                {"label": "Bigrams only (2,2)", "value": (2, 2)}
            ],
            description="Range of n-grams to extract from text"
        ),
        AlgorithmParameter(
            name="min_df",
            label="Min Document Frequency",
            type="range",
            default=1,
            min=1,
            max=5,
            step=1,
            description="Minimum number of documents a term must appear in"
        ),
        AlgorithmParameter(
            name="max_df",
            label="Max Document Frequency",
            type="range",
            default=1.0,
            min=0.5,
            max=1.0,
            step=0.1,
            description="Maximum document frequency as proportion (0.5-1.0)"
        ),
        AlgorithmParameter(
            name="binary",
            label="Binary Counts",
            type="select",
            default=False,
            options=[
                {"label": "Frequency counts", "value": False},
                {"label": "Binary (presence/absence)", "value": True}
            ],
            description="Use binary counts instead of term frequencies"
        ),
    ],
    dataset_name="mixed_text_corpus",
    visualization_type="heatmap",
    theory=(
        "Bag of Words (BoW) is one of the simplest and most fundamental text representation methods "
        "in natural language processing. It represents text as an unordered collection (or 'bag') of "
        "words, disregarding grammar and word order but keeping track of word frequencies. Each document "
        "is converted into a fixed-length vector where each dimension corresponds to a unique word in "
        "the vocabulary, and the value represents the frequency (or presence) of that word in the document.\n\n"
        "The process involves:\n"
        "1. Building a vocabulary of unique words from the entire corpus\n"
        "2. Creating a document-term matrix where rows represent documents and columns represent words\n"
        "3. Populating the matrix with word counts (or binary values for presence/absence)\n\n"
        "BoW creates sparse, high-dimensional vectors that can be used as input for machine learning "
        "algorithms. Despite its simplicity and limitations (ignoring word order and semantics), BoW "
        "remains effective for many text classification and information retrieval tasks."
    ),
    pros=[
        "Simple and intuitive representation",
        "Fast computation with low overhead",
        "Works well for text classification and clustering",
        "No training required - deterministic transformation",
        "Effective baseline for many NLP tasks"
    ],
    cons=[
        "Ignores word order and grammar",
        "Creates high-dimensional sparse vectors",
        "No semantic understanding of words",
        "Sensitive to vocabulary size",
        "Cannot handle out-of-vocabulary words",
        "Treats all words equally (no context weighting)"
    ],
    related_algorithms=["tfidf", "word2vec", "n-grams"]
)

AlgorithmRegistry.register(bag_of_words_metadata)


# Register NER metadata
ner_metadata = AlgorithmMetadata(
    id="ner",
    name="Named Entity Recognition (NER)",
    slug="ner",
    category=AlgorithmCategory.NLP,
    description="Identify and classify named entities (people, organizations, locations, dates) in text",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["nlp", "ner", "entity-extraction", "information-extraction", "spacy"],
    use_cases=[
        "Information extraction",
        "Document indexing",
        "Knowledge graph construction",
        "Search enhancement",
        "Content categorization",
        "Resume parsing",
        "News analytics"
    ],
    complexity=AlgorithmComplexity(
        time="O(n)",
        space="O(entities)"
    ),
    parameters=[
        AlgorithmParameter(
            name="model_name",
            label="spaCy Model",
            type="select",
            default="en_core_web_sm",
            options=[
                {"label": "Small (en_core_web_sm)", "value": "en_core_web_sm"},
                {"label": "Medium (en_core_web_md)", "value": "en_core_web_md"}
            ],
            description="spaCy model to use for entity extraction"
        ),
        AlgorithmParameter(
            name="entity_types",
            label="Entity Types",
            type="multiselect",
            default=["PERSON", "ORG", "GPE", "DATE"],
            options=[
                {"label": "Person", "value": "PERSON"},
                {"label": "Organization", "value": "ORG"},
                {"label": "Geopolitical Entity", "value": "GPE"},
                {"label": "Date", "value": "DATE"},
                {"label": "Time", "value": "TIME"},
                {"label": "Money", "value": "MONEY"},
                {"label": "Percent", "value": "PERCENT"},
                {"label": "Product", "value": "PRODUCT"},
                {"label": "Event", "value": "EVENT"},
                {"label": "Location", "value": "LOC"},
                {"label": "Facility", "value": "FAC"},
                {"label": "Nationality/Religion/Politics", "value": "NORP"},
                {"label": "Work of Art", "value": "WORK_OF_ART"},
                {"label": "Law", "value": "LAW"},
                {"label": "Language", "value": "LANGUAGE"}
            ],
            description="Types of entities to extract"
        ),
        AlgorithmParameter(
            name="confidence_threshold",
            label="Confidence Threshold",
            type="range",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.1,
            description="Minimum confidence score for entity extraction"
        ),
        AlgorithmParameter(
            name="text_index",
            label="Sample Text",
            type="range",
            default=0,
            min=0,
            max=29,
            step=1,
            description="Sample text selector (0-29)"
        ),
        AlgorithmParameter(
            name="merge_entities",
            label="Merge Adjacent Entities",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Merge adjacent entities of the same type"
        )
    ],
    dataset_name="diverse_sample_texts",
    visualization_type="annotated_text",
    theory=(
        "Named Entity Recognition (NER) is a fundamental natural language processing task that automatically "
        "identifies and classifies named entities in text into predefined categories. These categories typically "
        "include PERSON (people and fictional characters), ORG (companies, agencies, institutions), GPE (geopolitical "
        "entities like countries, cities, and states), DATE (absolute or relative dates), TIME (times smaller than a day), "
        "MONEY (monetary values), PERCENT (percentages), PRODUCT (objects, vehicles, goods), EVENT (named events like "
        "wars or sports competitions), LOC (non-GPE locations), FAC (buildings and facilities), NORP (nationalities, "
        "religious or political groups), WORK_OF_ART (titles of creative works), LAW (named laws and legal documents), "
        "and LANGUAGE (named languages).\n\n"
        "Modern NER systems, like those implemented in spaCy, use deep learning models trained on large annotated corpora. "
        "spaCy's pre-trained models employ a transition-based algorithm with neural networks that process text sequentially, "
        "maintaining an internal state and making predictions for each token. The model uses both local context (surrounding "
        "words) and learned patterns from training data to classify entities. The neural network architecture typically includes "
        "word embeddings, convolutional or recurrent layers for context encoding, and a classification layer for entity type "
        "prediction.\n\n"
        "NER is crucial for many downstream NLP applications including information extraction, knowledge graph construction, "
        "question answering systems, document indexing and search, content recommendation, and automated text analysis. By "
        "identifying and categorizing key entities, NER enables structured information extraction from unstructured text, "
        "making it possible to build knowledge bases, enhance search engines, and power intelligent applications."
    ),
    pros=[
        "Accurate entity extraction from pre-trained models",
        "Supports multiple entity types out-of-the-box",
        "Fast inference on CPU and GPU",
        "Can be fine-tuned for domain-specific entities",
        "Integrates well with other NLP pipelines"
    ],
    cons=[
        "May struggle with rare or domain-specific entities",
        "Requires pre-trained models or training data",
        "Performance depends on text quality and formatting",
        "May miss entities with unusual capitalization or context",
        "Limited to predefined entity categories"
    ],
    related_algorithms=["pos-tagging", "dependency-parsing", "bert"]
)

AlgorithmRegistry.register(ner_metadata)


# Register BERT Fine-tuning metadata
bert_finetuning_metadata = AlgorithmMetadata(
    id="bert-finetuning",
    name="BERT Fine-tuning",
    slug="bert-finetuning",
    category=AlgorithmCategory.NLP,
    description="Fine-tune pre-trained BERT for text classification",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["nlp", "transformers", "transfer-learning", "text-classification", "deep-learning"],
    use_cases=[
        "Sentiment analysis",
        "Text classification",
        "Question answering",
        "Named entity recognition",
        "Intent detection"
    ],
    complexity=AlgorithmComplexity(
        time="O(T²×d×L) where T=sequence length, d=hidden dim, L=layers",
        space="O(model_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="model_name",
            label="BERT Model Variant",
            type="select",
            default="bert-base-uncased",
            options=[
                {"label": "BERT Base Uncased", "value": "bert-base-uncased"},
                {"label": "DistilBERT Base Uncased", "value": "distilbert-base-uncased"}
            ],
            description="Pre-trained BERT model variant to fine-tune"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=2e-5,
            min=1e-5,
            max=5e-5,
            step=5e-6,
            description="Learning rate for fine-tuning (smaller is more stable)"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=3,
            min=1,
            max=10,
            step=1,
            description="Number of training epochs (more epochs = longer training)"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="range",
            default=16,
            min=8,
            max=32,
            step=8,
            description="Batch size for training (larger = more memory)"
        ),
        AlgorithmParameter(
            name="max_length",
            label="Max Sequence Length",
            type="range",
            default=128,
            min=64,
            max=512,
            step=64,
            description="Maximum sequence length for tokenization"
        )
    ],
    dataset_name="sentiment_classification",
    visualization_type="multi_chart",
    theory=(
        "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained deep learning "
        "model that revolutionized natural language processing. Unlike traditional models that process text "
        "unidirectionally, BERT reads text bidirectionally, understanding context from both left and right. "
        "Fine-tuning BERT involves taking a pre-trained BERT model (trained on massive text corpora) and "
        "adapting it to a specific task with a small amount of labeled data. This transfer learning approach "
        "achieves state-of-the-art results on various NLP tasks. The model uses self-attention mechanisms to "
        "weigh the importance of different words in a sequence, capturing complex linguistic patterns and "
        "relationships. For text classification, a classification layer is added on top of BERT's output, "
        "and the entire model is fine-tuned with a small learning rate to preserve the pre-trained knowledge "
        "while adapting to the new task. The attention mechanism allows visualization of which words the model "
        "focuses on when making predictions, providing interpretability."
    ),
    pros=[
        "State-of-the-art performance on NLP tasks",
        "Requires less labeled data due to pre-training",
        "Bidirectional context understanding",
        "Captures complex linguistic patterns",
        "Attention weights provide interpretability",
        "Pre-trained models available for many languages"
    ],
    cons=[
        "High computational requirements (GPU recommended)",
        "Large model size (110M+ parameters)",
        "Longer training time compared to simpler models",
        "Requires careful hyperparameter tuning",
        "May overfit on very small datasets",
        "Black-box nature despite attention visualizations"
    ],
    related_algorithms=["word2vec", "glove", "sentiment-analysis", "ner"]
)

AlgorithmRegistry.register(bert_finetuning_metadata)


# Register Text Classification metadata
text_classification_metadata = AlgorithmMetadata(
    id="text-classification",
    name="Text Classification",
    slug="text-classification",
    category=AlgorithmCategory.NLP,
    description="Classify text into predefined categories using ML or deep learning",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["nlp", "classification", "supervised", "text-mining"],
    use_cases=[
        "Sentiment analysis",
        "Spam detection",
        "Topic classification",
        "Intent detection",
        "Document categorization"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*m) where n=docs, m=features",
        space="O(vocab*classes)"
    ),
    parameters=[
        AlgorithmParameter(
            name="classifier_type",
            label="Classifier Type",
            type="select",
            default="naive_bayes",
            options=[
                {"label": "Naive Bayes", "value": "naive_bayes"},
                {"label": "Logistic Regression", "value": "logistic_regression"},
                {"label": "Support Vector Machine", "value": "svm"}
            ],
            description="Machine learning model for classification"
        ),
        AlgorithmParameter(
            name="max_features",
            label="Max Features",
            type="range",
            default=1000,
            min=100,
            max=5000,
            step=100,
            description="Maximum number of TF-IDF features to extract"
        ),
        AlgorithmParameter(
            name="test_size",
            label="Test Set Size",
            type="range",
            default=0.2,
            min=0.1,
            max=0.4,
            step=0.05,
            description="Proportion of data to use for testing"
        ),
        AlgorithmParameter(
            name="ngram_range",
            label="N-gram Range",
            type="select",
            default=(1, 2),
            options=[
                {"label": "Unigrams (1,1)", "value": (1, 1)},
                {"label": "Unigrams + Bigrams (1,2)", "value": (1, 2)},
                {"label": "Unigrams + Bigrams + Trigrams (1,3)", "value": (1, 3)}
            ],
            description="Range of n-grams to extract from text"
        )
    ],
    dataset_name="news_categories",
    visualization_type="confusion_matrix",
    theory=(
        "Text Classification is a supervised machine learning task that assigns predefined categories "
        "to text documents. The process involves converting text into numerical features using techniques "
        "like TF-IDF (Term Frequency-Inverse Document Frequency), which measures word importance in documents. "
        "Common classifiers include Naive Bayes, which uses probabilistic modeling based on Bayes' theorem; "
        "Logistic Regression, which learns linear decision boundaries; and Support Vector Machines (SVM), "
        "which find optimal hyperplanes to separate classes. The model is trained on labeled examples, learning "
        "patterns that distinguish different categories. During prediction, new texts are vectorized using the "
        "same feature extraction method and classified based on learned patterns. Performance is evaluated using "
        "metrics like precision (accuracy of positive predictions), recall (coverage of actual positives), and "
        "F1-score (harmonic mean of precision and recall). The confusion matrix visualizes correct and incorrect "
        "predictions across all classes, helping identify which categories are often confused."
    ),
    pros=[
        "Effective for various text categorization tasks",
        "Interpretable features show important words per class",
        "Fast training and prediction on large datasets",
        "Works well with limited training data",
        "Multiple algorithms available for different use cases",
        "Can handle multi-class classification naturally"
    ],
    cons=[
        "Requires labeled training data",
        "Performance depends on feature representation quality",
        "May struggle with nuanced or ambiguous texts",
        "Bag-of-words approach loses word order information",
        "Requires feature engineering and parameter tuning",
        "Cannot adapt to new categories without retraining"
    ],
    related_algorithms=["sentiment-analysis", "tfidf", "naive-bayes", "svm"]
)

AlgorithmRegistry.register(text_classification_metadata)


# Register Topic Modeling metadata
topic_modeling_metadata = AlgorithmMetadata(
    id="topic-modeling",
    name="Topic Modeling (LDA)",
    slug="topic-modeling",
    category=AlgorithmCategory.NLP,
    description="Discover abstract topics in document collections using probabilistic modeling",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["nlp", "topic-modeling", "lda", "unsupervised", "text-mining"],
    use_cases=[
        "Document clustering",
        "Content recommendation",
        "Trend analysis",
        "Research paper categorization",
        "Customer feedback analysis",
        "News article organization"
    ],
    complexity=AlgorithmComplexity(
        time="O(iterations × documents × vocabulary)",
        space="O(topics × vocabulary)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_topics",
            label="Number of Topics",
            type="range",
            default=5,
            min=2,
            max=20,
            step=1,
            description="Number of topics to discover in the corpus"
        ),
        AlgorithmParameter(
            name="max_iterations",
            label="Maximum Iterations",
            type="range",
            default=100,
            min=20,
            max=500,
            step=10,
            description="Maximum number of LDA training iterations"
        ),
        AlgorithmParameter(
            name="alpha",
            label="Document-Topic Density (Alpha)",
            type="select",
            default="auto",
            options=[
                {"label": "Auto (1/n_topics)", "value": "auto"},
                {"label": "Sparse (0.1)", "value": 0.1},
                {"label": "Medium (0.5)", "value": 0.5},
                {"label": "Dense (1.0)", "value": 1.0}
            ],
            description="Controls document-topic density (higher = more topics per document)"
        ),
        AlgorithmParameter(
            name="beta",
            label="Topic-Word Density (Beta)",
            type="select",
            default="auto",
            options=[
                {"label": "Auto (1/n_topics)", "value": "auto"},
                {"label": "Sparse (0.01)", "value": 0.01},
                {"label": "Medium (0.1)", "value": 0.1},
                {"label": "Dense (1.0)", "value": 1.0}
            ],
            description="Controls topic-word density (higher = more words per topic)"
        ),
        AlgorithmParameter(
            name="min_df",
            label="Minimum Document Frequency",
            type="range",
            default=2,
            min=1,
            max=10,
            step=1,
            description="Minimum number of documents a term must appear in"
        ),
        AlgorithmParameter(
            name="max_df",
            label="Maximum Document Frequency",
            type="range",
            default=0.95,
            min=0.5,
            max=1.0,
            step=0.05,
            description="Maximum document frequency as proportion (filters common words)"
        )
    ],
    dataset_name="multi_topic_corpus",
    visualization_type="multi_chart",
    theory=(
        "Latent Dirichlet Allocation (LDA) is a generative probabilistic model for discovering "
        "abstract topics in document collections. The key idea is that documents are represented "
        "as random mixtures over latent topics, and each topic is characterized by a distribution "
        "over words. LDA assumes a generative process: for each document, a distribution over topics "
        "is chosen; for each word in the document, a topic is chosen from this distribution, and a "
        "word is chosen from the topic's word distribution. The model works backwards from observed "
        "words to infer the hidden topic structure using variational inference or Gibbs sampling. "
        "The alpha parameter controls document-topic density (higher values mean documents contain "
        "more topics), while beta controls topic-word density (higher values mean topics contain "
        "more words). LDA is unsupervised and widely used for text mining, content recommendation, "
        "and exploratory data analysis. Unlike clustering, documents can belong to multiple topics "
        "with different probabilities, making it more flexible for real-world text analysis."
    ),
    pros=[
        "Unsupervised learning - no labeled data required",
        "Documents can belong to multiple topics",
        "Interpretable topics with top keywords",
        "Flexible model with adjustable parameters",
        "Scales to large document collections",
        "Useful for exploratory data analysis"
    ],
    cons=[
        "Requires choosing number of topics in advance",
        "Sensitive to preprocessing and parameters",
        "Topics may not always be coherent or meaningful",
        "Computationally intensive for large corpora",
        "Difficult to evaluate quality objectively",
        "Cannot handle short documents well"
    ],
    related_algorithms=["tfidf", "word2vec", "bag-of-words", "nmf"]
)

AlgorithmRegistry.register(topic_modeling_metadata)


# Register Tokenization metadata
tokenization_metadata = AlgorithmMetadata(
    id="tokenization",
    name="Tokenization",
    slug="tokenization",
    category=AlgorithmCategory.NLP,
    description="Break text into tokens (words, subwords, characters) with multiple tokenization strategies",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["nlp", "tokenization", "preprocessing", "text-processing"],
    use_cases=[
        "NLP preprocessing",
        "Search indexing",
        "Language modeling",
        "Machine translation",
        "Text analysis",
        "Feature extraction"
    ],
    complexity=AlgorithmComplexity(
        time="O(n)",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="tokenizer_type",
            label="Tokenizer",
            type="select",
            default="word",
            options=[
                {"label": "Whitespace", "value": "whitespace"},
                {"label": "Word (NLTK)", "value": "word"},
                {"label": "Sentence (NLTK)", "value": "sentence"},
                {"label": "WordPiece (BERT)", "value": "wordpiece"},
                {"label": "BPE (GPT-2)", "value": "bpe"},
                {"label": "Character", "value": "character"},
                {"label": "spaCy", "value": "spacy"}
            ],
            description="Type of tokenization strategy to use"
        ),
        AlgorithmParameter(
            name="lowercase",
            label="Convert to Lowercase",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Convert text to lowercase before tokenization"
        ),
        AlgorithmParameter(
            name="remove_punctuation",
            label="Remove Punctuation",
            type="select",
            default=False,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Remove punctuation marks from tokens"
        ),
        AlgorithmParameter(
            name="remove_stopwords",
            label="Remove Stop Words",
            type="select",
            default=False,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Remove common stop words (e.g., 'the', 'is', 'and')"
        ),
        AlgorithmParameter(
            name="max_tokens",
            label="Max Tokens to Display",
            type="range",
            default=100,
            min=10,
            max=500,
            step=10,
            description="Limit the number of output tokens"
        ),
        AlgorithmParameter(
            name="text_index",
            label="Sample Text",
            type="range",
            default=0,
            min=0,
            max=19,
            step=1,
            description="Select a sample text (0-19)"
        ),
        AlgorithmParameter(
            name="compare_mode",
            label="Compare Tokenizers",
            type="select",
            default=False,
            options=[
                {"label": "Single tokenizer", "value": False},
                {"label": "Compare all tokenizers", "value": True}
            ],
            description="Compare all tokenization strategies side-by-side"
        )
    ],
    dataset_name="diverse_sample_texts",
    visualization_type="token_breakdown",
    theory=(
        "Tokenization is the fundamental process of breaking text into smaller units called tokens, "
        "which can be words, subwords, characters, or sentences. It is the first step in most NLP pipelines. "
        "Different tokenization strategies serve different purposes:\n\n"
        "1. Whitespace Tokenization: Simple split on whitespace, fast but basic.\n"
        "2. Word Tokenization: Uses linguistic rules to handle punctuation, contractions, and special cases.\n"
        "3. Sentence Tokenization: Splits text into sentences using punctuation and abbreviation rules.\n"
        "4. WordPiece (BERT): Subword tokenization that splits rare words into common subword units. "
        "Handles out-of-vocabulary words by decomposing them into known pieces (e.g., 'unhappiness' → 'un', '##happiness').\n"
        "5. Byte Pair Encoding (BPE, GPT-2): Iteratively merges the most frequent character pairs to create subwords. "
        "Balances vocabulary size with representation capability.\n"
        "6. Character Tokenization: Splits text into individual characters, useful for character-level models.\n"
        "7. spaCy Tokenization: Advanced tokenization with linguistic features like part-of-speech tags, "
        "dependency parsing, and named entity recognition.\n\n"
        "The choice of tokenization strategy impacts model performance, vocabulary size, and ability to "
        "handle rare or out-of-vocabulary words. Modern transformer models typically use subword tokenization "
        "(WordPiece or BPE) to balance vocabulary size and coverage."
    ),
    pros=[
        "Essential preprocessing step for all NLP tasks",
        "Multiple strategies available for different use cases",
        "Modern subword tokenizers handle out-of-vocabulary words",
        "Fast and deterministic processing",
        "Reduces text complexity for downstream tasks",
        "Enables consistent text representation"
    ],
    cons=[
        "Language-specific rules may not generalize across languages",
        "Different tokenizers produce different results",
        "Subword tokenization can split meaningful words",
        "May lose important context (e.g., 'New York' → ['New', 'York'])",
        "Requires careful choice of strategy for specific task",
        "Some tokenizers require pre-trained models"
    ],
    related_algorithms=["bag-of-words", "tfidf", "word2vec", "bert-finetuning"]
)

AlgorithmRegistry.register(tokenization_metadata)


# Register Seq2Seq metadata
seq2seq_metadata = AlgorithmMetadata(
    id="seq2seq",
    name="Seq2Seq (Sequence-to-Sequence)",
    slug="seq2seq",
    category=AlgorithmCategory.NLP,
    description="Encoder-decoder architecture for sequence transformation tasks",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["nlp", "seq2seq", "encoder-decoder", "neural-translation", "attention"],
    use_cases=[
        "Machine translation",
        "Text summarization",
        "Question answering",
        "Chatbots",
        "Code generation",
        "Speech recognition"
    ],
    complexity=AlgorithmComplexity(
        time="O(sequence_length²)",
        space="O(hidden_size × layers)"
    ),
    parameters=[
        AlgorithmParameter(
            name="task",
            label="Task Type",
            type="select",
            default="translation",
            options=[
                {"label": "Machine Translation", "value": "translation"},
                {"label": "Sequence Reversal", "value": "reversal"},
                {"label": "Date Format Conversion", "value": "date-conversion"}
            ],
            description="Type of sequence transformation task"
        ),
        AlgorithmParameter(
            name="hidden_size",
            label="Hidden Size",
            type="range",
            default=256,
            min=64,
            max=512,
            step=64,
            description="Number of hidden units in LSTM layers"
        ),
        AlgorithmParameter(
            name="num_layers",
            label="Number of Layers",
            type="range",
            default=2,
            min=1,
            max=4,
            step=1,
            description="Number of LSTM layers in encoder/decoder"
        ),
        AlgorithmParameter(
            name="dropout",
            label="Dropout Rate",
            type="range",
            default=0.3,
            min=0.0,
            max=0.5,
            step=0.1,
            description="Dropout rate for regularization"
        ),
        AlgorithmParameter(
            name="attention",
            label="Use Attention",
            type="select",
            default=True,
            options=[
                {"label": "Enable Attention", "value": True},
                {"label": "Disable Attention", "value": False}
            ],
            description="Whether to use Bahdanau attention mechanism"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for Adam optimizer"
        ),
        AlgorithmParameter(
            name="teacher_forcing_ratio",
            label="Teacher Forcing Ratio",
            type="range",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            description="Probability of using teacher forcing during training"
        ),
        AlgorithmParameter(
            name="language_pair",
            label="Language Pair",
            type="select",
            default="en-fr",
            options=[
                {"label": "English-French", "value": "en-fr"},
                {"label": "English-Spanish", "value": "en-es"}
            ],
            description="Language pair for translation task"
        )
    ],
    dataset_name="machine_translation_pairs",
    visualization_type="attention_heatmap",
    theory=(
        "Sequence-to-Sequence (Seq2Seq) models are neural architectures designed for transforming one sequence "
        "into another. The model consists of two main components: an encoder that processes the input sequence and "
        "creates a context representation, and a decoder that generates the output sequence from this context. "
        "Both encoder and decoder typically use recurrent neural networks (RNNs), Long Short-Term Memory (LSTM), "
        "or Gated Recurrent Units (GRU) to handle sequential data.\n\n"
        "The encoder processes the input sequence token by token, updating its hidden state at each step. The final "
        "hidden state becomes the context vector, which encapsulates the meaning of the entire input sequence. "
        "The decoder then uses this context vector to generate the output sequence, also token by token.\n\n"
        "A key innovation is the attention mechanism (Bahdanau attention), which allows the decoder to focus on "
        "different parts of the input sequence at each decoding step. Instead of relying solely on the fixed context "
        "vector, attention computes a weighted sum of all encoder hidden states, where the weights indicate which "
        "input positions are most relevant for generating the current output token. This dramatically improves "
        "performance on long sequences by avoiding the information bottleneck of a single context vector.\n\n"
        "During training, teacher forcing is often used: the decoder receives the true previous token as input "
        "(rather than its own prediction) to stabilize training. At inference, the model uses its own predictions "
        "in an autoregressive manner. Seq2Seq models revolutionized machine translation and are foundational "
        "for many modern NLP tasks."
    ),
    pros=[
        "Can handle variable-length input and output sequences",
        "Attention mechanism provides interpretability",
        "End-to-end learning with no feature engineering",
        "Works for various sequence transformation tasks",
        "Can capture long-range dependencies with attention",
        "Flexible architecture adaptable to different domains"
    ],
    cons=[
        "Requires large amounts of parallel training data",
        "Computationally expensive to train",
        "Can suffer from exposure bias (teacher forcing vs. inference mismatch)",
        "May struggle with very long sequences despite attention",
        "Requires careful hyperparameter tuning",
        "Beam search decoding adds inference complexity"
    ],
    related_algorithms=["bert", "transformers", "word2vec", "ner"]
)

AlgorithmRegistry.register(seq2seq_metadata)


# Register POS Tagging metadata
pos_tagging_metadata = AlgorithmMetadata(
    id="pos-tagging",
    name="Part-of-Speech Tagging",
    slug="pos-tagging",
    category=AlgorithmCategory.NLP,
    description="Assign grammatical categories (noun, verb, adjective, etc.) to each word in text",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["nlp", "pos-tagging", "syntax", "grammar", "linguistics"],
    use_cases=[
        "Grammar checking",
        "Text-to-speech",
        "Information extraction",
        "Machine translation",
        "Question answering",
        "Syntax analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(n)",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="tagger",
            label="Tagger Type",
            type="select",
            default="spacy",
            options=[
                {"label": "spaCy", "value": "spacy"},
                {"label": "NLTK", "value": "nltk"},
                {"label": "Universal Tags", "value": "universal"}
            ],
            description="POS tagger type to use"
        ),
        AlgorithmParameter(
            name="text_index",
            label="Sample Text",
            type="range",
            default=0,
            min=0,
            max=24,
            step=1,
            description="Sample text selector (0-24)"
        ),
        AlgorithmParameter(
            name="show_fine_grained",
            label="Show Fine-grained Tags",
            type="select",
            default=True,
            options=[
                {"label": "Yes (Penn Treebank)", "value": True},
                {"label": "No (Universal only)", "value": False}
            ],
            description="Show detailed Penn Treebank tags"
        ),
        AlgorithmParameter(
            name="show_dependencies",
            label="Show Dependencies",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Include dependency parsing"
        ),
        AlgorithmParameter(
            name="tag_scheme",
            label="Tag Scheme",
            type="select",
            default="penn",
            options=[
                {"label": "Penn Treebank", "value": "penn"},
                {"label": "Universal", "value": "universal"}
            ],
            description="POS tag scheme to use"
        )
    ],
    dataset_name="diverse_grammatical_structures",
    visualization_type="multi_chart",
    theory=(
        "Part-of-Speech (POS) Tagging is a fundamental NLP task that assigns grammatical categories "
        "(parts of speech) to each word in a text. Common tags include nouns (NN), verbs (VB), "
        "adjectives (JJ), adverbs (RB), pronouns (PR), and more. Modern POS taggers use two main "
        "tag schemes:\n\n"
        "1. Penn Treebank Tags: Fine-grained tags (45 tags) that distinguish between different forms "
        "of the same part of speech (e.g., NN for singular noun, NNS for plural noun, VBD for past tense verb).\n\n"
        "2. Universal POS Tags: Coarse-grained tags (17 tags) that provide broader categorization "
        "(e.g., NOUN, VERB, ADJ) and work across languages.\n\n"
        "spaCy's POS tagger uses a statistical model trained on large annotated corpora. It employs "
        "a transition-based algorithm that processes text sequentially, considering both local context "
        "(surrounding words) and learned patterns from training data. The tagger also performs dependency "
        "parsing, which identifies syntactic relationships between words (e.g., which word modifies which). "
        "These relationships form a dependency tree where each word (except the root) has a head word it "
        "depends on and a dependency relation label (e.g., 'nsubj' for nominal subject, 'dobj' for direct object). "
        "POS tagging is crucial for many NLP applications including grammar checking, information extraction, "
        "machine translation, and text-to-speech systems."
    ),
    pros=[
        "Fast and accurate with pre-trained models",
        "Provides both coarse and fine-grained tags",
        "Includes dependency parsing for syntax analysis",
        "Works well on diverse text types",
        "Essential preprocessing for many NLP tasks",
        "Interpretable linguistic features"
    ],
    cons=[
        "Performance depends on text quality and domain",
        "May struggle with informal or domain-specific text",
        "Requires understanding of grammatical categories",
        "Tag ambiguity for some words in different contexts",
        "Limited to predefined tag sets"
    ],
    related_algorithms=["ner", "dependency-parsing", "tokenization"]
)

AlgorithmRegistry.register(pos_tagging_metadata)


@router.get("/")
async def nlp_root():
    return {"message": "Natural Language Processing algorithms endpoint"}


@router.get("/algorithms")
async def list_nlp_algorithms():
    """Get all registered NLP algorithms.

    Returns:
        List of algorithm metadata for all registered NLP algorithms
    """
    algorithms = AlgorithmRegistry.get_by_category(AlgorithmCategory.NLP)
    return [algo.model_dump() for algo in algorithms]


@router.post("/word2vec/train", response_model=Word2VecResponse)
async def train_word2vec(request: Word2VecParameters):
    """Train a Word2Vec model on text corpus.

    This endpoint trains a Word2Vec model using either the default sample
    corpus or a custom corpus provided by the user. Returns word embeddings,
    similar words, and word analogy examples.

    Args:
        request: Word2VecParameters containing training configuration

    Returns:
        Word2VecResponse with training results, embeddings, and examples

    Raises:
        HTTPException: If training fails due to invalid parameters or data issues
    """
    try:
        logger.info(f"Training Word2Vec with parameters: {request.model_dump()}")

        # Create and train model
        model = Word2VecModel()
        result = model.train(request)

        if not result.success:
            logger.error(f"Word2Vec training failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"Word2Vec training completed in {result.execution_time_ms:.2f}ms "
            f"with vocabulary size: {result.metrics.get('vocab_size', 0)}"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/word2vec/info")
async def get_word2vec_info() -> Dict[str, Any]:
    """Get Word2Vec algorithm information and metadata.

    Returns metadata, parameters, and information about the algorithm.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("word2vec")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Word2Vec metadata not found"
        )

    return {
        "metadata": metadata.model_dump(),
        "corpus_info": {
            "default_corpus": "Sample corpus with technology, science, and business topics",
            "supports_custom_corpus": True,
            "preprocessing": "Lowercase, punctuation removal, tokenization"
        }
    }


@router.post("/sentiment-analysis/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisParameters):
    """Analyze sentiment of text samples.

    This endpoint performs sentiment analysis using VADER (Valence Aware Dictionary
    and sEntiment Reasoner) to classify texts as positive, negative, or neutral.
    Returns sentiment predictions, confidence scores, and distribution statistics.

    Args:
        request: SentimentAnalysisParameters containing analysis configuration

    Returns:
        SentimentAnalysisResponse with predictions, distribution, and metrics

    Raises:
        HTTPException: If analysis fails due to invalid parameters or data issues
    """
    try:
        logger.info(f"Analyzing sentiment with parameters: {request.model_dump()}")

        # Create and run analysis
        model = SentimentAnalysisModel()
        result = model.analyze(request)

        if not result.success:
            logger.error(f"Sentiment analysis failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"Sentiment analysis completed in {result.execution_time_ms:.2f}ms "
            f"for {result.metrics.get('total_texts', 0)} texts"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/sentiment-analysis/info")
async def get_sentiment_analysis_info() -> Dict[str, Any]:
    """Get Sentiment Analysis algorithm information and metadata.

    Returns metadata, parameters, and information about the algorithm.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("sentiment-analysis")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Sentiment Analysis metadata not found"
        )

    return {
        "metadata": metadata.model_dump(),
        "model_info": {
            "default_model": "VADER (Valence Aware Dictionary and sEntiment Reasoner)",
            "supports_custom_texts": True,
            "sample_texts_count": 30,
            "sentiment_classes": ["positive", "negative", "neutral"],
            "score_range": "Compound score ranges from -1 (most negative) to +1 (most positive)"
        }
    }


@router.post("/tfidf/train", response_model=TFIDFResponse)
async def train_tfidf_endpoint(request: TFIDFRequest) -> TFIDFResponse:
    """Compute TF-IDF for document corpus.

    This endpoint computes TF-IDF scores for a collection of documents,
    identifying the most important terms in each document and across the corpus.

    Args:
        request: TF-IDF request with parameters and optional custom documents

    Returns:
        TFIDFResponse with TF-IDF matrix, top terms, and visualization data

    Raises:
        HTTPException: If computation fails
    """
    try:
        logger.info(f"Computing TF-IDF with parameters: {request.model_dump()}")
        response = train_tfidf(request)

        if not response.success:
            logger.error(f"TF-IDF computation failed: {response.error}")
            raise HTTPException(status_code=400, detail=response.error)

        logger.info(
            f"TF-IDF computation completed in {response.execution_time_ms:.2f}ms "
            f"with vocabulary size: {response.metrics.get('vocabulary_size', 0)}"
        )
        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Computation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@router.get("/tfidf/info")
async def get_tfidf_info() -> Dict[str, Any]:
    """Get TF-IDF algorithm information and metadata.

    Returns:
        Dictionary containing algorithm metadata, parameters, and dataset info
    """
    metadata = AlgorithmRegistry.get("tfidf")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="TF-IDF algorithm metadata not found"
        )

    dataset_info = get_tfidf_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/ner/extract", response_model=NERResponse)
async def extract_entities(request: NERParameters):
    """Extract named entities from text using spaCy.

    This endpoint performs Named Entity Recognition on either sample texts
    or custom text provided by the user. Returns extracted entities with
    their types, positions, and visualization data.

    Args:
        request: NER parameters including model, entity types, and text

    Returns:
        NERResponse with extracted entities and visualization data

    Raises:
        HTTPException: If NER fails due to invalid parameters or model issues
    """
    try:
        logger.info(f"Processing NER with parameters: {request.model_dump()}")

        # Create and run NER model
        model = NERModel()
        result = model.process(request)

        if not result.success:
            logger.error(f"NER extraction failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"NER completed in {result.execution_time_ms:.2f}ms "
            f"with {result.metrics.get('total_entities', 0)} entities extracted"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"NER error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"NER failed: {str(e)}")


@router.get("/ner/info")
async def get_ner_info() -> Dict[str, Any]:
    """Get Named Entity Recognition algorithm information and metadata.

    Returns metadata, parameters, entity types, and sample texts.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("ner")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="NER metadata not found"
        )

    sample_texts = get_default_texts()
    return {
        "metadata": metadata.model_dump(),
        "entity_types": get_entity_type_info(),
        "sample_texts": sample_texts,
        "total_samples": len(sample_texts),
        "model_info": {
            "default_model": "en_core_web_sm",
            "supported_models": ["en_core_web_sm", "en_core_web_md"],
            "requires_download": True,
            "library": "spaCy (en_core_web_sm model)",
            "entity_types_supported": list(get_entity_type_info().keys())
        }
    }


@router.get("/ner/samples")
async def get_ner_samples() -> Dict[str, Any]:
    """Get sample texts for NER demonstration.

    Returns:
        Dictionary with sample texts and entity type information
    """
    return {
        "samples": get_default_texts(),
        "entity_types": get_entity_type_info(),
        "total_samples": len(get_default_texts())
    }


@router.post("/bag-of-words/vectorize", response_model=BagOfWordsResponse)
async def vectorize_bag_of_words(request: BagOfWordsRequest) -> BagOfWordsResponse:
    """Create Bag of Words representation for document corpus.

    This endpoint creates a Bag of Words (BoW) representation from a collection
    of documents, converting text into word frequency vectors. Returns the
    document-term matrix, vocabulary, top terms per document, and visualization data.

    Args:
        request: Bag of Words request with parameters and optional custom documents

    Returns:
        BagOfWordsResponse with BoW matrix, vocabulary, and visualization data

    Raises:
        HTTPException: If vectorization fails due to invalid parameters or data issues
    """
    try:
        logger.info(f"Vectorizing with Bag of Words: {request.model_dump()}")

        # Create and run model
        model = BagOfWordsModel()
        result = model.vectorize(request)

        if not result.success:
            logger.error(f"Bag of Words vectorization failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"Bag of Words vectorization completed in {result.execution_time_ms:.2f}ms "
            f"with vocabulary size: {result.metrics.get('vocabulary_size', 0)}"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Vectorization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")


@router.get("/bag-of-words/info")
async def get_bag_of_words_info() -> Dict[str, Any]:
    """Get Bag of Words algorithm information and metadata.

    Returns metadata, parameters, and dataset information about the algorithm.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("bag-of-words")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Bag of Words metadata not found"
        )

    dataset_info = get_bow_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "vectorizer_info": {
            "library": "scikit-learn CountVectorizer",
            "supports_custom_documents": True,
            "preprocessing": "Lowercase, stopword removal (English)",
            "output_format": "Document-term matrix with word frequencies"
        }
    }


@router.post("/bert-finetuning/train", response_model=BERTFinetuningResponse)
async def train_bert_finetuning(request: BERTFinetuningParameters):
    """Fine-tune BERT model for text classification.

    This endpoint fine-tunes a pre-trained BERT model on a text classification dataset.
    Returns training history, predictions, confusion matrix, and attention visualizations.

    Args:
        request: BERT fine-tuning parameters including model variant, learning rate, epochs, etc.

    Returns:
        BERTFinetuningResponse with training results, predictions, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters or model issues
    """
    try:
        logger.info(f"Training BERT with parameters: {request.model_dump()}")

        # Create and train model
        model = BERTFinetuningModel()
        result = model.train(request)

        if not result.success:
            logger.error(f"BERT fine-tuning failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"BERT fine-tuning completed in {result.execution_time_ms:.2f}ms "
            f"with final accuracy: {result.metrics.get('final_val_accuracy', 0):.4f}"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/bert-finetuning/info")
async def get_bert_finetuning_info() -> Dict[str, Any]:
    """Get BERT Fine-tuning algorithm information and metadata.

    Returns metadata, parameters, dataset information, and model details.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("bert-finetuning")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="BERT Fine-tuning metadata not found"
        )

    dataset_info = get_bert_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "model_info": {
            "library": "Hugging Face Transformers",
            "supported_models": [
                "bert-base-uncased",
                "distilbert-base-uncased"
            ],
            "default_model": "bert-base-uncased",
            "parameters_bert_base": "110M",
            "parameters_distilbert": "66M",
            "supports_custom_dataset": True,
            "requires_gpu": "Recommended for faster training",
            "training_time": "3-10 minutes depending on parameters and hardware"
        }
    }


@router.post("/glove/query", response_model=GloVeResponse)
async def query_glove(request: GloVeParameters):
    """Query GloVe word embeddings for similar words and analogies.

    This endpoint loads pre-trained GloVe embeddings and performs various
    operations like finding similar words, solving word analogies, and
    computing cosine similarity between words.

    Args:
        request: GloVeParameters containing query configuration

    Returns:
        GloVeResponse with similar words, analogies, and visualizations

    Raises:
        HTTPException: If query fails due to invalid parameters or data issues
    """
    try:
        logger.info(f"Querying GloVe with parameters: {request.model_dump()}")

        # Create and query model
        model = GloVeModel()
        result = model.query(request)

        if not result.success:
            logger.error(f"GloVe query failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"GloVe query completed in {result.execution_time_ms:.2f}ms "
            f"with vocabulary size: {result.metrics.get('vocab_size', 0)}"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/glove/info")
async def get_glove_info() -> Dict[str, Any]:
    """Get GloVe algorithm information and metadata.

    Returns metadata, parameters, and information about the algorithm.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("glove")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="GloVe metadata not found"
        )

    return {
        "metadata": metadata.model_dump(),
        "embeddings_info": {
            "source": "Demo GloVe Embeddings",
            "available_dimensions": [50, 100, 200, 300],
            "vocabulary_size": "~200 words (demo subset)",
            "description": (
                "Pre-trained GloVe embeddings for demonstration. "
                "In production, load actual GloVe vectors from glove.6B or glove.42B files."
            ),
            "supports_queries": True,
            "supports_analogies": True,
            "sample_queries": {
                "royalty": ["king", "queen", "prince"],
                "geography": ["paris", "london", "berlin"],
                "technology": ["computer", "software", "algorithm"]
            },
            "sample_analogies": [
                "king - man + woman = queen",
                "paris - france + germany = berlin"
            ]
        }
    }


@router.post("/text-classification/train", response_model=TextClassificationResponse)
async def train_text_classification(request: TextClassificationParameters):
    """Train text classification model and evaluate on test set.

    This endpoint trains a text classification model using TF-IDF features
    and various classifiers (Naive Bayes, Logistic Regression, SVM).
    Returns predictions, confusion matrix, per-class metrics, and top features.

    Args:
        request: Text classification parameters including model type and features

    Returns:
        TextClassificationResponse with predictions, metrics, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters or data issues
    """
    try:
        logger.info(f"Training text classifier with parameters: {request.model_dump()}")

        # Create and train model
        model = TextClassificationModel()
        result = model.train(request)

        if not result.success:
            logger.error(f"Text classification training failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"Text classification training completed in {result.execution_time_ms:.2f}ms "
            f"with accuracy: {result.overall_metrics.get('accuracy', 0):.4f}"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/text-classification/info")
async def get_text_classification_info() -> Dict[str, Any]:
    """Get Text Classification algorithm information and metadata.

    Returns metadata, parameters, dataset information, and model details.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("text-classification")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Text Classification metadata not found"
        )

    # Get sample dataset for info
    from algorithms.nlp.text_classification import get_default_dataset
    texts, labels = get_default_dataset()
    dataset_info = get_text_classification_dataset_info(texts, labels)

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "categories": get_category_descriptions(),
        "model_info": {
            "library": "scikit-learn",
            "vectorizer": "TF-IDF (Term Frequency-Inverse Document Frequency)",
            "supported_classifiers": [
                {
                    "name": "Naive Bayes",
                    "value": "naive_bayes",
                    "description": "Probabilistic classifier based on Bayes' theorem, fast and effective"
                },
                {
                    "name": "Logistic Regression",
                    "value": "logistic_regression",
                    "description": "Linear model with multinomial classification, interpretable weights"
                },
                {
                    "name": "Support Vector Machine",
                    "value": "svm",
                    "description": "Linear SVM for finding optimal decision boundaries between classes"
                }
            ],
            "default_classifier": "naive_bayes",
            "features": "TF-IDF n-grams with configurable vocabulary size",
            "supports_custom_dataset": True,
            "training_time": "Fast, typically under 1 second for moderate datasets",
            "output_metrics": [
                "Confusion matrix",
                "Per-class precision, recall, F1-score",
                "Overall accuracy and averages",
                "Top discriminative features per class"
            ]
        }
    }


@router.post("/topic-modeling/train", response_model=TopicModelingResponse)
async def train_topic_modeling(request: TopicModelingParameters):
    """Train LDA model to discover topics in document corpus.

    This endpoint trains a Latent Dirichlet Allocation (LDA) model to discover
    abstract topics in a collection of documents. Returns discovered topics with
    keywords, document-topic distributions, coherence metrics, and visualizations.

    Args:
        request: Topic modeling parameters including n_topics, iterations, alpha, beta

    Returns:
        TopicModelingResponse with topics, distributions, metrics, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters or data issues
    """
    try:
        logger.info(f"Training Topic Modeling with parameters: {request.model_dump()}")

        # Create and train model
        model = TopicModelingModel()
        result = model.train(request)

        if not result.success:
            logger.error(f"Topic modeling training failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"Topic modeling training completed in {result.execution_time_ms:.2f}ms "
            f"with {len(result.topics)} topics discovered, "
            f"coherence: {result.coherence_score:.4f}"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/topic-modeling/info")
async def get_topic_modeling_info() -> Dict[str, Any]:
    """Get Topic Modeling (LDA) algorithm information and metadata.

    Returns metadata, parameters, dataset information, and model details.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("topic-modeling")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Topic Modeling metadata not found"
        )

    dataset_info = get_topic_modeling_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "model_info": {
            "library": "scikit-learn (LatentDirichletAllocation)",
            "algorithm": "Latent Dirichlet Allocation (LDA)",
            "method": "Online variational Bayes inference",
            "supports_custom_documents": True,
            "preprocessing": "Lowercase, stopword removal, CountVectorizer",
            "default_topics": 5,
            "topic_range": "2-20 topics",
            "output_visualizations": [
                "Word clouds per topic",
                "Topic-word distribution heatmap",
                "Top keywords bar charts",
                "Document-topic distributions",
                "Documents grouped by dominant topic"
            ],
            "metrics": [
                "Coherence score (topic quality)",
                "Perplexity (model fit)",
                "Topic distinctiveness",
                "Vocabulary size"
            ],
            "parameters_guide": {
                "alpha": "Controls document-topic density. Lower = fewer topics per document.",
                "beta": "Controls topic-word density. Lower = fewer words per topic.",
                "min_df": "Removes rare words appearing in few documents.",
                "max_df": "Removes common words appearing in most documents.",
                "n_topics": "Number of topics to discover. Higher = more granular topics.",
                "max_iterations": "Training iterations. More = better convergence but slower."
            },
            "use_case_examples": {
                "news_organization": "Group articles by topic for content categorization",
                "research": "Analyze academic papers to discover research themes",
                "customer_feedback": "Identify common themes in reviews and complaints",
                "social_media": "Detect trending topics in posts and discussions",
                "document_management": "Automatically tag and organize large document collections"
            }
        }
    }


@router.post("/tokenization/tokenize", response_model=TokenizationResponse)
async def tokenize_text(request: TokenizationRequest):
    """Tokenize text using specified tokenization strategy.

    This endpoint tokenizes text using various strategies including whitespace,
    word (NLTK), sentence (NLTK), WordPiece (BERT), BPE (GPT-2), character,
    and spaCy tokenization. Supports comparison mode to compare all tokenizers.

    Args:
        request: Tokenization parameters including strategy and text

    Returns:
        TokenizationResponse with tokens, statistics, and frequency distribution

    Raises:
        HTTPException: If tokenization fails due to invalid parameters
    """
    try:
        logger.info(f"Tokenizing text with parameters: {request.model_dump()}")

        # Create and run tokenization
        model = TokenizationModel()
        result = model.tokenize(request)

        if not result.success:
            logger.error(f"Tokenization failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        if request.compare_mode:
            logger.info(
                f"Tokenization comparison completed in {result.execution_time_ms:.2f}ms "
                f"with {len(result.comparison_results)} tokenizers"
            )
        else:
            logger.info(
                f"Tokenization completed in {result.execution_time_ms:.2f}ms "
                f"with {result.main_result.token_count if result.main_result else 0} tokens"
            )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Tokenization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")


@router.get("/tokenization/info")
async def get_tokenization_info() -> Dict[str, Any]:
    """Get Tokenization algorithm information and metadata.

    Returns metadata, parameters, dataset information, and tokenizer details.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("tokenization")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Tokenization metadata not found"
        )

    dataset_info = get_tokenization_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "sample_texts_count": len(get_sample_texts()),
        "tokenizers": {
            "whitespace": {
                "name": "Whitespace",
                "description": "Simple split on whitespace characters",
                "pros": ["Very fast", "Simple implementation"],
                "cons": ["No punctuation handling", "No linguistic awareness"],
                "use_cases": ["Quick prototyping", "Basic text splitting"]
            },
            "word": {
                "name": "Word (NLTK)",
                "description": "NLTK's word_tokenize using Penn Treebank tokenizer",
                "pros": ["Handles punctuation", "Deals with contractions"],
                "cons": ["English-focused", "Slower than whitespace"],
                "use_cases": ["Standard NLP preprocessing", "Text analysis"]
            },
            "sentence": {
                "name": "Sentence (NLTK)",
                "description": "NLTK's sent_tokenize for sentence segmentation",
                "pros": ["Handles abbreviations", "Punctuation-aware"],
                "cons": ["Limited to sentences", "May struggle with informal text"],
                "use_cases": ["Sentence boundary detection", "Document segmentation"]
            },
            "wordpiece": {
                "name": "WordPiece (BERT)",
                "description": "Subword tokenization used by BERT models",
                "pros": ["Handles OOV words", "Language-agnostic subwords"],
                "cons": ["Requires model download", "Can split words awkwardly"],
                "use_cases": ["BERT preprocessing", "Modern NLP models"]
            },
            "bpe": {
                "name": "BPE (GPT-2)",
                "description": "Byte Pair Encoding used by GPT models",
                "pros": ["Handles any text", "Efficient vocabulary"],
                "cons": ["Requires model download", "Non-intuitive tokens"],
                "use_cases": ["GPT preprocessing", "Language generation"]
            },
            "character": {
                "name": "Character",
                "description": "Character-level tokenization",
                "pros": ["No vocabulary limit", "Language-agnostic"],
                "cons": ["Very long sequences", "Loses word semantics"],
                "use_cases": ["Character-level models", "Language modeling"]
            },
            "spacy": {
                "name": "spaCy",
                "description": "Industrial-strength NLP with linguistic features",
                "pros": ["POS tags", "Rich linguistic info", "Fast"],
                "cons": ["Requires model download", "Memory intensive"],
                "use_cases": ["Advanced NLP", "Linguistic analysis"]
            }
        },
        "parameters_guide": {
            "tokenizer_type": "Choose tokenization strategy based on your use case",
            "lowercase": "Normalize text by converting to lowercase (reduces vocabulary)",
            "remove_punctuation": "Remove punctuation marks (simplifies tokens)",
            "remove_stopwords": "Filter common words like 'the', 'is', 'and'",
            "max_tokens": "Limit tokens displayed (useful for long texts)",
            "text_index": "Select from 20 diverse sample texts",
            "compare_mode": "Compare all 7 tokenizers side-by-side"
        },
        "model_info": {
            "libraries": ["NLTK", "spaCy", "Transformers (Hugging Face)"],
            "supports_custom_text": True,
            "supports_comparison": True,
            "available_tokenizers": 7,
            "features": [
                "Multiple tokenization strategies",
                "Token frequency distribution",
                "Statistics (count, unique, avg length)",
                "Compare mode for side-by-side comparison",
                "Stopword filtering",
                "Case normalization",
                "Punctuation handling"
            ]
        }
    }


@router.get("/tokenization/samples")
async def get_tokenization_samples() -> Dict[str, Any]:
    """Get sample texts for tokenization demonstration.

    Returns:
        Dictionary with sample texts and metadata
    """
    texts = get_sample_texts()

    samples_info = []
    for i, text in enumerate(texts):
        try:
            info = get_text_info(i)
            samples_info.append(info)
        except Exception as e:
            logger.warning(f"Failed to get info for text {i}: {e}")

    return {
        "samples": texts,
        "samples_info": samples_info,
        "total_samples": len(texts),
        "categories": [
            "Simple sentences",
            "Complex paragraphs",
            "Code snippets",
            "Punctuation-heavy text",
            "Social media posts",
            "Multilingual examples",
            "Scientific text",
            "Literary quotes",
            "News headlines"
        ]
    }


@router.post("/pos-tagging/tag", response_model=POSTaggingResponse)
async def tag_pos(request: POSTaggingParameters):
    """Perform Part-of-Speech tagging on text.

    This endpoint performs POS tagging using spaCy to assign grammatical
    categories to each word in text. Returns tagged words with POS tags,
    dependency parsing information, and visualization data.

    Args:
        request: POS tagging parameters including tagger type, text selection, etc.

    Returns:
        POSTaggingResponse with tagged words, POS distribution, and visualizations

    Raises:
        HTTPException: If tagging fails due to invalid parameters or model issues
    """
    try:
        logger.info(f"Processing POS tagging with parameters: {request.model_dump()}")

        # Create and run POS tagging model
        model = POSTaggingModel()
        result = model.process(request)

        if not result.success:
            logger.error(f"POS tagging failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"POS tagging completed in {result.execution_time_ms:.2f}ms "
            f"with {result.metrics.get('total_words', 0)} words tagged"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"POS tagging error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"POS tagging failed: {str(e)}")


@router.get("/pos-tagging/info")
async def get_pos_tagging_info() -> Dict[str, Any]:
    """Get Part-of-Speech Tagging algorithm information and metadata.

    Returns metadata, parameters, sample sentences, and tag descriptions.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("pos-tagging")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="POS Tagging metadata not found"
        )

    dataset_info = get_pos_tagging_dataset_info()
    tag_descriptions = get_pos_tag_descriptions()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "tag_descriptions": tag_descriptions,
        "model_info": {
            "library": "spaCy (en_core_web_sm) + NLTK",
            "tag_schemes": [
                {
                    "name": "Penn Treebank",
                    "description": "Fine-grained tags (45 tags)",
                    "examples": ["NN", "VBD", "JJ", "RB"]
                },
                {
                    "name": "Universal POS",
                    "description": "Coarse-grained tags (17 tags)",
                    "examples": ["NOUN", "VERB", "ADJ", "ADV"]
                }
            ],
            "supports_dependencies": True,
            "supports_custom_text": True,
            "sample_sentences_count": 25
        }
    }


@router.get("/pos-tagging/samples")
async def get_pos_tagging_samples() -> Dict[str, Any]:
    """Get sample sentences for POS tagging demonstration.

    Returns:
        Dictionary with sample sentences and metadata
    """
    sentences = get_sample_sentences()

    return {
        "samples": sentences,
        "total_samples": len(sentences),
        "categories": [
            "Simple declarative sentences",
            "Various verb tenses (present, past, future, perfect)",
            "Questions and imperatives",
            "Active and passive voice",
            "Complex and compound sentences",
            "Relative clauses",
            "Multiple adjectives and adverbs",
            "Prepositional phrases",
            "Modal verbs and phrasal verbs",
            "Gerunds, infinitives, and participles",
            "Comparatives and superlatives"
        ]
    }


@router.get("/pos-tagging/tags")
async def get_pos_tag_info() -> Dict[str, Any]:
    """Get information about POS tags and their descriptions.

    Returns:
        Dictionary with tag descriptions and color mappings
    """
    from algorithms.nlp.pos_tagging import get_pos_tag_colors

    tag_descriptions = get_pos_tag_descriptions()
    tag_colors = get_pos_tag_colors()

    # Group tags by category
    universal_tags = {
        tag: info for tag, info in tag_descriptions.items()
        if tag in ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ',
                   'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
                   'SYM', 'VERB', 'X', 'SPACE']
    }

    penn_tags = {
        tag: info for tag, info in tag_descriptions.items()
        if tag not in universal_tags
    }

    return {
        "universal_tags": universal_tags,
        "penn_treebank_tags": penn_tags,
        "tag_colors": tag_colors,
        "tag_categories": {
            "content_words": ["NOUN", "VERB", "ADJ", "ADV"],
            "function_words": ["DET", "ADP", "CONJ", "PRON"],
            "other": ["PUNCT", "NUM", "SYM", "X"]
        }
    }


@router.post("/seq2seq/train", response_model=Seq2SeqResponse)
async def train_seq2seq(request: Seq2SeqParameters):
    """Train Seq2Seq model for sequence transformation tasks.

    This endpoint trains an encoder-decoder model with optional attention mechanism
    for tasks like machine translation, sequence reversal, or date format conversion.
    Returns training history, sample predictions with BLEU scores, and attention visualizations.

    Args:
        request: Seq2Seq parameters including task, architecture, and training config

    Returns:
        Seq2SeqResponse with training results, predictions, attention weights, and metrics

    Raises:
        HTTPException: If training fails due to invalid parameters or model issues
    """
    try:
        logger.info(f"Training Seq2Seq with parameters: {request.model_dump()}")

        # Create and train model
        model = Seq2SeqModel()
        result = model.train(request)

        if not result.success:
            logger.error(f"Seq2Seq training failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)

        logger.info(
            f"Seq2Seq training completed in {result.execution_time_ms:.2f}ms "
            f"with final BLEU score: {result.metrics.get('final_bleu', 0):.4f}"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/seq2seq/info")
async def get_seq2seq_info() -> Dict[str, Any]:
    """Get Seq2Seq algorithm information and metadata.

    Returns metadata, parameters, dataset information, and model details.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("seq2seq")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Seq2Seq metadata not found"
        )

    # Get dataset info for default task
    dataset_info = get_seq2seq_dataset_info("translation", "en-fr")

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "model_info": {
            "library": "PyTorch",
            "encoder": "LSTM-based encoder",
            "decoder": "LSTM-based decoder with optional attention",
            "attention_type": "Bahdanau (additive) attention",
            "training_technique": "Teacher forcing with configurable ratio",
            "optimization": "Adam optimizer",
            "evaluation_metric": "BLEU score (unigram precision with brevity penalty)",
            "supported_tasks": [
                {
                    "name": "Machine Translation",
                    "value": "translation",
                    "description": "Translate between English-French or English-Spanish",
                    "dataset_size": "100+ parallel sentences per language pair"
                },
                {
                    "name": "Sequence Reversal",
                    "value": "reversal",
                    "description": "Reverse character sequences",
                    "dataset_size": "100 random sequences"
                },
                {
                    "name": "Date Format Conversion",
                    "value": "date-conversion",
                    "description": "Convert human-readable dates to ISO 8601 format",
                    "dataset_size": "300+ date pairs"
                }
            ],
            "architecture_details": {
                "encoder": "Multi-layer LSTM that processes input sequence",
                "decoder": "Multi-layer LSTM that generates output sequence",
                "attention": "Optional Bahdanau attention for better long-sequence handling",
                "embedding": "Learned embeddings for input and output vocabularies",
                "dropout": "Applied to embeddings and LSTM layers for regularization"
            },
            "training_details": {
                "teacher_forcing": "Uses ground truth tokens during training with configurable ratio",
                "inference": "Autoregressive generation with greedy decoding",
                "loss_function": "CrossEntropyLoss with padding token ignored",
                "vocabulary": "Built from training data with special tokens (PAD, SOS, EOS, UNK)"
            },
            "visualization_features": [
                "Training loss curve over epochs",
                "BLEU score progression",
                "Sample predictions with ground truth comparison",
                "Attention heatmap showing alignment between input and output",
                "Architecture diagram",
                "Encoder/decoder state visualization"
            ],
            "requires_gpu": "Optional - CPU training is supported but slower",
            "training_time": "1-5 minutes depending on parameters, task, and hardware"
        }
    }


@router.get("/seq2seq/datasets")
async def get_seq2seq_datasets() -> Dict[str, Any]:
    """Get information about available Seq2Seq datasets.

    Returns:
        Dictionary with dataset information for all supported tasks
    """
    datasets = {
        "translation": {
            "en-fr": get_seq2seq_dataset_info("translation", "en-fr"),
            "en-es": get_seq2seq_dataset_info("translation", "en-es")
        },
        "reversal": get_seq2seq_dataset_info("reversal"),
        "date-conversion": get_seq2seq_dataset_info("date-conversion")
    }

    return {
        "datasets": datasets,
        "total_tasks": 3,
        "supports_custom_data": True,
        "custom_data_format": {
            "input_sequences": ["List of input strings"],
            "target_sequences": ["List of corresponding target strings"]
        }
    }
