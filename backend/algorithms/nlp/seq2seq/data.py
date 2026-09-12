"""Data loading and preprocessing for Seq2Seq algorithm."""

from typing import List, Tuple, Dict, Any
import random


def get_translation_pairs(language_pair: str = "en-fr") -> List[Tuple[str, str]]:
    """Get English-French or English-Spanish translation pairs.

    Args:
        language_pair: Language pair to use ('en-fr' or 'en-es')

    Returns:
        List of (source, target) translation pairs
    """
    if language_pair == "en-fr":
        return [
            # Basic greetings and common phrases
            ("hello", "bonjour"),
            ("goodbye", "au revoir"),
            ("thank you", "merci"),
            ("please", "s'il vous plait"),
            ("yes", "oui"),
            ("no", "non"),
            ("good morning", "bonjour"),
            ("good evening", "bonsoir"),
            ("good night", "bonne nuit"),
            ("see you soon", "a bientot"),

            # Simple sentences
            ("i am happy", "je suis heureux"),
            ("you are beautiful", "tu es belle"),
            ("he is tall", "il est grand"),
            ("she is smart", "elle est intelligente"),
            ("we are friends", "nous sommes amis"),
            ("they are students", "ils sont etudiants"),

            # Questions
            ("how are you", "comment allez vous"),
            ("what is your name", "comment vous appelez vous"),
            ("where are you from", "d'ou venez vous"),
            ("how old are you", "quel age avez vous"),
            ("do you speak english", "parlez vous anglais"),

            # More complex sentences
            ("i love learning languages", "j'aime apprendre les langues"),
            ("this is a beautiful day", "c'est une belle journee"),
            ("i want to travel the world", "je veux voyager dans le monde"),
            ("she likes to read books", "elle aime lire des livres"),
            ("we enjoy eating together", "nous aimons manger ensemble"),
            ("they play soccer every weekend", "ils jouent au foot chaque weekend"),
            ("my favorite color is blue", "ma couleur preferee est le bleu"),
            ("the cat is sleeping", "le chat dort"),
            ("i need some water", "j'ai besoin d'eau"),
            ("can you help me", "pouvez vous m'aider"),

            # Daily activities
            ("i wake up early", "je me reveille tot"),
            ("he goes to work", "il va au travail"),
            ("she cooks dinner", "elle prepare le diner"),
            ("we watch movies", "nous regardons des films"),
            ("they study at night", "ils etudient la nuit"),

            # Time and numbers
            ("it is five o'clock", "il est cinq heures"),
            ("today is monday", "aujourd'hui c'est lundi"),
            ("tomorrow is tuesday", "demain c'est mardi"),
            ("i have two cats", "j'ai deux chats"),
            ("she has three brothers", "elle a trois freres"),

            # Food and drink
            ("i like coffee", "j'aime le cafe"),
            ("he drinks tea", "il boit du the"),
            ("she eats breakfast", "elle prend le petit dejeuner"),
            ("we love pizza", "nous adorons la pizza"),
            ("they want some bread", "ils veulent du pain"),

            # Weather and nature
            ("it is sunny today", "il fait beau aujourd'hui"),
            ("it is raining", "il pleut"),
            ("the sky is blue", "le ciel est bleu"),
            ("i see the moon", "je vois la lune"),
            ("flowers are blooming", "les fleurs fleurissent"),

            # More sentences for better training
            ("i am learning french", "j'apprends le francais"),
            ("this is my house", "c'est ma maison"),
            ("where is the library", "ou est la bibliotheque"),
            ("i need a taxi", "j'ai besoin d'un taxi"),
            ("the train is late", "le train est en retard"),
            ("she is a teacher", "elle est professeur"),
            ("he is a doctor", "il est medecin"),
            ("we are going home", "nous rentrons a la maison"),
            ("they are very happy", "ils sont tres heureux"),
            ("i have a question", "j'ai une question"),
            ("can i sit here", "puis je m'asseoir ici"),
            ("the restaurant is closed", "le restaurant est ferme"),
            ("i like this song", "j'aime cette chanson"),
            ("she plays the piano", "elle joue du piano"),
            ("he speaks three languages", "il parle trois langues"),
            ("we need more time", "nous avons besoin de plus de temps"),
            ("the book is interesting", "le livre est interessant"),
            ("i want to learn more", "je veux apprendre plus"),
            ("they live in paris", "ils habitent a paris"),
            ("the weather is nice", "le temps est agreable"),
            ("i am very tired", "je suis tres fatigue"),
            ("she works at the hospital", "elle travaille a l'hopital"),
            ("we have a meeting", "nous avons une reunion"),
            ("the dog is barking", "le chien aboie"),
            ("i forgot my keys", "j'ai oublie mes cles"),
            ("he loves his family", "il aime sa famille"),
            ("she runs every morning", "elle court tous les matins"),
            ("the children are playing", "les enfants jouent"),
            ("i bought a new car", "j'ai achete une nouvelle voiture"),
            ("they are eating lunch", "ils dejeunent"),
            ("the movie starts at eight", "le film commence a huit heures"),
            ("i will call you later", "je t'appellerai plus tard"),
            ("she is cooking pasta", "elle cuisine des pates"),
            ("we are waiting for you", "nous t'attendons"),
            ("he takes the bus", "il prend le bus"),
            ("the sun is shining", "le soleil brille"),
            ("i can swim well", "je sais bien nager"),
            ("they arrived yesterday", "ils sont arrives hier"),
            ("the store opens at nine", "le magasin ouvre a neuf heures"),
            ("i feel great today", "je me sens bien aujourd'hui"),
            ("she knows the answer", "elle connait la reponse"),
            ("we saw a beautiful bird", "nous avons vu un bel oiseau"),
            ("he always helps others", "il aide toujours les autres"),
            ("the music is loud", "la musique est forte"),
            ("i write emails daily", "j'ecris des emails tous les jours"),
            ("they celebrate every year", "ils celebrent chaque annee"),
            ("she smiles a lot", "elle sourit beaucoup"),
            ("the coffee is hot", "le cafe est chaud"),
            ("i understand everything", "je comprends tout"),
        ]

    elif language_pair == "en-es":
        return [
            # Basic greetings and common phrases
            ("hello", "hola"),
            ("goodbye", "adios"),
            ("thank you", "gracias"),
            ("please", "por favor"),
            ("yes", "si"),
            ("no", "no"),
            ("good morning", "buenos dias"),
            ("good afternoon", "buenas tardes"),
            ("good night", "buenas noches"),
            ("see you soon", "hasta pronto"),

            # Simple sentences
            ("i am happy", "estoy feliz"),
            ("you are beautiful", "eres hermosa"),
            ("he is tall", "el es alto"),
            ("she is smart", "ella es inteligente"),
            ("we are friends", "somos amigos"),
            ("they are students", "ellos son estudiantes"),

            # Questions
            ("how are you", "como estas"),
            ("what is your name", "como te llamas"),
            ("where are you from", "de donde eres"),
            ("how old are you", "cuantos anos tienes"),
            ("do you speak english", "hablas ingles"),

            # More complex sentences
            ("i love learning languages", "me encanta aprender idiomas"),
            ("this is a beautiful day", "este es un dia hermoso"),
            ("i want to travel the world", "quiero viajar por el mundo"),
            ("she likes to read books", "a ella le gusta leer libros"),
            ("we enjoy eating together", "disfrutamos comer juntos"),
            ("they play soccer every weekend", "juegan futbol cada fin de semana"),
            ("my favorite color is blue", "mi color favorito es azul"),
            ("the cat is sleeping", "el gato esta durmiendo"),
            ("i need some water", "necesito agua"),
            ("can you help me", "puedes ayudarme"),

            # Daily activities
            ("i wake up early", "me despierto temprano"),
            ("he goes to work", "el va al trabajo"),
            ("she cooks dinner", "ella cocina la cena"),
            ("we watch movies", "vemos peliculas"),
            ("they study at night", "ellos estudian de noche"),

            # Time and numbers
            ("it is five o'clock", "son las cinco"),
            ("today is monday", "hoy es lunes"),
            ("tomorrow is tuesday", "manana es martes"),
            ("i have two cats", "tengo dos gatos"),
            ("she has three brothers", "ella tiene tres hermanos"),

            # Food and drink
            ("i like coffee", "me gusta el cafe"),
            ("he drinks tea", "el bebe te"),
            ("she eats breakfast", "ella desayuna"),
            ("we love pizza", "nos encanta la pizza"),
            ("they want some bread", "ellos quieren pan"),

            # Weather and nature
            ("it is sunny today", "hace sol hoy"),
            ("it is raining", "esta lloviendo"),
            ("the sky is blue", "el cielo es azul"),
            ("i see the moon", "veo la luna"),
            ("flowers are blooming", "las flores estan floreciendo"),

            # More sentences
            ("i am learning spanish", "estoy aprendiendo espanol"),
            ("this is my house", "esta es mi casa"),
            ("where is the library", "donde esta la biblioteca"),
            ("i need a taxi", "necesito un taxi"),
            ("the train is late", "el tren esta retrasado"),
            ("she is a teacher", "ella es profesora"),
            ("he is a doctor", "el es doctor"),
            ("we are going home", "vamos a casa"),
            ("they are very happy", "ellos estan muy felices"),
            ("i have a question", "tengo una pregunta"),
            ("can i sit here", "puedo sentarme aqui"),
            ("the restaurant is closed", "el restaurante esta cerrado"),
            ("i like this song", "me gusta esta cancion"),
            ("she plays the piano", "ella toca el piano"),
            ("he speaks three languages", "el habla tres idiomas"),
            ("we need more time", "necesitamos mas tiempo"),
            ("the book is interesting", "el libro es interesante"),
            ("i want to learn more", "quiero aprender mas"),
            ("they live in madrid", "ellos viven en madrid"),
            ("the weather is nice", "el clima es agradable"),
            ("i am very tired", "estoy muy cansado"),
            ("she works at the hospital", "ella trabaja en el hospital"),
            ("we have a meeting", "tenemos una reunion"),
            ("the dog is barking", "el perro esta ladrando"),
            ("i forgot my keys", "olvide mis llaves"),
            ("he loves his family", "el ama a su familia"),
            ("she runs every morning", "ella corre todas las mananas"),
            ("the children are playing", "los ninos estan jugando"),
            ("i bought a new car", "compre un auto nuevo"),
            ("they are eating lunch", "estan almorzando"),
            ("the movie starts at eight", "la pelicula empieza a las ocho"),
            ("i will call you later", "te llamare mas tarde"),
            ("she is cooking pasta", "ella esta cocinando pasta"),
            ("we are waiting for you", "te estamos esperando"),
            ("he takes the bus", "el toma el autobus"),
            ("the sun is shining", "el sol esta brillando"),
            ("i can swim well", "puedo nadar bien"),
            ("they arrived yesterday", "llegaron ayer"),
            ("the store opens at nine", "la tienda abre a las nueve"),
            ("i feel great today", "me siento genial hoy"),
            ("she knows the answer", "ella sabe la respuesta"),
            ("we saw a beautiful bird", "vimos un pajaro hermoso"),
            ("he always helps others", "el siempre ayuda a otros"),
            ("the music is loud", "la musica esta fuerte"),
            ("i write emails daily", "escribo correos diariamente"),
            ("they celebrate every year", "celebran cada ano"),
            ("she smiles a lot", "ella sonrie mucho"),
            ("the coffee is hot", "el cafe esta caliente"),
            ("i understand everything", "entiendo todo"),
        ]

    return []


def get_reversal_pairs(num_pairs: int = 100) -> List[Tuple[str, str]]:
    """Generate character sequence reversal pairs.

    Args:
        num_pairs: Number of pairs to generate

    Returns:
        List of (sequence, reversed_sequence) pairs
    """
    pairs = []

    # Predefined interesting sequences
    predefined = [
        ("hello", "olleh"),
        ("world", "dlrow"),
        ("python", "nohtyp"),
        ("algorithm", "mhtirogla"),
        ("sequence", "ecneuqes"),
        ("neural", "laruen"),
        ("network", "krowten"),
        ("learning", "gninrael"),
        ("encoder", "redocne"),
        ("decoder", "redoced"),
        ("attention", "noitnetta"),
        ("translation", "noitalsnart"),
        ("language", "egaugnal"),
        ("computer", "retupmoc"),
        ("science", "ecneics"),
        ("machine", "enihcam"),
        ("data", "atad"),
        ("model", "ledom"),
        ("train", "niart"),
        ("test", "tset"),
    ]

    pairs.extend(predefined)

    # Generate random sequences
    chars = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(num_pairs - len(predefined)):
        length = random.randint(4, 12)
        sequence = ''.join(random.choice(chars) for _ in range(length))
        pairs.append((sequence, sequence[::-1]))

    return pairs


def get_date_conversion_pairs() -> List[Tuple[str, str]]:
    """Generate date format conversion pairs.

    Converts from human-readable format to ISO format.
    e.g., "May 3, 2024" -> "2024-05-03"

    Returns:
        List of (human_date, iso_date) pairs
    """
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    pairs = []

    # Generate diverse date examples
    for year in range(2020, 2026):
        for month_idx, month_name in enumerate(months, 1):
            # Generate a few dates per month
            for day in [1, 7, 15, 21, 28]:
                if month_idx == 2 and day > 28:  # February
                    continue
                if month_idx in [4, 6, 9, 11] and day > 30:  # 30-day months
                    continue

                human_date = f"{month_name} {day}, {year}"
                iso_date = f"{year}-{month_idx:02d}-{day:02d}"
                pairs.append((human_date, iso_date))

    return pairs


def get_dataset_by_task(task: str, language_pair: str = "en-fr") -> List[Tuple[str, str]]:
    """Get dataset for specific task.

    Args:
        task: Task type (translation, reversal, date-conversion)
        language_pair: Language pair for translation task

    Returns:
        List of (input, target) pairs
    """
    if task == "translation":
        return get_translation_pairs(language_pair)
    elif task == "reversal":
        return get_reversal_pairs(100)
    elif task == "date-conversion":
        return get_date_conversion_pairs()
    else:
        raise ValueError(f"Unknown task: {task}")


def get_dataset_info(task: str, language_pair: str = "en-fr") -> Dict[str, Any]:
    """Get information about the dataset for a task.

    Args:
        task: Task type
        language_pair: Language pair for translation

    Returns:
        Dictionary with dataset information
    """
    pairs = get_dataset_by_task(task, language_pair)

    info = {
        "task": task,
        "total_pairs": len(pairs),
        "sample_pairs": pairs[:5] if len(pairs) > 0 else [],
    }

    if task == "translation":
        info["language_pair"] = language_pair
        info["description"] = f"Machine translation from {language_pair.split('-')[0].upper()} to {language_pair.split('-')[1].upper()}"
    elif task == "reversal":
        info["description"] = "Character sequence reversal"
    elif task == "date-conversion":
        info["description"] = "Date format conversion (human-readable to ISO 8601)"

    return info
