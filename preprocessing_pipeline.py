# --------------- dependencies ---------------
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------- run_preprocessing_pipeline ---------------

def clean_tokens(
    tokens,
    remove_numbers=True,
    do_stemming=True
):
    """
    Cleans a list of tokens by removing stopwords, punctuation, and optionally numbers and applying stemming.

    Parameters:
        tokens (list): List of raw tokens (strings).
        remove_numbers (bool): Whether to remove numeric tokens.
        do_stemming (bool): Whether to apply stemming.

    Returns:
        list: Cleaned list of tokens.
    """
    # Initialize components
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    cleaned = []
    for word in tokens:
        word = word.lower()  # lowercase
        word = word.strip(string.punctuation)  # remove punctuation from edges

        # Remove if it's a stopword, empty, or (optionally) a number
        if (not word or
            word in stop_words or
            (remove_numbers and word.isdigit())):
            continue

        # Apply stemming
        if do_stemming:
            word = stemmer.stem(word)

        cleaned.append(word)

    return cleaned
