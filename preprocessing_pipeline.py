# --------------- dependencies ---------------
import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------- preprocess_dataframe ---------------

def preprocessing_pipeline(
    df,
    text_column='reviewText',
    remove_numbers=True,
    do_stemming=True
):
    """
    Apply text normalization and token cleaning to each review in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing review text.
        text_column (str): Column containing the raw review text.
        remove_numbers (bool): Whether to remove numeric tokens.
        do_stemming (bool): Whether to apply stemming.

    Returns:
        list of list: List of cleaned tokens per review.
        list: Flattened list of all tokens.
    """

    print("🧹 Preprocessing pipeline...")
    
    all_cleaned_tokens = df[text_column].apply(
        lambda review: preprocess_review(str(review), remove_numbers, do_stemming)
    )

    # flatten the token lists
    flattened_tokens = [token for tokens in all_cleaned_tokens for token in tokens]

    return all_cleaned_tokens, flattened_tokens
    
# --------------- preprocess_review ---------------

def preprocess_review(review, remove_numbers=True, do_stemming=True):
    """
    Preprocess a review by normalizing it and cleaning the tokens.

    Parameters:
        review (str): The raw review text.
        remove_numbers (bool): Whether to remove numbers in the review.
        do_stemming (bool): Whether to apply stemming.

    Returns:
        list: Cleaned tokens from the review.
    """
    # normalize the review text
    normalized_review = sentence_normalization(review)

    # tokenize the review text
    tokens = normalized_review.split()

    # clean the tokens
    cleaned_tokens = clean_tokens(tokens, remove_numbers, do_stemming)

    return cleaned_tokens

# --------------- sentence_normalization ---------------

def sentence_normalization(sentence):
    """
    Normalize the sentence by converting to ASCII, lowercase, and removing non-alphabetic characters.

    Parameters:
        sentence (str): The raw sentence to normalize.

    Returns:
        str: Normalized sentence.
    """
    # Normalize the sentence
    sentence = unicodedata.normalize('NFKD', sentence).lower().encode('ascii', errors='ignore').decode('utf-8')
    sentence = re.sub(' +', ' ', ' '.join([word if word.isalpha() else '' for word in sentence.split()])).strip()
    return sentence

# --------------- clean_tokens ---------------

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
