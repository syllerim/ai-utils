# --------------- dependencies ---------------
import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------- preprocess_dataframe ---------------

def run_preprocessing_pipeline(
    df,
    text_column='reviewText',
    token_column='cleaned_tokens',
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
        df (pd.DataFrame): The DataFrame result of preprocessing reviews.
        list of list: List of cleaned tokens per review.
        list: Flattened list of all tokens.
    """

    print("🧹 Starting preprocessing of reviews...")

    # apply preprocessing to each review
    all_cleaned_tokens = df[text_column].apply(
        lambda review: preprocess_review(str(review), remove_numbers, do_stemming)
    )
    print("\n- Tokenization and preprocessing complete.")

    # flatten the token lists to get a single list of all tokens
    flattened_tokens = [token for tokens in all_cleaned_tokens for token in tokens]
    print(f"\n- Flattened all tokens — total tokens: {len(flattened_tokens)}")

    # store tokenized reviews in a new column
    df[token_column] = all_cleaned_tokens
    print(f"\n- Stored cleaned tokens in column '{token_column}'.")

    # remove duplicates
    original_count = len(df)
    df = remove_duplicate_reviews(df, token_column)
    print(f"\n- Removed {original_count - len(df)} duplicate reviews.")

    return df, all_cleaned_tokens, flattened_tokens

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
    # normalize the sentence
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
    # initialize components
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # define junk fragments to remove
    junk_tokens = {'co', 'com', 'www', 'htt', 'https', 'th'}

    cleaned = []
    for word in tokens:
        word = word.lower()  # lowercase
        word = word.strip(string.punctuation)  # remove punctuation from edges

        # remove if it's a stopword, empty, or (optionally) a number
        if (not word or
            word in stop_words or
            word in junk_tokens or
            (remove_numbers and word.isdigit()) or
            len(word) <= 2
        ):
            continue

        # apply stemming
        if do_stemming:
            word = stemmer.stem(word)

        cleaned.append(word)

    return cleaned

# --------------- remove_duplicate_reviews ---------------

def remove_duplicate_reviews(df, token_column='cleaned_tokens'):
    """
    Removes duplicate reviews from a DataFrame based on the cleaned token list.

    Parameters:
        df (pd.DataFrame): DataFrame containing tokenized and cleaned reviews.
        token_column (str): Name of the column containing cleaned tokens.

    Returns:
        pd.DataFrame: DataFrame with duplicate reviews removed.
    """

    # create a temporary column by joining cleaned tokens into a single string
    df['cleaned_review_str'] = df[token_column].apply(lambda tokens: ' '.join(tokens))

    # drop duplicates based on the joined string
    df = df.drop_duplicates(subset='cleaned_review_str').reset_index(drop=True)

    # drop the helper column to keep things clean
    df = df.drop(columns=['cleaned_review_str'])

    return df
