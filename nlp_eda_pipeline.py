# --------------- dependencies ---------------
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import numpy as np

from collections import Counter
from nltk.probability import FreqDist
from nltk import FreqDist
from nltk.util import ngrams
from wordcloud import WordCloud

# --------------- run_eda_pipeline ---------------

def run_eda_pipeline(file_name):
    # dataset loading and preprocessing
    df = process_dataset(file_name)

    # analyze and add review length information to the dataframe
    df = analyze_review_lengths(df)

    # vocabulary creation from tokenized text
    vocabulary, words = create_vocabulary(df)

    # wordCloud to visually highlight the most frequent words in the reviews
    generate_wordcloud(vocabulary, title="Most Frequent Words in Reviews")

    # distribution of reviews by sentiment
    plot_sentiment_distribution(df)

    # encode sentiment labels based on the 'overall' rating and visualize the class distribution
    df = encode_and_plot_sentiment(df, label_sentiment)

    # top 10 bigrams
    top_bigrams = get_top_ngrams(words, ngram_size=2, num_top_ngrams=10)

    # top 10 trigrams
    top_trigrams = get_top_ngrams(words, ngram_size=3, num_top_ngrams=10)

    return df, vocabulary, words, top_bigrams, top_trigrams


# --------------- get_top_ngrams ---------------

def get_top_ngrams(
    words,
    ngram_size=2,
    num_top_ngrams=10
):
    """
    Generates n-grams from a list of words and returns the top-N most frequent ones.

    Parameters:
        words (list): List of tokens.
        ngram_size (int): Size of the n-gram (e.g., 2 for bigrams, 3 for trigrams).
        num_top_ngrams (int): Number of most common n-grams to return.

    Returns:
        list of tuples: Each tuple contains an n-gram and its frequency.
    """
    ngram_list = ngrams(words, ngram_size)
    freq_dist = FreqDist(ngram_list)
    top_ngrams = freq_dist.most_common(num_top_ngrams)
    print(top_ngrams)

    print(f"\nTop {num_top_ngrams} {ngram_size}-grams:")
    for x_gram, freq in top_ngrams:
      print(f"{freq:<5}  {' '.join(x_gram)}")

    # prepare data for the frequency diagram
    n_grams = [' '.join(ngram[0]) for ngram in top_ngrams]
    frequencies = [ngram[1] for ngram in top_ngrams]

    # Plot the frequency diagram
    plt.figure(figsize=(10, 5))
    plt.barh(n_grams, frequencies, color='skyblue')
    plt.xlabel("Frequency")
    plt.ylabel(f"{ngram_size}-grams")
    plt.title(f"Top {num_top_ngrams} {ngram_size}-grams by Frequency")
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.show()

    return top_ngrams

# --------------- encode_and_plot_sentiment ---------------

def encode_and_plot_sentiment(
    df,
    label_func,
    column='overall',
    sentiment_label_column='sentiment_label'
):
    """
    Encodes sentiment labels into a new 'sentiment_label' column using a custom labeling function,
    and plots the distribution of positive vs negative reviews.

    Parameters:
        df (pd.DataFrame): The DataFrame containing review data.
        label_func (function): Function to convert original ratings into sentiment labels.
        column (str): The name of the column with rating scores to base the sentiment on.
        sentiment_label_column (str): The name of the new column to store sentiment labels.

    Returns:
        pd.DataFrame: The DataFrame with the new 'sentiment_label' column added.
    """
    # Apply sentiment labeling
    df[sentiment_label_column] = df.apply(lambda row: label_func(row), axis=1)

    # pptionally preview
    print("Encoded 'sentiment_label' column:")
    print(df[['reviewText', column, sentiment_label_column]].head())
    print("\n")

    # plot sentiment distribution
    plot_sentiment_distribution(
        df,
        column='sentiment_label',
        title='Sentiment Distribution (Final Corpus) - Positive vs Negative Reviews',
        xLabel='0: Negative  1: Positive',
        yLabel='Count'
    )

    return df

# --------------- plot_sentiment_distribution ---------------

def plot_sentiment_distribution(
    df,
    column='overall',
    title='Overall Sentiment Distribution (Review)',
    xLabel='Rating Values',
    yLabel='Count',
    figsize=(8, 4)
):
    """
    Plots the distribution of review scores, preserving the original score order.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the sentiment scores.
        column (str): The column name with the sentiment or rating values.
        title (str): The title of the plot.
        figsize (tuple): Size of the figure.
    """
    sentiment_counts = df[column].value_counts(sort=False).sort_index()
    sentiment_counts.plot(kind='bar', title=title, figsize=figsize, color='skyblue')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    print("Plotting distribution of review scores (1 to 5, with 3 removed), keeping original score order.")

# --------------- generate_wordcloud ---------------

def generate_wordcloud(
    counter,
    title="Word Cloud",
    remove_stopwords=True,
    max_words=200
):
    """
    Generates and displays a word cloud from a Counter of word frequencies.

    Parameters:
        counter (Counter): A Counter object with word frequencies.
        title (str): Title of the plot.
        remove_stopwords (bool): Whether to remove common English stopwords.
        max_words (int): Maximum number of words to display in the cloud.

    Returns:
        None
    """
    # generate the word cloud
    wordcloud = WordCloud(
        width=800, height=400, background_color='white',
        max_words=max_words
    ).generate_from_frequencies(counter)

    # plot the word cloud
    plt.figure(figsize=(10, 5))
    # make the edges of words less pixelated and more visually appealing
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --------------- create_vocabulary ---------------

# from collections import Counter
# import pandas as pd

def create_vocabulary(
    df,
    text_column='reviewText'
):
    """
    Tokenizes the specified text column by lowercasing and splitting by spaces,
    then builds and returns a vocabulary as a Counter object.

    Parameters:
        df (pd.DataFrame): DataFrame containing the review text.
        text_column (str): Name of the column containing the text to process.

    Returns:
        Counter: A dictionary-like object with word frequencies.
        Words: the words in the text.
    """
    # lowercase and split reviews
    tokenized = df[text_column].str.lower().str.split()
    print("Sample tokenized reviews:\n")
    print(tokenized.head(5))

    # flatten list of tokens
    words = tokenized.apply(pd.Series).stack().reset_index(drop=True)
    print("\nFlattened list of all tokens (first 20):")
    print(words.head(5))

    # count word frequencies
    vocabulary = Counter(words)
    print(f"\nVocabulary size: {len(vocabulary)}\n")
    print("Top 20 most frequent words:")
    for word, freq in vocabulary.most_common(30):
      print(f"{freq:<5}  {word}")

    return vocabulary, words

# --------------- label_sentiment ---------------

def label_sentiment(row):
    """
    Labels sentiment as binary: 0 (negative) or 1 (positive).
    A sentiment score lower than 3 is considered negative.
    There are no neutral reviews, so they have been already deleted.

    Parameters:
        row (pd.Series): A row from a DataFrame containing a 'overall' column.

    Returns:
        int: 0 for negative sentiment, 1 for positive sentiment.
    """
    return 1 if int(row['overall']) > 3 else 0

# --------------- analyze_review_lengths ---------------

def analyze_review_lengths(df):
    """
    Analyzes the review lengths in a DataFrame by adding a new column for review length and plotting the distribution of review lengths.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'reviewText' column.

    Returns:
        pd.DataFrame: The updated DataFrame with the new 'review_length' column.
    """

    lengths_clean = df['reviewText'].astype(str).apply(len)

    max_len = lengths_clean.max()


    counts, bins = np.histogram(lengths_clean, bins=50)
    max_count = counts.max()

    print(f"Max frecuency: {max_count}")

    # plot the histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(lengths_clean, bins=50, kde=True, color='purple')
    plt.xlim(0, max_len)
    plt.ylim(0, max_count + 10)  # Add a small buffer for better visualization
    plt.title("Distribution of Review Lengths")
    plt.xlabel("Number of Characters (not words!)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    return df

# --------------- process_dataset ---------------

def process_dataset(file_name):
    """
    Processes the dataset by loading it, analyzing review lengths,
    and cleaning rows with missing or empty reviews.

    Parameters:
        file_name (str): The path to the .gz file containing the dataset.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # load the dataset directly from the .gz file
    df = pd.read_json(file_name, lines=True)

    print("\nFirst two rows:")
    print(df.head(2).to_string(index=False))

    # display the shape of the DataFrame
    print(f"\nDataset shape: {df.shape}\n")

    # calculate the ds lenghts
    lengths = df['reviewText'].astype(str).apply(len)

    min_len = lengths.min()
    print(f"- Min review length: {min_len}")

    max_len = lengths.max()
    print(f"- Max review length: {max_len}")

    df['reviewLength'] = lengths
    print("- Added new column 'reviewLength' with the number of characters in each review.")

    df = df[df['reviewLength'] > 0]
    print("- Removed reviews with zero or invalid length.")

    df = df[['overall', 'reviewText', 'reviewLength']]
    print("- Filtered dataset to include only relevant columns: overall, reviewText, reviewLength.")

    # I decide to remove the reviews with rating == 3, since I consider them neutral.
    df = df[df['overall'] != 3]
    print("- Removed neutral reviews (overall == 3); dataset now contains only positive and negative reviews.")

    print("\nCleaned Dataset first two rows:")
    print(df.head(2).to_string(index=False))

    print(f"\nDataset shape: {df.shape}")

    return df
