from sklearn.model_selection import train_test_split
import random
import pandas as pd

# --------------- load_dataframe_csv ---------------

def load_dataframe_csv(path):
    df = pd.read_csv(path, sep=',', decimal='.')
    print(f"DataFrame loaded from: {path} with shape: {df.shape}")
    return df

# --------------- save_dataframe_csv ---------------

def save_dataframe_csv(df, path):
    df.to_csv(path, sep=',', decimal='.', index=False)
    print(f"DataFrame with shape: {df.shape} saved to: {path}")

# --------------- process_and_save_splits ---------------

def process_and_save_splits(df, name, full_path, train_path, test_path, test_size=0.25, random_state=0):
    print(f"\n📁 Processing dataset: {name}")

    # save the full dataframe
    save_dataframe_csv(df, full_path)

    # split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=True, random_state=random_state)
    print(f"{name} train shape: {train_df.shape}, test shape: {test_df.shape}")

    # save train and test splits
    save_dataframe_csv(train_df, train_path)
    save_dataframe_csv(test_df, test_path)


# --------------- load_and_extract_tokens ---------------

def load_and_extract_tokens(path, column_x, column_y, preview_rows=5):
    """
    Load a DataFrame from CSV and extract feature and label columns.

    Parameters:
        path (str): Path to the CSV file.
        column_x (str): Name of the feature column (e.g. tokenized reviews).
        column_y (str): Name of the label column (e.g. sentiment labels).
        preview_rows (int): Number of rows to preview after loading (optional).

    Returns:
        pd.DataFrame: The full loaded DataFrame.
        pd.Series: The feature column (X).
        pd.Series: The label column (y).
    """
    df = load_dataframe_csv(path)

    X = df[column_x]
    y = df[column_y]

    print(f"\nPreview of features ({column_x}):\n{X.head(preview_rows)}")
    print(f"\nPreview of labels ({column_y}):\n{y.head(preview_rows)}")

    return df, X, y

# --------------- inspect_review_tfidf ---------------

def inspect_review_tfidf(X_tokens, y_labels, X_tfidf, vectorizer, index=None, top_n=10):
    """
    Inspect the TF-IDF scores of words in a specific review.

    Parameters:
        X_tokens (pd.Series): Series of tokenized reviews.
        y_labels (pd.Series): Corresponding sentiment labels (0 or 1).
        X_tfidf (scipy.sparse matrix): TF-IDF matrix (from vectorizer.transform).
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        index (int or None): Review index to inspect. If None, selects a random one.
        top_n (int): Number of top and bottom TF-IDF words to display.

    Returns:
        None
    """
    if index is None:
        index = random.randint(0, len(X_tokens) - 1)

    sentiment = 'positive' if y_labels.iloc[index] == 1 else 'negative'
    print(f"\n📝 Review ID: {index}")
    print(f"Sentiment: {sentiment}")
    print(f"Review tokens: {X_tokens.iloc[index]}")

    # get the TF-IDF vector for this review
    doc_vector = X_tfidf[index]
    df_tfidf = pd.DataFrame(
        doc_vector.T.todense(),
        index=vectorizer.get_feature_names_out(),
        columns=['tfidf']
    )
    df_tfidf = df_tfidf[df_tfidf['tfidf'] > 0]

    if df_tfidf.empty:
        print("\n⚠️ No non-zero TF-IDF values in this review (likely all words were filtered).")
        return

    print(f"\n🔝 Top {top_n} words with highest TF-IDF:")
    print(df_tfidf.sort_values(by="tfidf", ascending=False).head(top_n))

    print(f"\n🔻 Top {top_n} words with lowest (non-zero) TF-IDF:")
    print(df_tfidf.sort_values(by="tfidf", ascending=False).tail(top_n))
