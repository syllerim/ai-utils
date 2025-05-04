from sklearn.model_selection import train_test_split


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
