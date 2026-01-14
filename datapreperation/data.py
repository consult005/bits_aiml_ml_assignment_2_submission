import pandas as pd
from datapreperation.config import DATA_PATH, TARGET_COL

def load_dataset():
    # bank-additional-full.csv uses semicolon separator
    df = pd.read_csv(DATA_PATH, sep=";")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    # Map yes/no -> 1/0 for binary classification
    df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0})

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
