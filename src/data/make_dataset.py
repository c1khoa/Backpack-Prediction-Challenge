import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def load_data(path):
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path)

def save_data(df, path):
    logging.info(f"Saving data from {path}")
    df.to_csv(path, index=False)