import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from sklearn.preprocessing import MinMaxScaler

def null_process(df):
    categorical_cols = ['Brand', 'Size', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
    numeric_cols = ['Compartments','Weight Capacity (kg)', 'Price']

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    return df

def remove_outlier(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5*iqr
    upper_bound = q1 + 1.5*iqr
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df

def outlier_process(df):
    numeric_cols = ['Compartments','Weight Capacity (kg)', 'Price']

    for col in numeric_cols:
        df = remove_outlier(df, col)

    df.reset_index(drop=True, inplace=True)
    return df

def encoding(df):
    df.drop("id", axis=1, inplace=True)
    onehot_cols = ['Brand', 'Material', 'Style', 'Color']
    ordinal_cols = ['Size', 'Laptop Compartment', 'Waterproof']

    for col in onehot_cols: # One-hot Encoding
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
        df = pd.concat([df, dummies], axis=1)

    for col in ordinal_cols: # Ordinal Encoding
        df[col] = df[col].astype('category').cat.codes
    df.drop(onehot_cols, axis=1, inplace=True)

    return df

def scaling(df):
    minmax = MinMaxScaler()
    df[df.columns[:5]] = minmax.fit_transform(df[df.columns[:5]])
    return df

def build_feature(df):
    logging.info(f"Null Processing...")
    df = null_process(df)

    logging.info(f"Outlier processing...")
    df = outlier_process(df)

    logging.info(f"Encoding values ...")
    df = encoding(df)

    logging.info(f"Scaling values...")
    df = scaling(df)

    logging.info("Feature engineering successfully!!")
    return df
