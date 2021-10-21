import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def read_data(data_path, sep=","):
    df = pd.read_csv(data_path, sep=sep)

    return df


def binning_data(df, columns, n_q):
    if type(columns) == list or type(columns) == np.ndarray:
        for column in columns:
            feature = df[column]
    else:
        raise ValueError("columns parameter must be type of list or array")


def encode_data(df, columns):
    if type(columns) != list:
        raise ValueError("columns parameter must be type of list")

    output_df = pd.DataFrame()
    for column in columns:
        le = LabelEncoder()
        feature = df[column]

        output_df[column] = le.fit_transform(feature)

    return output_df


def onehot_data(df, columns):
    if type(columns) != list:
        raise ValueError("columns parameter must be type of list")

    one_hot = OneHotEncoder()
    encoded = one_hot.fit_transform(df[columns]).toarray()

    output_df = pd.DataFrame(encoded)

    return output_df
