import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from cleaning.cleaning_data import encode_data


def plot_correlation(df):
    df = encode_data(df, list(df.columns))

    corr = df.corr()

    sns.heatmap(round(abs(corr), 2), annot=True)
