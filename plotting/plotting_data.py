import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation(df):
    corr = df.corr()

    sns.heatmap(round(abs(corr), 2), annot=True)
