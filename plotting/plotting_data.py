import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation(df):
    corr = df.corr()

    sns.heatmap(abs(corr), annot=True)
