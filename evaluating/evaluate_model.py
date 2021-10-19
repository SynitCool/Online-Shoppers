import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, ax=None):
    matrix = confusion_matrix(y_true, y_pred)

    df_matrix = pd.DataFrame(matrix)

    if ax:
        sns.heatmap(df_matrix, annot=True, ax=ax)
    else:
        sns.heatmap(df_matrix, annot=True)


def plot_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
