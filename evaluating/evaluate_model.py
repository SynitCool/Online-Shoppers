import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def plot_confusion_matrix(y_true, y_pred, ax=None):
    matrix = confusion_matrix(y_true, y_pred)

    df_matrix = pd.DataFrame(matrix)

    if ax:
        sns.heatmap(df_matrix, annot=True, ax=ax)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
    else:
        sns.heatmap(df_matrix, annot=True)

        plt.xlabel("Predicted Label")
        plt.y_label("Actual Label")


def plot_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
