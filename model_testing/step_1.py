import pandas as pd
import matplotlib.pyplot as plt

from model_testing.config import MODEL
from model_testing.config import MODEL_TITLE
from model_testing.config import DATASET_PATH
from model_testing.config import OBJECT_COLUMNS
from model_testing.config import Y_COLUMN
from model_testing.config import X_OBJECT_COLUMNS
from model_testing.config import X_CATEGORICAL_COLUMNS
from model_testing.config import X_CONTINUES_COLUMNS
from model_testing.config import CATEGORICAL_COLUMNS
from model_testing.config import CONTINUES_COLUMNS

from cleaning.cleaning_data import read_data
from cleaning.cleaning_data import encode_data
from cleaning.cleaning_data import onehot_data

from modelling.model_selection import train_test

from evaluating.evaluate_model import plot_classification_report
from evaluating.evaluate_model import plot_confusion_matrix

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def encode_object_columns():
    # Loading Dataset
    df = read_data(DATASET_PATH)

    # Cleaning Dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Modelling
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))

    fig, ax = plt.subplots(figsize=(15, 9), nrows=2, ncols=2)

    for i in range(4):
        y_pred = MODEL[i].predict(X_test)
        if i < 2:
            plot_confusion_matrix(y_test, y_pred, ax=ax[0, i])

            ax[0, i].set_title(MODEL_TITLE[i])
        else:
            i_ = i - 2

            plot_confusion_matrix(y_test, y_pred, ax=ax[1, i_])

            ax[1, i_].set_title(MODEL_TITLE[i])


def one_hot_object_columns():
    # Loading Dataset
    df = read_data(DATASET_PATH)

    # Cleaning Dataset
    one_hot_df = onehot_data(df, X_OBJECT_COLUMNS)
    encoded_df = encode_data(df, [Y_COLUMN])

    df = df.drop(columns=X_OBJECT_COLUMNS)
    df = df.drop(columns=[Y_COLUMN])

    df = pd.concat([df, one_hot_df, encoded_df], axis=1)

    # Modelling
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))

    fig, ax = plt.subplots(figsize=(15, 9), nrows=2, ncols=2)

    for i in range(4):
        y_pred = MODEL[i].predict(X_test)
        if i < 2:
            plot_confusion_matrix(y_test, y_pred, ax=ax[0, i])

            ax[0, i].set_title(MODEL_TITLE[i])
        else:
            i_ = i - 2

            plot_confusion_matrix(y_test, y_pred, ax=ax[1, i_])

            ax[1, i_].set_title(MODEL_TITLE[i])


def one_hot_categorical_columns():
    # Loading Dataset
    df = read_data(DATASET_PATH)

    # Cleaning Dataset
    one_hot_df = onehot_data(df, X_CATEGORICAL_COLUMNS)
    encoded_df = encode_data(df, [Y_COLUMN])

    df = df.drop(columns=X_CATEGORICAL_COLUMNS)
    df = df.drop(columns=[Y_COLUMN])

    df = pd.concat([df, one_hot_df, encoded_df], axis=1)

    # Modelling
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))

    fig, ax = plt.subplots(figsize=(15, 9), nrows=2, ncols=2)

    for i in range(4):
        y_pred = MODEL[i].predict(X_test)
        if i < 2:
            plot_confusion_matrix(y_test, y_pred, ax=ax[0, i])

            ax[0, i].set_title(MODEL_TITLE[i])
        else:
            i_ = i - 2

            plot_confusion_matrix(y_test, y_pred, ax=ax[1, i_])

            ax[1, i_].set_title(MODEL_TITLE[i])

def one_hot_all():
    # Loading Dataset
    df = read_data(DATASET_PATH)

    X_columns = list(df.columns)
    X_columns.remove(Y_COLUMN)

    # Cleaning Dataset
    one_hot_df = onehot_data(df, X_columns)
    encoded_df = encode_data(df, [Y_COLUMN])

    df = df.drop(columns=X_columns)
    df = df.drop(columns=[Y_COLUMN])

    df = pd.concat([df, one_hot_df, encoded_df], axis=1)

    # Modelling
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))

    fig, ax = plt.subplots(figsize=(15, 9), nrows=2, ncols=2)

    for i in range(4):
        y_pred = MODEL[i].predict(X_test)
        if i < 2:
            plot_confusion_matrix(y_test, y_pred, ax=ax[0, i])

            ax[0, i].set_title(MODEL_TITLE[i])
        else:
            i_ = i - 2

            plot_confusion_matrix(y_test, y_pred, ax=ax[1, i_])

            ax[1, i_].set_title(MODEL_TITLE[i])

def encode_categorical_model():
    # Loading Dataset
    df = read_data(DATASET_PATH)
    df = df[CATEGORICAL_COLUMNS]

    # Cleaning Dataset
    encoded_df = encode_data(df, CATEGORICAL_COLUMNS)

    df = df.drop(columns=CATEGORICAL_COLUMNS)

    df = pd.concat([df, encoded_df], axis=1)

    # Modelling
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))

    fig, ax = plt.subplots(figsize=(15, 9), nrows=2, ncols=2)

    for i in range(4):
        y_pred = MODEL[i].predict(X_test)
        if i < 2:
            plot_confusion_matrix(y_test, y_pred, ax=ax[0, i])

            ax[0, i].set_title(MODEL_TITLE[i])
        else:
            i_ = i - 2

            plot_confusion_matrix(y_test, y_pred, ax=ax[1, i_])

            ax[1, i_].set_title(MODEL_TITLE[i])

def one_hot_categorical_model():
    # Loading Dataset
    df = read_data(DATASET_PATH)
    df = df[CATEGORICAL_COLUMNS]

    X_columns = list(df.columns)
    X_columns.remove(Y_COLUMN)

    # Cleaning Dataset
    one_hot_df = onehot_data(df, X_columns)
    encoded_df = encode_data(df, [Y_COLUMN])

    df = df.drop(columns=CATEGORICAL_COLUMNS)

    df = pd.concat([df, one_hot_df, encoded_df], axis=1)

    # Modelling
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))

    fig, ax = plt.subplots(figsize=(15, 9), nrows=2, ncols=2)

    for i in range(4):
        y_pred = MODEL[i].predict(X_test)
        if i < 2:
            plot_confusion_matrix(y_test, y_pred, ax=ax[0, i])

            ax[0, i].set_title(MODEL_TITLE[i])
        else:
            i_ = i - 2

            plot_confusion_matrix(y_test, y_pred, ax=ax[1, i_])

            ax[1, i_].set_title(MODEL_TITLE[i])

def continues_model():
    # Loading Dataset
    df = read_data(DATASET_PATH)
    df = df[CONTINUES_COLUMNS]

    # Cleaning Dataset
    encoded_df = encode_data(df, [Y_COLUMN])

    df = df.drop(columns=[Y_COLUMN])

    df = pd.concat([df, encoded_df], axis=1)

    # Modelling
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))

    fig, ax = plt.subplots(figsize=(15, 9), nrows=2, ncols=2)

    for i in range(4):
        y_pred = MODEL[i].predict(X_test)
        if i < 2:
            plot_confusion_matrix(y_test, y_pred, ax=ax[0, i])

            ax[0, i].set_title(MODEL_TITLE[i])
        else:
            i_ = i - 2

            plot_confusion_matrix(y_test, y_pred, ax=ax[1, i_])

            ax[1, i_].set_title(MODEL_TITLE[i])

def one_hot_continues_model():
    # Loading Dataset
    df = read_data(DATASET_PATH)
    df = df[CONTINUES_COLUMNS]

    X_columns = list(df.columns)
    X_columns.remove(Y_COLUMN)

    # Cleaning Dataset
    one_hot_df = onehot_data(df, X_columns)
    encoded_df = encode_data(df, [Y_COLUMN])

    df = df.drop(columns=CONTINUES_COLUMNS)

    df = pd.concat([df, one_hot_df, encoded_df], axis=1)

    # Modelling
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))

    fig, ax = plt.subplots(figsize=(15, 9), nrows=2, ncols=2)

    for i in range(4):
        y_pred = MODEL[i].predict(X_test)
        if i < 2:
            plot_confusion_matrix(y_test, y_pred, ax=ax[0, i])

            ax[0, i].set_title(MODEL_TITLE[i])
        else:
            i_ = i - 2

            plot_confusion_matrix(y_test, y_pred, ax=ax[1, i_])

            ax[1, i_].set_title(MODEL_TITLE[i])