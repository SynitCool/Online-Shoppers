import pandas as pd
import matplotlib.pyplot as plt

from model_testing.config import MODEL
from model_testing.config import MODEL_TITLE
from model_testing.config import DATASET_PATH
from model_testing.config import OBJECT_COLUMNS
from model_testing.config import Y_COLUMN

from cleaning.cleaning_data import read_data
from cleaning.cleaning_data import encode_data

from selection_feature.select_best_feature import selection_anova
from selection_feature.select_best_feature import selection_chi2
from selection_feature.select_best_feature import selection_decision_tree
from selection_feature.select_best_feature import selection_mutual_info
from selection_feature.select_best_feature import selection_variance
from selection_feature.select_best_feature import selection_pearson_correlation
from selection_feature.select_change import sc_onehot_pearson_correlation

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


def selection_anova_model(n_k_best=10):
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature Engineering
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    selected_df = selection_anova(X, y, n_k_best=n_k_best)

    # Modelling
    X_train, X_test, y_train, y_test = train_test(selected_df, y, MODEL)

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


def selection_chi2_model(n_k_best=10):
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature Engineering
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    selected_df = selection_chi2(X, y, n_k_best=n_k_best)

    # Modelling
    X_train, X_test, y_train, y_test = train_test(selected_df, y, MODEL)

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


def selection_mutual_info_model(n_k_best=10):
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature Engineering
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    selected_df = selection_mutual_info(X, y, n_k_best=n_k_best)

    # Modelling
    X_train, X_test, y_train, y_test = train_test(selected_df, y, MODEL)

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


def selection_variance_model(threshold=0):
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature Engineering
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    selected_df = selection_variance(X, y, threshold=threshold)

    # Modelling
    X_train, X_test, y_train, y_test = train_test(selected_df, y, MODEL)

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


def selection_decision_tree_model(n_k_best=10):
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature Engineering
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    selected_df = selection_decision_tree(X, y, n_k_best=n_k_best)

    # Modelling
    X_train, X_test, y_train, y_test = train_test(selected_df, y, MODEL)

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


def selection_p_correlation(thresh_corr=0.1):
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature Engineering
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    selected_df = selection_pearson_correlation(X, y, thresh_corr=thresh_corr)

    # Modelling
    X_train, X_test, y_train, y_test = train_test(selected_df, y, MODEL)

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


def select_change_corr_onehot(thresh_corr=0.1):
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature Engineering
    X = df.drop(columns=[Y_COLUMN])
    y = df[Y_COLUMN]

    selected_df = sc_onehot_pearson_correlation(X, y, thresh_corr=thresh_corr)

    # Modelling
    X_train, X_test, y_train, y_test = train_test(selected_df, y, MODEL)

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
