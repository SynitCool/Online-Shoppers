import pandas as pd

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

from modelling.model_selection import train_test

from evaluating.evaluate_model import plot_classification_report

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))


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
    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))


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
    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))


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
    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))


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
    X_train, X_test, y_train, y_test = train_test(X, y, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))
