import pandas as pd

from model_testing.config import MODEL
from model_testing.config import MODEL_TITLE
from model_testing.config import DATASET_PATH
from model_testing.config import OBJECT_COLUMNS
from model_testing.config import Y_COLUMN

from cleaning.cleaning_data import read_data
from cleaning.cleaning_data import encode_data

from sampling.sampling_technique import over_sampling_random
from sampling.sampling_technique import over_sampling_smote
from sampling.sampling_technique import over_sampling_svmsmote
from sampling.sampling_technique import over_sampling_adasyn
from sampling.sampling_technique import under_sampling_random
from sampling.sampling_technique import under_sampling_cluster_centroids

from modelling.model_selection import packing_run_model

from evaluating.evaluate_model import plot_classification_report

from sklearn.model_selection import train_test_split

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def over_sampling_smote_model():
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature engineering
    X, y = df.drop(columns=[Y_COLUMN]), df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train, y_train = over_sampling_smote(X_train, y_train)

    # Modelling
    packing_run_model(X_train, X_test, y_train, y_test, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))


def over_sampling_random_model():
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature engineering
    X, y = df.drop(columns=[Y_COLUMN]), df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train, y_train = over_sampling_random(X_train, y_train)

    # Modelling
    packing_run_model(X_train, X_test, y_train, y_test, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))


def over_sampling_adasyn_model():
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature engineering
    X, y = df.drop(columns=[Y_COLUMN]), df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train, y_train = over_sampling_adasyn(X_train, y_train)

    # Modelling
    packing_run_model(X_train, X_test, y_train, y_test, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))


def over_sampling_svmsmote_model():
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature engineering
    X, y = df.drop(columns=[Y_COLUMN]), df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train, y_train = over_sampling_svmsmote(X_train, y_train)

    # Modelling
    packing_run_model(X_train, X_test, y_train, y_test, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))


def under_sampling_random_model():
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature engineering
    X, y = df.drop(columns=[Y_COLUMN]), df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train, y_train = under_sampling_random(X_train, y_train)

    # Modelling
    packing_run_model(X_train, X_test, y_train, y_test, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))


def under_sampling_cluster_centroids_model():
    # Loading dataset
    df = read_data(DATASET_PATH)

    # Cleaning dataset
    encoded_df = encode_data(df, OBJECT_COLUMNS)
    df = df.drop(columns=OBJECT_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    # Feature engineering
    X, y = df.drop(columns=[Y_COLUMN]), df[Y_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train, y_train = under_sampling_cluster_centroids(X_train, y_train)

    # Modelling
    packing_run_model(X_train, X_test, y_train, y_test, MODEL)

    for i in range(len(MODEL)):
        print(f"\nTitle : {MODEL_TITLE[i]}")
        plot_classification_report(y_test, MODEL[i].predict(X_test))
