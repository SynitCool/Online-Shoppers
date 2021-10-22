import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_testing.config import DATASET_PATH
from model_testing.config import X_CATEGORICAL_COLUMNS
from model_testing.config import CATEGORICAL_COLUMNS
from model_testing.config import Y_COLUMN
from model_testing.config import MODEL
from model_testing.config import MODEL_TITLE
from model_testing.config import OBJECT_COLUMNS
from model_testing.config import X_OBJECT_COLUMNS

from cleaning.cleaning_data import read_data
from cleaning.cleaning_data import onehot_data
from cleaning.cleaning_data import encode_data

from modelling.model_selection import train_test

from evaluating.evaluate_model import plot_classification_report
from evaluating.evaluate_model import plot_confusion_matrix


def onehot_fe_categorical_columns():
    """
    Process testing

    read data --> encode data --> feature engineering --> modelling --> evaluation
    feature engineering
    combining columns
    [Administrative_Duration, Informational_Duration, ProductRelated_Duration] --> time spent
    [BounceRates, ExitRates] --> dislike rates
    [SpecialDay] --> y_n_special_day >= 0.5 = 1 and < 0.5 = 0
    [Month] --> semester = [1, 2]

    preprocessing
    one hot categorical columns
    """

    sems_1 = ["Jan", "Feb", "Mar", "Apr", "May", "June"]
    sems_2 = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    duration_columns = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
    ]
    rates_columns = ["BounceRates", "ExitRates"]
    new_cat_columns = ["y_n_special_day", "Semester"]
    cat_columns = X_CATEGORICAL_COLUMNS.copy()
    cat_columns.extend(new_cat_columns)

    # Loading data
    df = read_data(DATASET_PATH)

    # Feature Engineering
    df["TimeSpent"] = df[duration_columns].sum(axis=1)
    df["DislikeRates"] = df[rates_columns].sum(axis=1)
    df["y_n_special_day"] = df["SpecialDay"].apply(lambda x: 1 if x >= 0.5 else 0)
    df["Semester"] = df["Month"].apply(lambda x: 1 if x in sems_1 else 2)

    # Preprocessing
    one_hot_df = onehot_data(df, cat_columns)

    df = df.drop(columns=cat_columns)

    df = pd.concat([df, one_hot_df], axis=1)

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


def encode_fe_categorical_columns():
    """
    Process testing

    read data --> encode data --> feature engineering --> modelling --> evaluation
    feature engineering
    combining columns
    [Administrative_Duration, Informational_Duration, ProductRelated_Duration] --> time spent
    [BounceRates, ExitRates] --> dislike rates
    [SpecialDay] --> y_n_special_day >= 0.5 = 1 and < 0.5 = 0
    [Month] --> semester = [1, 2]

    preprocessing
    encode categorical columns
    """

    sems_1 = ["Jan", "Feb", "Mar", "Apr", "May", "June"]
    sems_2 = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    duration_columns = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
    ]
    rates_columns = ["BounceRates", "ExitRates"]
    new_cat_columns = ["y_n_special_day", "Semester"]
    cat_columns = CATEGORICAL_COLUMNS.copy()
    cat_columns.extend(new_cat_columns)

    # Loading data
    df = read_data(DATASET_PATH)

    # Feature Engineering
    df["TimeSpent"] = df[duration_columns].sum(axis=1)
    df["DislikeRates"] = df[rates_columns].sum(axis=1)
    df["y_n_special_day"] = df["SpecialDay"].apply(lambda x: 1 if x >= 0.5 else 0)
    df["Semester"] = df["Month"].apply(lambda x: 1 if x in sems_1 else 2)

    # Preprocessing
    encoded_df = encode_data(df, cat_columns)

    df = df.drop(columns=cat_columns)

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


def onehot_fe_object_columns():
    """
    Process testing

    read data --> encode data --> feature engineering --> modelling --> evaluation
    feature engineering
    combining columns
    [Administrative_Duration, Informational_Duration, ProductRelated_Duration] --> time spent
    [BounceRates, ExitRates] --> dislike rates
    [SpecialDay] --> y_n_special_day >= 0.5 = 1 and < 0.5 = 0
    [Month] --> semester = [1, 2]

    preprocessing
    one hot object columns
    """

    sems_1 = ["Jan", "Feb", "Mar", "Apr", "May", "June"]
    sems_2 = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    duration_columns = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
    ]
    rates_columns = ["BounceRates", "ExitRates"]
    new_cat_columns = ["y_n_special_day", "Semester"]
    obj_columns = X_OBJECT_COLUMNS.copy()
    obj_columns.extend(new_cat_columns)

    # Loading data
    df = read_data(DATASET_PATH)

    # Feature Engineering
    df["TimeSpent"] = df[duration_columns].sum(axis=1)
    df["DislikeRates"] = df[rates_columns].sum(axis=1)
    df["y_n_special_day"] = df["SpecialDay"].apply(lambda x: 1 if x >= 0.5 else 0)
    df["Semester"] = df["Month"].apply(lambda x: 1 if x in sems_1 else 2)

    # Preprocessing
    one_hot_df = onehot_data(df, obj_columns)

    df = df.drop(columns=obj_columns)

    df = pd.concat([df, one_hot_df], axis=1)

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


def encode_fe_object_columns():
    """
    Process testing

    read data --> encode data --> feature engineering --> modelling --> evaluation
    feature engineering
    combining columns
    [Administrative_Duration, Informational_Duration, ProductRelated_Duration] --> time spent
    [BounceRates, ExitRates] --> dislike rates
    [SpecialDay] --> y_n_special_day >= 0.5 = 1 and < 0.5 = 0
    [Month] --> semester = [1, 2]

    preprocessing
    encoded object columns
    """

    sems_1 = ["Jan", "Feb", "Mar", "Apr", "May", "June"]
    sems_2 = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    duration_columns = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
    ]
    rates_columns = ["BounceRates", "ExitRates"]
    new_cat_columns = ["y_n_special_day", "Semester"]
    obj_columns = X_OBJECT_COLUMNS.copy()
    obj_columns.extend(new_cat_columns)

    # Loading data
    df = read_data(DATASET_PATH)

    # Feature Engineering
    df["TimeSpent"] = df[duration_columns].sum(axis=1)
    df["DislikeRates"] = df[rates_columns].sum(axis=1)
    df["y_n_special_day"] = df["SpecialDay"].apply(lambda x: 1 if x >= 0.5 else 0)
    df["Semester"] = df["Month"].apply(lambda x: 1 if x in sems_1 else 2)

    # Preprocessing
    encoded_df = encode_data(df, obj_columns)

    df = df.drop(columns=obj_columns)

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


def onehot_select_categorical_columns(thress_sum=10):
    """
    Feature engineering after one hot and remove useless values

    process
    read data --> one hot categorical columns --> remove columns that sum < 10 --> Modelling
    """
    # Loading Data and initialize
    df = read_data(DATASET_PATH)

    # Feature Engineering
    encoded_df = encode_data(df, [Y_COLUMN])
    df = df.drop(columns=[Y_COLUMN])

    df = pd.concat([df, encoded_df], axis=1)

    onehot_df = onehot_data(df, X_CATEGORICAL_COLUMNS)
    onehot_df_sum = np.array(onehot_df.sum(axis=0))
    onehot_df_sum_thress = np.array(
        [col_sum for col_sum in onehot_df_sum if col_sum >= thress_sum]
    )
    index_sum_thress = np.where(
        np.reshape(onehot_df_sum_thress, (-1, 1)) == onehot_df_sum
    )[1]

    onehot_df = np.array(onehot_df)
    onehot_df = onehot_df[:, index_sum_thress]
    onehot_df = pd.DataFrame(onehot_df)

    df = df.drop(columns=X_CATEGORICAL_COLUMNS)

    df = pd.concat([df, onehot_df], axis=1)

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
