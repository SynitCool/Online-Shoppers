import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def stratified_kfold(X, y, model, n_splits=5):
    if type(model) != list:
        raise ValueError("type of model supposed to be list")

    skf = StratifiedKFold(n_splits=n_splits)

    mean_accuracy = {}

    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"\nFold : {fold}")

        X_train, X_test = X.loc[train_index], X.loc[val_index]

        y_train, y_test = y[train_index], y[val_index]

        for i in range(len(model)):
            mdl = model[i]
            mdl.fit(X_train, y_train)

            accuracy = {
                "training": mdl.score(X_train, y_train),
                "validation": mdl.score(X_test, y_test),
            }

            print(
                f"Model {i + 1} | train_accuracy : {accuracy['training']} | val_accuracy : {accuracy['validation']}"
            )

            try:
                mean_accuracy[f"Model {i + 1}"]["training_accuracy"].append(
                    accuracy["training"]
                )
                mean_accuracy[f"Model {i + 1}"]["validation_accuracy"].append(
                    accuracy["validation"]
                )

            except KeyError:
                mean_accuracy[f"Model {i + 1}"] = {
                    "training_accuracy": [],
                    "validation_accuracy": [],
                }

                mean_accuracy[f"Model {i + 1}"]["training_accuracy"].append(
                    accuracy["training"]
                )
                mean_accuracy[f"Model {i + 1}"]["validation_accuracy"].append(
                    accuracy["validation"]
                )

    print("-" * 50)

    for model in mean_accuracy:
        print(
            f"{model} | mean_train_accuracy : {np.mean(mean_accuracy[model]['training_accuracy'])} | mean_val_accuracy : {np.mean(mean_accuracy[model]['validation_accuracy'])}"
        )


def train_test(X, y, model, test_size=0.33, random_state=42):
    if type(model) != list:
        raise ValueError("type of model parameter supposed to be list")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state
    )

    for i in range(len(model)):
        mdl = model[i]
        mdl.fit(X_train, y_train)

        train_accuracy, val_accuracy = mdl.score(X_train, y_train), mdl.score(
            X_test, y_test
        )

        print(
            f"Model {i + 1} | training_accuracy : {train_accuracy} | validation_accuracy : {val_accuracy}"
        )

    return X_train, X_test, y_train, y_test
