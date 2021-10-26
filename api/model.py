import pandas as pd
import numpy as np
import pickle

from .config import X_OBJECT_COLUMNS
from .config import DATASET_PATH
from .config import Y_COLUMN

from .preprocessing import encoder_object
from .preprocessing import save_object
from .preprocessing import load_object

from sklearn.ensemble import RandomForestClassifier


def encode_object_columns():
    # Loading Dataset
    df = pd.read_csv(DATASET_PATH)

    # Save object and load
    for column in X_OBJECT_COLUMNS:
        file_path = f"api/object_save/encoder_columns_{column}"
        encoder = encoder_object(df[column])

        save_object(encoder, file_path)

        encoder = load_object(file_path)

        df[column] = encoder.transform(df[column])

    # Save object and load label
    file_path = "api/object_save/encoder_columns_label"

    encoder = encoder_object(df[Y_COLUMN])

    save_object(encoder, file_path)

    encoder = load_object(file_path)

    df[Y_COLUMN] = encoder.transform(df[Y_COLUMN])

    # Modelling
    X, y = df.drop(columns=[Y_COLUMN]), df[Y_COLUMN]
    file_path = "api/object_save/model_object_rfc"

    model = RandomForestClassifier()
    model.fit(X, y)

    save_object(model, file_path)
