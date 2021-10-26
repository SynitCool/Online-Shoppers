import numpy as np

from flask import Blueprint
from flask import request
from flask import jsonify

from .preprocessing import load_object

from .config import X_OBJECT_INDEX
from .config import X_OBJECT_COLUMNS

model_api = Blueprint("model_api", __name__)


@model_api.route("/api_test", methods=["POST"])
def modelling():
    # loading json
    data = request.get_json(force=True)

    X_data = data["input"]

    # declare model
    model_path = "api/object_save/model_object_rfc"
    model = load_object(model_path)

    # cleaning data
    for index, column in zip(X_OBJECT_INDEX, X_OBJECT_COLUMNS):
        file_path = f"api/object_save/encoder_columns_{column}"
        le = load_object(file_path)

        X_data[index] = le.transform([X_data[index]])

    X_data = np.reshape(X_data, (1, -1))

    # modelling
    y_pred = model.predict(X_data)

    # encode label
    file_path = "api/object_save/encoder_columns_label"
    le = load_object(file_path)
    y_pred = le.transform([y_pred[0]])

    return jsonify(results=str(y_pred[0]))
