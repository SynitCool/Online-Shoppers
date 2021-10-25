from flask import Blueprint

model_api = Blueprint("model_api", __name__)

@model_api.route("/api_test", methods=["POST"])
def modelling():
    pass