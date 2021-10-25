from flask import Flask


def create_app():
    from .model_api import model_api

    app = Flask(__name__)

    from .model_api import model_api

    app.register_blueprint(model_api)

    return app
