import logging
import os

from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_restful import Api

from reviews_classifier.config import Config
from reviews_classifier.middleware import Middleware
from reviews_classifier.rest.routes import DetectLang, Predict
from reviews_classifier.views.routes import classifier


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config())
    #app.wsgi_app = Middleware(app.wsgi_app)
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    api = Api(app)
    app.register_blueprint(classifier,
                           static_folder='static',
                           static_url_path='/static')
    api.add_resource(DetectLang, "/detect_lang")
    api.add_resource(Predict, "/predict")
    return app
