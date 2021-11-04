from flask import Flask
from flask_restful import Api

from reviews_classifier.config import Config
from reviews_classifier.middleware import Middleware
from reviews_classifier.rest.routes import DetectLang
from reviews_classifier.views.routes import classifier


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config())
    #app.wsgi_app = Middleware(app.wsgi_app)
    api = Api(app)
    api.register_blueprint(classifier,
                           static_folder='static',
                           static_url_path='/static')
    api.add_resource(DetectLang, "/detect_lang")
    return app
