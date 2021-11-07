from flask import request
from flask_restful import Resource

from reviews_classifier.service.classifier import predict


class DetectLang(Resource):
    def get(self):

        return 'test_detect_lang', 200
    def post(self):
        response = request.get_json()
        if not response:
            return {'message': "Request empty"}, 200

        return {'message': 'Pet entity has been created'}, 201


class Predict(Resource):
    def get(self):

        return 'test_detect_lang', 200

    def post(self):
        response = request.get_json()
        if not response:
            return {'message': "Request empty"}, 200

        score = predict(response['vector'])
        return {'Score': score}, 200
