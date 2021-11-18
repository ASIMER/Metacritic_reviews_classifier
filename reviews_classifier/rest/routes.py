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
        if request.form:
            response = request.form
        else:
            response = request.get_json()
        if not response:
            return {'message': "Request empty"}, 200
        try:
            score, language = predict(response['review_text'][:4000])
            result = {'score': score, 'language': language}
        except:
            result = {'score': 'Error', 'language': 'Error'}
        return result, 200
