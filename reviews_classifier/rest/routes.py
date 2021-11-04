from flask import request
from flask_restful import Resource


class DetectLang(Resource):
    def get(self):

        return 'test_detect_lang', 200
    def post(self):
        response = request.get_json()
        if not response:
            return {'message': "Request empty"}, 200

        return {'message': 'Pet entity has been created'}, 201
