import falcon

from .predictions import PredictionResource

api = application = falcon.API()
prediction_api = PredictionResource()
api.add_route('/prediction_api', prediction_api)