import falcon
from wsgiref import simple_server
from .predictions import PredictionResource

app = falcon.App()
prediction_api = PredictionResource()
app.add_route('/prediction_api', prediction_api)
