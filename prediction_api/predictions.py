
### Import libraries ###
import falcon
import time
from prediction_api.model import ModelClass
from apscheduler.schedulers.blocking import BlockingScheduler

class PredictionResource(object):

    def __init__(self):
        self.model = ModelClass()

        scheduler = BlockingScheduler()
        scheduler.add_job(self.main_function, 'interval' ,seconds=10, kwargs={"past_steps":2})
        scheduler.start()

    
    def main_function(self, past_steps):
        self.model.get_predicted_value()
