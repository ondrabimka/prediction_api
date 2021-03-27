
### Import libraries ###
import falcon
import time
import datetime
from prediction_api.model import ModelClass
from apscheduler.schedulers.blocking import BlockingScheduler

class PredictionResource(object):

    def __init__(self):
        self.model = ModelClass(24)

        ## Runs function every minute 
        scheduler = BlockingScheduler()
        scheduler.add_job(self.main_function, "cron", second=8, kwargs={"past_steps":2})  ## 8 seconds because data is incoplete when loading sooner.
        scheduler.start()

    
    def main_function(self, past_steps):
        self.model.get_predicted_value()
