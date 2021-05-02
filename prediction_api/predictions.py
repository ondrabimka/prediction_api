
### Import libraries ###
import falcon
import time
import datetime
from prediction_api.model import ModelClass
from apscheduler.schedulers.blocking import BlockingScheduler

class PredictionResource(object):

    def __init__(self):
        self.model = ModelClass(24)

        print("keu1")
        ## Runs function every minute.  
        scheduler = BlockingScheduler()
        scheduler.add_job(self.main_function, "cron", second=9, kwargs={"past_steps":2}, id='main_func')  ## 9 seconds because data is incoplete when loading sooner. Use cron when you want to run the job periodically at certain time(s) of day.
        scheduler.start()
        print("keu2")

        print("before 3 min")
        ## Retrain model after one week
        retrain_cheduler = BlockingScheduler()
        retrain_cheduler.add_job(self.retrain_function, "interval", minute=10, id="retrain_func")  ## 
        retrain_cheduler.start()

        print("after 3 min")


    
    def main_function(self, past_steps):
        print("main function")
        self.model.get_predicted_value()

    def retrain_function(self):
        print("retraining")
        time.sleep(5)
        self.model.retrain_function()
        print("retraining")
        pass

