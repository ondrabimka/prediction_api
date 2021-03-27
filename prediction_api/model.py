
### Import libraries ###
import time
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from prediction_api import credentials
import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.preprocessing import StandardScaler


class ModelClass():

    """
    Main model class which contains function for data manipulation and RNN models.

    Parameters
    ----------
    window_size: int 
        size of loaded model window.
    """


    def __init__(self, window_size=int):

        ## Create binance client for data api
        b_client = Client(api_key=credentials.binance_api_key, api_secret=credentials.binance_api_secret)
        self.b_client = b_client

        ## Get data 
        init_df = self.get_data(window_size)
        print(init_df)
        self.main_df = init_df

        ## Load model
        path = "C:/Users/Admin/OneDrive - České vysoké učení technické v Praze/Plocha/Python/GraphPrediction/models/models_24_11_2020/lstm.model"
        self.model = load_model(path)

    
    ## List of symbols I am interested in
    symbols = ["BTCBUSD", "ETHBUSD", "BCHBUSD", "LTCBUSD", "XLMBUSD", "ADABUSD"]

    ## Empty main dataframe
    main_df = pd.DataFrame()


    def get_data(self, past_steps=int):

        """
        This functoins gets data from Binance and creates dataframe from it. 

        Parameters
        ----------
        past_steps: int
            Number of minutes to the past we want to obtain data.

        Returns
        -------
        Dataframe with close prices of defined crypto
        """

        df = pd.DataFrame()

        for symbol in self.symbols:

            ## Get data 
            recived_data = self.b_client.get_klines(symbol=symbol, interval= '1m', limit=past_steps)

            ## Convert data to pd dataframe
            data = pd.DataFrame(recived_data, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data = data[["timestamp","close"]]
            data.set_index('timestamp', inplace=True)
            data.rename(columns={"close":f"{symbol}_close"}, inplace=True)

            if len(df.index) == 0:
                df = data
            else: 
                df = df.join(data)
 
        return df


    def update_df(self):
        
        """
        Updates main dataframe with latest data

        Parameters
        ----------
        None

        """

        df_update = self.get_data(3)
        print("df update")
        print(time.strftime("%H:%M:%S"))
        print(df_update)
        self.main_df.update(df_update)
        self.main_df = pd.concat([self.main_df, df_update])



    def get_predicted_value(self):

        """
        Makes a prediction using loaded model

        Parameters
        ----------
        None

        """

        self.update_df()
        print(self.main_df)

        my_df = pd.DataFrame(StandardScaler().fit_transform(self.main_df),columns = self.main_df.columns)
        to_predict = np.array(my_df[["BTCBUSD_close","LTCBUSD_close","ETHBUSD_close","BCHBUSD_close"]].tail(24))
        to_predict = to_predict[None, :, :]
        prediction = self.model.predict(to_predict)
        print(prediction)


    def retrain_model(self, window, patience, epochs):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
        history = self.model.fit(window.train, epochs=epochs, validation_data=window.val, callbacks=[early_stopping])
        self.history = history

    
    












