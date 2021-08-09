
### Import libraries ###
from datetime import datetime, timedelta
import time
from typing import Dict
from google.protobuf.text_format import PrintField
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from tensorflow.python.keras.backend import print_tensor
from prediction_api import credentials
import tensorflow as tf
from tensorflow.keras.models import load_model
from prediction_api.window import TrainWindowGenerator
import joblib
from sqlalchemy import create_engine

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

        ## Load init model
        path = "C:/Users/Admin/OneDrive - České vysoké učení technické v Praze/Plocha/Python/GraphPrediction/models_pct/models_08_04_2021/lstm.model"
        self.model = load_model(path)

        ## Load scaler 
        self.loaded_scaler = joblib.load("C:/Users/Admin/OneDrive - České vysoké učení technické v Praze/Plocha/Python/GraphPrediction/scalers/scaler_my_pct.gz")

        self.stop_saving = False

    
    ## List of symbols I am interested in
    symbols = ["BTCBUSD", "ETHBUSD", "BCHBUSD", "LTCBUSD"]

    ## Empty main dataframe
    main_df = pd.DataFrame()

    ## Save predictions to this list
    predictions = []


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
            data = pd.DataFrame(recived_data, columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore" ])
            data["timestamp"] = pd.to_datetime(data['timestamp'], unit='ms')
            data = data[["timestamp","close"]]
            data["close"] = pd.to_numeric(data["close"], errors="coerce")
            data.set_index("timestamp", inplace=True)
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
        self.main_df = self.main_df.combine_first(df_update)
        self.main_df.update(df_update)
        

    def get_predicted_value(self):

        """
        Makes a prediction using loaded model

        Parameters
        ----------
        None

        """

        self.update_df()

        ## Select desired part of the DF
        main_df_pct = self.main_df.tail(25).copy()
        main_df_pct = main_df_pct[["BTCBUSD_close","LTCBUSD_close","ETHBUSD_close","BCHBUSD_close"]].pct_change(fill_method="ffill")
        main_df_pct = main_df_pct[1:]

        ## Crate prediction
        my_df = pd.DataFrame(self.loaded_scaler.transform(main_df_pct),columns = main_df_pct.columns)
        to_predict = np.array(my_df)
        to_predict = to_predict[None, :, :]
        print("TO PREDICT:")
        print(to_predict)
        prediction = self.model.predict(to_predict)

        ## Append predictions with time.
        prediction_time = datetime.utcnow() + timedelta(minutes = 1)  ## data is in UTC + O
        prediction_time.strftime("%Y-%m-%d %H:%M")
        to_append_dict = {"timestamp": prediction_time.replace(second=0, microsecond=0), "predicted_value":prediction[0][23][0]}
        self.predictions.append(to_append_dict)

        print("Predicted value: " + str(prediction[0][23][0]))

        if self.main_df.shape[0] > 35:
            self.save_data()
            self.stop_saving = True


    def retrain_function(self):

        """
        Added function for possibility of retraining added model

        Parameters
        ----------
        None

        """

        print("inside retrain fucntion")

        my_df = pd.DataFrame(self.loaded_scaler.transform(self.main_df),columns = self.main_df.columns)
        window = TrainWindowGenerator(24, 1, 2, my_df)

        self.save_data()
        self.retrain_model(window, 3, 80)



    def retrain_model(self, window, patience, epochs):

        """
        Added function for possibility of retraining added model

        Parameters
        ----------
        window: TrainWindowGenerator
            window class which wraps data

        patience: Int
            number of epochs when model can be not iproven

        epochs: Int
            max number of epochs

        """


        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min", restore_best_weights=True)
        history = self.model.fit(window.train, epochs=epochs, validation_data=window.val, callbacks=[early_stopping])
        self.history = history


    def save_data(self):

        """
        Save data stored in dataframes to database

        Parameters
        ----------
        None

        """

        predictions_df = pd.DataFrame(self.predictions, columns=["timestamp", "predicted_value"])
        predictions_df.set_index("timestamp", inplace=True)

        print(predictions_df)

        to_save_df = pd.concat([self.main_df[["BTCBUSD_close","LTCBUSD_close","ETHBUSD_close","BCHBUSD_close"]], predictions_df], axis=1)

        self.save_to_sql(to_save_df)

        self.main_df = self.main_df.tail(30) ## Data is saved now I can reduce main_df size

        ## self.save_as_csv(to_save_df)



    def save_to_sql(self, df_to_sql):

        """
        Save data stored in dataframes to a csv file

        Parameters
        ----------
        df_to_csv: DataFrame
            dataframe we would like to save 

        """

        print("Saving to sql")
        engine = create_engine("mysql://root:@localhost/crypto_api_data")
        con = engine.connect()

        sql_get_time = "SELECT timestamp FROM predictions WHERE BTCBUSD_close IS NOT NULL ORDER BY timestamp DESC LIMIT 1"
        latest_date = engine.execute(sql_get_time)
        fetched_date = latest_date.fetchall()
        print("fetched date")
        print(fetched_date)


        if len(fetched_date) == 0:

            df_to_sql.to_sql("predictions",con=con, if_exists='append', index_label="timestamp")
            con.close()
            engine.dispose()

        else:

            sql_delete = "DELETE FROM predictions WHERE timestamp > '%s'" % fetched_date[0][0]
            engine.execute(sql_delete)

            df_to_sql = df_to_sql[df_to_sql.index > fetched_date[0][0]]
            df_to_sql.to_sql("predictions",con=con, if_exists='append', index_label="timestamp")
            con.close()
            engine.dispose()


    def save_as_csv(self, df_to_csv):

        """
        Save data stored in dataframes to a csv file

        Parameters
        ----------
        df_to_csv: DataFrame
            dataframe we would like to save 

        """


        csv_file_path = 'C:/Users/Admin/OneDrive - České vysoké učení technické v Praze/Plocha/Python/GraphPrediction/prediction_api_files/predictions_' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.csv'
        df_to_csv.to_csv(csv_file_path)
        







    
    












