
### Import libraries ###
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
from data_class import DataClass
import tensorflow as tf
from tensorflow.keras.models import load_model
from window import TrainWindowGenerator
import joblib
from sqlalchemy import create_engine
import asyncio
from optim_pred import OptimPredClass 
import credentials

###

class ModelClass():

    """
    Main model class which contains function for data manipulation and RNN models.

    Parameters
    ----------
    window_size: int 
        Size of loaded model window.
        
    symbol_to_predic: str
        Symbol we are predicting
    
    """

    def __init__(self, window_size:int, symbol_to_predict:str):
        
        ## Window size
        self.window_size = window_size
        
        ## Symbol we are predicting 
        self.symbol_to_predict = symbol_to_predict

        ## Create binance client for data api
        self.data_client = DataClass(api_key=credentials.binance_api_key, api_secret=credentials.binance_api_secret)

        ## Get data 
        init_df = asyncio.run(self.data_client.get_data(self.symbols, window_size))
        print(init_df)
        self.main_df = init_df

        ## Load init model
        path = "C:/Users/Admin/OneDrive - České vysoké učení technické v Praze/Plocha/Python/GraphPrediction/models_pct/models_02_01_2022_10_b_3_w_3_0_crypto_DOGEBUSD_all_18_oct_w_volume_trades_1024x5/g_b_s.model"
        self.model = load_model(path)

        ## Load scaler 
        self.loaded_scaler = joblib.load("C:/Users/Admin/OneDrive - České vysoké učení technické v Praze/Plocha/Python/GraphPrediction/scalers/scaler_02_01_2022_4_b_3_w_3_0_crypto_DOGEBUSD_all_18_oct_w_volume_trades_1024x5.gz")

        self.stop_saving = False
        

    ## List of symbols I am interested in
    symbols = ["DOGEBUSD","BTCBUSD","ETHBUSD","FTTBUSD","SNXBUSD","SRMBUSD","ADABUSD","LINKBUSD","ZECBUSD","MASKBUSD","CRVBUSD","SYSBUSD","GALABUSD","SANDBUSD"]

    ## Empty main dataframe
    main_df = pd.DataFrame()

    ## Save predictions to this list
    predictions = []


    def update_df(self):
        
        """
        Updates main dataframe with latest data

        Parameters
        ----------
        None

        """

        df_update = asyncio.run(self.data_client.get_data(self.symbols))
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
        
        ## Add _close to list
        string = '_close'
        symbols_close = [x + string for x in self.symbols]
        

        ## Select desired part of the DF
        main_df_pct = self.main_df.tail(self.window_size+1).copy()
        main_df_pct[symbols_close] = main_df_pct[symbols_close].pct_change(fill_method="ffill")
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
        #to_append_dict = {"timestamp": prediction_time.replace(second=0, microsecond=0), "predicted_value":prediction[0][self.window_size-1][0]}
        to_append_dict = {"timestamp": prediction_time.replace(second=0, microsecond=0), "predicted_value":prediction[0][0]}
        self.predictions.append(to_append_dict)

        #print("Predicted value: " + str(prediction[0][self.window_size-1][0]))
        print("Predicted value: " + str(prediction[0][0]))

        if self.main_df.shape[0] > 55:
            
            #upper, lower, val = OptimPredClass.get_optimized_thresholds(self.main_df)
            print("uper")
            #print(upper)
            #print(lower)
            #print(val)
            
            self.save_data()


    ##################################
    ###  Retraining model function ###    
    ##################################
    
    #region Retraining model function
    
    def retrain_function(self):

        """
        Added function for possibility of retraining added model

        Parameters
        ----------
        None

        """

        print("inside retrain function")

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
        
    #endregion
        
        
    ##############################
    ###  Saving data functions ###    
    ##############################
    
    #region Saving data functions

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
        
        ## Add _close to list
        string = '_close'
        symbols_close = [x + string for x in self.symbols]

        to_save_df = pd.concat([self.main_df[symbols_close], predictions_df], axis=1)

        #self.save_to_sql(to_save_df)

        self.main_df = self.main_df.tail(30) ## Data is saved now I can reduce main_df size

        self.save_as_csv(to_save_df)



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
        
    #endregion







    
    












