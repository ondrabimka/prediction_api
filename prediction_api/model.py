
### Import libraries ###
import pandas as pd
from binance.client import Client
from binance.enums import *
from prediction_api import credentials
import time

class ModelClass():

    def __init__(self):
        b_client = Client(api_key=credentials.binance_api_key, api_secret=credentials.binance_api_secret)
        self.b_client = b_client
        init_df = self.get_data(24)
        self.main_df = init_df

    
    ## List of symbols I am interested in
    symbols = ["BTCBUSD", "ETHBUSD", "BCHBUSD", "LTCBUSD", "XLMBUSD", "ADABUSD"]

    ## Empty main dataframe
    main_df = pd.DataFrame()


    def get_data(self, past_steps=int):

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
        df_update = self.get_data(2)
        print("df update")
        print(time.strftime("%H:%M"))
        print(time.localtime)
        print(df_update)
        return pd.merge(self.main_df, df_update, how="outer")



    def get_predicted_value(self):
        updated_df = self.update_df()
        #print(updated_df.tail(24))
        pass








