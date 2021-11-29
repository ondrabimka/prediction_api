import asyncio
from binance.client import Client
from binance.enums import *
import pandas as pd
import time

class DataClass(Client):

    async def get_data_for_symbol(self, symbol:str, past_steps:int, interval='1m'):
        
        """
        This function gets data from Binance and creates dataframe from it. 

        Parameters
        ----------
        symbol: str
            Desired symbol
        
        past_steps: int
            Number of minutes to the past we want to obtain data.
            
        interval: str
            Inetval of minutes into the past

        Returns
        -------
        Dataframe with close price, volume and trades of defined crypto
        """
        
        print(symbol)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, self.get_historical_klines, symbol, interval, f'{past_steps} minutes ago UTC')

        ## Convert data to pd dataframe
        data = pd.DataFrame(response, columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore" ])
        data["timestamp"] = pd.to_datetime(data['timestamp'], unit='ms')
        data = data[["timestamp","close","volume","trades"]]
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data.set_index("timestamp", inplace=True)
        data.rename(columns={"close":f"{symbol}_close", "volume":f"{symbol}_volume", "trades":f"{symbol}_trades"}, inplace=True)
        
        return data
    

    async def get_data(self):
        res = await asyncio.gather(self.get_data_for_symbol("BTCBUSD", 3),  self.get_data_for_symbol("ETHBUSD", 3),)
        
        df = pd.DataFrame()
        
        for data in res:

            if len(df.index) == 0:
                df = data
            else: 
                df = df.join(data)
                
        return df
        
        



if __name__ == '__main__':
    

    print("start")
    
    binance_api_key = 'trmKnqy43p8lFOOlTpEX1u6nbY34j2fr5J3DiU7TanFAPIhKxg5rxZqB2LWDWv2o'    # API-key 
    binance_api_secret = '3oNacOMuytB4cHG0PCCW43TEZJZ5CPH7HzfkWtJp0nO2ligtvcyPGSM1XZLlWNoh' # API-secret
    
    my_class = DataClass(api_key=binance_api_key, api_secret=binance_api_secret)
    result_df = asyncio.run(my_class.get_data())
    print(result_df)
    