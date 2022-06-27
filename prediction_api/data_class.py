import asyncio
from binance.client import Client
from binance.enums import *
import pandas as pd
import credentials

class DataClass(Client):

    async def get_data_for_symbol(self, symbol:str, past_steps:int, interval='1m', to_keep_from_data = ["timestamp","close","volume","trades"]):
        
        """
        This function gets data from Binance for a specific symbol. 

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
        
        Example
        -------
        
        >>> get_data_for_symbol("BTCBUSD", 3, interval='1m', to_keep_from_data = ["timestamp","close","volume","trades"])
        
        """
        
        print(symbol)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, self.get_historical_klines, symbol, interval, f'{past_steps} minutes ago UTC')

        ## Convert data to pd dataframe
        data = pd.DataFrame(response, columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore" ])
        data["timestamp"] = pd.to_datetime(data['timestamp'], unit='ms')
        data = data[to_keep_from_data]
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data.set_index("timestamp", inplace=True)
        data.rename(columns={"close":f"{symbol}_close", "volume":f"{symbol}_volume", "trades":f"{symbol}_trades"}, inplace=True)
        
        return data
    

    async def get_data(self, symbols:list, past_steps=3):
        
        """
        This function specifies which sumybol to take and takes obtained dataframes and puts theem to one. 

        Parameters
        ----------
        symbols: list
            List of symbols
        
        past_steps: int
            Number of minutes to the past we want to obtain data.

        Returns
        -------
        Dataframe with close price, volume and trades of defined crypto
        """
        
        res = await asyncio.gather(*(self.get_data_for_symbol(symbol, past_steps) for symbol in symbols),)
        
        df = pd.DataFrame()
        
        for data in res:

            if len(df.index) == 0:
                df = data
            else: 
                df = df.join(data)
                
        return df
        
        

if __name__ == '__main__':
    
    my_class = DataClass(api_key=credentials.binance_api_key, api_secret=credentials.binance_api_secret)
    result_df = asyncio.run(my_class.get_data(["BTCBUSD","ETHBUSD","LTCBUSD"]))
    print(result_df)
    
    