import numpy as np
from pandas.core.frame import DataFrame
import scipy.optimize as optimize


class OptimPredClass():
    
    ##################
    #### Optimize ####
    ##################
    
    #region Optimize model function
    
    @classmethod
    def get_optimized_thresholds(cls, data:DataFrame, print_out=True):
        
        """
        Simulate continous optimization on DataFrame. 

        Parameters
        ----------
        in_df: Dataframe 
            DataFrame which contains data.

        init_optim_window_size: int 
            Initial used lenght of in_df on which will be optimized threshold.

        retrain_interval: int 
            Iterval after which will be thresholds reoptimized.
        
        needed_profit: int 
            How big must profit be to consider trading.

        Returns
        ----------
        >>> Tuple (upper_threshold, lower_threshold, optim_result)
        
        Example
        ----------
        >>> get_optimized_thresholds(my_df, False)

        """
        
        
        """

        about = data.describe(percentiles = [0.15, 0.85])

        result = optimize.minimize(fun=cls.optimize_profit, x0=np.array([about["predictions"]["85%"], about["predictions"]["15%"]]), args=data ,bounds=[(about["predictions"]["mean"],about["predictions"]["max"]),(about["predictions"]["min"],about["predictions"]["mean"])], method = 'Nelder-Mead', options={'maxiter':10000, 'adaptive':True})
        if result.success:

            if print_out == True:
                print("Threshold:"+ str(result.x) +" correct_pct:"+ str(abs(result.fun))+ "")

            return result.x[0], result.x[1], abs(result.fun)

        else:
            
            return about["predictions"]["85%"], about["predictions"]["15%"], 0
            
        """
        
          
    @staticmethod
    def optimize_profit(in_arr ,in_df:DataFrame,  fee=0.00036, init_balance=10000):
        
        """
        Calculate profit based on input DataFrame. 

        Parameters
        ----------
        in_arr:  
            Initial guess.

        in_df: DataFrame 
            Dataframe with data containing predictions.

        fee: int 
            Fee.
        
        init_balance: int 
            Initial balance.
             
        Returns
        ----------
        >>> int (calculated balance)

        Example
        ----------
        >>> optimize_profit(sequential_6)
        """
        
        trade_df = in_df.copy()
        trade_df['prediction_when_created'] = trade_df['predictions'].shift(-1)
        trade_df.dropna(inplace=True)
        
        curr_balance = init_balance
        bought = 0
        
        a, b = in_arr
        
        for row in trade_df.itertuples():
            
            if  a >= row.predictions >= b:
                ## skip if threshold
                continue
            
            if row.predictions > 0 and row.scaled > 0:
                ## if bigger then buy !!? (based on value amount)
            
                if curr_balance > 0:
                    bought = curr_balance/row.org
                    bought = bought - bought*fee
                    curr_balance = 0
                
            elif row.predictions < 0 and row.scaled < 0:
                
                if bought > 0:
                    curr_balance = bought*row.org
                    curr_balance = curr_balance - curr_balance*fee
                    bought = 0

        
        if curr_balance == 0 and bought > 0:
            curr_balance = in_df["org"][-1]*bought
            bought = 0
        
        return -curr_balance
    
    
    #endregion

if __name__ == '__main__':
    
    print("keu")