import os
import datetime
import pandas as pd

def save_ds(transactions_df: pd.DataFrame, path: str, path1: str) -> None:
    """
        Saves the dataset into a CSV and divides the transactions per days into pickle files.

        Parameters:

        transactions_df (pd.DataFrame): The dataframe that contains all the transactions dataset.
        path (str): The path for the CSV.
        path1 (str): The path for the pickle files.
    """

    if not os.path.exists(path):
        os.makedirs(path)
        filename_output: str = 'transactions.csv'
        transactions_df.to_csv(path+filename_output, index=False)

    if not os.path.exists(path1):
        os.makedirs(path1)
    
    begin: datetime = datetime.datetime.strptime("2024-07-02", "%Y-%m-%d")

    for day in range(transactions_df.TRX_DAYS.max()+1):
        
        transactions_day: pd.DataFrame = transactions_df[transactions_df.TRX_DAYS==day].sort_values('TRX_SECONDS')
        
        date: datetime = begin + datetime.timedelta(days=day)
        filename_output: str = date.strftime("%Y-%m-%d")+'.pkl'
        
        transactions_day.to_pickle(path1+filename_output)