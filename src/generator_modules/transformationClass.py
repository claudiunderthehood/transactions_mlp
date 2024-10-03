import datetime
import pandas as pd

class Transform:

    def __init__(self) -> None:
        """The constructor for the Transform class."""
        pass

    def __check_if_weekend(self, transaction_date: datetime) -> bool:
        """
        Determines if the given date is a weekend.

        Parameters:
            transaction_date (datetime): The date of the transaction.

        Returns:
            bool: True if the date is a weekend (Saturday or Sunday), False otherwise.
        """

        day_of_week: int = transaction_date.isoweekday()
        return day_of_week in (6, 7)
    
    def weekend_indicator(self, transaction_date: datetime) -> int:
        """
        Provides a binary indicator of whether the given date is a weekend.

        Parameters:
            transaction_date (datetime): The date of the transaction.

        Returns:
            int: 1 if the date is a weekend, 0 otherwise.
        """
        return int(self.__check_if_weekend(transaction_date))
    

    def __check_if_night(self, transaction_time: datetime) -> bool:
        """
        Determines if the given time is during the night.

        Parameters:
            transaction_time (datetime): The datetime of the transaction.

        Returns:
            bool: True if the time is between 00:00 and 06:59, False otherwise.
        """
        return transaction_time.time() < datetime.datetime.strptime("07:00", "%H:%M").time()
    

    def night_indicator(self, transaction_time: datetime) -> int:
        """
        Provides a binary indicator of whether the given time is during the night.

        Parameters:
            transaction_time (datetime): The datetime of the transaction.

        Returns:
            int: 1 if the time is during the night, 0 otherwise.
        """
        return int(self.__check_if_night(transaction_time))
    
    def analyse_customer_spending(self, transactions: pd.DataFrame, time_windows: list = [1, 7, 30]) -> pd.DataFrame:
        """
            Provides an analysis of customer spending behaviour in specific time windows.

            Parameters:

            transactions (pd.DataFrame): The transactions dataframe.
            time_windows (list): Time windows.

            Returns:

            pd.DataFrame: Returns the dataframe with the new features regarding the client spending behaviour.
        """


        transactions = transactions.sort_values('TRX_DATETIME') 
        transactions = transactions.set_index('TRX_DATETIME')  

        for window_size in time_windows:
            sum: pd.Series = transactions['TRX_AMOUNT'].rolling(f'{window_size}d').sum()
            tx_window: pd.Series = transactions['TRX_AMOUNT'].rolling(f'{window_size}d').count()
            mean: pd.Series = sum / tx_window

            transactions[f'CLIENT_TX_{window_size}DAY_WINDOW'] = tx_window
            transactions[f'CLIENT_MEAN_{window_size}DAY_WINDOW'] = mean

        transactions.reset_index(inplace=True)
        transactions.set_index('TRX_ID')  
        return transactions


    def analyse_terminal_risk(self, transactions: pd.DataFrame, detection_delay: int=7, time_windows: list=[1, 7, 30], feature: str="TERMINAL_ID") -> pd.DataFrame:
        """
            Calculates risk score for terminals

            Parameters:

            transactions (pd.DataFrame): The Transactions dataframe.
            detection_delay (int): The delay factor for detecting the frauds.
            time_windows (list): Time Windows.
            feature (str): The Feature to observe the data.

            Returns:

            pd.DataFrame: returns the whole dataframe with the new features.
        """
        
        transactions = transactions.sort_values('TRX_DATETIME')
        transactions = transactions.set_index('TRX_DATETIME')

        fraud_count_delay: pd.Series = transactions['IS_FRAUD'].rolling(f'{detection_delay}d').sum()
        total_tx_delay: pd.Series = transactions['IS_FRAUD'].rolling(f'{detection_delay}d').count()

        for window_size in time_windows:
            fraud_count_window: pd.Series = transactions['IS_FRAUD'].rolling(f'{detection_delay + window_size}d').sum()
            total_tx_window: pd.Series = transactions['IS_FRAUD'].rolling(f'{detection_delay + window_size}d').count()
            risk_window: pd.Series = (fraud_count_window - fraud_count_delay) / (total_tx_window - total_tx_delay)

            transactions[f'{feature}_TX_{window_size}DAY_WINDOW'] = list(total_tx_window - total_tx_delay)
            transactions[f'{feature}_RISK_SCORE_{window_size}DAY_WINDOW'] = list(risk_window)

        transactions.reset_index(inplace=True)
        transactions.index = transactions.TRX_ID
        transactions.fillna(0, inplace=True)

        return transactions