import datetime

import time

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler

class Classification:
    def __init__(self) -> None:
        """
            Constructor for Classification class.
        """
        pass


    def split_train_test(self, transactions: pd.DataFrame,
                     training_start_date: str,
                     training_duration: int = 7,
                     delay_duration: int = 7,
                     test_duration: int = 7
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split transactions into training and test sets.

        Parameters:

            transactions (pd.DataFrame): DataFrame containing transaction data.
            training_start_date (datetime): The start date for the training period.
            training_duration (int): The number of days for the training period.
            delay_duration (int): The number of days for the delay period.
            test_duration (int): The number of days for the test period.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: DataFrames for the training set and the test set.
        """
        
        if isinstance(training_start_date, str):
            training_start_date = datetime.datetime.strptime(training_start_date, "%Y-%m-%d")
        
        training_end_date = training_start_date + datetime.timedelta(days=training_duration)
        
        training_set: pd.DataFrame = transactions[(transactions['TRX_DATETIME'] >= training_start_date) &
                                    (transactions['TRX_DATETIME'] < training_end_date)]
        
        test_set_list: list = []

        defrauded_customers: set = set(training_set[training_set['IS_FRAUD'] == 1]['CLIENT_ID'])

        training_start_day: datetime = training_set['TRX_DAYS'].min()

        for day in range(test_duration):
            current_test_day: datetime = training_start_day + training_duration + delay_duration + day
            test_day_data: datetime = transactions[transactions['TRX_DAYS'] == current_test_day]
            
            delay_period_day: datetime = training_start_day + training_duration + day - 1
            delay_period_data: pd.DataFrame = transactions[transactions['TRX_DAYS'] == delay_period_day]
            
            new_defrauded_customers: set = set(delay_period_data[delay_period_data['IS_FRAUD'] == 1]['CLIENT_ID'])
            defrauded_customers.update(new_defrauded_customers)
            
            filtered_test_day_data: datetime = test_day_data[~test_day_data['CLIENT_ID'].isin(defrauded_customers)]
            
            test_set_list.append(filtered_test_day_data)
        
        test_set: pd.DataFrame = pd.concat(test_set_list)
        
        training_set = training_set.sort_values('TRX_ID')
        test_set = test_set.sort_values('TRX_ID')
        
        return training_set, test_set


    def train_and_predict(self, model: ClassifierMixin, 
                      train_data: pd.DataFrame, 
                      test_data: pd.DataFrame, 
                      features: list, 
                      target: str = "IS_FRAUD", 
                      should_scale: bool = True) -> dict[str, any]:
        """
        Train a model and get predictions for both training and test datasets.

        Parameters:
            model (ClassifierMixin): The classifier to be trained.
            train_data (pd.DataFrame): DataFrame containing the training data.
            test_data (pd.DataFrame): DataFrame containing the test data.
            features (list): List of input features.
            target (str): The target feature to predict.
            should_scale (bool): Flag to indicate if scaling is required.

        Returns:
            dict[str, any]: Dictionary containing the trained model, predictions, and execution times.
        """
        
        if should_scale:
            scaler = StandardScaler()
            train_data[features] = scaler.fit_transform(train_data[features])
            test_data[features] = scaler.transform(test_data[features])
        
        start_time = time.time()
        model.fit(train_data[features], train_data[target])
        training_time = time.time() - start_time

        start_time = time.time()
        test_predictions = model.predict_proba(test_data[features])[:, 1]
        prediction_time = time.time() - start_time
        
        train_predictions = model.predict_proba(train_data[features])[:, 1]

        test_predictions_proba: int = (test_predictions >= 0.5).astype(int)
        train_predictions_proba: int = (train_predictions >= 0.5).astype(int)

        train_accuracy: float = metrics.accuracy_score(train_data[target], train_predictions_proba)
        test_accuracy: float = metrics.accuracy_score(test_data[target], test_predictions_proba)
        train_precision: float = metrics.precision_score(train_data[target], train_predictions_proba)
        test_precision: float = metrics.precision_score(test_data[target], test_predictions_proba)

        results = {
            'model': model,
            'test_predictions': test_predictions,
            'train_predictions': train_predictions,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision
        }
        
        return results
    

    def __calculate_top_k_precision(self, df: pd.DataFrame, top_k: int) -> tuple[list[int], float]:
        """
        Computes the top-k accuracy for a single day

        Parameters:
            df (pd.DataFrame): The dataframe that contains the transactions.
            top_k (int): Number of k cards to investigate.

        Returns:
            tuple[list[int], float]: List of all the compromised client's cards.
        """
        grouped_df: pd.DataFrame = df.groupby('CLIENT_ID').agg({'predictions': 'max', 'IS_FRAUD': 'max'}).reset_index()
        sorted_df: pd.DataFrame = grouped_df.sort_values(by="predictions", ascending=False).head(top_k)
        
        detected_cards: list = sorted_df[sorted_df['IS_FRAUD'] == 1]['CLIENT_ID'].tolist()
        precision_top_k: float = len(detected_cards) / top_k
    
        return detected_cards, precision_top_k
    

    def __compute_daily_precision(self, predictions_df: pd.DataFrame, top_k: int, exclude_detected: bool = True) -> tuple[list[int], list[float], float]:
        """
        Computes the daily Top-K precision.

        Args:
            predictions_df (pd.DataFrame): DataFrame containing the transactions.
            top_k (int): Number of suspicious cards.
            exclude_detected (bool): Check that excludes already detected cards.

        Returns:
            tuple[list[int], list[float], float]: Number of daily compromised cards, daily top-k precision, daily average top-k precision.
        """
        unique_days: list = sorted(predictions_df['TRX_DAYS'].unique())
        detected_cards: set = set()
        
        daily_precision_list: list = []
        daily_compromised_counts: list = []
        
        for day in unique_days:
            day_df: pd.DataFrame = predictions_df[predictions_df['TRX_DAYS'] == day][['predictions', 'CLIENT_ID', 'IS_FRAUD']]
            
            if exclude_detected:
                day_df = day_df[~day_df['CLIENT_ID'].isin(detected_cards)]
            
            compromised_count: int = day_df[day_df['IS_FRAUD'] == 1]['CLIENT_ID'].nunique()
            daily_compromised_counts.append(compromised_count)
            
            new_detected, precision_top_k = self.__calculate_top_k_precision(day_df, top_k)
            daily_precision_list.append(precision_top_k)
            
            detected_cards.update(new_detected)
        
        mean_precision_top_k: float = np.mean(daily_precision_list)
        
        return daily_compromised_counts, daily_precision_list, mean_precision_top_k
    

    def evaluate_model_performance(self, predictions_df: pd.DataFrame, output_col: str = 'IS_FRAUD', 
                               prediction_col: str = 'predictions', top_k_values: list[int] = [100], 
                               round_results: bool = True) -> pd.DataFrame:
        """
        Evaluates model perfomances.

        Args:
            predictions_df (pd.DataFrame): The transactions dataframe.
            output_col (str): Target Column to predict.
            prediction_col (str): Column containing the predictions.
            top_k_values (list[int]): Top-k values to compute the precision.
            round_results (bool): If true, rounds the results.

        Returns:
            pd.DataFrame: Dataframe with the performance evaluations.
        """
        auc_roc: float = metrics.roc_auc_score(predictions_df[output_col], predictions_df[prediction_col])
        avg_precision: float = metrics.average_precision_score(predictions_df[output_col], predictions_df[prediction_col])
        
        performance_metrics = {
            'AUC ROC': [auc_roc],
            'Average Precision': [avg_precision]
        }
        
        for top_k in top_k_values:
            _, _, mean_precision_top_k = self.__compute_daily_precision(predictions_df, top_k)
            performance_metrics[f'Accuracy for {top_k}'] = [mean_precision_top_k]
        
        performance_df: pd.DataFrame = pd.DataFrame(performance_metrics)
        
        if round_results:
            performance_df = performance_df.round(3)
        
        return performance_df


    def assess_models_performance(self, models_predictions: dict[str, any], 
                              transactions: pd.DataFrame, 
                              dataset_type: str = 'test', 
                              top_k_list: list[int] = [100]) -> pd.DataFrame:
        """
        Assess the performance of multiple models based on their predictions.

        Args:
            models_predictions (dict[str, any]): Dictionary containing models and their predictions.
            transactions (pd.DataFrame): DataFrame containing transaction data.
            dataset_type (str): Specify 'train' or 'test' to select the dataset type.
            top_k_list (list[int]): List of top-k values for precision calculation.

        Returns:
            pd.DataFrame: DataFrame containing performance metrics for each model.
        """
        performances: pd.DataFrame = pd.DataFrame() 
    
        for classifier_name, model_and_predictions in models_predictions.items():
        
            predictions_df: pd.DataFrame = transactions 
                
            predictions_df['predictions']=model_and_predictions[dataset_type+'_predictions']
            
            performances_model: pd.DataFrame = self.evaluate_model_performance(predictions_df, output_col='IS_FRAUD', 
                                                    prediction_col='predictions', top_k_values=top_k_list)
            performances_model.index=[classifier_name]
            
            performances: pd.DataFrame = pd.concat([performances, performances_model])
            
        return performances
    

    def collect_execution_times(self, models_predictions: dict) -> pd.DataFrame:
        """
        Collects and organizes execution times for training and prediction from a dictionary of fitted models.

        Args:
            models_predictions (dict): A dictionary where the keys are classifier names and the values 
                                                            are dictionaries containing 'training_execution_time' and 
                                                            'prediction_execution_time'.

        Returns:
            pd.DataFrame: A dataframe containing the execution times for training and prediction for each classifier.
        """
        execution_data: list = []
        
        for model_name, metrics in models_predictions.items():
            execution_data.append({
                'Classifier': model_name,
                'Training Execution Time': metrics['training_time'],
                'Prediction Execution Time': metrics['prediction_time']
            })
        
        execution_times_df: pd.DataFrame = pd.DataFrame(execution_data)
        
        execution_times_df.set_index('Classifier', inplace=True)
        
        return execution_times_df