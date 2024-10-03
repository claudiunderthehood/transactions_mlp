from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(
    training_data: pd.DataFrame, 
    testing_data: pd.DataFrame, 
    feature_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scales the specified feature columns in both the training and testing datasets 
    using the StandardScaler from scikit-learn.

    Parameters:
        training_data (pd.DataFrame): The training dataset.
        testing_data (pd.DataFrame): The testing dataset.
        feature_columns (List[str]): List of column names to be normalized.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the scaled training and testing datasets.
    """
    
    # Initialize the StandardScaler
    standard_scaler: StandardScaler = StandardScaler()

    # Fit the scaler on the training data and apply the transformation
    standard_scaler.fit(training_data[feature_columns])
    
    # Scale the training and testing data
    training_data[feature_columns] = standard_scaler.transform(training_data[feature_columns])
    testing_data[feature_columns] = standard_scaler.transform(testing_data[feature_columns])
    
    return training_data, testing_data
