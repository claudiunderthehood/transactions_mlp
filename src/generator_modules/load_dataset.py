import os
from datetime import datetime
import pandas as pd


def get_files_within_date_range(directory: str, start_date: str, end_date: str) -> list:
    """
    Get list of files within the specified date range.

    Parameters:
        directory (str): The directory containing the files.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        List[str]: List of file paths within the specified date range.
    """
    start_datetime: datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime: datetime = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_files: list = os.listdir(directory)
    filtered_files: list = [
        os.path.join(directory, file)
        for file in all_files
        if start_datetime <= datetime.strptime(file.split('.')[0], '%Y-%m-%d') <= end_datetime
    ]
    return filtered_files


def load_ds(directory: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Read and concatenate data from a list of pickle files.

    Parameters:
        directory (str): The directory containing the files.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """

    file_paths: list = get_files_within_date_range(directory=directory, start_date=start_date, end_date=end_date)

    data_frames: list = [pd.read_pickle(file) for file in file_paths]
    concatenated_df: pd.DataFrame = pd.concat(data_frames)
    concatenated_df.sort_values('TRX_ID', inplace=True)
    concatenated_df.reset_index(drop=True, inplace=True)
    concatenated_df.replace([-1], 0, inplace=True)
    return concatenated_df