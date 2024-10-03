import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np

class TransactionDatasetForPipeline(Dataset):
    """
    A dataset class for handling transaction data for pipelines, compatible with PyTorch DataLoader.
    
    This class supports both labeled and unlabeled data. If labels are not provided, 
    the target will default to -1.

    Attributes:
        features (torch.FloatTensor): Input features for the dataset.
        labels (Optional[torch.LongTensor]): Target labels (if available). Set to None for unlabeled data.
    """
    
    def __init__(self, features: Union[pd.DataFrame, np.ndarray], labels: Optional[pd.Series] = None):
        """
        Initializes the dataset with features and optional labels.

        Parameters:
            features (Union[pd.DataFrame, np.ndarray]): The input features for the dataset, which can be either 
                                                        a pandas DataFrame or a numpy array.
            labels (Optional[pd.Series]): The target labels (can be None for unlabeled data).
        """
        # Convert features to FloatTensor
        if isinstance(features, pd.DataFrame):
            self.features = torch.FloatTensor(features.values)
        elif isinstance(features, np.ndarray):
            self.features = torch.FloatTensor(features)
        else:
            raise ValueError("Features must be a pandas DataFrame or a numpy array")
        
        # Convert labels to LongTensor if provided, otherwise set to None
        self.labels = torch.LongTensor(labels.values) if labels is not None else None

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Generates one sample of data (features and label, or -1 if no label).

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the feature tensor and the label.
                                      If no label is provided, returns -1 as the label.
        """
        # Get feature data
        feature_sample = self.features[index]
        
        # Get label if available, otherwise return -1
        if self.labels is not None:
            return feature_sample, int(self.labels[index].item())
        else:
            return feature_sample, -1

