import torch
from typing import Optional, Tuple

class TransactionDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling transaction data,
    with optional labels for supervised learning.
    
    Attributes:
        features (torch.Tensor): The input features for the dataset.
        labels (Optional[torch.Tensor]): The target labels. Can be None for inference.
        device (torch.device): The device (CPU/GPU) to which tensors should be moved.
    """

    def __init__(self, features: torch.Tensor, labels: Optional[torch.Tensor], device: torch.device):
        """
        Initializes the dataset with input features, labels (optional), and the device.

        Parameters:
            features (torch.Tensor): The feature data for each sample.
            labels (Optional[torch.Tensor]): The corresponding labels. Set to None for inference.
            device (torch.device): The device where the data will be loaded (e.g., 'cpu' or 'cuda').
        """
        self.features = features
        self.labels = labels
        self.device = device

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generates one sample of data based on the given index.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the feature tensor 
                                                        and the label tensor (or None if no labels).
        """
        feature_sample: torch.Tensor = self.features[idx].to(self.device)
        
        if self.labels is not None:
            label_sample: torch.Tensor = self.labels[idx].to(self.device)
            return feature_sample, label_sample
        else:
            return feature_sample, None
