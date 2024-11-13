import torch
from torch import Tensor
from typing import Optional, Tuple, Union

class TransactionDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to manage transaction data with optional labels.

    Attributes:
        x (Tensor): The input data tensor.
        y (Optional[Tensor]): The target labels tensor, or None if not available.
        device (torch.device): The device on which tensors should be stored.
    """
    
    def __init__(self, x: Tensor, y: Optional[Tensor], device: torch.device) -> None:
        """
        Initializes the TransactionDataset instance.

        Args:
            x (Tensor): Input data tensor.
            y (Optional[Tensor]): Target labels tensor, or None if labels are unavailable.
            device (torch.device): Device to which tensors will be moved.
        """
        self.x = x.to(device)
        self.y = y.to(device) if y is not None else None
        self.device = device

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.x.size(0)

    def __getitem__(self, index: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Retrieves a sample of data (and label, if available) by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: 
                - A tuple containing the input and target tensors if `y` is not None.
                - Only the input tensor if `y` is None.
        """
        x_sample = self.x[index]
        
        if self.y is not None:
            y_sample = self.y[index]
            return x_sample, y_sample
        return x_sample
