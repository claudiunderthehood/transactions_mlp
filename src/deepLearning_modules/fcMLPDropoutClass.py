import torch
from torch import nn

class SimpleTransactionMLPWithDropout(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model with two fully connected layers of 5 neurons each and Dropout for binary classification.
    """

    def __init__(self, input_dim: int = 15, dropout_rate: float = 0.3):
        """
        Initializes the MLP model with two fully connected layers of 5 neurons each and Dropout.

        Parameters:
            input_dim (int): Number of input features (default is 15).
            dropout_rate (float): Dropout probability (default is 0.3).
        """
        super(SimpleTransactionMLPWithDropout, self).__init__()
        
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        # Two fully connected layers with Dropout
        self.fc1 = nn.Linear(self.input_dim, 5)  # First layer from input_dim to 5 neurons
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.dropout_rate)  # Dropout after first layer

        self.fc2 = nn.Linear(5, 5)  # Second layer with 5 neurons
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=self.dropout_rate)  # Dropout after second layer

        # Output layer for binary classification
        self.fc3 = nn.Linear(5, 1)  # Final output layer with 1 neuron
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the MLP model with Dropout.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 1), containing
                          the predicted probabilities for binary classification.
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  # Apply dropout
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)  # Apply dropout

        # Output layer
        x = self.fc3(x)
        output = self.sigmoid(x)
        
        return output
