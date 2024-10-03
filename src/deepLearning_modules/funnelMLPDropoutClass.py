import torch
from torch import nn

class FunnelTransactionMLPWithDropout(nn.Module):
    """
    A deep Multi-Layer Perceptron (MLP) model with a funnel-like architecture and Dropout layers
    to prevent overfitting. The model expands and then contracts the number of neurons before reaching the output.
    """

    def __init__(self, input_dim: int = 15, dropout_rate: float = 0.3):
        """
        Initializes the MLP model with a deep funnel architecture and Dropout layers.

        Parameters:
            input_dim (int): Number of input features (default is 15).
            dropout_rate (float): Dropout probability (default is 0.3).
        """
        super(FunnelTransactionMLPWithDropout, self).__init__()
        
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        # Funnel architecture with Dropout
        # Expanding phase
        self.fc1 = nn.Linear(self.input_dim, 30)  # From input_dim (15) to 30
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.dropout_rate)  # Dropout after first layer
        
        self.fc2 = nn.Linear(30, 60)  # From 30 to 60
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=self.dropout_rate)  # Dropout after second layer

        self.fc3 = nn.Linear(60, 90)  # From 60 to 90
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=self.dropout_rate)  # Dropout after third layer

        self.fc4 = nn.Linear(90, 120)  # From 90 to 120 (maximum expansion)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=self.dropout_rate)  # Dropout after fourth layer
        
        # Contracting phase
        self.fc5 = nn.Linear(120, 90)  # From 120 to 90
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=self.dropout_rate)  # Dropout during contraction

        self.fc6 = nn.Linear(90, 60)  # From 90 to 60
        self.relu6 = nn.ReLU()

        self.fc7 = nn.Linear(60, 30)  # From 60 to 30
        self.relu7 = nn.ReLU()

        self.fc8 = nn.Linear(30, 15)  # From 30 to 15
        self.relu8 = nn.ReLU()

        self.fc9 = nn.Linear(15, 10)  # From 15 to 10
        self.relu9 = nn.ReLU()

        self.fc10 = nn.Linear(10, 5)  # From 10 to 5
        self.relu10 = nn.ReLU()

        # Output layer
        self.fc11 = nn.Linear(5, 1)  # Final output layer with 1 neuron for binary classification
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
        # Expanding phase with Dropout
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  # Apply dropout
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)  # Apply dropout
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)  # Apply dropout
        
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)  # Apply dropout
        
        # Contracting phase with Dropout
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.dropout5(x)  # Apply dropout

        x = self.fc6(x)
        x = self.relu6(x)

        x = self.fc7(x)
        x = self.relu7(x)

        x = self.fc8(x)
        x = self.relu8(x)

        x = self.fc9(x)
        x = self.relu9(x)

        x = self.fc10(x)
        x = self.relu10(x)

        # Output layer
        x = self.fc11(x)
        output = self.sigmoid(x)
        
        return output