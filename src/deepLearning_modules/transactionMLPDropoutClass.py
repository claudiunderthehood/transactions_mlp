import torch
from torch import nn

class TransactionMLPWithDropout(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model with Batch Normalization and Dropout for binary classification tasks.

    This model includes one hidden layer with ReLU activation, followed by batch normalization and dropout,
    and an output layer with a sigmoid activation function for binary classification.
    
    Attributes:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of neurons in the hidden layer.
        dropout_prob (float): The probability of an element to be zeroed during dropout.
        fc1 (nn.Linear): Fully connected layer mapping input features to the hidden layer.
        bn1 (nn.BatchNorm1d): Batch Normalization layer applied after the first fully connected layer.
        relu (nn.ReLU): ReLU activation function applied after the first layer.
        fc2 (nn.Linear): Fully connected layer mapping hidden layer to the output.
        sigmoid (nn.Sigmoid): Sigmoid activation function for binary classification.
        dropout (nn.Dropout): Dropout layer applied after the hidden layer.
    """

    def __init__(self, input_dim: int, dropout: float):
        """
        Initializes the MLP model with one hidden layer, batch normalization, dropout, and a sigmoid output layer.

        Parameters:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of neurons in the hidden layer.
            dropout_prob (float): The dropout probability to be applied after the hidden layer.
        """
        super(TransactionMLPWithDropout, self).__init__()
        
        # Model architecture
        self.input_dim = input_dim
        self.dropout_proba = dropout
        
        # Input to hidden layer
        self.fc1: nn.Linear = nn.Linear(self.input_dim, 10)
        self.relu: nn.ReLU = nn.ReLU()
        
        # Hidden to output layer
        self.fc2: nn.Linear = nn.Linear(10, 5)
        self.relu2: nn.ReLU = nn.ReLU()

        self.fc3: nn.Linear = nn.Linear(5, 4)
        self.relu3: nn.ReLU = nn.ReLU()

        self.fc4: nn.Linear = nn.Linear(4, 3)
        self.relu4: nn.ReLU = nn.ReLU()

        self.fc5: nn.Linear = nn.Linear(3, 2)
        self.relu5: nn.ReLU = nn.ReLU()

        self.fc6: nn.Linear = nn.Linear(2, 1)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

        self.dropout = torch.nn.Dropout(self.dropout_proba)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the MLP model.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 1), containing the predicted
                          probabilities for binary classification.
        """
        # Pass through the first layer
        hidden: torch.Tensor = self.fc1(x)
        activated_hidden: torch.Tensor = self.relu(hidden)
        activated_hidden= self.dropout(activated_hidden)

        hidden1: torch.Tensor = self.fc2(activated_hidden)
        activated_hidden1: torch.Tensor = self.relu2(hidden1)
        activated_hidden1 = self.dropout(activated_hidden1)

        hidden2: torch.Tensor = self.fc3(activated_hidden1)
        activated_hidden2: torch.Tensor = self.relu3(hidden2)
        activated_hidden2 = self.dropout(activated_hidden2)

        hidden3: torch.Tensor = self.fc4(activated_hidden2)
        activated_hidden3: torch.Tensor = self.relu4(hidden3)
        activated_hidden3 = self.dropout(activated_hidden3)

        hidden4: torch.Tensor = self.fc5(activated_hidden3)
        activated_hidden4: torch.Tensor = self.relu5(hidden4)
        activated_hidden4 = self.dropout(activated_hidden4)
        
        output: torch.Tensor = self.fc6(activated_hidden4)
        
        probability: torch.Tensor = self.sigmoid(output)
        
        return probability
