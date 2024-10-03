import torch
from torch import nn

class TransactionMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for transaction classification tasks,
    specifically designed for binary classification problems.
    
    Attributes:
        input_dim (int): Size of the input layer.
        hidden_dim (int): Size of the hidden layer.
        fc1 (torch.nn.Linear): Fully connected layer mapping input to hidden layer.
        relu (torch.nn.ReLU): Activation function applied after the first layer.
        fc2 (torch.nn.Linear): Fully connected layer mapping hidden layer to output.
        sigmoid (torch.nn.Sigmoid): Sigmoid activation function for binary classification.
    """

    def __init__(self, input_dim: int):
        """
        Initializes the MLP model with one hidden layer and a sigmoid output layer.

        Parameters:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of neurons in the hidden layer.
        """
        super(TransactionMLP, self).__init__()
        
        self.input_dim = input_dim
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the MLP model.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 1), containing 
                          the predicted probabilities for binary classification.
        """
        hidden: torch.Tensor = self.fc1(x)
        activated_hidden: torch.Tensor = self.relu(hidden)
        hidden1: torch.Tensor = self.fc2(activated_hidden)
        activated_hidden1: torch.Tensor = self.relu2(hidden1)
        hidden2: torch.Tensor = self.fc3(activated_hidden1)
        activated_hidden2: torch.Tensor = self.relu3(hidden2)
        hidden3: torch.Tensor = self.fc4(activated_hidden2)
        activated_hidden3: torch.Tensor = self.relu4(hidden3)
        hidden4: torch.Tensor = self.fc5(activated_hidden3)
        activated_hidden4: torch.Tensor = self.relu5(hidden4)
        output: torch.Tensor = self.fc6(activated_hidden4)
        
        probability: torch.Tensor = self.sigmoid(output)
        
        return probability
