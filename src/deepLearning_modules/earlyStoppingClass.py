import os

import torch

MODEL_PATH: str = './models/best/'


class EarlyStopping:
    """
    Implements early stopping to halt training when the validation loss stops improving 
    after a certain number of epochs. It can also save the best model as a checkpoint.
    
    Attributes:
        patience (int): Number of epochs to wait for an improvement in the score before stopping.
        verbose (bool): Whether to print updates when the score improves or when patience is exceeded.
        counter (int): Counts how many consecutive epochs without improvement have occurred.
        best_score (float): The best observed score (typically validation loss).
        checkpoint_path (str): The path to save the model checkpoint.
    """
    
    def __init__(self, patience: int = 2, verbose: bool = False, checkpoint_path: str = None):
        """
        Initializes the EarlyStopping object with patience value, verbosity, and checkpoint path.

        Parameters:
            patience (int): Number of epochs to wait before stopping if no improvement. Default is 2.
            verbose (bool): Whether to print updates. Default is False.
            checkpoint_path (str): The file path where the best model should be saved. Default is None.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = float('inf')  # Represents the lowest score (best validation loss)
        self.checkpoint_path = checkpoint_path  # Path to save model checkpoint
        self.best_model_state = None  # Store the best model's state_dict

    def should_continue(self, current_score: float, model: torch.nn.Module) -> bool:
        """
        Checks if training should continue based on the current score and the best score seen so far.
        Saves the model if the current score is the best.

        Parameters:
            current_score (float): The current validation score (e.g., validation loss).
            model (torch.nn.Module): The model to save if the current score is the best.

        Returns:
            bool: Returns True if training should continue, False if it should stop based on patience.
        """
        if current_score < self.best_score:
            self.best_score = current_score
            self.counter = 0
            if self.verbose:
                print(f"New best score: {current_score}")

            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
                filename_output: str = MODEL_PATH+self.checkpoint_path
                torch.save(model.state_dict(), filename_output)
                if self.verbose:
                    print(f"Model checkpoint saved at: {filename_output}")
            else:
                filename_output: str = MODEL_PATH+self.checkpoint_path
                torch.save(model.state_dict(), filename_output)
                if self.verbose:
                    print(f"Model checkpoint saved at: {filename_output}")

        else:
            self.counter += 1
            if self.verbose:
                print(f"{self.counter} iterations since the best score.")
        
        # Continue training if the counter has not exceeded the patience threshold
        return self.counter <= self.patience
