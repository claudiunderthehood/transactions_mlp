import os

import time
import datetime

import sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import numpy as np
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
from typing import Any, Callable, List, Tuple, Dict, Optional

from classification_modules.classificationClass import Classification
from deepLearning_modules.earlyStoppingClass import EarlyStopping
from deepLearning_modules.transactionsDatasetClass import TransactionDataset
from classification_modules.classificationClass import Classification

def compute_model_loss(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> float:
    """
    Evaluates the model on a given data loader and computes the average loss over all batches.

    Parameters:
        model (torch.nn.Module): The neural network model to be evaluated.
        data_loader (torch.utils.data.DataLoader): A DataLoader containing the batches of data to evaluate.
        loss_function (Callable): The loss function used to compute the error between predictions and targets.

    Returns:
        float: The average loss computed over all batches in the data loader.
    """
    
    # Set the model to evaluation mode
    model.eval()

    # List to store loss values for each batch
    losses_per_batch: list = []

    # No gradient calculation needed during evaluation
    with torch.no_grad():
        for features, targets in data_loader:
            # Get the model's predictions
            predictions: torch.nn.Module = model(features)
            
            # Calculate the loss between predictions and actual targets
            loss: torch.Tensor = loss_function(predictions.squeeze(), targets)
            
            # Append the loss to the list
            losses_per_batch.append(loss.item())
    
    # Calculate the mean loss over all batches
    avg_loss: float = float(np.mean(losses_per_batch))
    
    return avg_loss



def train_model(
    model: torch.nn.Module,
    trainer: torch.utils.data.DataLoader,
    tester: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_epochs: int
) -> tuple[List[float], List[float], float]:
    """
    Trains a PyTorch model and evaluates it after each epoch on a test set.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        trainer (torch.utils.data.DataLoader): DataLoader providing the training batches.
        tester (torch.utils.data.DataLoader): DataLoader providing the testing batches.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's weights.
        criterion (Callable): The loss function used to compute the loss between predictions and targets.
        n_epochs (int): The number of training epochs.

    Returns:
        tuple[List[float], List[float], float]: A tuple containing the list of training losses, the list of test losses, 
                                                and the total training execution time.
    """
    
    # Set model to training mode
    model.train()
    
    # To store epoch-wise train and test losses
    epochs_train_losses: List[float] = []
    epochs_test_losses: List[float] = []
    
    # Start the training time counter
    start_time: float = time.time()
    
    # Training loop over epochs
    for epoch in range(n_epochs):
        # Set model to training mode
        model.train()
        
        # To store batch-wise training losses
        train_loss: List[float] = []
        
        # Loop over each batch in the training set
        for x_batch, y_batch in trainer:
            # Zero the gradients from the previous step
            optimizer.zero_grad()
            
            # Forward pass: make predictions
            y_pred: torch.nn.Module = model(x_batch)
            
            # Compute loss between predictions and actual labels
            loss: torch.Tensor = criterion(y_pred.squeeze(), y_batch)
            
            # Backward pass: compute the gradients
            loss.backward()
            
            # Update the model's weights
            optimizer.step()
            
            # Store the current batch loss
            train_loss.append(loss.item())
        
        # Compute the average loss for the epoch
        avg_train_loss: List[float] = np.mean(train_loss)
        epochs_train_losses.append(avg_train_loss)
        print(f'Epoch {epoch + 1}/{n_epochs}: train loss: {avg_train_loss}')
        
        # Evaluate the model on the test set using the compute_model_loss function
        val_loss: float = compute_model_loss(model, tester, criterion)
        epochs_test_losses.append(val_loss)
        print(f'Test loss: {val_loss}\n')

    # Compute total training execution time
    training_execution_time: float = time.time() - start_time
    
    return epochs_train_losses, epochs_test_losses, training_execution_time


def top_k_accuracy(preds, labels, k=100):
    """ 
    Compute top-k accuracy. If there are fewer than k samples, return the accuracy for all available samples.
    """
    num_samples = min(k, len(preds))  # Ensure we don't go out of bounds
    top_k_indices = np.argsort(preds)[-num_samples:]  # Get indices of the top k predictions
    top_k_labels = labels[top_k_indices]  # Get the true labels for those top k predictions
    top_k_hits = np.sum(top_k_labels == 1)  # Count how many of the top k are actual fraud cases
    return top_k_hits / num_samples

def train_model_earlystopping_metrics(
    model: torch.nn.Module,
    trainer: torch.utils.data.DataLoader,
    validator: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_epochs: int = 100,
    apply_early_stopping: bool = True,
    patience: int = 2,
    verbose: bool = False,
    model_name: str = 'model.pth',
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, any]:
    """
    Trains a PyTorch model and evaluates it after training on the validation set.
    Optionally applies early stopping and a learning rate scheduler.
    
    Additionally, it calculates AUC-ROC, Average Precision, and Top K accuracy (K=100).
    """

    # Setting the model in training mode
    model.train()

    # If early stopping is applied, initialize the EarlyStopping object
    if apply_early_stopping:
        early_stopping = EarlyStopping(verbose=verbose, patience=patience, checkpoint_path=model_name)
    
    # Lists to store training and validation losses across all epochs
    all_train_losses: List[float] = []
    all_valid_losses: List[float] = []
    
    # Start timing the training process
    start_time = time.time()
    
    # Main training loop
    for epoch in range(max_epochs):
        model.train()  # Set the model to training mode
        train_loss: List[float] = []
        
        # Training over batches
        for x_batch, y_batch in trainer:
            optimizer.zero_grad()  # Clear gradients from previous step
            y_pred = model(x_batch)  # Forward pass
            loss = criterion(y_pred.squeeze(), y_batch)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            train_loss.append(loss.item())  # Track batch loss
        
        # Compute the average training loss for this epoch
        avg_train_loss = np.mean(train_loss)
        all_train_losses.append(avg_train_loss)
        
        if verbose:
            print(f'Epoch {epoch + 1}/{max_epochs}: train loss: {avg_train_loss}')
        
        # Validation phase: evaluate model on validation set and compute metrics
        model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_val, y_val in validator:
                y_pred = model(x_val).squeeze()
                loss = criterion(y_pred, y_val)
                total_loss += loss.item() * len(y_val)
                total_samples += len(y_val)
                all_preds.append(y_pred.detach().cpu().numpy())  # Ensure we collect raw probabilities
                all_labels.append(y_val.detach().cpu().numpy())
        
        # Average validation loss
        avg_valid_loss = total_loss / total_samples
        all_valid_losses.append(avg_valid_loss)
        
        # Early stopping based on validation loss
        if apply_early_stopping:
            if not early_stopping.should_continue(avg_valid_loss, model):
                if verbose:
                    print("Early stopping triggered.")
                break

        # Step the learning rate scheduler based on validation loss (for ReduceLROnPlateau)
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_valid_loss)
            else:
                scheduler.step()

    # Calculate the total training time
    training_execution_time = time.time() - start_time
    
    # Final Metrics Calculation after training is complete
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Convert probabilities to binary predictions (for binary classification)
    all_preds_binary = (all_preds >= 0.5).astype(int)

    # Compute final metrics using sklearn
    accuracy = metrics.accuracy_score(all_labels, all_preds_binary)
    precision = metrics.precision_score(all_labels, all_preds_binary)
    recall = metrics.recall_score(all_labels, all_preds_binary)
    f1 = metrics.f1_score(all_labels, all_preds_binary)

    # Debugging print for predictions
    print(f"Debugging - Predicted Probabilities (first 10): {all_preds[:10]}")
    print(f"Debugging - True Labels (first 10): {all_labels[:10]}")

    # AUC-ROC and Average Precision
    try:
        auc_roc = metrics.roc_auc_score(all_labels, all_preds)
        avg_precision = metrics.average_precision_score(all_labels, all_preds)
    except ValueError as e:
        print(f"Error calculating AUC-ROC or Average Precision: {e}")
        auc_roc = None
        avg_precision = None

    # Top K Accuracy (K=100)
    top_k_acc = top_k_accuracy(all_preds, all_labels, k=100) if len(all_preds) >= 100 else None
    
    # Print the final metrics
    print(f"Final Validation Metrics:\n"
          f"Accuracy: {accuracy:.4f}\n"
          f"Precision: {precision:.4f}\n"
          f"Recall: {recall:.4f}\n"
          f"F1 Score: {f1:.4f}")
    
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.4f}")
    if avg_precision is not None:
        print(f"Average Precision: {avg_precision:.4f}")
    if top_k_acc is not None:
        print(f"Top K Accuracy (K=100): {top_k_acc:.4f}")
    else:
        print("Top K Accuracy could not be computed due to insufficient data.")

    # Return results: model, time, losses, and final metrics
    return {
        'model': model,
        'training_time': training_execution_time,
        'train_losses': all_train_losses,
        'valid_losses': all_valid_losses,
        'final_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'average_precision': avg_precision,
            'top_k_accuracy': top_k_acc
        }
    }

def prepare_data_loaders(
    train_df: pd.DataFrame, 
    valid_df: pd.DataFrame, 
    input_features: list, 
    output_feature: str, 
    batch_size: int = 64,
    device: str = 'cuda'
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepares PyTorch DataLoader generators for training and validation datasets.

    Parameters:
        train_df (pd.DataFrame): The training dataframe containing input and output features.
        valid_df (pd.DataFrame): The validation dataframe containing input and output features.
        input_features (list): A list of column names representing input features.
        output_feature (str): The column name representing the target feature (label).
        batch_size (int): The batch size to use for data loading. Default is 64.
        device (str): The device used for training.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
        A tuple containing the DataLoader for the training set and the validation set.
    """
    
    # Convert DataFrame columns to FloatTensors for both features and labels
    x_train = torch.FloatTensor(train_df[input_features].values)
    x_valid = torch.FloatTensor(valid_df[input_features].values)
    y_train = torch.FloatTensor(train_df[output_feature].values)
    y_valid = torch.FloatTensor(valid_df[output_feature].values)

    # Parameters for DataLoader for training and validation sets
    train_loader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 0
    }
    valid_loader_params = {
        'batch_size': batch_size,
        'shuffle': False,  # No need to shuffle validation set
        'num_workers': 0
    }
    
    # Prepare datasets using the custom FraudDataset class
    training_set = TransactionDataset(x_train, y_train, device)
    validation_set = TransactionDataset(x_valid, y_valid, device)
    
    # Create DataLoader generators
    trainer = torch.utils.data.DataLoader(training_set, **train_loader_params)
    validator = torch.utils.data.DataLoader(validation_set, **valid_loader_params)
    
    return trainer, validator