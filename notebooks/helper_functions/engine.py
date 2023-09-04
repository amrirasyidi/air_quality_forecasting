"""
A module contains necessary classes and functions
for training LSTNet Model
"""
import logging
# import math
import numpy as np
import torch
import torch.optim as optim
import time
import mlflow
import mlflow.pytorch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

class EarlyStopper:
    """OOP implementation of early stopping
    
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    def __init__(self, patience=1, min_delta=0, doi=4):
        self.patience = patience
        self.min_delta = min_delta
        self.doi = doi # degree of insignificancy
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, doi):
        validation_loss = round(validation_loss, doi)
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# https://aahansingh.com/experimentation-tracking-with-mlflow-part-1
def train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    ) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Args:
        model (torch.nn.Module): A PyTorch model to be trained.
        train_loader (torch.utils.data.Dataset): A DataLoader instance for the model to be trained on.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        optimizer (optim.Optimizer): A PyTorch optimizer to help minimize the loss function.

    Returns:
        Tuple[float, float, float]: A tuple of training loss and training accuracy metrics.
        In the form (epoch_loss, epoch_rmse, epoch_total_time).
    """
    # Put model in train mode
    model.train()

    # Setup metrics initial values
    epoch_loss=0.0
    # running_mse=0.0
    start_time = time.time()

    for batch_no, (x, y_true) in enumerate(train_dataloader):
        x, y_true = x.to(device), y_true.to(device)

        # CLEAR GRADIENT TO PREVENT ACCUMULATION
        optimizer.zero_grad()
        # COMPUTE OUTPUT
        y_pred = model(x)
        # COMPUTE LOSS
        loss = loss_fn(y_pred, y_true)
        # FIND GRADIENTS
        loss.backward()
        # UPDATE WEIGHTS
        optimizer.step()

        # # Calculate RMSE for this batch
        # mse = torch.mean((y_pred - y_true) ** 2)
        # running_mse += mse.item()

        # Update epoch loss
        epoch_loss += loss.item()

    # Calculate Loss for the entire epoch
    epoch_loss /= len(train_dataloader)
    # # Calculate RMSE for the entire epoch
    # epoch_rmse = math.sqrt(running_mse / len(train_dataloader))
    # Calculate time taken for 1 epoch
    epoch_total_time = time.time() - start_time

    # return epoch_loss, epoch_rmse, epoch_total_time
    return epoch_loss, epoch_total_time

# model: torch.nn.Module,
# train_dataloader: torch.utils.data.DataLoader,
# loss_fn: torch.nn.Module,
# optimizer: optim.Optimizer,
# device: torch.device,

def test_step(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
    ) -> Tuple[float, float,]:
    """Tests a PyTorch model for a single epoch.

    Args:
        model (torch.nn.Module): A PyTorch model to be tested.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        loss_fn (torch.nn.Module): A PyTorch loss function to calculate loss on the test data.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        Tuple[float, float, float]: A tuple of testing loss and testing accuracy metrics.
        In the form (epoch_loss, epoch_rmse, epoch_total_time).
    """
    # Put model in eval mode
    model.eval()

    # Setup metrics initial values
    epoch_loss=0.0
    # running_mse=0.0
    start_time = time.time()

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch_no, (x, y_true) in enumerate(test_dataloader):
            # Send data to target device
            x, y_true = x.to(device), y_true.to(device)

            # 1. Forward pass
            y_pred = model(x)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y_true)
            epoch_loss += loss.item()

            # # Calculate RMSE for this batch
            # mse = torch.mean((y_pred - y_true) ** 2)
            # running_mse += mse.item()

    # Calculate Loss for the entire epoch
    epoch_loss /= len(test_dataloader)
    # # Calculate RMSE for the entire epoch
    # epoch_rmse = math.sqrt(running_mse / len(test_dataloader))
    # Calculate time taken for 1 epoch
    epoch_total_time = time.time() - start_time

    # return epoch_loss, epoch_rmse, epoch_total_time
    return epoch_loss, epoch_total_time

def train(
    model: torch.nn.Module,
    model_name: str,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    patience: int
    ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, logs, and stores evaluation metrics throughout.

    Args:
        model (torch.nn.Module): A PyTorch model to be trained and tested.
        model_name (str): Name of the model for the saving purpose
        train_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        loss_fn (torch.nn.Module): A PyTorch loss function to calculate loss on both datasets.
        epochs (int): An integer indicating how many epochs to train for.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").
        patience (int): For earlystopping implementation. Use patience = epoch to make no earlystopping

    Returns:
        Dict[str, List]: A dictionary of training and testing loss as well as training and
        testing calculation time metrics. Each metric has a value in a list for
        each epoch.
        In the form: {
            "train_loss": [],
            "train_time": [],
            "test_loss": [],
            "test_time": []
            }
        For example if training for epochs=2:
        {"train_loss": [2.0616, 1.0537],
        "train_time": [0.3945, 0.3945],
        "test_loss": [0.2345, 1.3945],
        "test_time": [1.0212, 1.4357]}
    """
    # Create empty results dictionary
    results = {
        "train_loss": [],
        # "train_rmse": [],
        "train_time": [],
        "test_loss": [],
        # "test_rmse": [],
        "test_time": []
    }

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    for epoch in tqdm(range(epochs)):
        train_epoch_loss, train_epoch_total_time = train_step(
            model=model,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            )

        test_epoch_loss, test_epoch_total_time = test_step(
            model=model,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            )

        # Logging statement using lazy formatting with %
        train_log_message = "Model Config: %s | Epoch: %d | train_loss: %.4f | train_time: %.2f"
        logging.info(train_log_message, model_name, epoch+1, train_epoch_loss, train_epoch_total_time)

        test_log_message = "Model Config: %s | Epoch: %d | test_loss: %.4f | test_time: %.2f \n"
        logging.info(test_log_message, model_name, epoch+1, test_epoch_loss, test_epoch_total_time)

        # Update results dictionary
        results["train_loss"].append(train_epoch_loss)
        # results["train_rmse"].append(train_epoch_rmse)
        results["train_time"].append(train_epoch_total_time)
        results["test_loss"].append(test_epoch_loss)
        # results["test_rmse"].append(test_epoch_rmse)
        results["test_time"].append(test_epoch_total_time)

        mlflow.log_metric("Train Loss", train_epoch_loss, step=epoch)
        # mlflow.log_metric("Train RMSE", train_epoch_rmse, step=epoch)
        mlflow.log_metric("Train Time Taken", train_epoch_total_time, step=epoch)

        mlflow.log_metric("Test Loss", test_epoch_loss, step=epoch)
        # mlflow.log_metric("Test RMSE", test_epoch_rmse, step=epoch)
        mlflow.log_metric("Test Time Taken", test_epoch_total_time, step=epoch)

        if early_stopper.early_stop(train_epoch_loss, doi=4):
            early_stopper_log_message = "Stopped at Epoch %d after %d epochs with insignificant/no improvement \n"
            logging.info(early_stopper_log_message, epoch+1, patience)
            break

    # Return the filled results at the end of the epochs
    return results
