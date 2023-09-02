"""
"""
import math
import torch
import torch.optim as optim
import time
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from tqdm.auto import tqdm

def train_step(
    model: torch.nn.Module,
    train_loader: torch.utils.data.Dataset,
    device: torch.device,
    loss_fn: torch.nn.Module,
    optimizer:optim.Optimizer,
    epoch:int,
    batch_size:int
    ):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        train_loader (torch.utils.data.Dataset): _description_
        device (torch.device): _description_
        loss_fn (torch.nn.Module): _description_
        optimizer (optim.Optimizer): _description_
        epoch (int): _description_
        batch_size (int): _description_
    """
    # training
    avg_loss, running_mae, running_mse = 0.0, 0.0, 0.0
    start_time = time.time()
    for batch_no, (x, y_true) in tqdm(enumerate(train_loader), total=len(train_loader)):

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
        # OBTAIN RMSE ON BATCH
        # https://stackoverflow.com/questions/63391113/pytorch-calculate-mse-and-mae
        error = torch.abs(y_pred - y_true).sum().data
        squared_error = ((y_pred - y_true)*(y_pred - y_true)).sum().data
        running_mae += error
        running_mse += squared_error

        mlflow.log_metric(
            "Batch RMSE",
            math.sqrt(running_mse.item() / batch_size),
            # step=math.ceil(epoch * len(train_loader) / batch_size) + batch_no,
        )

        mlflow.log_metric(
            "Loss",
            loss.item(),
            # step=math.ceil(epoch * len(train_loader) / batch_size) + batch_no,
        )
        avg_loss += loss.item()

    total_time = time.time() - start_time
    avg_loss /= len(train_loader)
    rmse = math.sqrt(running_mse/len(train_loader))
    
    mlflow.log_metric("RMSE", rmse, step=epoch)
    mlflow.log_metric("Average Loss", avg_loss, step=epoch)
    mlflow.log_metric("Time Taken", total_time, step=epoch)
