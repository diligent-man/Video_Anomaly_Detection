import torch
import mlflow
import torcheval
import torchvision

from tqdm import tqdm
from typing import List
from torcheval import metrics
from torchvision.transforms import v2, Compose


__all__ = ["get_dataset", "train", "val"]


def get_dataset(root: str = "./MNIST"):
    transform = Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    train = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=False,
        transform=transform
    )

    test = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=False,
        transform=transform
    )
    return train, test


def train(epochs: int,
          curr_epoch: int,
          model: torch.nn.Module,
          loss: torch.nn.Module,
          metric_lst: List[torcheval.metrics.Metric],
          optimizer: torch.optim.Optimizer,
          DataLoader: DataLoader,
          device: str
          ):
    """
    Simple training loop
    """
    model.train()
    curr_step = 0 + curr_epoch * len(DataLoader)

    for i, (X, y) in tqdm(enumerate(DataLoader), initial=curr_step, total=len(DataLoader) * epochs, desc="Training"):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        curr_loss = loss(input=pred, target=y)
        _ = [metric.update(input=pred, target=y) for metric in metric_lst]

        # Backpropagation.
        curr_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        curr_loss = curr_loss.item()
        # Only applicable to metric that returns scalar
        curr_metrics: List[float] = [metric.compute().item() for metric in metric_lst]

        curr_step += 1
        print(curr_loss, curr_metrics)
        mlflow.log_metric("train_loss", curr_loss, step=curr_step)

        for i in range(len(metric_lst)):
            mlflow.log_metric(f"train_{metric_lst[i].__class__.__name__}", curr_metrics[i], step=curr_step)


def val(model: torch.nn.Module,
        loss: torch.nn.Module,
        metric_lst: List[torcheval.metrics.Metric],
        epoch: int,
        DataLoader: DataLoader,
        device: str
        ):
    """
    Validate model per epoch.
    """
    model.eval()
    num_batches = len(DataLoader)
    eval_loss, eval_accuracy = 0., 0.

    with torch.no_grad():
        for X, y in tqdm(DataLoader, total=num_batches, desc="Validating"):
            X, y = X.to(device), y.to(device)

            pred = model(X)

            eval_loss += loss(input=pred, target=y).item()
            _ = [metric.update(input=pred, target=y) for metric in metric_lst]

    eval_loss /= num_batches
    curr_metrics: List[float] = [metric.compute().item() for metric in metric_lst]

    mlflow.log_metric("val_loss", eval_loss, step=epoch)

    for i in range(len(metric_lst)):
        mlflow.log_metric(f"val_{metric_lst[i].__class__.__name__}", curr_metrics[i], step=epoch)
