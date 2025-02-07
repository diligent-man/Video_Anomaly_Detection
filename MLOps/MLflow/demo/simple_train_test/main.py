import os
import copy
import shutil
import datetime

from typing import List


import torch
import mlflow
import torcheval
import numpy as np
import pandas as pd

from tqdm import tqdm
from torchinfo import summary

from mlflow.entities import ViewType
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from torcheval.metrics.classification import MulticlassAccuracy, MulticlassF1Score


from MLOps.MLflow.demo.simple_train_test.train import train, val, get_dataset
from MLOps.MLflow.demo.simple_train_test.ImageClassifier import ImageClassifier

# Auth by env var for mlflow
os.environ["MLFLOW_TRACKING_USERNAME"] = "root"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "Root123!"

mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment("pytorch_demo")
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_with_mlflow(run_name: str,
                      epochs: int,
                      lr: float,
                      batch_size: int,
                      model: torch.nn.Module,
                      loss: torch.nn.Module,
                      metric_lst: List[torcheval.metrics.Metric],
                      optimizer: torch.optim.Optimizer,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader,
                      signature: ModelSignature
                      ) -> None:
    # Note: Signatures and Input Examples are not set
    # Ref: https://mlflow.org/docs/latest/model/signatures.html
    with mlflow.start_run(log_system_metrics=True,
                          run_name=run_name
                          ):
        params = {
            "Epochs": epochs,
            "Lr": lr,
            "Batch size": batch_size,
            "Loss": loss.__class__.__name__,
            "Optimizer": optimizer.__class__.__name__,
            **{f" Metric {i}": metric_lst[i].__class__.__name__ for i in range(len(metric_lst))}
        }

        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(epochs, t, model, loss, metric_lst, optimizer, train_loader, device)
            val(model, loss, metric_lst, t, val_loader, device)

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, artifact_path="model", signature=signature)


def test_with_mlflow(test_loader: torch.utils.data.DataLoader,
                     signature: ModelSignature
                     ) -> None:
    """
    Search latest run and test model with it
    """
    def convert_tz(df: pd.DataFrame) -> pd.DataFrame:
        date_cols = df.select_dtypes(include=['datetime64[ns, UTC]']).columns

        for col in date_cols:
            df[col] = df[col].dt.tz_convert('Asia/Ho_Chi_Minh')
            df[col] = df[col].apply(lambda row: datetime.datetime.strftime(row,"%Y-%m-%d %H:%M:%S"))
        return df

    experiment_id: str = mlflow.search_experiments(ViewType.ACTIVE_ONLY, 1,"name = 'pytorch_demo'")[0].experiment_id
    df: pd.DataFrame = mlflow.search_runs([experiment_id], order_by=["start_time DESC"], max_results=1)
    df = convert_tz(df)
    df.to_excel("./runs_info.xlsx", index=False)

    model_uri: str = f"runs:/{df['run_id'][0]}/model"
    model: torch.nn.Module = mlflow.pytorch.load_model(model_uri)
    model.eval()
    print(model)

    metric_lst: List[torcheval.metrics.Metric] = [
        MulticlassAccuracy(num_classes=10, device=torch.device(device)),
        MulticlassF1Score(num_classes=10, device=torch.device(device)),
    ]

    with torch.no_grad():
        for X, y in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _ = [metric.update(input=pred, target=y) for metric in metric_lst]

    metric_vals: List[float] = [metric.compute().item() for metric in metric_lst]

    print("--------------- Testing result --------------")
    for i in range(len(metric_lst)):
        print(f"{metric_lst[i].__class__.__name__}: {metric_vals[i]}")


def main() -> None:
    epochs = 1
    lr = 1e-3
    batch_size = 1024

    train_data, test_data = get_dataset(root="/home/trong/Downloads/Dataset/MNIST/raw")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=False)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, drop_last=False)
    test_loader = copy.deepcopy(val_loader)

    # Manually specify. Current caveat: not directly accept tensor datatype
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1, 28, 28), "input")])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10), "output")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    for i in range(4):
        run_name = f"run {i}"
        model = ImageClassifier().to(device)
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        metric_lst = [
            MulticlassAccuracy(num_classes=10, device=torch.device(device)),
            MulticlassF1Score(num_classes=10, device=torch.device(device))
        ]
        train_with_mlflow(run_name, epochs, lr, batch_size,
                          model, loss, metric_lst, optimizer,
                          train_loader, val_loader, signature
                          )
        lr *= .1

    test_with_mlflow(test_loader, signature)

    # if os.path.exists("./MNIST"):
    #     shutil.rmtree("./MNIST")
    return None


if __name__ == '__main__':
    main()
