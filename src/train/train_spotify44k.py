from pathlib import Path

import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from datasets.data_pipeline import make_Spotify44k_loaders


def train_one_epoch( 
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion,
        optimizer,
        metrics: dict[str, MetricCollection],
        train_loss_metric: MeanMetric,
        validation_loss_metric: MeanMetric,
        device: torch.device,
    ):


    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device) 
        optimizer.zero_grad()
        logits = model(inputs)
        loss, _= criterion(logits, targets)
        loss.backward()
        optimizer.step()
        metrics['train'].update(logits, targets)
        train_loss_metric.update(loss.item())

    model.eval()
    with torch.no_grad(): 
        for input, targets in val_loader:
            input, targets = input.to(device), targets.to(device)
            logits = model(input)
            loss = criterion(logits, targets)
            metrics['val'].update(logits)
            validation_loss_metric.update(loss.item())


def format_big_number(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000:
            # Format to 1 decimal place if it's a decimal, otherwise keep it clean
            return f"{num:.1f}".rstrip('0').rstrip('.') + unit if unit else str(num)
        num /= 1000.0
    return f"{num:.1f}Q"

def fit_model(
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metrics: dict[str, MetricCollection],
        epochs: int,
        log_dir: Path,
    ) -> pl.DataFrame:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG_DIR = log_dir / f"{model.__class__.__name__}{format_big_number(num_params)}_training_data.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_loss_metric = MeanMetric().to(device)
    val_loss_metric = MeanMetric().to(device)
    history = []

    for epoch in range(epochs):
        train_one_epoch(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            metrics,
            train_loss_metric,
            val_loss_metric,
            device,
        )
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss_metric.compute().item(),
            "val_loss": val_loss_metric.compute().item(),
        }

        # record epoch metric results in polars frame

        for metric_name, value in metrics['train'].items():
            epoch_data[metric_name] = value.compute().item()
         
        for metric_name, value in metrics['val'].items():
            epoch_data[metric_name] = value.compute().item()

        history.append(epoch_data)

        # reset metrics in preparation for next epoch
        metrics['train'].reset()
        metrics['val'].reset()
        train_loss_metric.reset()
        val_loss_metric.reset()
    
    # return metrics to the caller
    df_history = pl.DataFrame(history)
    df_history.write_csv(LOG_DIR)

    return df_history

def test_model(
        model: nn.Module,
        test_loader: DataLoader,
        metrics: dict[str, MetricCollection],
        log_dir: Path,
    ) -> pl.DataFrame:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG_DIR = log_dir / f"{model.__class__.__name__}{format_big_number(num_params)}_testing_data.csv"

    device = "gpu" if torch.cuda.is_available() else "cpu"
    model.eval()  
    criterion = nn.CrossEntropyLoss()
    test_loss_metric = MeanMetric().to(device)

    model.eval()
    with torch.no_grad(): 
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(input)
            loss, _= criterion(logits, targets)
            metrics['test'].update(logits)
            test_loss_metric.update(loss.item())

    test_loss = test_loss_metric.compute().item()

    test_data = {
        "test_loss": test_loss,
    }

    test_results = metrics['test'].compute()
    for metric_name, value in test_results.items():
        test_data[metric_name] = value.item()

    
    # return metrics to the caller
    df_history = pl.DataFrame(test_data)
    df_history.write_csv(LOG_DIR)

    return df_history

def run_train_test(
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_epochs: int,
        log_dir: str,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:

    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, test_loader = make_Spotify44k_loaders(shuffle=True)
    num_classes = len(train_loader.dataset.classes)

    base_metrics = MetricCollection({
        "accuracy":         MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "top3_acc":         MulticlassAccuracy(num_classes=num_classes, topk=3, average="micro"),
        "precision":        MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall":           MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1":               MulticlassF1Score(num_classes=num_classes, average="macro"),
    })
    
    metrics = {
        "train": base_metrics.clone("train_").to(device),
        "val":   base_metrics.clone("val_").to(device),
        "test":  base_metrics.clone("test_").to(device),
    }

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    

    train_results = fit_model(
        model, 
        optimizer,
        train_loader,
        val_loader,
        metrics,
        train_epochs,
        log_path,
    )

    test_results = test_model(
        model,
        test_loader,
        metrics,
        log_path,
    )

    return train_results, test_results
