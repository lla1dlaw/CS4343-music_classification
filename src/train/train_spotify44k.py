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


def _train_one_epoch( 
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

    model.train()


    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device) 
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        metrics['train'].update(logits, targets)
        train_loss_metric.update(loss.item())

    model.eval()
    with torch.no_grad(): 
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            metrics['val'].update(logits, targets)
            validation_loss_metric.update(loss.item())
    


def _format_big_number(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000:
            # Format to 1 decimal place if it's a decimal, otherwise keep it clean
            return f"{num:.1f}".rstrip('0').rstrip('.') + unit if unit else str(num)
        num /= 1000.0
    return f"{num:.1f}Q"

def _fit_model(
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metrics: dict[str, MetricCollection],
        epochs: int,
        log_dir: Path,
    ) -> pl.DataFrame:

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG_DIR = log_dir / f"{model.__class__.__name__}{_format_big_number(num_params)}_training_data.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_loss_metric = MeanMetric().to(device)
    val_loss_metric = MeanMetric().to(device)
    history = []

    for epoch in range(epochs):
        _train_one_epoch(
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

        print(
            f"Epoch {epoch+1:03d}/{epochs:03d} | "
            f"Train Loss: {epoch_data['train_loss']:.4f} | "
            f"Val Loss: {epoch_data['val_loss']:.4f} | "
            f"Train Acc: {epoch_data['train_accuracy']:.4f} | "
            f"Val Acc: {epoch_data['val_accuracy']:.4f} | "
            f"Train F1: {epoch_data['train_f1']:.4f} | "
            f"Val F1: {epoch_data['val_f1']:.4f}"
        )

        # reset metrics in preparation for next epoch
        metrics['train'].reset()
        metrics['val'].reset()
        train_loss_metric.reset()
        val_loss_metric.reset()
    
    # return metrics to the caller
    df_history = pl.DataFrame(history)
    df_history.write_csv(LOG_DIR)

    return df_history

def _test_model(
        model: nn.Module,
        test_loader: DataLoader,
        metrics: dict[str, MetricCollection],
        log_dir: Path,
    ) -> pl.DataFrame:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG_DIR = log_dir / f"{model.__class__.__name__}{_format_big_number(num_params)}_testing_data.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()  
    criterion = nn.CrossEntropyLoss()
    test_loss_metric = MeanMetric().to(device)

    model.eval()
    with torch.no_grad(): 
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            metrics['test'].update(logits, targets)
            test_loss_metric.update(loss.item())

    test_loss = test_loss_metric.compute().item()

    test_data = {
        "test_loss": test_loss,
    }

    test_results = metrics['test'].compute()
    for metric_name, value in test_results.items():
        test_data[metric_name] = value.item()

    print(
        f"Test Results | "
        f"Loss: {test_data['test_loss']:.4f} | "
        f"Acc: {test_data['test_accuracy']:.4f} | "
        f"F1: {test_data['test_f1']:.4f}"
    )
    
    # return metrics to the caller
    df_history = pl.DataFrame([test_data])
    df_history.write_csv(LOG_DIR)

    return df_history

def run_train_test(
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_epochs: int,
        log_dir: str,
        dummy_dataloader: DataLoader | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:

    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if dummy_dataloader is None:
        train_loader, val_loader, test_loader = make_Spotify44k_loaders(shuffle=True)
    else:
        train_loader, val_loader, test_loader = dummy_dataloader, dummy_dataloader, dummy_dataloader

    num_classes = 14

    base_metrics = MetricCollection({
        "accuracy":  MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "top3_acc":  MulticlassAccuracy(num_classes=num_classes, top_k=3, average="micro"),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall":    MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1":        MulticlassF1Score(num_classes=num_classes, average="macro"),
    })
    
    metrics = {
        "train": base_metrics.clone("train_").to(device),
        "val":   base_metrics.clone("val_").to(device),
        "test":  base_metrics.clone("test_").to(device),
    }

    train_results = _fit_model(
        model, 
        optimizer,
        train_loader,
        val_loader,
        metrics,
        train_epochs,
        log_path,
    )

    test_results = _test_model(
        model,
        test_loader,
        metrics,
        log_path,
    )

    return train_results, test_results
