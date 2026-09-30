import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
from datasets.data_pipeline import make_loaders
from torch.utils.data import DataLoader


def update_metrics(metrics, logits: torch.tensor):
    pass


def evaluate(model: nn.Module, data_loader: DataLoader):
    model.eval()


def train_one_epoch( 
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion,
        optimizer,
        metrics: dict[str, MetricCollection],
        # need to add a separate loss metric with torchmetric mean aggregrator
        device: torch.device,
    ):

    model.train()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        metrics['train'].update(logits, labels)

    train_results = metrics['train'].compute()

    # complete validation

def fit_model(model: nn.Module, optimizer: torch.optim.Optimizer, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, test_loader = make_loaders(shuffle=True)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, val_loader, criterion, optimizer, metrics, device) 


if __name__ == "__main__":
    main()

