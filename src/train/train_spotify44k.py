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
from torchvision.transforms import v2
from datasets.data_pipeline import make_loaders
from torch.utils.data import DataLoader




def update_metrics(metrics, logits: torch.tensor):
    pass


def evaluate(model: nn.Module, data_loader: DataLoader):
    model.eval()


def train_one_epoch(model: nn.Model, data_loader: DataLoader)
    model.train()

def fit_model(model: nn.Module, epochs: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    train_loader, val_loader, test_loader = make_loaders(shuffle=True)
    num_classes = len(train_loader.dataset..classes)

    base_metrics = MetricCollection({
        "accuracy":         MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "top3_acc":         MulticlassAccuracy(num_classes=num_classes, topk=3, average="micro"),
        "precision":        MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall":           MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1":               MulticlassF1Score(num_classes=num_classes, average="macro"),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=num_classes, average="macro"),
    })
    
    metrics = {
        "train": base_metrics.clone("train_").to(device),
        "val":   base_metrics.clone("val_").to(device),
        "test":  base_metrics.clone("test_").to(device),
    }

    for epoch in range(epochs):
    


if __name__ == "__main__":
    main()

