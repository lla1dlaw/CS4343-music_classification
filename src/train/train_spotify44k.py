import torch
from torch import nn
import torchmetrics
from torchvision.transforms import v2


METRICS = [
    torchmetrics.Accuracy, 
    torchmetrics.Precision,
    torchmetrics.Recall,
    torchmetrics.F1Score,
    torchmetrics.ConfusionMatrix,
]


def update_metrics(metrics, logits: torch.tensor):



def train(model: nn.Module, epochs: int):
    # load dataset
    
    for epoch in range(epochs):

