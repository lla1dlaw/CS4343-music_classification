import torch
import torchvision
from torch.optim import Adam

from train import train_spotify44k

import os
from pathlib import Path

import time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 14 # spotify 44k has 14 classes
    model = torchvision.models.resnet18(num_classes=num_classes).to(device)
    model_id = time.time()
    model_name = model.__class__.__name__
    optimizer = Adam(
        model.parameters(),
        lr = 0.004,
    )
    epochs = 100

    log_dir = Path(os.getcwd()).parent.parent / "results"
    log_dir.mkdir(exist_ok=True)

    print(f"Starting training and testing loops for model: {model_name}")
    train_results, test_results = train_spotify44k.run_train_test(
        model,
        optimizer,
        epochs,
        str(log_dir)
    )
     
    print(f"{model_name} training and testing completed.")
    
    folder_path = Path(os.getcwd()) / "models"
    folder_path.mkdir(exist_ok=True)
    file_name = f"{model_name}_{model_id}_spotify44k.pt"
    save_path =  folder_path / file_name   
    torch.save(model.state_dict(), save_path) 


if __name__ == "__main__":
     main()
