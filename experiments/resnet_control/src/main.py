import torch
import torchvision
from torch.optim import Adam

from train import train_spotify44k


import os
from pathlib import Path


def main():
    num_classes = 14 # spotify 44k has 14 classes
    model = torchvision.models.resnet18(num_classes=num_classes)
    model_name = model.__class__.__name__
    optimizer = Adam(
        model.parameters(),
        lr = 0.004,
    )
    epochs = 2

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
    folder_path = "models"
    file_name = f"{model_name}_spotify44k.pt"
    save_path = os.path.join(folder_path, file_name)    
    torch.save(model.state_dict(), save_path) 


if __name__ == "__main__":
     main()
