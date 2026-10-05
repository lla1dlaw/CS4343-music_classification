import torchvision
from torch.optim import Adam
from train import train_spotify44k

import os
from pathlib import Path


def main():
    model = torchvision.models.alexnet()
    optimizer = Adam(
        model.parameters(),
        lr = 0.004,
    )
    epochs = 2

    log_dir = Path(os.getcwd()).parent.parent / "results"
    log_dir.mkdir(exist_ok=True)

    train_results, test_results = train_spotify44k.run_train_test(
        model,
        optimizer,
        epochs,
        str(log_dir)
    )

if __name__ == "__main__":
     main()
