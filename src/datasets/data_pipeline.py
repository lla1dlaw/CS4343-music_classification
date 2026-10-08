# prepares the dataset found here: https://www.kaggle.com/datasets/mrodriguez2/spotify-3-second-mel-spectrograms

import torch
import torchvision
from torch.utils.data import DataLoader as DL
from torchvision import datasets as ds, transforms as tf
from pathlib import Path
import numpy as np
from PIL import Image
from pathlib import Path 
import os
import kagglehub


# use this function in the training loop function
def make_Spotify44k_loaders(shuffle: bool = True) -> tuple[DL, DL, DL]:
    _download_spotify44k() # will only download if it needs to

    return (
        _make_loader("training", shuffle=shuffle),
        _make_loader("validation", shuffle=shuffle),
        _make_loader("testing", shuffle=shuffle),
    )


def _download_spotify44k():
    project_root = Path(os.getenv("PROJECT_ROOT", default=os.getcwd()))
    handle = "mrodriguez2/spotify-3-second-mel-spectrograms"
    name = "spotify44k"
    root = "data/spotify44k"
    
    dataset_root_path = Path(project_root, root)
    dataset_path = dataset_root_path / name
    
    try: 
        # requires KAGGLE_API_KEY variable to be set in the runtime env
        #kagglehub.dataset_download(handle=handle, path=name, output_dir=root)    
        kagglehub.dataset_download(handle=handle, output_dir=root)    
    except FileExistsError: # dataset already exists!
        print(f"Using existing dataset found at {dataset_path}")

def _make_loader(split: str, shuffle: bool = True) -> DL:
    transform = tf.Compose([
        tf.Grayscale(num_output_channels=3),
        tf.Resize((224, 224)),
        tf.ToTensor()
    ])
    dataset = ds.ImageFolder(root=f"data/spotify44k/dataset/{split}", transform=transform)
    return DL(dataset, batch_size=32, shuffle=shuffle, num_workers=4)


if __name__ == "__main__":
    train, val, test = make_Spotify44k_loaders()
    
    sample_input, sample_labels = next(iter(train))
    train_num_samples = len(train.dataset)
    val_num_samples = len(val.dataset)
    test_num_samples = len(test.dataset)

    total_samples = train_num_samples + val_num_samples + test_num_samples

    train_prop = train_num_samples / total_samples
    val_prop = val_num_samples / total_samples
    test_prop = test_num_samples / total_samples

    batch_shape = sample_input.shape
    
    print(f"Batch Shape: {batch_shape}")
    print(f"Num Training Samples: {train_num_samples}")
    print(f"Num Validation Samples: {val_num_samples}")
    print(f"Num Testing Samples: {test_num_samples}")
    
    print(f"Proportions [train, val, test]: [{train_prop:.1f}, {val_prop:.1f}, {test_prop:.1f}]")

