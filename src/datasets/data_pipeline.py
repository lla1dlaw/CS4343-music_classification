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
def make_loaders(shuffle: bool = True) -> tuple[DL, DL, DL]:
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
    root = "data"
    
    dataset_root_path = Path(project_root, root)
    dataset_path = dataset_root_path / name
    
    try: 
        # requires KAGGLE_API_KEY variable to be set in the runtime env
        kagglehub.dataset_download(handle=handle, path=name, output_dir=root)    
    except FileExistsError: # dataset already exists!
        print(f"Using existing dataset found at {dataset_path}")

def _make_loader(split: str, shuffle: bool = True) -> DL:
    transform = tf.Compose([
        tf.Grayscale(num_output_channels=3),
        tf.Resize((224, 224)),
        tf.PILToTensor()
    ])
    dataset = ds.ImageFolder(root=f"data/spotify44k/{split}", transform=transform)
    return DL(dataset, batch_size=32, shuffle=shuffle, num_workers=4)


<<<<<<< HEAD
def build_tensor(split: str): 
=======
<<<<<<< Updated upstream
def build_tensor(split: str):
>>>>>>> c179334d631ee747bd5e506b9a536e46e9d8e56b
# this function actually isn't needed anymore because of 
# the use of the ds.ImageFolder() function above which takes care of walking the 
# directories for us
=======
# this actually isn't needed anymore because of 
# the use of the ds.ImageFolder() function above which takes care of walking the 
# directories for us
def build_tensor(split: str):
>>>>>>> Stashed changes
    root = Path("data/spotify44k") / split
    genres = sorted(p.name for p in root.iterdir() if p.is_dir())   # folder names = genres

    # every (image path, genre number) pair, genre taken from the folder the image is in
    files = [(img, label)
             for label, genre in enumerate(genres)
             for img in sorted((root / genre).glob("*.jpg"))]

    data = torch.zeros((len(files), 225, 224), dtype=torch.uint8)
    for n, (path, label) in enumerate(files):
        img = Image.open(path).convert("L").resize((224, 224))   # "L" = grayscale, so 2D
        data[n, :224, :] = torch.from_numpy(np.array(img))       # the image
        data[n, 224, :] = label                                  # the label row
        if n % 5000 == 0:
            print(f"{n}/{len(files)}")

    return data, genres


if __name__ == "__main__":
<<<<<<< Updated upstream
    train, val, test = make_loaders()
    
    sample_input, sample_labels = next(iter(train))
    train_num_samples = len(train.dataset)
    val_num_samples = len(val.dataset)
    test_num_samples = len(test.dataset)
=======
    print("-" * 10)
    loader = make_loader("")

    data, genres = build_tensor("training")
    print(data.shape)                       # torch.Size([N, 225, 224])
>>>>>>> Stashed changes

    total_samples = train_num_samples + val_num_samples + test_num_samples

<<<<<<< Updated upstream
    train_prop = train_num_samples / total_samples
    val_prop = val_num_samples / total_samples
    test_prop = test_num_samples / total_samples

    batch_shape = sample_input.shape
    
    print(f"Batch Shape: {batch_shape}")
    print(f"Num Training Samples: {train_num_samples}")
    print(f"Num Validation Samples: {val_num_samples}")
    print(f"Num Testing Samples: {test_num_samples}")
    
    print(f"Proportions [train, val, test]: [{train_prop:.1f}, {val_prop:.1f}, {test_prop:.1f}]")


# if __name__ == "__main__":
#     print("-" * 10)
#     loader = make_loader("")
#
#     data, genres = build_tensor("training")
#     print(data.shape)                       # torch.Size([N, 225, 224])
#     training_data, training_genres = build_tensor("training")
#     testing_data, testing_genre = build_tensor("testing")
#     val_data, val_genre = build_tensor("validation")
#
#     print(training_data.shape)                       # torch.Size([N, 225, 224])
#
#     for i in range(3):
#         label = training_data[i, 224, 0].item()
#         print(f"sample {i}: genre {label} = {training_genres[label]}")
#         print(training_data[i, :224])                # the 2D image
#
#     torch.save({"data": training_data, "genres": training_genres}, "data/spotify44k/train_tensor.pt")
#     torch.save({"data": val_data, "genres": val_genre}, "data/spotify44k/val_tensor.pt")
#     torch.save({"data": testing_data, "genres": testing_genre}, "data/spotify44k/testing_tensor.pt")
#
#     torch.save({"data": data, "genres": genres}, "data/spotify44k/train_tensor.pt")
=======
    torch.save({"data": training_data, "genres": training_genres}, "data/spotify44k/train_tensor.pt")
    torch.save({"data": val_data, "genres": val_genre}, "data/spotify44k/val_tensor.pt")
    torch.save({"data": testing_data, "genres": testing_genre}, "data/spotify44k/testing_tensor.pt")

    torch.save({"data": data, "genres": genres}, "data/spotify44k/train_tensor.pt")
>>>>>>> Stashed changes
