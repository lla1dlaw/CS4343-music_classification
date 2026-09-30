import torch
import torchvision
from torch.utils.data import DataLoader as DL
from torchvision import datasets as ds, transforms as tf
from pathlib import Path
import numpy as np
from PIL import Image


# use this function in the training loop function
def make_loaders(shuffle: bool = True):
    return _make_loader("training"), _make_loader("validation"), _make_loader("testing")


def _make_loader(split: str, shuffle: bool = True):
    transform = tf.Compose([
        tf.Grayscale(num_output_channels=3),
        tf.Resize((224, 224)),
        tf.PILToTensor()
    ])
<<<<<<< HEAD
=======

def make_loader(split: str, shuffle: bool):
>>>>>>> 8ba2b0e1d7bb38ec10a4c07b6e8853d9066627f7
    dataset = ds.ImageFolder(root=f"data/spotify44k/{split}", transform=transform)
    return DL(dataset, batch_size=32, shuffle=shuffle, num_workers=4), dataset.classes


<<<<<<< HEAD
# this actually isn't needed anymore because of 
# the use of the ds.ImageFolder() function above which takes care of walking the 
# directories for us
=======
>>>>>>> 8ba2b0e1d7bb38ec10a4c07b6e8853d9066627f7
def build_tensor(split: str):
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
<<<<<<< HEAD
    print("-" * 10)
    loader = make_loader("")

    data, genres = build_tensor("training")
    print(data.shape)                       # torch.Size([N, 225, 224])
=======
    training_data, training_genres = build_tensor("training")
    testing_data, testing_genre = build_tensor("testing")
    val_data, val_genre = build_tensor("validation")
    
    print(training_data.shape)                       # torch.Size([N, 225, 224])
>>>>>>> 8ba2b0e1d7bb38ec10a4c07b6e8853d9066627f7

    for i in range(3):
        label = training_data[i, 224, 0].item()
        print(f"sample {i}: genre {label} = {training_genres[label]}")
        print(training_data[i, :224])                # the 2D image

    torch.save({"data": training_data, "genres": training_genres}, "data/spotify44k/train_tensor.pt")
    torch.save({"data": val_data, "genres": val_genre}, "data/spotify44k/val_tensor.pt")
    torch.save({"data": testing_data, "genres": testing_genre}, "data/spotify44k/testing_tensor.pt")

<<<<<<< HEAD
    torch.save({"data": data, "genres": genres}, "data/spotify44k/train_tensor.pt")
=======


        
        

    

>>>>>>> 8ba2b0e1d7bb38ec10a4c07b6e8853d9066627f7
