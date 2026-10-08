import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from train.train_spotify44k import run_train_test

class DummyGarbageDataset(Dataset):
    def __init__(self, num_samples=10, input_dim=20, num_classes=14):
        self.inputs = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.classes = [f"class_{i}" for i in range(num_classes)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class GarbageClassifier(nn.Module):
    def __init__(self, input_dim=20, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def main():
    dataset = DummyGarbageDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = GarbageClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train_results, test_results = run_train_test(
        model=model,
        optimizer=optimizer,
        train_epochs=2,
        log_dir="results",
        dummy_dataloader=dataloader
    )
    
    unique_id = str(int(time.time()))
    model_name = model.__class__.__name__
    
    os.makedirs("results", exist_ok=True)
    
    train_csv_path = f"results/{model_name}_train_{unique_id}.csv"
    test_csv_path = f"results/{model_name}_test_{unique_id}.csv"
    
    train_results.write_csv(train_csv_path)
    test_results.write_csv(test_csv_path)
    
    print(f"Data saved to {train_csv_path} and {test_csv_path}")

if __name__ == "__main__":
    main()
