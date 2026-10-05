# Cross validation Balance Accuracy = 0.00%

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ==========================================
# 1. DATASET & DATALOADER SETUP
# ==========================================
class DryBeanDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class BeanMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0):
        super(BeanMLP, self).__init__()
        # TODO: Define layers (Input -> Hidden -> Output)
        # Remember to include Batch Normalization and Dropout if enabled
        pass

    def forward(self, x):
        # TODO: Define forward pass logic
        pass


# ==========================================
# 3. CUSTOM LOSS FUNCTION
# ==========================================
def custom_loss_fn(outputs, targets):
    """
    Custom loss function normalized by input values / batch size.
    """
    # TODO: Implement manual loss calculation normalized by batch/input size
    pass


# ==========================================
# 4. TRAINING & EVALUATION LOOPS
# ==========================================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # TODO: Zero gradients, forward pass, calculate custom loss, backward pass, step optimizer
        
    return running_loss


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            # TODO: Get model predictions
            pass

    # TODO: Calculate and return Balanced Accuracy Score
    return 0.0


# ==========================================
# 5. MAIN EXECUTION & INFERENCE
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: Load train and test CSVs, apply feature scaling (StandardScaler)
    # TRAIN_FILE = "dry_bean_train.csv"
    # TEST_FILE  = "dry_bean_test.csv"

    # TODO: Perform Cross-Validation to determine best params

    # TODO: Save state_dict after final training
    # torch.save(model.state_dict(), "mlp_model.pth")

    # TODO: Generate test set predictions and save to 'network.csv'
    # Required columns: keep original columns from dry_bean_test.csv + add "Target"
    print("Template ready. Replace placeholders with implementation.")