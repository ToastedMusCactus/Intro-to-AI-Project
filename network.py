# Cross validation Balance Accuracy = 0.00%

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. DATASET SETUP (PERSON A)
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
# 2. NEURAL NETWORK ARCHITECTURE (PERSON A)
# ==========================================
class BeanMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0):
        super(BeanMLP, self).__init__()
        
        # Input Layer -> Hidden Layer -> Output Layer
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),       # Batch Normalization
            nn.ReLU(),                        # Activation function
            nn.Dropout(dropout_rate),         # For Person B's regularization tests
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. CUSTOM LOSS FUNCTION (PERSON B)
# ==========================================
def custom_loss_fn(outputs, targets):
    """
    Custom loss function normalized by input values / batch size.
    """
    # TODO: Implement manual loss calculation normalized by batch/input size
    pass

# ==========================================
# 4. TRAINING & EVALUATION LOOPS (PERSON B)
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

    # ------------------------------------------
    # DATA PREP (PERSON A)
    # ------------------------------------------
    TRAIN_FILE = "dry_bean_train.csv"
    TEST_FILE  = "dry_bean_test.csv"

    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    # Separate features and labels
    FEATURE_COLS = [c for c in train_df.columns if c != "Class"]
    X_train_raw = train_df[FEATURE_COLS].values
    X_test_raw  = test_df[FEATURE_COLS].values

    # Create mapping for string labels to integers
    classes = train_df["Class"].unique()
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    y_train_raw = np.array([class_to_idx[cls] for cls in train_df["Class"]])

    # Apply Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    # Instantiate the Datasets
    train_dataset_full = DryBeanDataset(X_train_scaled, y_train_raw)
    test_dataset = DryBeanDataset(X_test_scaled)
    
    # ------------------------------------------
    # MODEL INSTANTIATION, CV, & TRAINING (PERSON B)
    # ------------------------------------------
    # TODO: Wrap datasets in DataLoaders (e.g., DataLoader(train_dataset_full, batch_size=..., shuffle=True))
    # TODO: Instantiate BeanMLP, optimizer, and run cross-validation / training loops
    # TODO: Ensure the final trained model is saved to the variable 'model' before the export step below
    
    
    # ------------------------------------------
    # CSV EXPORT (PERSON A)
    # ------------------------------------------
    """ 
    # TODO Person B: Uncomment this block once 'model' is fully trained and ready for inference
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_outputs = model(X_test_tensor)
        _, predicted_idx = torch.max(test_outputs, 1)
        predicted_idx = predicted_idx.cpu().numpy()

    # Map integers back to strings (e.g., 0 -> 'SEKER')
    predicted_labels = [idx_to_class[idx] for idx in predicted_idx]

    # Keep original columns and add "Target"
    output_df = test_df.copy()
    output_df["Target"] = predicted_labels
    output_df.to_csv("network.csv", index=False)

    print("\nWrote predictions to network.csv")
    print(output_df.head())
    """