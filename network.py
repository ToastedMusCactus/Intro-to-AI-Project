# Cross validation Balance Accuracy = 0.00%

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
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
def custom_loss_function(outputs, targets):
    """
    A Custom loss function that is normalized by the input values / batch size.
    """
    # TODO: Implement manual loss calculation normalized by batch/input size
    pass


# ==========================================
# 4. TRAINING & EVALUATION LOOPS (PERSON B)
# ==========================================
def train_epoch(model, dataloader, optimizer, device):
    # 1) Begin training mode and tracking total loss
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        # 2) Move the batch data into the device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # TODO: Zero gradients, forward pass, calculate custom loss, backward pass, step optimizer
        
    return running_loss


def evaluate(model, dataloader, device):
    # 1) Set the model to evaluation mode and initialize prediction and target lists
    model.eval()
    predictions = []
    targets = []

    # 2) Turns off the gradient calculator engine to run faster and save memory
    with torch.no_grad():

        # 3) Loop through all batches in the dataloader
        for X_batch, y_batch in dataloader:

            # 4) Move the batch into the device
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


    # Instantiate Hyperparameters
    INPUT_SIZE = X_train_scaled.shape[1]
    NUMBER_OF_CLASSES = len(classes)
    HIDDEN_SIZE = 128
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 30
    WEIGHT_DECAY = 1e-4

    val_size = int(0.2 * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset_full, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = BeanMLP(INPUT_SIZE, hidden_dim=HIDDEN_SIZE, output_dim=NUMBER_OF_CLASSES).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )

    best_value_accuracy = 0.0

    print("\nBeginning Training...")
    for epoch in range(1, EPOCHS + 1):
        training_loss = train_epoch(model, train_loader, optimizer, device)
        value_accuracy = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {training_loss:.4f} | Val Balanced Acc: {value_accuracy:.4f}")
    
        if value_accuracy > best_value_accuracy:
            best_value_accuracy = value_accuracy
            torch.save(model.state_dict(), "best_bean_mlp.pth")

    print(f"\nTraining Completed -> Best Validation Balanced Accuracy: {best_value_accuracy:.4f}")

    model.load_state_dict(torch.load("best_bean_mlp.pth"))
    # ------------------------------------------
    # CSV EXPORT (PERSON A)
    # ------------------------------------------
    
    #TODO Person B: Uncomment this block once 'model' is fully trained and ready for inference
    
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
    