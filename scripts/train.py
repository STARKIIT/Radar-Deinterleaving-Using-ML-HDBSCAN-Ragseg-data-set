import torch, yaml, numpy as np, os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model.embedding_transformer import PDWEncoder
from model.losses import batch_all_triplet

def create_sequences(pdw_data, labels, seq_len=100, step=50):
    """Creates overlapping sequences using a sliding window."""
    X, Y = [], []
    for i in range(0, len(pdw_data) - seq_len, step):
        X.append(pdw_data[i:i+seq_len])
        Y.append(labels[i:i+seq_len])
    return np.array(X), np.array(Y)

# --- Configuration ---
cfg = yaml.safe_load(open("config/train.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Load and Prepare Data ---
print("Loading data...")
pdws = np.load(cfg["train_pdw"])
labels = np.load(cfg["train_labels"])
if pdws.size == 0 or labels.size == 0:
    raise ValueError("Training data or labels are empty.")
if len(np.unique(labels)) < 2:
    raise ValueError("Training labels must contain at least two classes for triplet/contrastive loss.")

# Create sequences for the transformer using a sliding window
X, Y = create_sequences(pdws, labels, seq_len=100, step=50)
if len(X) == 0:
    raise ValueError("Not enough data to create sequences for training.")

# Train/Validation Split
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long())

train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg["batch"], shuffle=False)
print(f"Training data: {len(train_ds)} sequences. Validation data: {len(val_ds)} sequences.")

# --- Model, Optimizer, Scheduler ---
net = PDWEncoder().to(DEVICE)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5) # Use AdamW
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.1, patience=5)

# --- Training Loop ---
best_val_loss = float('inf')
print("Starting training...")
for epoch in range(cfg["epochs"]):
    net.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        z = net(x)
        loss = batch_all_triplet(z, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    # Validation Loop
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            z = net(x)
            loss = batch_all_triplet(z, y)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{cfg['epochs']} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        os.makedirs(os.path.dirname(cfg["ckpt"]), exist_ok=True)
        torch.save(net.state_dict(), cfg["ckpt"])
        print(f"  -> Model saved to {cfg['ckpt']} (Val Loss: {best_val_loss:.4f})")

print("Training complete.")
