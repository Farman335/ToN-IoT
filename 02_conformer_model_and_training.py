import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the Dataset
dataset_path = r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_Network_dataset\train_test_network.csv"
print(f"Loading dataset from: {dataset_path}...")
df = pd.read_csv(dataset_path, low_memory=False)

# Convert Zeek dashes into NaNs
df.replace('-', np.nan, inplace=True)

# ---------------------------------------------------------
# STRATEGY 1: DROP EMPTY COLUMNS (> 60% missing)
# ---------------------------------------------------------
missing_percentages = df.isnull().mean() * 100
threshold = 60.0
columns_to_drop = missing_percentages[missing_percentages > threshold].index
df.drop(columns=columns_to_drop, inplace=True)

# Drop duplicates & remaining NaNs
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# ---------------------------------------------------------
# STRATEGY 2: ISOLATE IDENTIFIERS (Metadata for Wazuh)
# ---------------------------------------------------------
identifiers = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
df.drop(columns=identifiers, inplace=True, errors='ignore')

# ---------------------------------------------------------
# STRATEGY 3: TRANSLATE TEXT TO NUMBERS
# ---------------------------------------------------------
text_columns = df.select_dtypes(include=['object']).columns
for col in text_columns:
    if col != 'type':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Encode multiclass label 'type'
type_encoder = LabelEncoder()
df['type'] = type_encoder.fit_transform(df['type'].astype(str))
num_classes = len(type_encoder.classes_)

# Split features (X) and targets (y)
y_label = df['label'].reset_index(drop=True)
y_type = df['type'].reset_index(drop=True)
X_features = df.drop(columns=['label', 'type']).reset_index(drop=True)

# ---------------------------------------------------------
# STRATEGY 4: SCALING (Leakage-Free)
# ---------------------------------------------------------
split_idx = int(0.8 * len(X_features))
scaler = MinMaxScaler()
scaler.fit(X_features.iloc[:split_idx])
X_scaled = pd.DataFrame(scaler.transform(X_features), columns=X_features.columns)

# ---------------------------------------------------------
# STRATEGY 5: SEQUENCE GENERATION (Temporal Preservation)
# ---------------------------------------------------------
def create_sequences(X, y_bin, y_mul, time_steps=10):
    Xs, y_bins, y_muls = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        y_bins.append(y_bin[i + time_steps])
        y_muls.append(y_mul[i + time_steps])
    return np.array(Xs, dtype=np.float32), np.array(y_bins, dtype=np.float32), np.array(y_muls, dtype=np.int64)

TIME_STEPS = 10
X_seq, y_bin_seq, y_mul_seq = create_sequences(X_scaled, y_label.values, y_type.values, TIME_STEPS)

# Split sequences
X_train_seq, X_val_seq, y_bin_train_seq, y_bin_val_seq, y_mul_train_seq, y_mul_val_seq = train_test_split(
    X_seq, y_bin_seq, y_mul_seq, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_bin_seq
)

print(f"Train sequences shape: {X_train_seq.shape}")
print(f"Val sequences shape: {X_val_seq.shape}")

# DataLoaders
train_dataset = TensorDataset(
    torch.tensor(X_train_seq), 
    torch.tensor(y_bin_train_seq).unsqueeze(1), 
    torch.tensor(y_mul_train_seq)
)
val_dataset = TensorDataset(
    torch.tensor(X_val_seq), 
    torch.tensor(y_bin_val_seq).unsqueeze(1), 
    torch.tensor(y_mul_val_seq)
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# ---------------------------------------------------------
# Step 6: PYTORCH CONFORMER ARCHITECTURE
# ---------------------------------------------------------
class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_model * expansion_factor)
        self.w_2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()  # Swish activation is standard in Conformer

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return residual + 0.5 * x  # Half-step residual

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # Depthwise Conv1d (groups = d_model)
        self.depthwise = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, 
            groups=d_model
        )
        self.pointwise = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=1
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = x.transpose(1, 2)  # Conv1d expects (batch, d_model, seq_len)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x

class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, dropout=dropout)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model, dropout=dropout)
        self.post_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # 1. FFN half-step
        x = self.ff1(x)
        
        # 2. Self-Attention
        residual = x
        x_ln = self.attn_ln(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = residual + attn_out
        
        # 3. Conv module
        x = self.conv(x)
        
        # 4. FFN half-step
        x = self.ff2(x)
        
        # 5. Post Norm
        x = self.post_ln(x)
        return x

class ConformerSentinelModel(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_classes=10):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Single Conformer Block (extremely lightweight and fast)
        self.conformer = ConformerBlock(d_model=d_model, nhead=nhead, kernel_size=3, dropout=0.1)
        
        self.fc = nn.Linear(d_model, 32)
        self.dropout = nn.Dropout(0.2)
        
        self.binary_head = nn.Linear(32, 1)
        self.multiclass_head = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        x = self.input_projection(x)  # shape: (batch_size, seq_len, d_model)
        x = self.conformer(x)         # shape: (batch_size, seq_len, d_model)
        
        # Global average pooling over the sequence dimension (dim 1)
        x = torch.mean(x, dim=1)      # shape: (batch_size, d_model)
        
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        
        bin_out = self.binary_head(x)
        mul_out = self.multiclass_head(x)
        return bin_out, mul_out

num_features = X_train_seq.shape[2]
model = ConformerSentinelModel(num_features=num_features, d_model=64, nhead=4, num_classes=num_classes).to(device)

criterion_bin = nn.BCEWithLogitsLoss()
criterion_mul = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------
# Step 7: MODEL TRAINING (15 epochs)
# ---------------------------------------------------------
print("\n--- STARTING CONFORMER TRAINING PROCESS ---")
epochs = 15
best_val_loss = float('inf')

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y_bin, batch_y_mul in train_loader:
        batch_x, batch_y_bin, batch_y_mul = batch_x.to(device), batch_y_bin.to(device), batch_y_mul.to(device)
        
        optimizer.zero_grad()
        bin_pred, mul_pred = model(batch_x)
        
        loss_bin = criterion_bin(bin_pred, batch_y_bin)
        loss_mul = criterion_mul(mul_pred, batch_y_mul)
        
        loss = loss_bin + loss_mul
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_x.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    # Validation Loop
    model.eval()
    val_loss = 0.0
    correct_bin = 0
    with torch.no_grad():
        for batch_x, batch_y_bin, batch_y_mul in val_loader:
            batch_x, batch_y_bin, batch_y_mul = batch_x.to(device), batch_y_bin.to(device), batch_y_mul.to(device)
            bin_pred, mul_pred = model(batch_x)
            
            loss_bin = criterion_bin(bin_pred, batch_y_bin)
            loss_mul = criterion_mul(mul_pred, batch_y_mul)
            val_loss += (loss_bin.item() + loss_mul.item()) * batch_x.size(0)
            
            # Accuracy metric
            bin_preds = (torch.sigmoid(bin_pred) >= 0.5).float()
            correct_bin += (bin_preds == batch_y_bin).sum().item()
            
    val_loss /= len(val_loader.dataset)
    val_acc = correct_bin / len(val_loader.dataset)
    
    print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Bin Acc: {val_acc:.4f}")
    
    # Save best weights
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'conformer_sentinel_model.pth')
        print(" -> Saved new best Conformer model weights.")

# ---------------------------------------------------------
# Step 8: EVALUATION
# ---------------------------------------------------------
print("\n--- LOADING BEST CONFORMER MODEL FOR EVALUATION ---")
model.load_state_dict(torch.load('conformer_sentinel_model.pth'))
model.eval()

all_y_bin_pred, all_y_bin_true = [], []
all_y_mul_pred, all_y_mul_true = [], []

with torch.no_grad():
    for batch_x, batch_y_bin, batch_y_mul in val_loader:
        batch_x = batch_x.to(device)
        bin_pred, mul_pred = model(batch_x)
        
        bin_probs = torch.sigmoid(bin_pred).cpu().numpy()
        bin_preds = (bin_probs >= 0.5).astype(int)
        
        mul_preds = torch.argmax(mul_pred, dim=1).cpu().numpy()
        
        all_y_bin_pred.extend(bin_preds.flatten())
        all_y_bin_true.extend(batch_y_bin.numpy().flatten())
        
        all_y_mul_pred.extend(mul_preds)
        all_y_mul_true.extend(batch_y_mul.numpy())

print("\n======================================================")
print("      ZERO-DAY DETECTION (BINARY) EVALUATION REPORT")
print("======================================================")
print(classification_report(all_y_bin_true, all_y_bin_pred, target_names=['Normal (0)', 'Attack (1)']))

print("\n======================================================")
print("   FORENSIC CLASSIFICATION (MULTICLASS) EVALUATION REPORT")
print("======================================================")
print(classification_report(all_y_mul_true, all_y_mul_pred, target_names=type_encoder.classes_))
