import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

df = pd.read_csv(r"C:\Users\ALI\Downloads\MVS\data_processed\final\MASTER_PI_FINAL.csv")

pnp_cols = ["tx","ty","tz","rx","ry","rz"]

for col in pnp_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=pnp_cols + ["cwt_path","label"])

df["rul"] = 1 - df["label"]

# normalize tabular features
scaler = StandardScaler()
df[pnp_cols] = scaler.fit_transform(df[pnp_cols])

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

class FusionDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def preprocess(self, cwt):
        cwt = np.nan_to_num(cwt)

        for i in range(cwt.shape[0]):
            c = cwt[i]
            c = (c - c.mean()) / (c.std() + 1e-6)
            cwt[i] = np.clip(c, -3, 3)

        return cwt

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cwt = np.load(row["cwt_path"])
        cwt = self.preprocess(cwt)

        cwt = torch.tensor(cwt, dtype=torch.float32)

        pnp = torch.tensor(np.array(row[pnp_cols], dtype=np.float32))
        label = torch.tensor(row["label"], dtype=torch.float32)
        rul = torch.tensor(row["rul"], dtype=torch.float32)

        return cwt, pnp, label, rul

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten()
        )

        dummy = torch.zeros(1,6,128,128)
        cnn_out = self.cnn(dummy).shape[1]

        # tabular branch
        self.pnp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        fusion_dim = cnn_out + 64

        # shared representation
        self.shared = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # regression head 
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # classification head
        self.class_head = nn.Linear(128, 1)

    def forward(self, cwt, pnp):
        x = torch.cat([self.cnn(cwt), self.pnp(pnp)], dim=1)
        x = self.shared(x)

        rul = self.reg_head(x)
        cls = self.class_head(x)

        return rul, cls

train_loader = DataLoader(FusionDataset(train_df, augment=True), batch_size=8, shuffle=True)
test_loader = DataLoader(FusionDataset(test_df, augment=False), batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionNet().to(device)

mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()

alpha = 0.85
beta = 0.15

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for cwt, pnp, label, rul in train_loader:
        cwt, pnp = cwt.to(device), pnp.to(device)
        label = label.view(-1,1).to(device)
        rul = rul.view(-1,1).to(device)

        rul_pred, cls_pred = model(cwt, pnp)

        loss_rul = mse(rul_pred, rul)
        loss_cls = bce(cls_pred, label)

        loss = alpha * loss_rul + beta * loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

model.eval()

rul_preds, rul_true = [], []
cls_preds, labels = [], []

with torch.no_grad():
    for cwt, pnp, label, rul in test_loader:
        cwt, pnp = cwt.to(device), pnp.to(device)

        r_pred, c_pred = model(cwt, pnp)

        rul_preds.extend(r_pred.cpu().numpy().squeeze())
        rul_true.extend(rul.numpy())

        cls_preds.extend(torch.sigmoid(c_pred).cpu().numpy().squeeze())
        labels.extend(label.numpy())

rul_preds = np.array(rul_preds)
rul_true = np.array(rul_true)

cls_preds = np.array(cls_preds)
labels = np.array(labels)

# classification
pred_class = (cls_preds > 0.5).astype(int)
acc = (pred_class == labels).mean()

# regression metrics
mae = np.mean(np.abs(rul_true - rul_preds))
rmse = np.sqrt(np.mean((rul_true - rul_preds)**2))

print("\n===== FINAL RESULTS =====")
print("Accuracy:", acc)
print("MAE:", mae)
print("RMSE:", rmse)

torch.save(model.state_dict(), "fusion_model_best.pth")

print("\nTRAINING COMPLETE")

pd.DataFrame({
    "rul_true": np.array(rul_true).squeeze(),
    "cnn_pred": np.array(rul_preds).squeeze(),
    "label": np.array(labels).squeeze()
}).to_csv("cnn_final_predictions.csv", index=False)
