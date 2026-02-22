import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class Hw1PosDataset(Dataset):
    def __init__(self, path_pt: str):
        assert os.path.exists(path_pt), f"Missing file: {path_pt}"
        d = torch.load(path_pt)

        if "img_before" not in d:
            raise KeyError("Dataset must contain 'img_before'. Run merge_hw1.py with imgs_before_*.")

        self.img = d["img_before"].float() / 255.0   # [N,3,128,128]
        self.act = d["actions"].long()               # [N]
        self.pos = d["positions"].float()            # [N,2]

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        return self.img[idx], self.act[idx], self.pos[idx]

class CNNModel(nn.Module):
    def __init__(self, act_emb_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),   # 128->64
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),  # 64->32
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # 32->16
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU() # 16->8
        )
        self.act_emb = nn.Embedding(4, act_emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8 + act_emb_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, img, act):
        f = self.conv(img).view(img.size(0), -1)
        a = self.act_emb(act)
        x = torch.cat([f, a], dim=1)
        return self.fc(x)

@torch.no_grad()
def eval_mse(model, loader, device):
    model.eval()
    s = 0.0
    n = 0
    for img, act, pos in loader:
        img, act, pos = img.to(device), act.to(device), pos.to(device)
        pred = model(img, act)
        mse = (pred - pos).pow(2).mean(dim=1)
        s += mse.sum().item()
        n += mse.numel()
    return s / max(n, 1)

def train(model, train_loader, val_loader, device, epochs=15, lr=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        k = 0
        for img, act, pos in train_loader:
            img, act, pos = img.to(device), act.to(device), pos.to(device)
            pred = model(img, act)
            loss = F.mse_loss(pred, pos)

            opt.zero_grad()
            loss.backward()
            opt.step()

            run += loss.item()
            k += 1

        val_mse = eval_mse(model, val_loader, device)
        print(f"[CNN] Epoch {ep:02d} | train_loss={run/max(k,1):.6f} | val_MSE={val_mse:.6f}")

def main():
    path = "hw1_dataset_1000.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = Hw1PosDataset(path)
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = CNNModel(act_emb_dim=16)
    train(model, train_loader, val_loader, device, epochs=15, lr=1e-3)

    torch.save(model.state_dict(), "cnn_pos.pt")
    print("Saved: cnn_pos.pt")

if __name__ == "__main__":
    main()
