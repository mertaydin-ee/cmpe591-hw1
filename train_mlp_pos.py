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

class MLPModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        in_dim = 3 * 128 * 128 + 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, img, act):
        x = img.view(img.size(0), -1)
        a = F.one_hot(act, num_classes=4).float()
        x = torch.cat([x, a], dim=1)
        return self.net(x)

@torch.no_grad()
def eval_mse(model, loader, device):
    model.eval()
    s = 0.0
    n = 0
    for img, act, pos in loader:
        img, act, pos = img.to(device), act.to(device), pos.to(device)
        pred = model(img, act)
        mse = (pred - pos).pow(2).mean(dim=1)  # per-sample, mean over 2 dims
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
        print(f"[MLP] Epoch {ep:02d} | train_loss={run/max(k,1):.6f} | val_MSE={val_mse:.6f}")

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

    model = MLPModel(hidden=512)
    train(model, train_loader, val_loader, device, epochs=15, lr=1e-3)

    torch.save(model.state_dict(), "mlp_pos.pt")
    print("Saved: mlp_pos.pt")
    
def test_only(seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Hw1PosDataset(DATASET)

    
    n_train = int(0.8 * len(ds))
    n_val = int(0.1 * len(ds))
    n_test = len(ds) - n_train - n_val
    _, _, test_ds = random_split(ds, [n_train, n_val, n_test],
                                 generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = MLP().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    mse = test(model, test_loader, device)
    print("TEST MSE (loaded model):", mse)
    return mse

if __name__ == "__main__":
    main()
     # test_only()
