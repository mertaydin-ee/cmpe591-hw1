import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# -------------------------
# Dataset
# -------------------------
class Hw1ImagePredictionDataset(Dataset):
    def __init__(self, path_pt: str):
        assert os.path.exists(path_pt), f"Missing file: {path_pt}"
        d = torch.load(path_pt)
        if "img_before" not in d or "img_after" not in d:
            raise KeyError("Dataset must contain 'img_before' and 'img_after'.")
        self.x_img = d["img_before"].float() / 255.0
        self.x_act = d["actions"].long()
        self.y_img = d["img_after"].float() / 255.0

    def __len__(self):
        return self.x_img.shape[0]

    def __getitem__(self, idx):
        return self.x_img[idx], self.x_act[idx], self.y_img[idx]

# -------------------------
# Small U-Net-ish model (skip connections)
# Condition: action one-hot as extra channels
# Output: residual (delta) in [-1,1]
# -------------------------
class ImagePredictionNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder
        self.e1 = nn.Sequential(nn.Conv2d(7, 32, 3, 1, 1), nn.ReLU(),
                                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.p1 = nn.Conv2d(32, 32, 4, 2, 1)   # 128->64

        self.e2 = nn.Sequential(nn.ReLU(),
                                nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        self.p2 = nn.Conv2d(64, 64, 4, 2, 1)   # 64->32

        self.e3 = nn.Sequential(nn.ReLU(),
                                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
                                nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.p3 = nn.Conv2d(128, 128, 4, 2, 1) # 32->16

        self.bott = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())

        # decoder
        self.u3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 16->32
        self.d3 = nn.Sequential(nn.ReLU(),
                                nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(),
                                nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())

        self.u2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 32->64
        self.d2 = nn.Sequential(nn.ReLU(),
                                nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())

        self.u1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 64->128
        self.d1 = nn.Sequential(nn.ReLU(),
                                nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
                                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())

        # residual head
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, img_before, act):
        B, _, H, W = img_before.shape
        a = F.one_hot(act, num_classes=4).float().to(img_before.device)  # [B,4]
        a_map = a.view(B, 4, 1, 1).expand(-1, -1, H, W)                  # [B,4,H,W]
        x = torch.cat([img_before, a_map], dim=1)                        # [B,7,H,W]

        s1 = self.e1(x)             # [B,32,128,128]
        x1 = self.p1(s1)            # [B,32,64,64]
        s2 = self.e2(x1)            # [B,64,64,64]
        x2 = self.p2(s2)            # [B,64,32,32]
        s3 = self.e3(x2)            # [B,128,32,32]
        x3 = self.p3(s3)            # [B,128,16,16]

        b  = self.bott(x3)          # [B,256,16,16]

        u3 = self.u3(b)             # [B,128,32,32]
        d3 = self.d3(torch.cat([u3, s3], dim=1))  # [B,128,32,32]

        u2 = self.u2(d3)            # [B,64,64,64]
        d2 = self.d2(torch.cat([u2, s2], dim=1))  # [B,64,64,64]

        u1 = self.u1(d2)            # [B,32,128,128]
        d1 = self.d1(torch.cat([u1, s1], dim=1))  # [B,32,128,128]

        res = torch.tanh(self.out(d1))            # [-1,1]
        return res

def to_png(x_chw, path):
    x = (x_chw.clamp(0,1) * 255).byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(x, mode="RGB").save(path)

@torch.no_grad()
def save_examples(model, ds, device, out_dir="imgpred_examples", k=8):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    idxs = torch.randperm(len(ds))[:k].tolist()
    for i, idx in enumerate(idxs):
        xb, a, y = ds[idx]
        xb = xb.unsqueeze(0).to(device)
        a  = torch.tensor([a], dtype=torch.long, device=device)
        y  = y.to(device)

        res = model(xb, a).squeeze(0)
        pred = (xb.squeeze(0) + 0.5 * res).clamp(0,1)  # scale residual

        to_png(ds[idx][0], os.path.join(out_dir, f"{i:02d}_before_a{int(ds[idx][1])}.png"))
        to_png(ds[idx][2], os.path.join(out_dir, f"{i:02d}_gt_after.png"))
        to_png(pred,        os.path.join(out_dir, f"{i:02d}_pred_after.png"))

    print(f"Saved examples to: {out_dir}/")

def main():
    path = "hw1_dataset_1000.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = Hw1ImagePredictionDataset(path)
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = ImagePredictionNetV2().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # weighted loss params
    alpha = 10.0     # change pixels weight
    thresh = 0.02    # change threshold on mean abs residual

    for ep in range(1, 26):
        model.train()
        run = 0.0
        k = 0

        for xb, a, y in train_loader:
            xb, a, y = xb.to(device), a.to(device), y.to(device)

            # residual ground truth
            res_gt = y - xb

            # predict residual
            res_pred = model(xb, a)
            pred = (xb + 0.5 * res_pred).clamp(0,1)

            # build change mask from GT residual (training only)
            change = res_gt.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
            mask = (change > thresh).float()
            w = 1.0 + alpha * mask

            # weighted L1 on final image (you can also do on residual)
            loss = (w * (pred - y).abs()).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            run += loss.item()
            k += 1

        # quick val metric
        model.eval()
        with torch.no_grad():
            val_l1 = 0.0
            n = 0
            for xb, a, y in val_loader:
                xb, a, y = xb.to(device), a.to(device), y.to(device)
                res_pred = model(xb, a)
                pred = (xb + 0.5 * res_pred).clamp(0,1)
                val_l1 += F.l1_loss(pred, y, reduction="mean").item()
                n += 1
            val_l1 /= max(n,1)

        print(f"[image_prediction_v2] Epoch {ep:02d} | train_loss={run/max(k,1):.6f} | val_L1={val_l1:.6f}")

    torch.save(model.state_dict(), "image_prediction.pt")
    print("Saved: image_prediction.pt")

    # save qualitative samples
    save_examples(model, ds, device, out_dir="imgpred_examples", k=8)

if __name__ == "__main__":
    main()
