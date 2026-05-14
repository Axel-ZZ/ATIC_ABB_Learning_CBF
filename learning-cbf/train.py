import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
SET_DIR = REPO_ROOT / "learning-cbf/generated-sets/set_01"
CKPT_ROOT = REPO_ROOT / "learning-cbf/checkpoints"

X_safe_df = pd.read_csv(SET_DIR / "X_safe.csv")
N_df = pd.read_csv(SET_DIR / "N.csv")

X_safe = torch.tensor(X_safe_df.values, dtype=torch.float32)
X_unsafe = torch.tensor(N_df.values, dtype=torch.float32)

print(f"Columns: {list(X_safe_df.columns)}")
print(f"X_safe:   {tuple(X_safe.shape)}")
print(f"X_unsafe: {tuple(X_unsafe.shape)}")


class CBFNet(nn.Module):
    def __init__(self, in_dim=5, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def cbf_loss(h_safe, h_unsafe, margin=0.1):
    loss_safe = F.relu(margin - h_safe).mean()
    loss_unsafe = F.relu(margin + h_unsafe).mean()
    return loss_safe + loss_unsafe, loss_safe.item(), loss_unsafe.item()


def save_checkpoint(path, model, opt, epoch, config, acc):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "config": config,
        "acc": acc,
    }, path)


config = dict(
    in_dim=X_safe.shape[1],
    hidden=64,
    lr=1e-3,
    batch_size=512,
    epochs=200,
    margin=0.1,
    dataset="set_01",
)

run_name = datetime.now().strftime("cbf_%Y%m%d_%H%M%S")
ckpt_dir = CKPT_ROOT / run_name
ckpt_dir.mkdir(parents=True, exist_ok=True)

wandb.init(project="atic-cbf", name=run_name, config=config)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CBFNet(in_dim=config["in_dim"], hidden=config["hidden"]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

X_safe_d = X_safe.to(device)
X_unsafe_d = X_unsafe.to(device)

best_acc = 0.0
for epoch in range(config["epochs"]):
    idx_s = torch.randint(0, X_safe_d.size(0), (config["batch_size"],), device=device)
    idx_u = torch.randint(0, X_unsafe_d.size(0), (config["batch_size"],), device=device)

    h_s = model(X_safe_d[idx_s])
    h_u = model(X_unsafe_d[idx_u])
    loss, ls, lu = cbf_loss(h_s, h_u, margin=config["margin"])

    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        acc_s = (model(X_safe_d) > 0).float().mean().item()
        acc_u = (model(X_unsafe_d) < 0).float().mean().item()
    acc = 0.5 * (acc_s + acc_u)

    wandb.log({
        "epoch": epoch,
        "loss": loss.item(),
        "loss_safe": ls,
        "loss_unsafe": lu,
        "acc_safe": acc_s,
        "acc_unsafe": acc_u,
        "acc": acc,
    })

    if (epoch + 1) % 20 == 0:
        print(f"ep {epoch+1:4d} | loss {loss.item():.4f} (safe {ls:.4f}, unsafe {lu:.4f}) | acc safe {acc_s:.3f} unsafe {acc_u:.3f}")

    if acc > best_acc:
        best_acc = acc
        save_checkpoint(ckpt_dir / "best.pt", model, opt, epoch, config, acc)

save_checkpoint(ckpt_dir / "last.pt", model, opt, config["epochs"] - 1, config, acc)
wandb.save(str(ckpt_dir / "best.pt"))
wandb.finish()
print(f"checkpoints saved to {ckpt_dir} (best acc {best_acc:.3f})")
