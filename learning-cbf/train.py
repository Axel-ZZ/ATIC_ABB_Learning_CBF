"""
NN CBF training. Consumes a stage-2 dataset by tag.

Usage:
    python train.py --dataset <ds_tag>
        [--hidden 64] [--lr 1e-3] [--epochs 200] [--margin 0.1] [--batch 512]
        [--no-wandb]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
import pipeline_io as pio  # noqa: E402


class CBFNet(nn.Module):
    def __init__(self, in_dim=3, hidden=64):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="stage-2 dataset tag")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    ds_dir = pio.resolve_existing(pio.STAGE_DATA, args.dataset)
    X_safe_df = pd.read_csv(ds_dir / "safe.csv")
    X_unsafe_df = pd.read_csv(ds_dir / "unsafe.csv")
    X_safe = torch.tensor(X_safe_df.values, dtype=torch.float32)
    X_unsafe = torch.tensor(X_unsafe_df.values, dtype=torch.float32)
    print(f"[stage 3/nn] parent dataset = {args.dataset}")
    print(f"  X_safe   : {tuple(X_safe.shape)}")
    print(f"  X_unsafe : {tuple(X_unsafe.shape)}")

    CFG = dict(
        method="nn", dataset_tag=args.dataset,
        in_dim=X_safe.shape[1], hidden=args.hidden,
        lr=args.lr, batch_size=args.batch,
        epochs=args.epochs, margin=args.margin,
    )
    human = f"h{args.hidden}_lr{args.lr:g}_e{args.epochs}"
    TAG = pio.make_tag(human, CFG)
    OUT = pio.run_dir(pio.STAGE_CBF, TAG, method="nn")
    print(f"  tag = {TAG}\n  out = {OUT}")

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="atic-cbf", name=TAG, config=CFG, dir=str(OUT))
        except Exception as e:
            print(f"(wandb disabled: {e})")
            use_wandb = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CBFNet(in_dim=CFG["in_dim"], hidden=CFG["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    X_safe_d = X_safe.to(device)
    X_unsafe_d = X_unsafe.to(device)

    t0 = time.monotonic()
    best_acc = 0.0
    history = []
    for epoch in range(CFG["epochs"]):
        idx_s = torch.randint(0, X_safe_d.size(0), (CFG["batch_size"],), device=device)
        idx_u = torch.randint(0, X_unsafe_d.size(0), (CFG["batch_size"],), device=device)
        h_s = model(X_safe_d[idx_s])
        h_u = model(X_unsafe_d[idx_u])
        loss, ls, lu = cbf_loss(h_s, h_u, margin=CFG["margin"])
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            acc_s = (model(X_safe_d) > 0).float().mean().item()
            acc_u = (model(X_unsafe_d) < 0).float().mean().item()
        acc = 0.5 * (acc_s + acc_u)
        history.append(dict(epoch=epoch, loss=loss.item(),
                            loss_safe=ls, loss_unsafe=lu,
                            acc_safe=acc_s, acc_unsafe=acc_u, acc=acc))
        if use_wandb:
            wandb.log(history[-1])
        if (epoch + 1) % 20 == 0:
            print(f"  ep {epoch+1:4d} | loss {loss.item():.4f} | "
                  f"acc safe {acc_s:.3f} unsafe {acc_u:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "config": CFG, "acc": acc}, OUT / "model_best.pt")

    torch.save({"epoch": CFG["epochs"] - 1, "model_state": model.state_dict(),
                "config": CFG, "acc": acc}, OUT / "model_last.pt")
    runtime_s = time.monotonic() - t0

    metrics = dict(best_acc=best_acc, final_acc=acc,
                   final_acc_safe=acc_s, final_acc_unsafe=acc_u,
                   final_loss=loss.item(), runtime_s=runtime_s,
                   n_epochs=CFG["epochs"])
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "history.json").write_text(json.dumps(history))
    pio.save_config(OUT, CFG)
    pio.save_meta(OUT, metrics)
    pio.append_index_row(pio.STAGE_CBF, dict(
        tag=TAG, method="nn", dataset_tag=args.dataset,
        hidden=args.hidden, lr=args.lr, epochs=args.epochs, margin=args.margin,
        best_acc=f"{best_acc:.4f}", final_acc=f"{acc:.4f}",
        runtime_s=f"{runtime_s:.1f}", path=str(OUT),
    ))
    if use_wandb:
        wandb.finish()
    print(f"\n[done] best_acc={best_acc:.3f}  → {OUT}")


if __name__ == "__main__":
    main()
