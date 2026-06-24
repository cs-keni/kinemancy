"""Phase F — Dynamic Gesture LSTM Training.

Architecture: LSTM(input=126, hidden=128, num_layers=2, dropout=0.3) → FC(128,64) → FC(64,N)
Data: data/dynamic_gestures_aug/{gesture}/*.npy  (T=30, D=126 float32 sequences)
Split: 70/15/15 by shuffled index within each class
Loss: CrossEntropyLoss
Optimizer: Adam lr=5e-4, weight_decay=1e-4
MLflow: loss curves, val_acc, confusion matrix artifact

Usage:
    python train_dynamic.py
    python train_dynamic.py --epochs 200 --lr 3e-4
    python train_dynamic.py --eval-only
    mlflow ui   → http://localhost:5000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.constants import DYNAMIC_GESTURES, GestureLabel

_DATA_DIR = Path("data/dynamic_gestures_aug")
_MODEL_DIR = Path("models")
_MODEL_PATH = _MODEL_DIR / "dynamic_gesture_lstm.pt"

DYNAMIC_LABEL_NAMES: list[str] = sorted(
    [g.name for g in GestureLabel if g in DYNAMIC_GESTURES],
    key=lambda n: GestureLabel[n].value,
)
N_CLASSES = len(DYNAMIC_LABEL_NAMES)   # 7
WINDOW = 30
INPUT_SIZE = 126


class DynamicGestureLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=0.3 if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, input_size)
        out, _ = self.lstm(x)    # (batch, T, hidden)
        return self.fc(out[:, -1, :])  # classify on last timestep


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load all augmented sequences. Returns X (N, 30, 126) and y (N,)."""
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for label_idx, name in enumerate(DYNAMIC_LABEL_NAMES):
        class_dir = _DATA_DIR / name
        if not class_dir.exists():
            print(f"  WARN: no data for '{name}' — directory missing")
            continue
        files = sorted(class_dir.glob("*.npy"))
        if not files:
            print(f"  WARN: no .npy files in {class_dir}")
            continue
        for f in files:
            seq = np.load(f).astype(np.float32)
            if seq.shape == (WINDOW, INPUT_SIZE):
                X_list.append(seq)
                y_list.append(label_idx)

    if not X_list:
        raise RuntimeError(
            "No training data found.\n"
            "Run: python scripts/collect_dynamic.py\n"
            "Then: python scripts/augment_dynamic.py"
        )

    return np.stack(X_list), np.array(y_list, dtype=np.int64)


def _temporal_split(
    X: np.ndarray, y: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n = len(X)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    return (X[:i1], y[:i1]), (X[i1:i2], y[i1:i2]), (X[i2:], y[i2:])


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total if total > 0 else 0.0


def _confusion_matrix(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            for p, t in zip(preds, yb.numpy()):
                cm[t, p] += 1
    return cm


def train(
    epochs: int = 150,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    print("Loading data...")
    X, y = _load_data()
    print(f"  Total sequences: {len(X)}")
    for i, name in enumerate(DYNAMIC_LABEL_NAMES):
        print(f"  {name:<15} {(y == i).sum():>4} samples")

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _temporal_split(X, y)
    print(f"\nSplit: train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader  = _make_loader(X_te,  y_te,  batch_size, shuffle=False)

    model = DynamicGestureLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_experiment("kinemancy-dynamic-gesture")
    run_name = f"lstm_{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
            "batch_size": batch_size, "n_classes": N_CLASSES,
            "train_samples": len(X_tr), "val_samples": len(X_val),
            "test_samples": len(X_te), "device": str(device),
            "hidden_size": 128, "num_layers": 2, "input_size": INPUT_SIZE,
        })

        best_val_acc = 0.0
        best_epoch = 0

        print(f"\nTraining {epochs} epochs...\n{'─'*55}")
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            scheduler.step()

            avg_loss = epoch_loss / len(X_tr)
            val_acc = _accuracy(model, val_loader, device)

            mlflow.log_metrics({"train_loss": avg_loss, "val_acc": val_acc,
                                 "lr": scheduler.get_last_lr()[0]}, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                _MODEL_DIR.mkdir(exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "label_names": DYNAMIC_LABEL_NAMES,
                    "n_classes": N_CLASSES,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "hidden_size": 128,
                    "num_layers": 2,
                    "input_size": INPUT_SIZE,
                    "window": WINDOW,
                }, _MODEL_PATH)

            if epoch % 10 == 0 or epoch <= 5:
                best_marker = " ← best" if epoch == best_epoch else ""
                print(f"  Epoch {epoch:>4}/{epochs}  loss={avg_loss:.4f}"
                      f"  val_acc={val_acc:.4f}{best_marker}")

        print(f"{'─'*55}")
        print(f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")

        # Final evaluation
        checkpoint = torch.load(_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        test_acc = _accuracy(model, test_loader, device)
        cm = _confusion_matrix(model, test_loader, device)

        mlflow.log_metrics({"test_acc": test_acc, "best_val_acc": best_val_acc})
        mlflow.log_artifact(str(_MODEL_PATH))

        cm_path = _MODEL_DIR / "confusion_matrix_dynamic.json"
        cm_path.write_text(json.dumps({
            "labels": DYNAMIC_LABEL_NAMES, "matrix": cm.tolist()
        }, indent=2))
        mlflow.log_artifact(str(cm_path))

        print(f"\nTest accuracy: {test_acc:.4f}")
        target = 0.90
        if test_acc >= target:
            print(f"Target ≥{int(target*100)}% REACHED")
        else:
            miss = [DYNAMIC_LABEL_NAMES[i] for i in range(N_CLASSES)
                    if cm[i].sum() > 0 and cm[i, i] / cm[i].sum() < 0.90]
            print(f"Below target — under-performing: {miss}")
            print("Collect 10+ more samples for those classes, re-augment, re-train.")

        print("\nConfusion matrix (rows=true, cols=pred):")
        header = "           " + "".join(f"{n[:6]:>7}" for n in DYNAMIC_LABEL_NAMES)
        print(header)
        for i, row_name in enumerate(DYNAMIC_LABEL_NAMES):
            row = "".join(f"{cm[i,j]:>7}" for j in range(N_CLASSES))
            print(f"  {row_name:<10} {row}")

        print(f"\nModel saved: {_MODEL_PATH.resolve()}")
        print("To use: set 'dynamic_classifier': 'trained' in config/actions.json")

        example_input = torch.zeros(1, WINDOW, INPUT_SIZE)
        mlflow.pytorch.log_model(model, "model", input_example=example_input,
                                 serialization_format="pickle")


def evaluate_only() -> None:
    if not _MODEL_PATH.exists():
        print(f"No model at {_MODEL_PATH}")
        sys.exit(1)
    device = torch.device("cpu")
    ckpt = torch.load(_MODEL_PATH, map_location=device, weights_only=False)
    model = DynamicGestureLSTM(n_classes=ckpt["n_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    X, y = _load_data()
    _, _, (X_te, y_te) = _temporal_split(X, y)
    loader = _make_loader(X_te, y_te, 256, shuffle=False)
    acc = _accuracy(model, loader, device)
    print(f"Test accuracy: {acc:.4f}  ({len(X_te)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.eval_only:
        evaluate_only()
    else:
        train(args.epochs, args.lr, args.weight_decay, args.batch_size)
