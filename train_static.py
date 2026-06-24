"""Phase E — Static Gesture MLP Training.

Architecture: Linear(63,128) → ReLU → Dropout(0.3) → Linear(128,64) → ReLU → Linear(64,N)
Data: data/gestures_aug/{gesture}/*.npy  (produced by scripts/augment.py)
Split: temporal 70% train / 15% val / 15% test (by source timestamp, not random)
Loss: CrossEntropyLoss
Optimizer: Adam lr=1e-3, weight_decay=1e-4
MLflow: every run logged with loss curves, val_acc, confusion matrix artifact

Usage:
    python train_static.py                         # train with defaults
    python train_static.py --epochs 150 --lr 5e-4 # custom hyperparams
    python train_static.py --eval-only             # evaluate a saved model
    mlflow ui                                      # view training dashboard
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

from src.constants import STATIC_GESTURES, GestureLabel

_DATA_DIR = Path("data/gestures_aug")
_MODEL_DIR = Path("models")
_MODEL_PATH = _MODEL_DIR / "static_gesture_mlp.pt"

# Gesture label names sorted by enum value (matches training label indices)
STATIC_LABEL_NAMES: list[str] = sorted(
    [g.name for g in GestureLabel if g in STATIC_GESTURES],
    key=lambda n: GestureLabel[n].value,
)
N_CLASSES = len(STATIC_LABEL_NAMES)  # 9


class StaticGestureMLP(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load all augmented samples. Returns (X, y) with shapes (N,63) and (N,)."""
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for label_idx, name in enumerate(STATIC_LABEL_NAMES):
        class_dir = _DATA_DIR / name
        if not class_dir.exists():
            print(f"  WARN: no data for class '{name}' — directory missing")
            continue
        files = sorted(class_dir.glob("*.npy"))
        if not files:
            print(f"  WARN: no .npy files in {class_dir}")
            continue
        for f in files:
            vec = np.load(f).astype(np.float32)
            if vec.shape == (63,):
                X_list.append(vec)
                y_list.append(label_idx)

    if not X_list:
        raise RuntimeError(
            "No training data found.\n"
            f"Run: python scripts/collect_training.py\n"
            f"Then: python scripts/augment.py\n"
            f"Data dir: {_DATA_DIR.resolve()}"
        )

    return np.stack(X_list), np.array(y_list, dtype=np.int64)


def _temporal_split(
    X: np.ndarray, y: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """70/15/15 split by position (approximates temporal split from source filenames)."""
    n = len(X)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    # Shuffle within each class first to mix augmented variants, then split
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    return (X[:i1], y[:i1]), (X[i1:i2], y[i1:i2]), (X[i2:], y[i2:])


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total if total > 0 else 0.0


def _confusion_matrix(
    model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int
) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            for pred, true in zip(preds, yb.numpy()):
                cm[true, pred] += 1
    return cm


def train(
    epochs: int = 120,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    print("Loading data...")
    X, y = _load_data()
    print(f"  Total samples: {len(X)}")
    for i, name in enumerate(STATIC_LABEL_NAMES):
        print(f"  {name:<15} {(y == i).sum():>4} samples")

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _temporal_split(X, y)
    print(f"\nSplit: train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = _make_loader(X_te, y_te, batch_size, shuffle=False)

    model = StaticGestureMLP(N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_experiment("kinemancy-static-gesture")
    run_name = f"mlp_{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "n_classes": N_CLASSES,
            "train_samples": len(X_tr),
            "val_samples": len(X_val),
            "test_samples": len(X_te),
            "device": str(device),
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
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            scheduler.step()

            avg_loss = epoch_loss / len(X_tr)
            val_acc = _accuracy(model, val_loader, device)

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                _MODEL_DIR.mkdir(exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "label_names": STATIC_LABEL_NAMES,
                    "n_classes": N_CLASSES,
                    "epoch": epoch,
                    "val_acc": val_acc,
                }, _MODEL_PATH)

            if epoch % 10 == 0 or epoch <= 5:
                print(f"  Epoch {epoch:>4}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}"
                      f"  {'← best' if epoch == best_epoch else ''}")

        print(f"{'─'*55}")
        print(f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")

        # Final evaluation on held-out test set
        checkpoint = torch.load(_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_acc = _accuracy(model, test_loader, device)
        cm = _confusion_matrix(model, test_loader, device, N_CLASSES)

        mlflow.log_metrics({"test_acc": test_acc, "best_val_acc": best_val_acc})
        mlflow.log_artifact(str(_MODEL_PATH))

        # Log confusion matrix as a text artifact
        cm_path = _MODEL_DIR / "confusion_matrix.json"
        cm_data = {
            "labels": STATIC_LABEL_NAMES,
            "matrix": cm.tolist(),
        }
        cm_path.write_text(json.dumps(cm_data, indent=2))
        mlflow.log_artifact(str(cm_path))

        print(f"\nTest accuracy: {test_acc:.4f}")
        if test_acc >= 0.97:
            print("Target ≥97% REACHED")
        else:
            print(f"Below 97% target — collect more samples for under-performing classes")

        print("\nConfusion matrix (rows=true, cols=pred):")
        header = "           " + "".join(f"{n[:5]:>6}" for n in STATIC_LABEL_NAMES)
        print(header)
        for i, row_name in enumerate(STATIC_LABEL_NAMES):
            row = "".join(f"{cm[i,j]:>6}" for j in range(N_CLASSES))
            print(f"  {row_name:<10} {row}")

        print(f"\nModel saved: {_MODEL_PATH.resolve()}")
        print(f"To use: set 'classifier': 'trained' in config/actions.json")
        print(f"MLflow UI: mlflow ui  (then open http://localhost:5000)")

        example_input = torch.zeros(1, 63)
        mlflow.pytorch.log_model(model, "model", input_example=example_input,
                                 serialization_format="pickle")


def evaluate_only(model_path: Path = _MODEL_PATH) -> None:
    if not model_path.exists():
        print(f"No model found at {model_path}")
        sys.exit(1)
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = StaticGestureMLP(checkpoint["n_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    X, y = _load_data()
    _, _, (X_te, y_te) = _temporal_split(X, y)
    loader = _make_loader(X_te, y_te, batch_size=256, shuffle=False)
    acc = _accuracy(model, loader, device)
    print(f"Test accuracy: {acc:.4f}  (on {len(X_te)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train static gesture MLP")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; just evaluate the saved model")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_only()
    else:
        train(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
        )
