"""Adapt the pre-trained ST-GCN to squat, lunge and other, and evaluate it.

The checkpoint is a published ST-GCN trained on NTU60 with the COCO-17
keypoint layout. Only its classification head is replaced; every layer below
starts from the pre-trained weights, and the early blocks are frozen so that
roughly 1,600 windows from one participant adapt the task-specific layers
rather than overwrite general motion features.

Modes:

  --mode inspect  Print the checkpoint's tensors, to confirm the architecture
                  matches before training anything.

  --mode cv       Leave-one-session-out cross-validation. Sixteen folds, each
                  testing on a recording the model never saw, pooled into one
                  confusion matrix.

  --mode final    Fine-tune once on every session and save the model the
                  server will load. Run only after cross-validation, since
                  this has no held-out data and so no honest accuracy.

Typical sequence:

    python3 train.py --mode inspect --checkpoint stgcn_ntu60.pth
    python3 train.py --mode cv      --checkpoint stgcn_ntu60.pth
    python3 train.py --mode cv      --checkpoint stgcn_ntu60.pth --norm hip
    python3 train.py --mode final   --checkpoint stgcn_ntu60.pth --out recogniser.pt
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from augment import augment
from dataset import CLASSES, load_dataset, summarise
from evaluate import (format_confusion, format_report, leave_one_session_out,
                      per_class_report)
from graph import adjacency
from model import (ExerciseRecogniser, count_parameters, inspect_checkpoint,
                   load_pretrained)


class WindowDataset(Dataset):
    def __init__(self, X, y, training, mode="image", seed=0):
        self.X, self.y = X, y
        self.training = training
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = augment(self.X[i], self.rng,
                    self.mode) if self.training else self.X[i]
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])


def device():
    # Apple silicon exposes its GPU through the Metal backend; CUDA and CPU
    # are the fallbacks so the same script runs anywhere.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(y, n_classes=len(CLASSES)):
    """Inverse-frequency weights, so the larger squat class does not dominate
    the loss purely by having been recorded more."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return torch.from_numpy(counts.sum() / (n_classes * counts))


def build(args, verbose=False):
    model = ExerciseRecogniser(adjacency(), num_classes=len(CLASSES),
                               dropout=args.dropout)
    if args.checkpoint:
        load_pretrained(model, args.checkpoint, verbose=verbose)
    model.freeze_stages(args.freeze)
    return model


def train_one(X_train, y_train, args, dev, seed=0, verbose=False):
    torch.manual_seed(seed)
    model = build(args, verbose=verbose).to(dev)

    loader = DataLoader(
        WindowDataset(X_train, y_train, True, args.norm, seed),
        batch_size=args.batch_size, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train).to(dev),
                                    label_smoothing=0.05)
    # Only unfrozen parameters are handed to the optimiser, so the frozen
    # blocks cannot drift through weight decay either.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(1, len(loader)))

    model.train()
    for epoch in range(args.epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimiser.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimiser.step()
            schedule.step()

            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        if args.verbose and (epoch + 1) % 5 == 0:
            print(f"      epoch {epoch + 1:>3}  loss {loss_sum / total:.4f}  "
                  f"train acc {correct / total:.3f}")
    return model


@torch.no_grad()
def predict(model, X, dev, batch_size=128):
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(
            X[start:start + batch_size]))
        out.append(model(xb.to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="../recordings")
    p.add_argument("--mode", choices=("inspect", "cv", "final"), default="cv")
    p.add_argument("--checkpoint", default=None,
                   help="published ST-GCN .pth to adapt; omitting it trains "
                        "from random initialisation, which does not satisfy "
                        "the pre-trained model requirement")
    p.add_argument("--freeze", type=int, default=8,
                   help="number of leading blocks held fixed (of 10)")
    p.add_argument("--norm", choices=("image", "hip"), default="image",
                   help="input coordinate frame; image matches the "
                        "checkpoint's own preprocessing")
    p.add_argument("--window", type=int, default=30,
                   help="frames per window; 30 is two seconds at 15 Hz")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="kept low: this is adaptation, not training")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--out", default="recogniser.pt")
    p.add_argument("--results", default="recognition_results.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.mode == "inspect":
        if not args.checkpoint:
            p.error("--mode inspect needs --checkpoint")
        inspect_checkpoint(args.checkpoint)
        print("\nloading it into this model:")
        load_pretrained(ExerciseRecogniser(adjacency()), args.checkpoint)
        return

    if not args.checkpoint:
        print("WARNING: no --checkpoint given. The project brief requires the "
              "recogniser to be a pre-trained model, adapted. Training from "
              "random initialisation here would not meet that.\n")

    X, y, sessions, metas = load_dataset(args.data, args.window, args.stride,
                                         args.norm)
    dev = device()

    probe = build(args, verbose=True)
    print(f"\ndevice {dev} | {count_parameters(probe):,} trainable of "
          f"{count_parameters(probe, False):,} total | "
          f"{args.freeze} of 10 blocks frozen")
    print(f"normalisation {args.norm} | window {args.window} frames "
          f"({args.window / 15:.1f} s) | stride {args.stride}")
    print(summarise(y, metas) + "\n")

    if args.mode == "cv":
        started = time.perf_counter()

        def fit_predict(X_train, y_train, X_test):
            trained = train_one(X_train, y_train, args, dev)
            return predict(trained, X_test, dev)

        matrix, folds, accuracy = leave_one_session_out(X, y, sessions,
                                                        fit_predict)
        print()
        print(format_confusion(matrix))
        print()
        rows = per_class_report(matrix)
        print(format_report(rows, accuracy))
        votes = sum(f["vote_correct"] for f in folds)
        print(f"\nsession-level: {votes}/{len(folds)} sessions correctly "
              f"identified by majority vote")
        print(f"cross-validation took {time.perf_counter() - started:.0f} s")

        with open(args.results, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint, "frozen_blocks": args.freeze,
                "normalisation": args.norm, "window": args.window,
                "stride": args.stride, "epochs": args.epochs,
                "accuracy": accuracy, "confusion": matrix.tolist(),
                "classes": CLASSES, "per_class": rows, "folds": folds,
                "sessions": metas,
            }, f, indent=2)
        print(f"results written to {args.results}")

    else:
        trained = train_one(X, y, args, dev, verbose=True)
        torch.save({
            "state_dict": trained.state_dict(),
            "classes": CLASSES,
            "window": args.window,
            "capture_hz": 15,
            "normalisation": args.norm,
            "source_checkpoint": args.checkpoint,
            "frozen_blocks": args.freeze,
        }, args.out)
        print(f"\nadapted on all {len(y)} windows and saved to {args.out}")
        print("this run has no held-out data; report the cross-validation "
              "figures instead")


if __name__ == "__main__":
    main()
