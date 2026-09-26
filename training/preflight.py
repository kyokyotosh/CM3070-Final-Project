"""Checks to run before a full cross-validation.

  1. The graph computed by graph.py matches the adjacency in the checkpoint.
  2. A forward pass on real windows gives finite logits, and the
     frozen/trainable split is as requested.
  3. How long one fold takes on this machine.

Usage:
    python3 preflight.py --checkpoint stgcn_ntu60.pth --data ../training/sessions
"""

import argparse
import time

import numpy as np
import torch

from dataset import CLASSES, load_dataset, summarise
from graph import adjacency
from model import ExerciseRecogniser, count_parameters, load_pretrained


def check_graph(model, checkpoint_path):
    """Compare the computed adjacency with the checkpoint's own."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)

    keys = [k for k in state if k.endswith("gcn.A")]
    if not keys:
        print("  no adjacency buffer in the checkpoint; cannot compare")
        return

    theirs = state[keys[0]].numpy()
    ours = adjacency()

    print(f"  checkpoint adjacency {tuple(theirs.shape)} from {keys[0]}")
    print(f"  computed adjacency   {tuple(ours.shape)}")

    if theirs.shape != ours.shape:
        print("  SHAPE MISMATCH: graph.py does not describe this checkpoint")
        return

    difference = float(np.abs(theirs - ours).max())
    if difference < 1e-5:
        print(f"  identical (max difference {difference:.2e}) "
              "- graph.py reproduces the published graph")
    else:
        print(f"  DIFFERENT (max difference {difference:.4f})")
        print("  Training still works, because the checkpoint's buffer is "
              "loaded over the computed one,")
        print("  but graph.py is not describing this checkpoint and the "
              "report should not claim it does.")

    # Are all the blocks' adjacencies the same? They should be.
    if len({state[k].numpy().tobytes() for k in keys}) == 1:
        print(f"  all {len(keys)} blocks share one adjacency, as expected")
    else:
        print(f"  WARNING: the {len(keys)} blocks do not share one adjacency")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="../recordings")
    p.add_argument("--freeze", type=int, default=8)
    p.add_argument("--norm", choices=("image", "hip"), default="image")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    print("1. DATA")
    X, y, sessions, metas = load_dataset(args.data, args.window, args.stride,
                                         args.norm)
    print("  " + summarise(y, metas).replace("\n", "\n  "))
    held_out = len(set(sessions))
    smallest = min(m["windows"] for m in metas)
    print(f"  {held_out} folds; smallest session contributes {smallest} "
          f"test windows")

    print("\n2. GRAPH")
    model = ExerciseRecogniser(adjacency(), num_classes=len(CLASSES))
    check_graph(model, args.checkpoint)

    print("\n3. CHECKPOINT")
    report = load_pretrained(model, args.checkpoint)

    print("\n4. FREEZING")
    model.freeze_stages(args.freeze)
    trainable = count_parameters(model, True)
    total = count_parameters(model, False)
    print(f"  {args.freeze} of 10 blocks frozen")
    print(f"  {trainable:,} trainable of {total:,} "
          f"({trainable / total * 100:.1f}% adapting)")
    if trainable == total:
        print("  WARNING: nothing is frozen")
    frozen_bn = sum(1 for m in model.modules()
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
                    and not any(p.requires_grad for p in m.parameters()))
    print(f"  {frozen_bn} batch-norm layers held in evaluation mode")

    print("\n5. FORWARD PASS")
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    model = model.to(dev).eval()

    batch = torch.from_numpy(np.ascontiguousarray(X[:64])).to(dev)
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(batch)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"  device {dev} | input {tuple(batch.shape)} -> "
          f"output {tuple(logits.shape)} in {elapsed:.0f} ms")
    print(f"  logit range {logits.min().item():.2f} to "
          f"{logits.max().item():.2f}")
    if not torch.isfinite(logits).all():
        print("  PROBLEM: non-finite logits")
    predicted = logits.argmax(1).cpu().numpy()
    spread = np.bincount(predicted, minlength=3)
    print(f"  untrained head predicts {dict(zip(CLASSES, spread.tolist()))} "
          "(any split is fine; the head is random)")

    print("\n6. ONE TRAINING EPOCH, TIMED")
    from train import WindowDataset, class_weights
    from torch.utils.data import DataLoader

    mask = sessions != sorted(set(sessions))[0]
    loader = DataLoader(WindowDataset(X[mask], y[mask], True, args.norm, 0),
                        batch_size=args.batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights(y[mask]).to(dev), label_smoothing=0.05)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=5e-4)

    model.train()
    started = time.perf_counter()
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        optimiser.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimiser.step()
    epoch_s = time.perf_counter() - started

    fold_s = epoch_s * args.epochs
    print(f"  one epoch over {int(mask.sum())} windows: {epoch_s:.1f} s "
          f"({len(loader)} steps)")
    print(f"  one fold at {args.epochs} epochs: about {fold_s / 60:.1f} min")
    print(f"  {held_out} folds: about {fold_s * held_out / 60:.0f} min")
    if fold_s * held_out > 3600:
        print("  Over an hour. Consider --epochs 15, or --stride 10 to halve "
              "the windows,")
        print("  and keep the full setting for the run that goes in the "
              "report.")

    print("\nIf sections 2 to 5 are clean, start the cross-validation.")


if __name__ == "__main__":
    main()
