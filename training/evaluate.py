"""Leave-one-session-out cross-validation for the action recogniser.

With fourteen recordings across three classes, a single held-out split would
test on three sessions and report an accuracy with a very wide interval. Every
session is instead held out exactly once, so each prediction is made on a
recording the model never saw, and the confusion matrix pools predictions over
all fourteen sessions.

The split is by session, never by window. Windows overlap by design, so
neighbouring windows from one recording are near-identical; separating them
across a split reports memorisation rather than generalisation.

fit_predict is passed in, so the same harness evaluates any model. It takes
(X_train, y_train, X_test) and returns predicted labels for X_test.
"""

import numpy as np

from dataset import CLASSES


def confusion(y_true, y_pred, n_classes=len(CLASSES)):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def per_class_report(matrix):
    """Precision, recall and F1 per class from a confusion matrix."""
    rows = []
    for i, name in enumerate(CLASSES):
        tp = matrix[i, i]
        predicted = matrix[:, i].sum()
        actual = matrix[i, :].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        rows.append({"class": name, "precision": precision, "recall": recall,
                     "f1": f1, "support": int(actual)})
    return rows


def format_confusion(matrix):
    header = "actual\\predicted " + "".join(f"{c:>9}" for c in CLASSES)
    lines = [header]
    for i, name in enumerate(CLASSES):
        lines.append(f"{name:<17}" + "".join(f"{v:>9}" for v in matrix[i]))
    return "\n".join(lines)


def format_report(rows, accuracy):
    lines = [f"{'class':<8}{'precision':>11}{'recall':>9}{'f1':>8}{'support':>9}"]
    for r in rows:
        lines.append(f"{r['class']:<8}{r['precision']:>11.3f}"
                     f"{r['recall']:>9.3f}{r['f1']:>8.3f}{r['support']:>9}")
    macro = np.mean([r["f1"] for r in rows])
    lines.append(f"\naccuracy {accuracy:.3f}   macro F1 {macro:.3f}")
    return "\n".join(lines)


def leave_one_session_out(X, y, sessions, fit_predict, verbose=True):
    """Run the cross-validation and return (confusion matrix, per-fold rows)."""
    unique = sorted(set(sessions))
    all_true, all_pred, folds = [], [], []

    for held_out in unique:
        test_mask = sessions == held_out
        train_mask = ~test_mask

        y_pred = fit_predict(X[train_mask], y[train_mask], X[test_mask])
        y_true = y[test_mask]

        correct = int((y_pred == y_true).sum())
        accuracy = correct / len(y_true)
        majority = int(np.bincount(y_pred, minlength=len(CLASSES)).argmax())

        folds.append({
            "session": held_out,
            "true_class": CLASSES[y_true[0]],
            "windows": len(y_true),
            "accuracy": accuracy,
            "session_vote": CLASSES[majority],
            "vote_correct": bool(majority == y_true[0]),
        })

        all_true.append(y_true)
        all_pred.append(y_pred)

        if verbose:
            mark = "ok " if folds[-1]["vote_correct"] else "XX "
            print(f"  {mark}{held_out:<28} {folds[-1]['true_class']:<6} "
                  f"window acc {accuracy:>6.3f}  session vote "
                  f"{folds[-1]['session_vote']}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    matrix = confusion(y_true, y_pred)
    accuracy = float((y_true == y_pred).mean())
    return matrix, folds, accuracy
