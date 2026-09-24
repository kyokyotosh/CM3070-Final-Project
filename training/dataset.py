"""Dataset loading and windowing for the action recogniser.

Sessions recorded by frontend/record.html are read from disk and cut into
fixed-length windows. Two rules are enforced here rather than left to the
training script, because getting either wrong invalidates the reported
accuracy:

  1. A window never spans two sessions.
  2. Every window carries its source session id, so splits can be made by
     session. Splitting by window would place near-identical, overlapping
     samples on both sides and report memorisation of one recording rather
     than generalisation to a new one.

MediaPipe's 33 landmarks are reduced to the 17 COCO keypoints, which is the
layout the pre-trained checkpoint was trained on. Every one of the 17 maps
directly from a MediaPipe landmark, with no synthesised joints, so no
topology is invented at the boundary between the two models.
"""

import glob
import json
import os

import numpy as np

CLASSES = ["squat", "lunge", "other"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}

CAPTURE_HZ = 15

# COCO-17 keypoint order, given as the MediaPipe index supplying each one.
# COCO index:      0  1  2  3  4   5   6   7   8   9  10  11  12  13  14  15  16
# COCO joint:   nose lE rE lEar rEar lSh rSh lEl rEl lWr rWr lHip rHip lKn rKn lAn rAn
JOINTS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
NUM_JOINTS = len(JOINTS)
NUM_CHANNELS = 3        # x, y, confidence

# Indices within the COCO-17 layout, for the normalisation below.
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def load_session(path):
    """Read one recording into (frames, metadata).

    frames has shape (T, NUM_JOINTS, NUM_CHANNELS).
    """
    with open(path) as f:
        data = json.load(f)

    raw = np.asarray([frame["lm"] for frame in data["frames"]], dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1] != 33:
        raise ValueError(f"{path}: expected 33 landmarks per frame, got {raw.shape}")

    # Keep x, y and visibility. The pre-trained model's third channel is a
    # keypoint confidence, which is what visibility is, so the channel
    # meanings line up. z from a single camera is a weak estimate and is
    # dropped rather than given equal weight with the image-plane
    # coordinates the form analysis also relies on.
    frames = raw[:, JOINTS][:, :, [0, 1, 3]]

    meta = {
        "session_id": data["session_id"],
        "label": data["label"],
        "distance": data.get("distance", "unknown"),
        "lighting": data.get("lighting", "unknown"),
        "frame_count": len(frames),
    }
    return frames, meta


def normalise(window, mode="image"):
    """Put a window into the coordinate frame the model expects.

    Two modes, because the choice is a real trade-off worth measuring rather
    than assuming:

    "image" reproduces the preprocessing the pre-trained checkpoint was
    trained with: coordinates centred on the image and scaled to roughly
    [-1, 1]. MediaPipe already returns normalised coordinates, so this is a
    shift and a scale. Matching the checkpoint's input distribution is what
    makes its frozen early layers useful.

    "hip" centres each frame on the hip midpoint and scales by torso length,
    which makes a window invariant to where the person stands and how large
    they appear. It generalises better across camera distance but moves the
    input away from what the pre-trained layers saw.

    Compare the two under cross-validation and report which transferred
    better; do not assume.
    """
    out = window.copy()
    xy = out[:, :, :2]

    if mode == "image":
        out[:, :, :2] = (xy - 0.5) * 2.0
        return out

    if mode == "hip":
        hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0
        shoulder = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2.0
        torso = np.linalg.norm(shoulder - hip, axis=1)
        scale = float(np.median(torso))
        if scale < 1e-3:
            scale = 1e-3
        out[:, :, :2] = (xy - hip[:, None, :]) / scale
        return out

    raise ValueError(f"unknown normalisation mode: {mode}")


def window_session(frames, window, stride, mode="image"):
    """Cut one session into overlapping windows of fixed length."""
    if len(frames) < window:
        return np.empty((0, window, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    starts = range(0, len(frames) - window + 1, stride)
    return np.stack([normalise(frames[s:s + window], mode) for s in starts])


def load_dataset(directory, window=30, stride=5, mode="image"):
    """Load every recording in a directory and return windowed arrays.

    Returns X of shape (N, window, NUM_JOINTS, NUM_CHANNELS), y of class
    indices, sessions of source session ids, and per-session metadata.
    """
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not paths:
        raise FileNotFoundError(f"no recordings found in {directory}")

    xs, ys, sessions, metas = [], [], [], []
    for path in paths:
        frames, meta = load_session(path)
        if meta["label"] not in CLASS_TO_INDEX:
            print(f"skipping {meta['session_id']}: unknown label {meta['label']}")
            continue
        windows = window_session(frames, window, stride, mode)
        if len(windows) == 0:
            print(f"skipping {meta['session_id']}: shorter than one window")
            continue
        xs.append(windows)
        ys.append(np.full(len(windows), CLASS_TO_INDEX[meta["label"]],
                          dtype=np.int64))
        sessions.extend([meta["session_id"]] * len(windows))
        meta["windows"] = len(windows)
        metas.append(meta)

    return (np.concatenate(xs), np.concatenate(ys),
            np.asarray(sessions), metas)


def summarise(y, metas):
    lines = [f"{len(y)} windows from {len(metas)} sessions"]
    for index, name in enumerate(CLASSES):
        count = int((y == index).sum())
        n_sessions = sum(1 for m in metas if m["label"] == name)
        lines.append(f"  {name:<6} {count:>5} windows  {n_sessions} sessions  "
                     f"{count / len(y) * 100:>5.1f}%")
    return "\n".join(lines)
