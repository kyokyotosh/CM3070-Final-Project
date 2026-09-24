"""Training-time augmentation for single-participant skeleton data.

Every recording comes from one person in one room, so the main risk is
memorising that person's framing and tempo rather than learning the movement.
These transforms attack exactly that: they vary facing, apparent size,
position in frame and keypoint confidence, none of which should change which
exercise is being performed.

Mirroring is legitimate here because the pipeline downstream is already
facing-invariant: the lunge analyser identifies the front leg from the ankle
furthest from the hip centre line, in either direction.

Magnitudes are scaled to the coordinate frame in use. In "image" mode
coordinates span roughly [-1, 1] across the frame, so a 0.1 shift is a
twentieth of the frame; in "hip" mode the unit is one torso length and the
same number means something different.
"""

import numpy as np

# Left and right joint indices swapped, in the COCO-17 order used by
# dataset.JOINTS. Mirroring coordinates without swapping these would produce
# an anatomically impossible skeleton, with a left knee on the right side.
FLIP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


def augment(window, rng, mode="image"):
    """Return one randomly transformed copy of a window."""
    out = window.copy()
    unit = 1.0 if mode == "image" else 0.5

    # Facing: mirror the image plane and swap left for right.
    if rng.random() < 0.5:
        out = out[:, FLIP]
        out[:, :, 0] = -out[:, :, 0]

    # Apparent size: the person standing nearer or further than in training.
    out[:, :, :2] *= rng.uniform(0.85, 1.15)

    # Position in frame.
    out[:, :, 0] += rng.uniform(-0.12, 0.12) * unit
    out[:, :, 1] += rng.uniform(-0.12, 0.12) * unit

    # Small rotation, for a camera that is not perfectly level.
    theta = np.radians(rng.uniform(-8, 8))
    cos, sin = np.cos(theta), np.sin(theta)
    x, y = out[:, :, 0].copy(), out[:, :, 1].copy()
    out[:, :, 0] = cos * x - sin * y
    out[:, :, 1] = sin * x + cos * y

    # Keypoint jitter and confidence dropout, standing in for the tracking
    # failures seen in dim and backlit conditions. The recorded data already
    # shows the occluded far leg sitting near 0.4 confidence.
    out[:, :, :2] += rng.normal(0, 0.015 * unit, out[:, :, :2].shape)
    if rng.random() < 0.3:
        dropped = rng.random(out.shape[1]) < 0.1
        out[:, dropped, 2] *= 0.3

    return out.astype(np.float32)
