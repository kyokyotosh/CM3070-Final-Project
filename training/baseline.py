"""Hand-crafted geometric baseline for the action recogniser.

The learned model has to earn its place. This baseline classifies a window
from a handful of interpretable measurements, using the same windows, the same
normalisation and the same cross-validation as the network. If the network
does not beat it, that is a result worth reporting rather than hiding.

The features are deliberately the ones a person would name when asked how to
tell these movements apart: how far apart the feet are, how much the knees
bend, and how much the hips travel.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Indices into the COCO-17 layout built by dataset.JOINTS. These must be
# kept in step with that layout: the same integers point at different joints
# in a different keypoint order, and wrong indices still produce plausible
# looking accuracy from meaningless features.
NOSE = 0
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def _angle(a, b, c):
    """Angle at b, per frame, for arrays of shape (T, 2)."""
    ba, bc = a - b, c - b
    cos = ((ba * bc).sum(-1)
           / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def features(window):
    """Reduce one window to a fixed-length feature vector."""
    xy = window[:, :, :2]

    ankle_gap = np.abs(xy[:, L_ANKLE, 0] - xy[:, R_ANKLE, 0])
    left_knee = _angle(xy[:, L_HIP], xy[:, L_KNEE], xy[:, L_ANKLE])
    right_knee = _angle(xy[:, R_HIP], xy[:, R_KNEE], xy[:, R_ANKLE])
    knee_gap = np.abs(left_knee - right_knee)
    hip_y = (xy[:, L_HIP, 1] + xy[:, R_HIP, 1]) / 2.0
    shoulder = (xy[:, L_SH] + xy[:, R_SH]) / 2.0
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0
    trunk = np.degrees(np.arctan2(shoulder[:, 0] - hip[:, 0],
                                  -(shoulder[:, 1] - hip[:, 1])))
    ankle_y_gap = np.abs(xy[:, L_ANKLE, 1] - xy[:, R_ANKLE, 1])

    both = np.minimum(left_knee, right_knee)
    return np.array([
        ankle_gap.mean(), ankle_gap.std(), ankle_gap.max(),
        ankle_y_gap.mean(), ankle_y_gap.max(),
        both.min(), both.mean(), both.max(), both.std(),
        knee_gap.mean(), knee_gap.max(),
        hip_y.min(), hip_y.max(), hip_y.max() - hip_y.min(), hip_y.std(),
        np.abs(trunk).mean(), np.abs(trunk).max(),
        window[:, :, 2].mean(),
    ], dtype=np.float32)


def featurise(X):
    return np.stack([features(w) for w in X])


def fit_predict(X_train, y_train, X_test, seed=0):
    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    model.fit(featurise(X_train), y_train)
    return model.predict(featurise(X_test))
