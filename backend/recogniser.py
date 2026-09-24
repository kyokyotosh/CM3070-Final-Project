"""Live exercise recognition, and the gate that decides when to act on it.

Two responsibilities, kept separate because they fail differently.

`Recogniser` is inference: it holds a rolling window of landmark frames and
classifies it as squat, lunge or other. It knows nothing about the session.

`ExerciseGate` decides what the coaching system should do with a stream of
those classifications. This is where the real design work is, because a
classifier that is right 90% of the time on two-second windows still produces
a prediction that flickers, and acting on every flicker would be worse than
the manual selector it replaces. The gate requires a sustained majority at
sufficient confidence, and refuses to switch analysers in the middle of a
repetition.

The third class earns its place here. `other` does not select an analyser; it
suppresses verdicts. Sitting down or stretching flexes the knees enough to
drive a rep through the state machine, and without a way to recognise
not-exercising, the system would count those as repetitions and coach them.
Recognition is therefore a gate on the form analysis as much as a selector
for it.
"""

import collections
import os
import sys

import numpy as np
import torch

# The model definition and its graph live with the training code, so the
# architecture the server instantiates is the same file that produced the
# weights. Vendoring a copy here would let the two drift apart silently.
TRAINING_DIR = os.environ.get(
    "RECOGNISER_TRAINING_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(TRAINING_DIR))

from graph import adjacency          # noqa: E402  (after the path bootstrap)
from model import ExerciseRecogniser  # noqa: E402

DEFAULT_MODEL_PATH = os.environ.get(
    "RECOGNISER_MODEL",
    os.path.join(os.path.abspath(TRAINING_DIR), "recogniser.pt"))

# MediaPipe-33 indices supplying the COCO-17 layout, in COCO order. This MUST
# match dataset.JOINTS in the training package: the same integers mean
# different joints in a different order, and a mismatch produces confident
# nonsense rather than an error. The assertion in `Recogniser.__init__`
# catches a wrong joint count but not a wrong order, so treat this list as
# paired with the training code.
JOINTS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

CLASSES = ["squat", "lunge", "other"]
EXERCISE_CLASSES = ("squat", "lunge")


def _to_window(frames, mode):
    """Stack buffered frames into the model's input layout."""
    window = np.stack(frames)                      # (T, 17, 3)
    if mode == "image":
        window = window.copy()
        window[:, :, :2] = (window[:, :, :2] - 0.5) * 2.0
        return window
    if mode == "hip":
        window = window.copy()
        xy = window[:, :, :2]
        hip = (xy[:, 11] + xy[:, 12]) / 2.0
        shoulder = (xy[:, 5] + xy[:, 6]) / 2.0
        scale = float(np.median(np.linalg.norm(shoulder - hip, axis=1)))
        window[:, :, :2] = (xy - hip[:, None, :]) / max(scale, 1e-3)
        return window
    raise ValueError(f"unknown normalisation mode: {mode}")


class Recogniser:
    """Rolling-window classifier over the landmark stream."""

    def __init__(self, path, device=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.classes = checkpoint.get("classes", CLASSES)
        self.window_size = checkpoint.get("window", 30)
        self.mode = checkpoint.get("normalisation", "image")
        self.capture_hz = checkpoint.get("capture_hz", 15)

        if device is None:
            device = torch.device("mps") if torch.backends.mps.is_available() \
                else torch.device("cpu")
        self.device = device

        self.model = ExerciseRecogniser(adjacency(),
                                        num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()

        # The input normalisation layer is sized by channels times joints, so
        # it is a direct check that this joint list matches the trained model.
        expected = self.model.data_bn.num_features
        if expected != 3 * len(JOINTS):
            raise ValueError(
                f"model expects {expected} input features "
                f"({expected // 3} joints) but JOINTS has {len(JOINTS)}")

        self.frames = collections.deque(maxlen=self.window_size)

    def push(self, landmarks):
        """Add one frame of MediaPipe landmarks to the rolling window.

        Frames are appended at the full stream rate, not the rate the form
        analyser samples at, because the model was trained on 15 Hz windows
        and a sparser window would not match what it learned.
        """
        if not landmarks or len(landmarks) < 33:
            return False
        frame = np.asarray(
            [[landmarks[i]["x"], landmarks[i]["y"],
              landmarks[i].get("visibility", 0.0)] for i in JOINTS],
            dtype=np.float32)
        self.frames.append(frame)
        return True

    @property
    def ready(self):
        return len(self.frames) == self.window_size

    def snapshot(self):
        """The current window, ready for inference, or None if not full."""
        if not self.ready:
            return None
        return _to_window(list(self.frames), self.mode)

    @torch.no_grad()
    def classify(self, window):
        """Return (label, confidence, probabilities) for one window."""
        batch = torch.from_numpy(
            np.ascontiguousarray(window[None])).to(self.device)
        probabilities = torch.softmax(self.model(batch), dim=1)[0]
        index = int(probabilities.argmax())
        return (self.classes[index],
                float(probabilities[index]),
                {c: float(p) for c, p in zip(self.classes, probabilities)})

    def reset(self):
        self.frames.clear()


class ExerciseGate:
    """Turn a flickering stream of classifications into stable decisions.

    A decision is taken only when a class holds a majority of the recent
    window at or above a confidence floor. Switching the analyser additionally
    requires the current one to be between repetitions, so a switch cannot
    discard a half-finished rep or reset the count mid-set.
    """

    def __init__(self, default="squat", history=5, majority=3,
                 min_confidence=0.6):
        self.exercise = default
        self.history = collections.deque(maxlen=history)
        self.majority = majority
        self.min_confidence = min_confidence

        self.detected = None          # last stable class, including "other"
        self.confidence = 0.0
        self.suppressed = False       # True while "other" is the stable class
        self.pending = None           # class waiting for a gap between reps

    def _stable(self):
        """The class holding a sufficient majority, with its mean confidence."""
        if len(self.history) < self.majority:
            return None, 0.0
        labels = [label for label, _ in self.history]
        label, count = collections.Counter(labels).most_common(1)[0]
        if count < self.majority:
            return None, 0.0
        confidences = [c for l, c in self.history if l == label]
        mean = sum(confidences) / len(confidences)
        if mean < self.min_confidence:
            return None, mean
        return label, mean

    def update(self, label, confidence, phase):
        """Record one classification and return the resulting decision.

        `phase` is the active analyser's state machine phase, "up" between
        repetitions and "down" during one.
        """
        self.history.append((label, confidence))
        stable, mean = self._stable()

        result = {
            "detected": self.detected,
            "confidence": self.confidence,
            "exercise": self.exercise,
            "switched": False,
            "suppressed": self.suppressed,
        }

        if stable is None:
            return result

        self.detected = stable
        self.confidence = mean
        result["detected"] = stable
        result["confidence"] = mean

        # "other" never selects an analyser. It withholds verdicts, so that
        # incidental knee flexion is not coached as a repetition.
        if stable == "other":
            self.suppressed = True
            result["suppressed"] = True
            return result

        self.suppressed = False
        result["suppressed"] = False

        if stable == self.exercise:
            self.pending = None
            return result

        # A different exercise is being performed. Wait for a gap between
        # repetitions before switching, because switching rebuilds the
        # analyser and restarts the count.
        self.pending = stable
        if phase == "up":
            self.exercise = stable
            self.pending = None
            result["exercise"] = stable
            result["switched"] = True
        return result

    def reset(self, default=None):
        if default is not None:
            self.exercise = default
        self.history.clear()
        self.detected = None
        self.confidence = 0.0
        self.suppressed = False
        self.pending = None
