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
import time

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

    Acquiring an exercise and changing it are deliberately not symmetric.
    Deciding what the person has started is easy: a short majority at modest
    confidence. Deciding they have switched to a different exercise is hard,
    because a switch changes which analyser is authoritative, and a spurious
    one mid-set is worse than a slow correct one. A switch therefore needs a
    longer majority, higher mean confidence, a gap since the last switch, and
    the active analyser to be between repetitions.

    Observed confidences fall from about 0.96 early in a set to about 0.73 as
    form degrades, which is exactly when a single loose threshold starts to
    flap, so the switch floor sits above that range.

    The third class earns its place here. "other" never selects an analyser;
    it suppresses verdicts. Sitting down or stretching flexes the knees enough
    to complete a repetition, and without a way to recognise not-exercising
    the system would count and coach those.
    """

    def __init__(self, default="squat", history=7,
                 acquire_majority=3, min_confidence=0.6,
                 switch_majority=5, switch_confidence=0.8,
                 min_dwell_s=4.0):
        self.exercise = default
        self.history = collections.deque(maxlen=history)

        self.acquire_majority = acquire_majority
        self.min_confidence = min_confidence
        self.switch_majority = switch_majority
        self.switch_confidence = switch_confidence
        self.min_dwell_s = min_dwell_s

        self.detected = None          # last stable class, including "other"
        self.confidence = 0.0
        self.suppressed = False       # True while "other" is the stable class
        self.pending = None           # detected, but not yet allowed to switch
        self.last_switch = 0.0

    def _stable(self, majority, floor):
        """The class holding `majority` of the recent window at or above
        `floor` mean confidence, with that mean."""
        if len(self.history) < majority:
            return None, 0.0
        labels = [label for label, _ in self.history]
        label, count = collections.Counter(labels).most_common(1)[0]
        if count < majority:
            return None, 0.0
        confidences = [c for l, c in self.history if l == label]
        mean = sum(confidences) / len(confidences)
        if mean < floor:
            return None, mean
        return label, mean

    def update(self, label, confidence, phase, now=None):
        """Record one classification and return the resulting decision.

        `phase` is the active analyser's state machine phase, "up" between
        repetitions and "down" during one. `now` is injectable so the dwell
        rule can be tested without waiting.
        """
        now = time.monotonic() if now is None else now
        self.history.append((label, confidence))

        # Reporting and acquisition use the looser test, so the readout stays
        # responsive even when no switch is warranted.
        stable, mean = self._stable(self.acquire_majority, self.min_confidence)

        result = {
            "detected": self.detected,
            "confidence": self.confidence,
            "exercise": self.exercise,
            "switched": False,
            "suppressed": self.suppressed,
            "pending": self.pending,
        }

        if stable is None:
            return result

        self.detected = stable
        self.confidence = mean
        result["detected"] = stable
        result["confidence"] = mean

        if stable == "other":
            self.suppressed = True
            result["suppressed"] = True
            return result

        self.suppressed = False
        result["suppressed"] = False

        if stable == self.exercise:
            self.pending = None
            result["pending"] = None
            return result

        # A different exercise. Every one of these must hold before the
        # authoritative analyser changes.
        strict, _ = self._stable(self.switch_majority, self.switch_confidence)
        blocked = (strict != stable
                   or now - self.last_switch < self.min_dwell_s
                   or phase != "up")
        if blocked:
            self.pending = stable
            result["pending"] = stable
            return result

        self.exercise = stable
        self.last_switch = now
        self.pending = None
        result["exercise"] = stable
        result["switched"] = True
        result["pending"] = None
        return result

    def reset(self, default=None):
        if default is not None:
            self.exercise = default
        self.history.clear()
        self.detected = None
        self.confidence = 0.0
        self.suppressed = False
        self.pending = None
        self.last_switch = 0.0
