# Squat diagnosis thresholds, in degrees. These are starting values and
# are expected to need empirical calibration through self-testing.
DOWN_ENTER = 120.0     # knee angle below this: user is descending into a squat
UP_EXIT = 160.0        # knee angle above this: user has returned to standing
PARALLEL = 95.0        # knee angle at/below this counts as full depth
TRUNK_MAX = 50.0       # trunk lean above this flags excessive forward lean
MIN_VISIBILITY = 0.5   # below this, the reading is not trusted


class SquatAnalyzer:
    """Deterministic squat rep counter and form diagnoser.

    Holds per-session state. The knee angle drives a two-phase state
    machine (up / down) with hysteresis to avoid double-counting. Depth
    and trunk lean are judged once per completed rep.
    """

    def __init__(self):
        self.phase = "up"
        self.rep_count = 0
        self.min_knee_this_rep = 180.0
        self.max_trunk_this_rep = 0.0
        self.last_feedback = "Stand side-on to the camera and begin squatting."

    def update(self, angles):
        if angles is None:
            return self._status("No pose detected.")
        if angles["knee_visibility"] < MIN_VISIBILITY:
            return self._status("Step back so your legs are clearly visible.")

        knee = angles["knee"]
        trunk = angles["trunk_lean"]

        # Track the extremes of the current descent
        if self.phase == "down":
            self.min_knee_this_rep = min(self.min_knee_this_rep, knee)
            self.max_trunk_this_rep = max(self.max_trunk_this_rep, trunk)

        # State transitions with hysteresis
        if self.phase == "up" and knee < DOWN_ENTER:
            self.phase = "down"
            self.min_knee_this_rep = knee
            self.max_trunk_this_rep = trunk
        elif self.phase == "down" and knee > UP_EXIT:
            self.phase = "up"
            self.rep_count += 1
            self.last_feedback = self._judge_rep()

        return self._status()

    def _judge_rep(self):
        faults = []
        if self.min_knee_this_rep > PARALLEL:
            faults.append(
                "aim to squat deeper, until your thighs are parallel")
        if self.max_trunk_this_rep > TRUNK_MAX:
            faults.append("try to keep your chest more upright")
        if not faults:
            return "Good rep with full depth and a stable torso."
        return "Rep counted. " + " Also, ".join(f.capitalize() for f in faults) + "."

    def _status(self, message=None):
        return {
            "rep_count": self.rep_count,
            "phase": self.phase,
            "feedback": message if message is not None else self.last_feedback,
        }
