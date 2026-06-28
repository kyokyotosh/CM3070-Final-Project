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

    def update(self, angles):
        rep_completed = False
        if angles is None:
            return self._status("No pose detected."), None
        if angles["knee_visibility"] < MIN_VISIBILITY:
            return self._status("Step back so your legs are clearly visible."), None

        knee = angles["knee"]
        trunk = angles["trunk_lean"]

        if self.phase == "down":
            self.min_knee_this_rep = min(self.min_knee_this_rep, knee)
            self.max_trunk_this_rep = max(self.max_trunk_this_rep, trunk)

        verdict = None
        if self.phase == "up" and knee < DOWN_ENTER:
            self.phase = "down"
            self.min_knee_this_rep = knee
            self.max_trunk_this_rep = trunk
        elif self.phase == "down" and knee > UP_EXIT:
            self.phase = "up"
            self.rep_count += 1
            verdict = {
                "rep_number": self.rep_count,
                "depth_ok": self.min_knee_this_rep <= PARALLEL,
                "trunk_ok": self.max_trunk_this_rep <= TRUNK_MAX,
                "min_knee": round(self.min_knee_this_rep, 1),
                "max_trunk": round(self.max_trunk_this_rep, 1),
            }

        return self._status(), verdict

    def _status(self, message=None):
        return {
            "rep_count": self.rep_count,
            "phase": self.phase,
            "feedback": message,
        }
