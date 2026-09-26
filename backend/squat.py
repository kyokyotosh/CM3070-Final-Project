# Squat thresholds, in degrees, calibrated from self-recorded repetitions of
# one user.
DOWN_ENTER = 120.0     # knee angle below this: user is descending into a squat
UP_EXIT = 160.0        # knee angle above this: user has returned to standing
PARALLEL = 95.0        # knee angle at/below this counts as full depth
TRUNK_MAX = 50.0       # trunk lean above this flags excessive forward lean
MIN_VISIBILITY = 0.5   # below this, the reading is not trusted

# Knee readings below this come from bad frames, not real movement (an
# observed 9.2 degree reading once passed the depth check).
KNEE_MIN_PLAUSIBLE = 30.0


class SquatAnalyzer:
    """Rule-based squat repetition counter and form checker.

    Holds per-session state. The knee angle drives a two-phase state machine
    (up / down) with hysteresis so each repetition is counted once, and depth
    and trunk lean are judged once per completed repetition. Trunk lean is
    aggregated only from frames that pass the visibility gate.
    """

    def __init__(self):
        self.phase = "up"
        self.rep_count = 0
        self.min_knee_this_rep = 180.0
        self.max_trunk_this_rep = 0.0
        self.trunk_seen_this_rep = False

    def update(self, angles):
        if angles is None:
            return self._status("No pose detected."), None
        if angles["knee_visibility"] < MIN_VISIBILITY:
            return self._status("Step back so your legs are clearly visible."), None

        knee = angles["knee"]

        # Drop an implausible knee frame rather than letting it advance the
        # state machine or pollute the per-rep minimum.
        if knee < KNEE_MIN_PLAUSIBLE:
            return self._status(), None

        trunk = angles["trunk_lean"]
        trunk_valid = angles.get("trunk_valid", trunk is not None)

        if self.phase == "down":
            self.min_knee_this_rep = min(self.min_knee_this_rep, knee)
            if trunk_valid and trunk is not None:
                self.max_trunk_this_rep = max(self.max_trunk_this_rep, trunk)
                self.trunk_seen_this_rep = True

        verdict = None
        if self.phase == "up" and knee < DOWN_ENTER:
            self.phase = "down"
            self.min_knee_this_rep = knee
            if trunk_valid and trunk is not None:
                self.max_trunk_this_rep = trunk
                self.trunk_seen_this_rep = True
            else:
                self.max_trunk_this_rep = 0.0
                self.trunk_seen_this_rep = False
        elif self.phase == "down" and knee > UP_EXIT:
            self.phase = "up"
            self.rep_count += 1
            verdict = {
                "exercise": "squat",
                "rep_number": self.rep_count,
                "depth_ok": self.min_knee_this_rep <= PARALLEL,
                # Report posture as unknown rather than a silent pass when the
                # trunk was never confidently measured during the rep.
                "trunk_ok": (self.max_trunk_this_rep <= TRUNK_MAX)
                            if self.trunk_seen_this_rep else None,
                "min_knee": round(self.min_knee_this_rep, 1),
                "max_trunk": round(self.max_trunk_this_rep, 1)
                if self.trunk_seen_this_rep else None,
            }

        return self._status(), verdict

    def _status(self, message=None):
        return {
            "rep_count": self.rep_count,
            "phase": self.phase,
            "feedback": message,
        }
