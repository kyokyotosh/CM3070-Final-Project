# Forward-lunge thresholds, in degrees unless noted, calibrated from
# self-recorded repetitions of one user.
DOWN_ENTER = 130.0        # front-knee angle below this: user is descending
UP_EXIT = 160.0           # front-knee angle above this: user has stood up
DEPTH_TARGET = 110.0      # front-knee angle at/below this counts as adequate depth
TRUNK_MAX = 20.0          # trunk lean above this flags excessive forward lean
# front knee ahead of ankle by > 0.15 shin-lengths flags over-travel
KNEE_TRAVEL_MAX = 0.15
MIN_VISIBILITY = 0.5      # below this, the front-leg reading is not trusted

# Knee readings below this come from bad frames, not real movement. They are
# dropped before the state machine, so one frame cannot set a repetition's
# minimum.
KNEE_MIN_PLAUSIBLE = 30.0


class LungeAnalyzer:
    """Rule-based forward-lunge repetition counter and form checker.

    Same structure as SquatAnalyzer: the front-knee angle drives a two-phase
    state machine with hysteresis, and each completed repetition is judged once
    on depth, trunk lean and knee travel. Trunk lean is aggregated only from
    frames that pass the visibility gate.
    """

    def __init__(self):
        self.phase = "up"
        self.rep_count = 0
        self.min_knee_this_rep = 180.0
        self.max_trunk_this_rep = 0.0
        self.max_travel_this_rep = -999.0
        self.trunk_seen_this_rep = False
        self.front_side = None

    def update(self, angles):
        if angles is None:
            return self._status("No pose detected."), None
        if angles["front_visibility"] < MIN_VISIBILITY:
            return self._status("Step back so your front leg is fully in view."), None

        knee = angles["front_knee"]
        # Drop an implausible front-knee frame rather than letting it advance
        # the state machine or pollute the per-rep minimum.
        if knee < KNEE_MIN_PLAUSIBLE:
            return self._status(), None

        trunk = angles["trunk_lean"]
        travel = angles["knee_travel_ratio"]

        if self.phase == "down":
            self.min_knee_this_rep = min(self.min_knee_this_rep, knee)
            self.max_travel_this_rep = max(self.max_travel_this_rep, travel)
            if angles["trunk_valid"] and trunk is not None:
                self.max_trunk_this_rep = max(self.max_trunk_this_rep, trunk)
                self.trunk_seen_this_rep = True

        verdict = None
        if self.phase == "up" and knee < DOWN_ENTER:
            self.phase = "down"
            self.min_knee_this_rep = knee
            self.max_travel_this_rep = travel
            self.front_side = angles["front_side"]
            if angles["trunk_valid"] and trunk is not None:
                self.max_trunk_this_rep = trunk
                self.trunk_seen_this_rep = True
            else:
                self.max_trunk_this_rep = 0.0
                self.trunk_seen_this_rep = False
        elif self.phase == "down" and knee > UP_EXIT:
            self.phase = "up"
            self.rep_count += 1
            verdict = {
                "exercise": "lunge",
                "rep_number": self.rep_count,
                "front_side": self.front_side,
                "depth_ok": self.min_knee_this_rep <= DEPTH_TARGET,
                # If the trunk was never confidently measured this rep, treat
                # posture as unknown rather than silently passing it.
                "trunk_ok": (self.max_trunk_this_rep <= TRUNK_MAX)
                            if self.trunk_seen_this_rep else None,
                "knee_travel_ok": self.max_travel_this_rep <= KNEE_TRAVEL_MAX,
                "min_knee": round(self.min_knee_this_rep, 1),
                "max_trunk": round(self.max_trunk_this_rep, 1)
                if self.trunk_seen_this_rep else None,
                "max_knee_travel": round(self.max_travel_this_rep, 2),
            }

        return self._status(), verdict

    def _status(self, message=None):
        return {
            "rep_count": self.rep_count,
            "phase": self.phase,
            "feedback": message,
        }
