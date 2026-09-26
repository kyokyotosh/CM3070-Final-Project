from lunge import (
    LungeAnalyzer, DOWN_ENTER, UP_EXIT, DEPTH_TARGET,
    TRUNK_MAX, KNEE_TRAVEL_MAX, KNEE_MIN_PLAUSIBLE,
)
from angles import analyze_lunge, MIN_VISIBILITY
import unittest
import os
import sys

# Make the backend package importable whether this file is run directly
# (python tests/test_x.py) or through unittest, since the modules under test
# live one directory up in backend/.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

"""Unit tests for the forward-lunge analysis layer.

Two layers are tested separately:
  1. angles.analyze_lunge geometry (front-leg selection, knee-travel sign,
     trunk visibility gating), using hand-built landmark sets.
  2. lunge.LungeAnalyzer state machine (rep counting with hysteresis and the
     depth / trunk / knee-travel verdict), using pre-made analysis dicts so
     the state logic is tested independently of the geometry.

Run with:  python3 -m unittest test_lunge -v
"""


def _landmark(x, y, vis=1.0):
    return {"x": x, "y": y, "z": 0.0, "visibility": vis}


def _skeleton(overrides, default_vis=1.0):
    """Build a 33-landmark list, all at origin, then apply index overrides."""
    lm = [_landmark(0.5, 0.5, default_vis) for _ in range(33)]
    for idx, point in overrides.items():
        lm[idx] = point
    return lm


# Landmark indices, mirrored from angles.py for readability.
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK = 27, 28


class TestLungeGeometry(unittest.TestCase):

    def _lunge_pose(self, forward_left=True, knee_past_toe=0.0):
        """A plausible side-on lunge. The lead (left) foot steps forward; the
        trailing (right) foot stays near the centre line. forward_left=True
        means the step is toward smaller x (person faces image-left).

        knee_past_toe shifts the front knee ahead of the ankle in the step
        direction, to exercise the knee-travel sign."""
        step = -0.18 if forward_left else 0.18
        toe_x = 0.5 + step
        knee_x = toe_x + (step / abs(step)) * \
            knee_past_toe  # ahead in step dir
        return _skeleton({
            L_SH: _landmark(0.5, 0.30),
            R_SH: _landmark(0.5, 0.30),
            L_HIP: _landmark(0.5, 0.50),
            R_HIP: _landmark(0.5, 0.50),
            # front (left) leg: knee bent, shin roughly vertical
            L_KNEE: _landmark(knee_x, 0.62),
            L_ANK: _landmark(toe_x, 0.80),
            # trailing (right) leg near centre line, knee back
            R_KNEE: _landmark(0.52, 0.66),
            R_ANK: _landmark(0.54, 0.82),
        })

    def test_front_leg_identified_as_lead_foot(self):
        pose = self._lunge_pose(forward_left=True)
        out = analyze_lunge(pose)
        self.assertEqual(out["front_side"], "left")

    def test_front_leg_identified_when_facing_other_way(self):
        pose = self._lunge_pose(forward_left=False)
        out = analyze_lunge(pose)
        # left is still the lead foot
        self.assertEqual(out["front_side"], "left")

    def test_knee_over_ankle_gives_small_travel(self):
        pose = self._lunge_pose(forward_left=True, knee_past_toe=0.0)
        out = analyze_lunge(pose)
        self.assertLess(out["knee_travel_ratio"], KNEE_TRAVEL_MAX)

    def test_knee_past_toe_flagged_by_positive_travel(self):
        pose = self._lunge_pose(forward_left=True, knee_past_toe=0.12)
        out = analyze_lunge(pose)
        self.assertGreater(out["knee_travel_ratio"], KNEE_TRAVEL_MAX)

    def test_knee_travel_sign_is_facing_invariant(self):
        left = analyze_lunge(self._lunge_pose(True, knee_past_toe=0.12))
        right = analyze_lunge(self._lunge_pose(False, knee_past_toe=0.12))
        # Same fault magnitude regardless of which way the user faces.
        self.assertAlmostEqual(
            left["knee_travel_ratio"], right["knee_travel_ratio"], places=2)

    def test_trunk_gated_out_when_shoulder_dropped(self):
        pose = self._lunge_pose(forward_left=True)
        pose[L_SH] = _landmark(0.5, 0.30, vis=0.1)  # unreliable shoulder
        out = analyze_lunge(pose)
        self.assertFalse(out["trunk_valid"])
        self.assertIsNone(out["trunk_lean"])

    def test_trunk_present_when_landmarks_reliable(self):
        out = analyze_lunge(self._lunge_pose(forward_left=True))
        self.assertTrue(out["trunk_valid"])
        self.assertIsNotNone(out["trunk_lean"])

    def test_none_on_empty_input(self):
        self.assertIsNone(analyze_lunge([]))


class TestLungeStateMachine(unittest.TestCase):
    """Feed the analyzer pre-made analysis dicts to isolate rep-counting and
    verdict logic from the geometry."""

    @staticmethod
    def _a(front_knee, trunk=5.0, travel=0.2, vis=1.0, trunk_valid=True):
        return {
            "front_knee": front_knee,
            "front_side": "left",
            "front_visibility": vis,
            "knee_travel_ratio": travel,
            "trunk_lean": trunk if trunk_valid else None,
            "trunk_valid": trunk_valid,
        }

    def _do_rep(self, an, bottom_knee, trunk=5.0, travel=0.2):
        """Drive one full descent-and-return and return the verdict."""
        verdict = None
        # descend through the entry threshold to the bottom
        for k in (170, DOWN_ENTER - 1, bottom_knee):
            _, v = an.update(self._a(k, trunk=trunk, travel=travel))
            verdict = v or verdict
        # rise back above the exit threshold
        for k in (bottom_knee, DOWN_ENTER, UP_EXIT + 1):
            _, v = an.update(self._a(k, trunk=trunk, travel=travel))
            verdict = v or verdict
        return verdict

    def test_single_rep_counts_once(self):
        an = LungeAnalyzer()
        self._do_rep(an, bottom_knee=90)
        self.assertEqual(an.rep_count, 1)

    def test_good_rep_all_criteria_pass(self):
        an = LungeAnalyzer()
        v = self._do_rep(an, bottom_knee=90, trunk=8.0, travel=0.1)
        self.assertTrue(v["depth_ok"])
        self.assertTrue(v["trunk_ok"])
        self.assertTrue(v["knee_travel_ok"])

    def test_shallow_rep_fails_depth(self):
        an = LungeAnalyzer()
        v = self._do_rep(an, bottom_knee=125, trunk=8.0, travel=0.2)
        self.assertFalse(v["depth_ok"])

    def test_forward_lean_fails_trunk(self):
        an = LungeAnalyzer()
        v = self._do_rep(an, bottom_knee=90, trunk=TRUNK_MAX + 10, travel=0.2)
        self.assertFalse(v["trunk_ok"])

    def test_knee_drift_fails_travel(self):
        an = LungeAnalyzer()
        v = self._do_rep(an, bottom_knee=90, trunk=8.0,
                         travel=KNEE_TRAVEL_MAX + 0.3)
        self.assertFalse(v["knee_travel_ok"])

    def test_hysteresis_no_double_count_on_boundary_noise(self):
        """Jitter around the entry threshold must not rack up phantom reps."""
        an = LungeAnalyzer()
        for k in (170, DOWN_ENTER + 1, DOWN_ENTER - 1, DOWN_ENTER + 1,
                  DOWN_ENTER - 1, 95):
            an.update(self._a(k))
        # Still mid-rep: never rose above UP_EXIT, so no rep counted yet.
        self.assertEqual(an.rep_count, 0)
        # Complete the rep.
        for k in (DOWN_ENTER, UP_EXIT + 1):
            an.update(self._a(k))
        self.assertEqual(an.rep_count, 1)

    def test_two_reps_count_two(self):
        an = LungeAnalyzer()
        self._do_rep(an, bottom_knee=90)
        self._do_rep(an, bottom_knee=88)
        self.assertEqual(an.rep_count, 2)

    def test_low_visibility_blocks_analysis(self):
        an = LungeAnalyzer()
        status, verdict = an.update(self._a(90, vis=MIN_VISIBILITY - 0.1))
        self.assertIsNone(verdict)
        self.assertIn("view", status["feedback"].lower())

    def test_artefact_knee_frame_ignored_in_minimum(self):
        """A single sub-floor frame (e.g. 11 degrees) must not pass depth by
        polluting the per-rep minimum. Reproduces the observed 11.1 defect."""
        an = LungeAnalyzer()
        verdict = None
        seq = [170, DOWN_ENTER - 1, 11.0, 90, 90, DOWN_ENTER, UP_EXIT + 1]
        for k in seq:
            _, v = an.update(self._a(k))
            verdict = v or verdict
        self.assertEqual(an.rep_count, 1)
        self.assertGreaterEqual(verdict["min_knee"], KNEE_MIN_PLAUSIBLE)
        self.assertEqual(verdict["min_knee"], 90.0)

    def test_genuine_deep_rep_above_floor_still_counts(self):
        """A real deep lunge (observed 46.8 degrees) sits above the floor and
        must still count and pass depth."""
        an = LungeAnalyzer()
        v = self._do_rep(an, bottom_knee=46.8)
        self.assertEqual(an.rep_count, 1)
        self.assertTrue(v["depth_ok"])

    def test_travel_threshold_flags_observed_fault(self):
        """0.24 was an observed knee-over-toe fault; it must be flagged."""
        an = LungeAnalyzer()
        v = self._do_rep(an, bottom_knee=90, travel=0.24)
        self.assertFalse(v["knee_travel_ok"])

    def test_travel_threshold_passes_observed_good(self):
        """-0.66 was an observed good rep; it must still pass."""
        an = LungeAnalyzer()
        v = self._do_rep(an, bottom_knee=90, travel=-0.66)
        self.assertTrue(v["knee_travel_ok"])

    def test_trunk_unknown_when_never_valid(self):
        """If trunk was never reliably seen, verdict reports it as unknown."""
        an = LungeAnalyzer()
        verdict = None
        for k in (170, DOWN_ENTER - 1, 90):
            _, v = an.update(self._a(k, trunk_valid=False))
            verdict = v or verdict
        for k in (90, DOWN_ENTER, UP_EXIT + 1):
            _, v = an.update(self._a(k, trunk_valid=False))
            verdict = v or verdict
        self.assertIsNone(verdict["trunk_ok"])
        self.assertIsNone(verdict["max_trunk"])


if __name__ == "__main__":
    unittest.main()
