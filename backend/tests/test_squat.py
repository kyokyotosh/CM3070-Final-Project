from squat import (
    SquatAnalyzer, DOWN_ENTER, UP_EXIT, PARALLEL, TRUNK_MAX,
    KNEE_MIN_PLAUSIBLE,
)
from angles import analyze_landmarks, MIN_VISIBILITY
import unittest
import os
import sys

# Make the backend package importable whether this file is run directly
# (python tests/test_x.py) or through unittest, since the modules under test
# live one directory up in backend/.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

"""Unit tests for the squat analysis layer.

Mirrors test_lunge.py, and tests the same two layers separately:
  1. angles.analyze_landmarks geometry (leg selection by visibility, trunk
     visibility gating), using hand-built landmark sets.
  2. squat.SquatAnalyzer state machine (rep counting with hysteresis and the
     depth / trunk verdict), using pre-made analysis dicts so the state logic
     is tested independently of the geometry.

Several cases are anchored on values observed during calibration sessions,
including the 9.2 degree knee reading that passed depth before the
plausibility floor was applied to the squat.

Run with:  python3 -m unittest test_squat -v
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


class TestSquatGeometry(unittest.TestCase):

    def _squat_pose(self, left_vis=1.0, right_vis=1.0, shoulder_x=0.5,
                    shoulder_vis=1.0):
        """A side-on standing pose with a straight left leg and a bent right
        leg, so the two legs give clearly different knee angles and the
        selected side can be identified from the returned value.

        shoulder_x moves the shoulders ahead of the hips to produce trunk
        lean; shoulder_vis drops shoulder confidence to exercise the gate."""
        return _skeleton({
            L_SH: _landmark(shoulder_x, 0.30, shoulder_vis),
            R_SH: _landmark(shoulder_x, 0.30, shoulder_vis),
            L_HIP: _landmark(0.50, 0.50, left_vis),
            R_HIP: _landmark(0.50, 0.50, right_vis),
            # straight left leg: hip, knee and ankle are collinear
            L_KNEE: _landmark(0.50, 0.65, left_vis),
            L_ANK: _landmark(0.50, 0.80, left_vis),
            # bent right leg: knee displaced forward of the hip-ankle line
            R_KNEE: _landmark(0.40, 0.65, right_vis),
            R_ANK: _landmark(0.50, 0.80, right_vis),
        })

    def test_more_visible_leg_is_selected(self):
        out = analyze_landmarks(self._squat_pose(left_vis=0.9, right_vis=0.3))
        self.assertEqual(out["side"], "left")
        self.assertAlmostEqual(out["knee"], 180.0, places=1)

    def test_other_leg_selected_when_it_is_the_clearer_one(self):
        out = analyze_landmarks(self._squat_pose(left_vis=0.3, right_vis=0.9))
        self.assertEqual(out["side"], "right")
        self.assertLess(out["knee"], 180.0)

    def test_knee_visibility_reports_the_selected_leg(self):
        out = analyze_landmarks(self._squat_pose(left_vis=0.8, right_vis=0.2))
        self.assertAlmostEqual(out["knee_visibility"], 0.8, places=2)

    def test_upright_trunk_is_near_zero(self):
        out = analyze_landmarks(self._squat_pose())
        self.assertTrue(out["trunk_valid"])
        self.assertLess(out["trunk_lean"], 1.0)

    def test_forward_shoulders_increase_trunk_lean(self):
        upright = analyze_landmarks(self._squat_pose(shoulder_x=0.50))
        leaning = analyze_landmarks(self._squat_pose(shoulder_x=0.40))
        self.assertGreater(leaning["trunk_lean"], upright["trunk_lean"])

    def test_trunk_gated_out_when_shoulder_dropped(self):
        """The 175-179 degree prototype defect came from computing the trunk
        vector on unreliable landmarks. Low confidence must gate it out."""
        out = analyze_landmarks(self._squat_pose(shoulder_vis=0.1))
        self.assertFalse(out["trunk_valid"])
        self.assertIsNone(out["trunk_lean"])

    def test_none_on_empty_input(self):
        self.assertIsNone(analyze_landmarks([]))

    def test_none_on_short_landmark_list(self):
        self.assertIsNone(analyze_landmarks([_landmark(0.5, 0.5)] * 10))


def _a(knee, trunk=20.0, vis=1.0, trunk_valid=True):
    """One pre-made analysis dict, as angles.analyze_landmarks would return."""
    return {
        "knee": knee,
        "side": "left",
        "knee_visibility": vis,
        "trunk_lean": trunk if trunk_valid else None,
        "trunk_valid": trunk_valid,
    }


class TestSquatStateMachine(unittest.TestCase):
    """Feed the analyzer pre-made analysis dicts to isolate rep-counting and
    verdict logic from the geometry."""

    def _do_rep(self, an, bottom_knee, trunk=20.0):
        """Drive one full descent-and-return and return the verdict."""
        verdict = None
        # descend through the entry threshold to the bottom
        for k in (170, DOWN_ENTER - 1, bottom_knee):
            _, v = an.update(_a(k, trunk=trunk))
            verdict = v or verdict
        # rise back above the exit threshold
        for k in (bottom_knee, DOWN_ENTER, UP_EXIT + 1):
            _, v = an.update(_a(k, trunk=trunk))
            verdict = v or verdict
        return verdict

    def test_single_rep_counts_once(self):
        an = SquatAnalyzer()
        self._do_rep(an, bottom_knee=90)
        self.assertEqual(an.rep_count, 1)

    def test_two_reps_count_two(self):
        an = SquatAnalyzer()
        self._do_rep(an, bottom_knee=90)
        self._do_rep(an, bottom_knee=88)
        self.assertEqual(an.rep_count, 2)

    def test_rep_counted_only_after_returning_to_standing(self):
        an = SquatAnalyzer()
        for k in (170, DOWN_ENTER - 1, 90):
            an.update(_a(k))
        self.assertEqual(an.rep_count, 0)
        self.assertEqual(an.phase, "down")
        for k in (DOWN_ENTER, UP_EXIT + 1):
            an.update(_a(k))
        self.assertEqual(an.rep_count, 1)

    def test_good_rep_passes_both_criteria(self):
        an = SquatAnalyzer()
        v = self._do_rep(an, bottom_knee=89.2, trunk=29.0)
        self.assertTrue(v["depth_ok"])
        self.assertTrue(v["trunk_ok"])
        self.assertEqual(v["exercise"], "squat")
        self.assertEqual(v["rep_number"], 1)

    def test_shallow_rep_fails_depth(self):
        """119.4 degrees was an observed shallow rep against a 95 target."""
        an = SquatAnalyzer()
        v = self._do_rep(an, bottom_knee=119.4)
        self.assertFalse(v["depth_ok"])
        self.assertGreater(v["min_knee"], PARALLEL)

    def test_observed_borderline_depth_still_passes(self):
        """92.2 degrees passed against the 95 threshold by under three
        degrees. The margin is thin, and this records where it sits."""
        an = SquatAnalyzer()
        v = self._do_rep(an, bottom_knee=92.2)
        self.assertTrue(v["depth_ok"])
        self.assertLess(PARALLEL - v["min_knee"], 3.0)

    def test_forward_lean_fails_trunk(self):
        """62.9 degrees was an observed trunk fault against a 50 limit."""
        an = SquatAnalyzer()
        v = self._do_rep(an, bottom_knee=75.6, trunk=62.9)
        self.assertFalse(v["trunk_ok"])
        self.assertEqual(v["max_trunk"], 62.9)

    def test_trunk_at_the_threshold_passes(self):
        an = SquatAnalyzer()
        v = self._do_rep(an, bottom_knee=90, trunk=TRUNK_MAX)
        self.assertTrue(v["trunk_ok"])

    def test_hysteresis_no_double_count_on_boundary_noise(self):
        """Jitter around the entry threshold must not rack up phantom reps."""
        an = SquatAnalyzer()
        for k in (170, DOWN_ENTER + 1, DOWN_ENTER - 1, DOWN_ENTER + 1,
                  DOWN_ENTER - 1, 95):
            an.update(_a(k))
        # Still mid-rep: never rose above UP_EXIT, so no rep counted yet.
        self.assertEqual(an.rep_count, 0)
        # Complete the rep.
        for k in (DOWN_ENTER, UP_EXIT + 1):
            an.update(_a(k))
        self.assertEqual(an.rep_count, 1)

    def test_low_visibility_blocks_analysis(self):
        an = SquatAnalyzer()
        status, verdict = an.update(_a(90, vis=MIN_VISIBILITY - 0.1))
        self.assertIsNone(verdict)
        self.assertIn("visible", status["feedback"].lower())

    def test_no_pose_reports_no_detection(self):
        an = SquatAnalyzer()
        status, verdict = an.update(None)
        self.assertIsNone(verdict)
        self.assertIsNotNone(status["feedback"])


class TestSquatArtefactGuards(unittest.TestCase):
    """Frame-level bad readings must not reach rep-level aggregates. These
    reproduce observed defects rather than hypothetical ones."""

    def test_artefact_knee_frame_ignored_in_minimum(self):
        """A single sub-floor frame (observed at 9.2 degrees) must not pass
        depth by polluting the per-rep minimum. Same defect class as the
        lunge 11.1 reading."""
        an = SquatAnalyzer()
        verdict = None
        seq = [170, DOWN_ENTER - 1, 9.2, 90, 90, DOWN_ENTER, UP_EXIT + 1]
        for k in seq:
            _, v = an.update(_a(k))
            verdict = v or verdict
        self.assertEqual(an.rep_count, 1)
        self.assertGreaterEqual(verdict["min_knee"], KNEE_MIN_PLAUSIBLE)
        self.assertEqual(verdict["min_knee"], 90.0)

    def test_artefact_frame_does_not_start_a_descent(self):
        """An implausible frame while standing must not trigger a phase
        change, which would otherwise create a phantom rep."""
        an = SquatAnalyzer()
        for k in (170, 9.2, 170):
            an.update(_a(k))
        self.assertEqual(an.phase, "up")
        self.assertEqual(an.rep_count, 0)

    def test_genuine_deep_rep_above_floor_still_counts(self):
        """A real deep squat (observed 41.0 degrees) sits above the floor and
        must still count and pass depth."""
        an = SquatAnalyzer()
        verdict = None
        for k in (170, DOWN_ENTER - 1, 41.0, 41.0, DOWN_ENTER, UP_EXIT + 1):
            _, v = an.update(_a(k))
            verdict = v or verdict
        self.assertEqual(an.rep_count, 1)
        self.assertTrue(verdict["depth_ok"])

    def test_invalid_trunk_frame_does_not_pollute_maximum(self):
        """One gated-out trunk frame must not raise the per-rep maximum. This
        is the 173.4 defect: the corrupted value never enters the aggregate
        because angles.py returns None for it."""
        an = SquatAnalyzer()
        verdict = None
        _, v = an.update(_a(170))
        _, v = an.update(_a(DOWN_ENTER - 1, trunk=25.0))
        for k, valid in ((90, True), (90, False), (90, True)):
            _, v = an.update(_a(k, trunk=25.0, trunk_valid=valid))
            verdict = v or verdict
        for k in (DOWN_ENTER, UP_EXIT + 1):
            _, v = an.update(_a(k, trunk=25.0))
            verdict = v or verdict
        self.assertEqual(verdict["max_trunk"], 25.0)
        self.assertTrue(verdict["trunk_ok"])

    def test_trunk_unknown_when_never_valid(self):
        """If the trunk was never reliably seen, the verdict reports it as
        unknown rather than silently passing it."""
        an = SquatAnalyzer()
        verdict = None
        for k in (170, DOWN_ENTER - 1, 90, 90, DOWN_ENTER, UP_EXIT + 1):
            _, v = an.update(_a(k, trunk_valid=False))
            verdict = v or verdict
        self.assertIsNone(verdict["trunk_ok"])
        self.assertIsNone(verdict["max_trunk"])


if __name__ == "__main__":
    unittest.main()
