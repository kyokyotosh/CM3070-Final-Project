from quality import score, is_partial, BASE_PENALTY
import unittest
import os
import sys

# Make the backend package importable whether this file is run directly
# (python tests/test_quality.py) or through unittest, since the modules under
# test live one directory up in backend/.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

"""Unit tests for rep quality scoring.

The score is shown to the user, so it must behave predictably: clean reps score
full marks, every breached criterion costs something, and a worse breach costs
more than a narrow one. These tests use values observed during threshold
calibration so the scoring stays anchored to real repetitions.
"""


def squat_verdict(rep=1, depth_ok=True, trunk_ok=True,
                  min_knee=90.0, max_trunk=20.0):
    return {"exercise": "squat", "rep_number": rep,
            "depth_ok": depth_ok, "trunk_ok": trunk_ok,
            "min_knee": min_knee, "max_trunk": max_trunk}


def lunge_verdict(rep=1, depth_ok=True, trunk_ok=True, knee_travel_ok=True,
                  min_knee=100.0, max_trunk=10.0, max_knee_travel=0.0):
    return {"exercise": "lunge", "rep_number": rep, "depth_ok": depth_ok,
            "trunk_ok": trunk_ok, "knee_travel_ok": knee_travel_ok,
            "min_knee": min_knee, "max_trunk": max_trunk,
            "max_knee_travel": max_knee_travel}


class TestCleanReps(unittest.TestCase):

    def test_clean_squat_scores_full(self):
        self.assertEqual(score(squat_verdict()), 100)

    def test_clean_lunge_scores_full(self):
        self.assertEqual(score(lunge_verdict()), 100)

    def test_unmeasured_trunk_is_not_penalised(self):
        # trunk_ok None means the trunk was never confidently seen. It is not
        # a fault, so it must not cost score.
        v = squat_verdict(trunk_ok=None, max_trunk=None)
        self.assertEqual(score(v), 100)
        self.assertTrue(is_partial(v))


class TestSingleFaults(unittest.TestCase):

    def test_any_fault_costs_at_least_the_base_penalty(self):
        v = squat_verdict(depth_ok=False, min_knee=95.5)
        self.assertLessEqual(score(v), 100 - BASE_PENALTY)

    def test_worse_depth_scores_lower_than_narrow_miss(self):
        narrow = score(squat_verdict(depth_ok=False, min_knee=98.0))
        gross = score(squat_verdict(depth_ok=False, min_knee=125.0))
        self.assertLess(gross, narrow)

    def test_observed_borderline_lunge_depth_still_penalised(self):
        """114 degrees against a 110 target was an observed fault rep with a
        thin margin. It must lose score without collapsing it."""
        s = score(lunge_verdict(depth_ok=False, min_knee=114.0))
        self.assertLess(s, 100)
        self.assertGreater(s, 60)

    def test_observed_knee_travel_fault_is_penalised(self):
        """0.24 was an observed knee-over-toe fault against a 0.15 threshold."""
        s = score(lunge_verdict(knee_travel_ok=False, max_knee_travel=0.24))
        self.assertLess(s, 100 - BASE_PENALTY)

    def test_missing_metric_falls_back_to_base_penalty(self):
        v = squat_verdict(depth_ok=False, min_knee=None)
        self.assertEqual(score(v), int(round(100 - BASE_PENALTY)))

    def test_inconsistent_metric_still_costs_score(self):
        # The verdict is the authority. If it says depth failed, the rep never
        # scores 100, even when the stored value looks acceptable.
        v = squat_verdict(depth_ok=False, min_knee=80.0)
        self.assertLess(score(v), 100)


class TestMultipleFaults(unittest.TestCase):

    def test_two_faults_score_lower_than_one(self):
        one = score(squat_verdict(trunk_ok=False, max_trunk=60.0))
        two = score(squat_verdict(depth_ok=False, min_knee=120.0,
                                  trunk_ok=False, max_trunk=60.0))
        self.assertLess(two, one)

    def test_third_fault_still_costs_despite_the_cue_cap(self):
        """coach.select_faults surfaces at most two faults in the spoken cue.
        The score must still reflect all three, or the interface would show a
        better score than the rep deserves."""
        two = score(lunge_verdict(trunk_ok=False, max_trunk=35.0,
                                  knee_travel_ok=False, max_knee_travel=0.24))
        three = score(lunge_verdict(trunk_ok=False, max_trunk=35.0,
                                    knee_travel_ok=False, max_knee_travel=0.24,
                                    depth_ok=False, min_knee=130.0))
        self.assertLess(three, two)

    def test_score_never_goes_below_zero(self):
        v = lunge_verdict(depth_ok=False, min_knee=179.0,
                          trunk_ok=False, max_trunk=90.0,
                          knee_travel_ok=False, max_knee_travel=3.0)
        self.assertGreaterEqual(score(v), 0)


class TestExerciseThresholds(unittest.TestCase):

    def test_same_angle_scored_against_the_right_threshold(self):
        """A 108 degree minimum passes the lunge target but not the squat one,
        so the two exercises must not share a threshold."""
        squat = score(squat_verdict(depth_ok=False, min_knee=108.0))
        lunge = score(lunge_verdict(depth_ok=False, min_knee=108.0))
        self.assertLess(squat, lunge)

    def test_knee_travel_is_ignored_for_squat(self):
        v = squat_verdict()
        v["knee_travel_ok"] = None
        self.assertEqual(score(v), 100)


if __name__ == "__main__":
    unittest.main()
