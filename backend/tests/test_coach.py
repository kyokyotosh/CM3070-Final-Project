from coach import select_faults, _fallback, _verdict_to_text, MAX_FAULTS_SURFACED
import unittest
import os
import sys

# Make the backend package importable whether this file is run directly
# (python tests/test_x.py) or through unittest, since the modules under test
# live one directory up in backend/.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

"""Unit tests for deterministic fault selection in coach.py.

These test the selection and fallback logic only, which is pure and needs no
language model. The ollama import in coach.py is lazy, so this suite runs
with no server and no stub:  python3 -m unittest test_coach -v
"""


def squat_verdict(rep=1, depth_ok=True, trunk_ok=True):
    return {"exercise": "squat", "rep_number": rep,
            "depth_ok": depth_ok, "trunk_ok": trunk_ok}


def lunge_verdict(rep=1, depth_ok=True, trunk_ok=True, knee_travel_ok=True):
    return {"exercise": "lunge", "rep_number": rep, "depth_ok": depth_ok,
            "trunk_ok": trunk_ok, "knee_travel_ok": knee_travel_ok}


class TestFaultSelection(unittest.TestCase):

    def test_clean_rep_has_no_faults(self):
        self.assertEqual(select_faults(squat_verdict()), [])
        self.assertEqual(select_faults(lunge_verdict()), [])

    def test_unmeasured_trunk_is_not_a_fault(self):
        # trunk_ok None means "not measured", must not be surfaced as a fault
        v = squat_verdict(trunk_ok=None)
        self.assertEqual(select_faults(v), [])

    def test_single_fault_surfaced(self):
        faults = select_faults(squat_verdict(depth_ok=False))
        self.assertEqual(len(faults), 1)
        self.assertIn("deeper", faults[0][0])

    def test_priority_trunk_before_depth(self):
        # both fail: trunk (safety) must come first
        faults = select_faults(squat_verdict(depth_ok=False, trunk_ok=False))
        self.assertEqual(faults[0][0], "keep your chest up")

    def test_cap_limits_number_surfaced(self):
        # all three lunge criteria fail; only MAX_FAULTS_SURFACED are returned
        v = lunge_verdict(depth_ok=False, trunk_ok=False, knee_travel_ok=False)
        faults = select_faults(v)
        self.assertEqual(len(faults), MAX_FAULTS_SURFACED)
        # depth is lowest priority, so it is the one dropped
        cues = [c for c, _ in faults]
        self.assertNotIn("drop your front knee lower", cues)

    def test_observed_rep6_surfaces_both_faults(self):
        """Rep 6 from calibration: good depth, trunk AND knee-travel both fail.
        Previously the trunk fault was dropped from the sentence; both must now
        be surfaced."""
        v = lunge_verdict(depth_ok=True, trunk_ok=False, knee_travel_ok=False)
        faults = select_faults(v)
        cues = [c for c, _ in faults]
        self.assertIn("keep your chest up", cues)
        self.assertIn("keep your front knee over your ankle", cues)

    def test_lunge_depth_cue_differs_from_squat(self):
        lunge = select_faults(lunge_verdict(depth_ok=False))
        squat = select_faults(squat_verdict(depth_ok=False))
        self.assertNotEqual(lunge[0][0], squat[0][0])

    def test_knee_travel_only_applies_to_lunge(self):
        # a squat verdict never carries knee_travel, so it is never selected
        v = {"exercise": "squat", "rep_number": 1,
             "depth_ok": True, "trunk_ok": True, "knee_travel_ok": False}
        self.assertEqual(select_faults(v), [])


class TestFallbackText(unittest.TestCase):

    def test_clean_rep_praises(self):
        self.assertIn("great form", _fallback(squat_verdict()))

    def test_rep6_fallback_mentions_both(self):
        v = lunge_verdict(depth_ok=True, trunk_ok=False, knee_travel_ok=False)
        text = _fallback(v)
        self.assertIn("chest up", text)
        self.assertIn("front knee", text)

    def test_llm_message_lists_selected_faults(self):
        v = lunge_verdict(depth_ok=True, trunk_ok=False, knee_travel_ok=False)
        msg = _verdict_to_text(v)
        self.assertIn("Torso", msg)
        self.assertIn("Front knee", msg)

    def test_llm_message_clean_rep_asks_for_praise(self):
        msg = _verdict_to_text(squat_verdict())
        self.assertIn("passed", msg.lower())


if __name__ == "__main__":
    unittest.main()
