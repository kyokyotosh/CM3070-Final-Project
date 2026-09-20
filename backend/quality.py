"""Rep quality scoring.

The interface shows a 0-100 quality score for each repetition. That score is a
judgement about movement, so it is produced here, in the rule-based layer, and
never in the browser. The frontend renders the number it is given.

Scoring is a transparent penalty model rather than a learned one. A repetition
with no breached criterion scores 100. Each breached criterion subtracts a
fixed penalty plus a margin-scaled component, so a rep that misses depth by two
degrees does not score the same as one that misses it by twenty. Thresholds are
imported from the analyser modules so there is a single source of truth for the
calibrated values.

Every breached criterion is scored, including any fault beyond the two-fault
cap applied in coach.select_faults. The cap limits what is said aloud, not what
is measured. A criterion reported as None was not measured this rep and is not
penalised, in keeping with the visibility gating in angles.py.
"""

from squat import PARALLEL as SQUAT_DEPTH, TRUNK_MAX as SQUAT_TRUNK
from lunge import (DEPTH_TARGET as LUNGE_DEPTH, TRUNK_MAX as LUNGE_TRUNK,
                   KNEE_TRAVEL_MAX)

# A breach costs this much before its margin is considered, so that any fault
# is visible in the score even when the threshold was missed narrowly.
BASE_PENALTY = 15.0

# Upper bound on the margin-scaled component, so one gross fault cannot hide
# the presence of a second one by driving the score straight to zero.
MAX_MARGIN_PENALTY = 25.0

# Margin scaling. Angular criteria are scored per degree past the threshold.
# Knee travel is a shin-normalised ratio, where the observed fault margin is
# about 0.09, so it needs a much larger multiplier to be comparable.
DEGREE_SCALE = 1.0
TRAVEL_SCALE = 150.0


def _penalty(measured, threshold, scale, above_is_fault):
    """Penalty for one breached criterion.

    measured may be None when the criterion was flagged but the value was not
    recorded, in which case only the base penalty applies.
    """
    if measured is None:
        return BASE_PENALTY
    excess = (measured - threshold) if above_is_fault else (threshold - measured)
    if excess <= 0:
        # The verdict says this criterion failed but the stored value does not
        # show it. Charge the base penalty and trust the verdict, which is the
        # authority, rather than silently scoring the rep as clean.
        return BASE_PENALTY
    return BASE_PENALTY + min(MAX_MARGIN_PENALTY, excess * scale)


def score(verdict):
    """Return an integer quality score from 0 to 100 for a completed rep."""
    exercise = verdict.get("exercise", "squat")
    if exercise == "lunge":
        depth_threshold, trunk_threshold = LUNGE_DEPTH, LUNGE_TRUNK
    else:
        depth_threshold, trunk_threshold = SQUAT_DEPTH, SQUAT_TRUNK

    penalty = 0.0

    # Depth fails when the knee angle stays above the target, so a larger
    # measured minimum means a shallower rep.
    if verdict.get("depth_ok") is False:
        penalty += _penalty(verdict.get("min_knee"), depth_threshold,
                            DEGREE_SCALE, above_is_fault=True)

    if verdict.get("trunk_ok") is False:
        penalty += _penalty(verdict.get("max_trunk"), trunk_threshold,
                            DEGREE_SCALE, above_is_fault=True)

    if verdict.get("knee_travel_ok") is False:
        penalty += _penalty(verdict.get("max_knee_travel"), KNEE_TRAVEL_MAX,
                            TRAVEL_SCALE, above_is_fault=True)

    return int(round(max(0.0, 100.0 - penalty)))


def is_partial(verdict):
    """True when at least one criterion could not be measured this rep.

    The score is still returned, but it was computed from fewer criteria than
    usual and should be read with that in mind.
    """
    keys = ("depth_ok", "trunk_ok")
    if verdict.get("exercise") == "lunge":
        keys = keys + ("knee_travel_ok",)
    return any(verdict.get(k) is None for k in keys)
