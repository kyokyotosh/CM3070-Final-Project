"""Pose-tracking quality over the recorded sessions.

Answers the evaluation's pose-tracking target: in what share of recorded
frames would the live analysers accept the pose estimate as usable?

Every frame is passed through the same geometry functions the server uses
(angles.analyze_landmarks for squat, angles.analyze_lunge for lunge) and
checked against the same gates the analysers apply, so this measures the
deployed acceptance rule rather than a re-implementation of it:

  leg gate      the leg used for depth has mean visibility >= MIN_VISIBILITY
  trunk gate    all four shoulder and hip landmarks >= MIN_VISIBILITY
  plausibility  the knee angle is at or above KNEE_MIN_PLAUSIBLE

A frame is "usable" when all three hold. "other" sessions are measured with
the squat geometry for completeness but are excluded from the headline, since
the target concerns frames in which an exercise is being analysed.

Run from backend/:

    python3 pose_quality.py
    python3 pose_quality.py --dir ../training/sessions --out evaluations/pose_quality.csv
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

from angles import MIN_VISIBILITY, analyze_landmarks, analyze_lunge
from squat import KNEE_MIN_PLAUSIBLE

DEFAULT_DIR = os.path.join("..", "training", "sessions")
DEFAULT_OUT = os.path.join("evaluations", "pose_quality.csv")
TARGET = 0.95


def to_landmarks(frame):
    """Recorder frames hold 33 [x, y, z, visibility] arrays; the geometry
    functions expect the dict form the browser sends."""
    return [{"x": p[0], "y": p[1], "z": p[2], "visibility": p[3]}
            for p in frame["lm"]]


def measure_frame(landmarks, label):
    if label == "lunge":
        a = analyze_lunge(landmarks)
        if a is None:
            return None
        knee, leg_vis = a["front_knee"], a["front_visibility"]
    else:
        a = analyze_landmarks(landmarks)
        if a is None:
            return None
        knee, leg_vis = a["knee"], a["knee_visibility"]

    leg_ok = leg_vis >= MIN_VISIBILITY
    trunk_ok = bool(a["trunk_valid"])
    plausible = knee >= KNEE_MIN_PLAUSIBLE
    return {
        "leg_vis": leg_vis,
        "leg_ok": leg_ok,
        "trunk_ok": trunk_ok,
        "plausible": plausible,
        "usable": leg_ok and trunk_ok and plausible,
    }


def measure_session(path):
    with open(path) as f:
        data = json.load(f)

    label = data["label"]
    counts = defaultdict(int)
    vis_total = 0.0

    for frame in data["frames"]:
        counts["frames"] += 1
        m = measure_frame(to_landmarks(frame), label)
        if m is None:
            counts["no_pose"] += 1
            continue
        vis_total += m["leg_vis"]
        for key in ("leg_ok", "trunk_ok", "plausible", "usable"):
            counts[key] += int(m[key])

    measured = counts["frames"] - counts["no_pose"]
    return {
        "session_id": data["session_id"],
        "label": label,
        "distance": data.get("distance", "unknown"),
        "lighting": data.get("lighting", "unknown"),
        "frames": counts["frames"],
        "no_pose": counts["no_pose"],
        "leg_ok": counts["leg_ok"],
        "trunk_ok": counts["trunk_ok"],
        "plausible": counts["plausible"],
        "usable": counts["usable"],
        "mean_leg_vis": round(vis_total / measured, 3) if measured else 0.0,
    }


def rate(part, whole):
    return part / whole if whole else 0.0


def pooled(rows):
    frames = sum(r["frames"] for r in rows)
    return {
        "sessions": len(rows),
        "frames": frames,
        "leg": rate(sum(r["leg_ok"] for r in rows), frames),
        "trunk": rate(sum(r["trunk_ok"] for r in rows), frames),
        "plausible": rate(sum(r["plausible"] for r in rows), frames),
        "usable": rate(sum(r["usable"] for r in rows), frames),
        "no_pose": sum(r["no_pose"] for r in rows),
    }


def print_group(name, rows):
    p = pooled(rows)
    print(f"  {name:<22}{p['sessions']:>4}{p['frames']:>8}"
          f"{p['leg']:>8.3f}{p['trunk']:>8.3f}{p['plausible']:>8.3f}"
          f"{p['usable']:>8.3f}{p['no_pose']:>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if not paths:
        raise SystemExit(f"no recordings found in {args.dir}")

    rows = [measure_session(p) for p in paths]

    print(f"{'session':<30}{'frames':>7}{'leg':>7}{'trunk':>7}"
          f"{'plaus':>7}{'usable':>8}{'vis':>7}")
    for r in rows:
        f = r["frames"]
        print(f"{r['session_id']:<30}{f:>7}"
              f"{rate(r['leg_ok'], f):>7.3f}{rate(r['trunk_ok'], f):>7.3f}"
              f"{rate(r['plausible'], f):>7.3f}{rate(r['usable'], f):>8.3f}"
              f"{r['mean_leg_vis']:>7.2f}")

    exercise = [r for r in rows if r["label"] in ("squat", "lunge")]

    print(f"\n  {'group':<22}{'n':>4}{'frames':>8}{'leg':>8}{'trunk':>8}"
          f"{'plaus':>8}{'usable':>8}{'nopose':>8}")
    print_group("exercise (headline)", exercise)
    for label in ("squat", "lunge", "other"):
        print_group(label, [r for r in rows if r["label"] == label])
    for key in ("distance", "lighting"):
        for value in sorted({r[key] for r in exercise}):
            print_group(f"exercise, {value}",
                        [r for r in exercise if r[key] == value])

    headline = pooled(exercise)["usable"]
    verdict = "met" if headline >= TARGET else "NOT met"
    print(f"\nusable frames in exercise sessions: {headline:.3f} "
          f"against a target of {TARGET:.2f} ({verdict})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-session results written to {args.out}")


if __name__ == "__main__":
    main()
