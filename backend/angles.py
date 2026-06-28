import math

# MediaPipe Pose landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def midpoint(p, q):
    return {"x": (p["x"] + q["x"]) / 2, "y": (p["y"] + q["y"]) / 2}


def angle_at(a, b, c):
    """Angle in degrees at point b, formed by a-b-c, in the 2D image plane."""
    ang = math.degrees(
        math.atan2(c["y"] - b["y"], c["x"] - b["x"])
        - math.atan2(a["y"] - b["y"], a["x"] - b["x"])
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang


def trunk_lean(shoulder, hip):
    """Trunk angle away from vertical, in degrees.
    0 is perfectly upright; larger means leaning further forward."""
    dx = shoulder["x"] - hip["x"]
    dy = shoulder["y"] - hip["y"]
    # image vertical (up) is (0, -1)
    return abs(math.degrees(math.atan2(dx, -dy)))


def leg_visibility(landmarks, hip_idx, knee_idx, ankle_idx):
    vis = [
        landmarks[hip_idx].get("visibility", 0.0),
        landmarks[knee_idx].get("visibility", 0.0),
        landmarks[ankle_idx].get("visibility", 0.0),
    ]
    return sum(vis) / len(vis)


def analyze_landmarks(landmarks):
    """Compute squat-relevant angles, selecting the more visible leg."""
    if not landmarks or len(landmarks) < 33:
        return None

    left_knee = angle_at(landmarks[LEFT_HIP],
                         landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
    right_knee = angle_at(
        landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])

    left_vis = leg_visibility(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_vis = leg_visibility(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)

    if left_vis >= right_vis:
        knee, side, knee_vis = left_knee, "left", left_vis
    else:
        knee, side, knee_vis = right_knee, "right", right_vis

    mid_shoulder = midpoint(
        landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    mid_hip = midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
    trunk = trunk_lean(mid_shoulder, mid_hip)

    return {
        "knee": round(knee, 1),
        "side": side,
        "knee_visibility": round(knee_vis, 2),
        "trunk_lean": round(trunk, 1),
    }
