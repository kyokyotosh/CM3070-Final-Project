import math

# MediaPipe Pose landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Below this a landmark reading is not trusted. Shared by the depth and
# trunk gating so that both measurements apply the same confidence floor.
MIN_VISIBILITY = 0.5


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


def trunk_visibility(landmarks):
    """Lowest visibility among the four landmarks used for the trunk vector.

    A trunk angle computed from a dropped shoulder or hip is meaningless, so
    callers gate on this value."""
    return min(
        landmarks[LEFT_SHOULDER].get("visibility", 0.0),
        landmarks[RIGHT_SHOULDER].get("visibility", 0.0),
        landmarks[LEFT_HIP].get("visibility", 0.0),
        landmarks[RIGHT_HIP].get("visibility", 0.0),
    )


def analyze_landmarks(landmarks):
    """Compute squat angles, using the more visible leg.

    If the shoulder or hip landmarks are not confidently detected, trunk_lean
    is None and trunk_valid is False."""
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

    trunk_ok = trunk_visibility(landmarks) >= MIN_VISIBILITY
    trunk = round(trunk_lean(mid_shoulder, mid_hip), 1) if trunk_ok else None

    return {
        "knee": round(knee, 1),
        "side": side,
        "knee_visibility": round(knee_vis, 2),
        "trunk_lean": trunk,
        "trunk_valid": trunk_ok,
    }


def _sign(v):
    return -1.0 if v < 0 else 1.0


def analyze_lunge(landmarks):
    """Compute forward-lunge angles from a side-on view.

    Both knees bend during a lunge, so the front leg is taken as the ankle
    furthest from the hip midpoint along the horizontal axis. A session should
    use one lead leg throughout.

    Returns the front-knee angle, the front side and its visibility, the gated
    trunk lean, and the knee-travel ratio: how far the front knee sits ahead of
    the front ankle, divided by shin length so it does not depend on distance
    from the camera."""
    if not landmarks or len(landmarks) < 33:
        return None

    mid_hip = midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])

    left_off = landmarks[LEFT_ANKLE]["x"] - mid_hip["x"]
    right_off = landmarks[RIGHT_ANKLE]["x"] - mid_hip["x"]

    # Front leg is the ankle furthest from the hip centre line; its sign
    # gives the forward direction of the step.
    if abs(left_off) >= abs(right_off):
        front, forward = "left", _sign(left_off)
        hip_i, knee_i, ankle_i = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
    else:
        front, forward = "right", _sign(right_off)
        hip_i, knee_i, ankle_i = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE

    front_knee = angle_at(
        landmarks[hip_i], landmarks[knee_i], landmarks[ankle_i])
    front_vis = leg_visibility(landmarks, hip_i, knee_i, ankle_i)

    # Positive when the knee is past the toes. Divided by shin length
    # (vertical knee-to-ankle distance) to remove the effect of camera
    # distance.
    knee_ahead = (landmarks[knee_i]["x"] - landmarks[ankle_i]["x"]) * forward
    shin = abs(landmarks[knee_i]["y"] - landmarks[ankle_i]["y"])
    if shin < 1e-6:
        shin = 1e-6
    knee_travel_ratio = knee_ahead / shin

    mid_shoulder = midpoint(
        landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    trunk_ok = trunk_visibility(landmarks) >= MIN_VISIBILITY
    trunk = round(trunk_lean(mid_shoulder, mid_hip), 1) if trunk_ok else None

    return {
        "front_knee": round(front_knee, 1),
        "front_side": front,
        "front_visibility": round(front_vis, 2),
        "knee_travel_ratio": round(knee_travel_ratio, 2),
        "trunk_lean": trunk,
        "trunk_valid": trunk_ok,
    }
