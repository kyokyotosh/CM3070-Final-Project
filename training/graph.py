"""COCO-17 skeleton graph matching the pre-trained checkpoint.

Reproduces MMAction2's `coco` layout with `stgcn_spatial` partitioning. Graph
convolution weights only make sense with the adjacency they were trained on, so
this must match the checkpoint. model.load_pretrained uses the adjacency stored
in the checkpoint when present; this module is the fallback.

Each joint's neighbourhood is split into three: the joint itself, neighbours
nearer the graph centre and neighbours further from it.
"""

import numpy as np

# COCO-17 joint order. The recogniser's input is built in exactly this order
# by dataset.JOINTS.
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_JOINTS = len(JOINT_NAMES)

# Directed toward the graph centre, as MMAction2 defines them for this layout.
# Each pair is (child, parent).
INWARD = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
    (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
    (1, 0), (3, 1), (2, 0), (4, 2),
]

# The nose is the graph centre in this layout. It is not the body's centre of
# mass; it is simply the root the published adjacency is built around, and it
# must be kept to stay compatible with the pre-trained weights.
CENTRE = 0

MAX_HOP = 1


def hop_distance(num_node, edges, max_hop=MAX_HOP):
    """Shortest path length between joints, capped at max_hop."""
    adjacency = np.zeros((num_node, num_node))
    for i, j in edges:
        adjacency[i, j] = adjacency[j, i] = 1

    distance = np.zeros((num_node, num_node)) + np.inf
    transfer = [np.linalg.matrix_power(adjacency, d) for d in range(max_hop + 1)]
    arrived = (np.stack(transfer) > 0)
    for d in range(max_hop, -1, -1):
        distance[arrived[d]] = d
    return distance


def normalize_digraph(adjacency):
    """Column-normalise, so each convolution averages over a neighbourhood
    rather than summing over it."""
    degree = adjacency.sum(0)
    inverse = np.zeros_like(adjacency)
    for i in range(len(adjacency)):
        if degree[i] > 0:
            inverse[i, i] = degree[i] ** -1
    return adjacency @ inverse


def adjacency():
    """Return the (3, V, V) spatial adjacency: self, centripetal, centrifugal."""
    outward = [(j, i) for i, j in INWARD]
    neighbour = INWARD + outward

    distance = hop_distance(NUM_JOINTS, neighbour)
    reachable = np.zeros((NUM_JOINTS, NUM_JOINTS))
    reachable[distance <= MAX_HOP] = 1
    normalised = normalize_digraph(reachable)

    to_centre = distance[:, CENTRE]

    parts = []
    for hop in range(MAX_HOP + 1):
        closer = np.zeros((NUM_JOINTS, NUM_JOINTS))
        further = np.zeros((NUM_JOINTS, NUM_JOINTS))
        for i in range(NUM_JOINTS):
            for j in range(NUM_JOINTS):
                if distance[j, i] != hop:
                    continue
                if to_centre[j] >= to_centre[i]:
                    closer[j, i] = normalised[j, i]
                else:
                    further[j, i] = normalised[j, i]
        parts.append(closer)
        if hop > 0:
            parts.append(further)

    return np.stack(parts).astype(np.float32)
