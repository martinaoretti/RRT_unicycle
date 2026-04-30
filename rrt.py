"""
rrt.py – Rapidly-exploring Random Tree planner for a unicycle robot.

Each node stores the robot pose (x, y, θ), and each edge is a circular arc
that the unicycle can actually follow. If the closest node can’t reach a
sampled point, the planner tries the next few nearest nodes instead.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from shapely.geometry import Polygon as ShapelyPolygon

from robot import Robot
from unicycle import (
    ArcParams,
    PropagationResult,
    compute_arc,
    propagate,
    DEFAULT_DT,
    DEFAULT_VELOCITY,
    MAX_STEPS_PER_ARC,
    STEP_SIZE,
)
from collision import (
    ObstacleList,
    check_arc_collision,
    is_pose_free,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOAL_BIAS: float = 0.15           # probability of sampling the goal
GOAL_TOLERANCE: float = 4.0       # Euclidean threshold to declare success
NEAREST_FALLBACK_K: int = 5       # try up to K nearest nodes
MIN_ARC_STEPS: int = 2            # reject arcs shorter than this

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RRTNode:
    """A node in the RRT tree storing full unicycle state."""
    x: float
    y: float
    theta: float
    parent: Optional[int] = None          # index into tree list
    arc_poses: List[Tuple[float, float, float]] = field(default_factory=list)
    # intermediate poses along the arc from parent → this node (for drawing)


@dataclass
class RRTResult:
    """Container returned by the planner."""
    tree: List[RRTNode]
    path_indices: Optional[List[int]]     # node indices from start to goal
    success: bool
    snapshots: List[List[RRTNode]]        # tree snapshots for animation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance on (x, y) only."""
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _k_nearest(tree: List[RRTNode], px: float, py: float, k: int) -> List[int]:
    """Return indices of the *k* closest nodes by Euclidean (x, y) distance.

    Complexity: O(n log n) due to sort — acceptable per spec.
    """
    dists = [(_euclidean(n.x, n.y, px, py), idx) for idx, n in enumerate(tree)]
    dists.sort(key=lambda t: t[0])
    return [idx for _, idx in dists[:k]]


def _extract_path(tree: List[RRTNode], goal_idx: int) -> List[int]:
    """Walk parent pointers back to root and return forward-ordered indices."""
    path: List[int] = []
    idx: Optional[int] = goal_idx
    while idx is not None:
        path.append(idx)
        idx = tree[idx].parent
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Main RRT
# ---------------------------------------------------------------------------

def rrt(
    robot: Robot,
    obstacles: ObstacleList,
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    num_iterations: int = 2000,
    workspace: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
    dt: float = DEFAULT_DT,
    velocity: float = DEFAULT_VELOCITY,
    max_steps: int = MAX_STEPS_PER_ARC,
    snapshot_interval: int = 10,
    goal_check_interval: int = 50,
) -> RRTResult:
    """Run the RRT planner.

    Parameters
    ----------
    robot              : Robot polygon model.
    obstacles          : list of Shapely obstacle polygons.
    start              : (x, y, θ) initial pose.
    goal               : (x, y, θ) desired final pose (θ used only for the
                         last connection; sampling is 2-D in x, y).
    num_iterations     : maximum sampling rounds.
    workspace          : (x_min, y_min, x_max, y_max).
    dt                 : propagation time-step.
    velocity           : constant linear speed.
    max_steps          : cap on propagation steps per arc.
    snapshot_interval  : save tree snapshot every N iterations.
    goal_check_interval: try direct goal connection every N iterations.

    Returns
    -------
    ``RRTResult`` with tree, path, success flag, and animation snapshots.
    """
    x_min, y_min, x_max, y_max = workspace

    root = RRTNode(x=start[0], y=start[1], theta=start[2])
    tree: List[RRTNode] = [root]
    snapshots: List[List[RRTNode]] = [_snapshot(tree)]

    for iteration in range(1, num_iterations + 1):
        # ----- sample -----
        if random.random() < GOAL_BIAS:
            sx, sy = goal[0], goal[1]
        else:
            sx = random.uniform(x_min, x_max)
            sy = random.uniform(y_min, y_max)

        # ----- extend towards sample -----
        new_node = _try_extend(tree, sx, sy, robot, obstacles, workspace,
                               dt, velocity, max_steps)
        if new_node is not None:
            tree.append(new_node)

        # ----- periodic goal connection attempt -----
        if iteration % goal_check_interval == 0:
            goal_node = _try_connect_goal(tree, goal, robot, obstacles,
                                          workspace, dt, velocity, max_steps)
            if goal_node is not None:
                goal_idx = len(tree)
                tree.append(goal_node)
                path = _extract_path(tree, goal_idx)
                snapshots.append(_snapshot(tree))
                return RRTResult(tree=tree, path_indices=path,
                                 success=True, snapshots=snapshots)

        if iteration % snapshot_interval == 0:
            snapshots.append(_snapshot(tree))

    # exhausted iterations — no path found
    snapshots.append(_snapshot(tree))
    return RRTResult(tree=tree, path_indices=None,
                     success=False, snapshots=snapshots)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _snapshot(tree: List[RRTNode]) -> List[RRTNode]:
    """Cheap shallow copy of the tree list for animation."""
    return list(tree)


def _try_extend(
    tree: List[RRTNode],
    sx: float, sy: float,
    robot: Robot,
    obstacles: ObstacleList,
    bounds: Tuple[float, float, float, float],
    dt: float, velocity: float, max_steps: int,
) -> Optional[RRTNode]:
    """Attempt to grow the tree toward (sx, sy), trying up to K nearest
    nodes as kinematic fallback."""
    candidates = _k_nearest(tree, sx, sy, NEAREST_FALLBACK_K)

    for near_idx in candidates:
        near = tree[near_idx]
        arc = compute_arc(near.x, near.y, near.theta, sx, sy)
        if arc is None:
            continue

        prop = propagate(near.x, near.y, near.theta, arc,
                         dt=dt, velocity=velocity, max_steps=max_steps)
        if len(prop.poses) < MIN_ARC_STEPS:
            continue

        ok, last_free = check_arc_collision(robot, prop.poses, obstacles, bounds)

        if ok:
            # full arc is collision-free
            final = prop.poses[-1]
            node = RRTNode(x=final[0], y=final[1], theta=final[2],
                           parent=near_idx, arc_poses=list(prop.poses))
            return node
        elif last_free >= MIN_ARC_STEPS:
            # partial arc up to last collision-free pose
            final = prop.poses[last_free]
            node = RRTNode(x=final[0], y=final[1], theta=final[2],
                           parent=near_idx,
                           arc_poses=list(prop.poses[:last_free + 1]))
            return node

    return None  # none of the K nearest could extend


def _try_connect_goal(
    tree: List[RRTNode],
    goal: Tuple[float, float, float],
    robot: Robot,
    obstacles: ObstacleList,
    bounds: Tuple[float, float, float, float],
    dt: float, velocity: float, max_steps: int,
) -> Optional[RRTNode]:
    """Try connecting the closest tree nodes directly to the goal.

    For goal connection we allow a longer arc (up to 3× STEP_SIZE) so the
    planner can close the last gap without requiring the tree to land
    exactly on the goal.
    """
    gx, gy, g_theta = goal
    goal_max_dist = STEP_SIZE * 3.0  # allow longer reach for goal
    candidates = _k_nearest(tree, gx, gy, min(len(tree), 10))

    for near_idx in candidates:
        near = tree[near_idx]
        dist_to_goal = _euclidean(near.x, near.y, gx, gy)
        if dist_to_goal > goal_max_dist:
            continue

        arc = compute_arc(near.x, near.y, near.theta, gx, gy)
        if arc is None:
            continue

        # For goal connection: propagate the full arc (not clamped to STEP_SIZE)
        goal_arc_length = min(arc.arc_length, goal_max_dist)
        prop = propagate(near.x, near.y, near.theta, arc,
                         dt=dt, velocity=velocity, max_steps=max_steps,
                         override_length=goal_arc_length)
        if len(prop.poses) < MIN_ARC_STEPS:
            continue

        ok, _ = check_arc_collision(robot, prop.poses, obstacles, bounds)
        if not ok:
            continue

        final = prop.poses[-1]
        if _euclidean(final[0], final[1], gx, gy) < GOAL_TOLERANCE:
            node = RRTNode(x=final[0], y=final[1], theta=final[2],
                           parent=near_idx, arc_poses=list(prop.poses))
            return node

    return None
