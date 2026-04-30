"""
collision.py – Check the robot shape for collisions at each step of an arc.

Each discrete pose from the unicycle propagation is turned into the robot
footprint in the world, then tested against obstacles with Shapely. If a
collision happens, we keep the last pose that was still safe.
"""

from __future__ import annotations

from typing import List, Tuple

from shapely.geometry import Polygon as ShapelyPolygon

from robot import Robot

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Pose = Tuple[float, float, float]
ObstacleList = List[ShapelyPolygon]


# ---------------------------------------------------------------------------
# Workspace boundary check
# ---------------------------------------------------------------------------

def _in_bounds(x: float, y: float,
               x_min: float, y_min: float,
               x_max: float, y_max: float) -> bool:
    """Return True if (x, y) is within the rectangular workspace."""
    return x_min <= x <= x_max and y_min <= y <= y_max


# ---------------------------------------------------------------------------
# Core collision checking
# ---------------------------------------------------------------------------

def check_pose_collision(
    robot: Robot,
    pose: Pose,
    obstacles: ObstacleList,
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
) -> bool:
    """Return **True** if the robot at *pose* collides with any obstacle
    or is out of bounds.

    Parameters
    ----------
    robot     : Robot instance (polygon footprint).
    pose      : (x, y, θ).
    obstacles : list of Shapely polygons representing obstacles.
    bounds    : (x_min, y_min, x_max, y_max) workspace limits.
    """
    x, y, _ = pose
    if not _in_bounds(x, y, *bounds):
        return True

    footprint: ShapelyPolygon = robot.footprint_at(pose)
    for obs in obstacles:
        if footprint.intersects(obs):
            return True
    return False


def check_arc_collision(
    robot: Robot,
    poses: List[Pose],
    obstacles: ObstacleList,
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
) -> Tuple[bool, int]:
    """Check every pose along a discretised arc for collisions.

    Parameters
    ----------
    robot     : Robot instance.
    poses     : ordered sequence of (x, y, θ) from ``unicycle.propagate``.
    obstacles : Shapely polygons.
    bounds    : workspace rectangle.

    Returns
    -------
    ``(collision_free, last_free_index)``

    * *collision_free* is True when the **entire** arc is safe.
    * *last_free_index* is the index of the last valid pose
      (−1 if even the first step collides → arc is unusable).
    """
    last_free: int = -1

    for idx, pose in enumerate(poses):
        if check_pose_collision(robot, pose, obstacles, bounds):
            return False, last_free
        last_free = idx

    return True, last_free


def is_pose_free(
    robot: Robot,
    pose: Pose,
    obstacles: ObstacleList,
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
) -> bool:
    """Convenience wrapper: True when the pose is collision-free."""
    return not check_pose_collision(robot, pose, obstacles, bounds)
