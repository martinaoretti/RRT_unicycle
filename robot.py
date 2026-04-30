"""
robot.py – Polygonal robot footprint, moved and rotated in the world.

The robot is modelled as a simple convex polygon (by default a small
rectangle). Every collision check uses the full robot shape, not just a
single reference point.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Pose = Tuple[float, float, float]          # (x, y, theta)
Vertex = Tuple[float, float]

# ---------------------------------------------------------------------------
# Default robot shape (body-frame, centred at origin)
# ---------------------------------------------------------------------------
DEFAULT_ROBOT_VERTICES: List[Vertex] = [
    (-1.5, -1.0),
    ( 1.5, -1.0),
    ( 1.5,  1.0),
    (-1.5,  1.0),
]


class Robot:
    """Non-point-like mobile robot described by a polygon footprint."""

    def __init__(self, vertices: List[Vertex] | None = None) -> None:
        """
        Parameters
        ----------
        vertices : list of (x, y) tuples **in body frame** (centred at the
                   robot's reference point, typically the rear-axle midpoint).
                   If *None*, ``DEFAULT_ROBOT_VERTICES`` is used.
        """
        self.body_vertices: List[Vertex] = list(vertices or DEFAULT_ROBOT_VERTICES)

    # ------------------------------------------------------------------
    # Roto-translation
    # ------------------------------------------------------------------
    def footprint_at(self, pose: Pose) -> ShapelyPolygon:
        """Return the Shapely polygon of the robot placed at *pose*.

        Parameters
        ----------
        pose : (x, y, theta) – position and heading in world frame (rad).

        Returns
        -------
        ShapelyPolygon with the robot outline in world coordinates.
        """
        x, y, theta = pose
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        world_verts: List[Vertex] = []
        for bx, by in self.body_vertices:
            wx = x + bx * cos_t - by * sin_t
            wy = y + bx * sin_t + by * cos_t
            world_verts.append((wx, wy))

        return ShapelyPolygon(world_verts)

    def footprint_vertices_at(self, pose: Pose) -> np.ndarray:
        """Return an (N, 2) array of world-frame vertices – handy for
        matplotlib visualisation."""
        poly = self.footprint_at(pose)
        return np.array(poly.exterior.coords)
