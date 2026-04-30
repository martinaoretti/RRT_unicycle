"""
obstacles.py – Preset obstacle layouts and the random scene generator.

Each configuration function returns:
    (obstacles, vert_arrays, start_region, goal_region)

where:
    obstacles   : list of Shapely polygons
    vert_arrays : list of Nx2 numpy arrays (for matplotlib)
    start_region: (lo, hi) for sampling start pose x,y
    goal_region : (lo, hi) for sampling goal pose x,y
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Callable

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from collision import ObstacleList

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Region = Tuple[float, float]
ObstacleConfig = Tuple[ObstacleList, List[np.ndarray], Region, Region]
ConfigFactory = Callable[[], ObstacleConfig]

WORKSPACE_MIN: float = 0.0
WORKSPACE_MAX: float = 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poly(vertices: List[Tuple[float, float]]) -> ShapelyPolygon:
    """Create a Shapely polygon from a list of (x, y) vertices."""
    return ShapelyPolygon(vertices)


def _pack(
    polygons: List[ShapelyPolygon],
    start_region: Region = (5.0, 35.0),
    goal_region: Region = (65.0, 95.0),
) -> ObstacleConfig:
    """Convert a list of Shapely polygons into the standard return tuple."""
    vert_arrays = [np.array(p.exterior.coords) for p in polygons]
    return polygons, vert_arrays, start_region, goal_region


def _random_convex_polygon(
    cx: float, cy: float, size: float, n_verts: int,
) -> ShapelyPolygon:
    """Generate a random convex polygon centred roughly at (cx, cy)."""
    angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(n_verts)])
    verts = [(cx + random.uniform(size * 0.4, size) * math.cos(a),
              cy + random.uniform(size * 0.4, size) * math.sin(a))
             for a in angles]
    return ShapelyPolygon(verts).convex_hull


def _rect(x: float, y: float, w: float, h: float) -> ShapelyPolygon:
    """Axis-aligned rectangle with bottom-left corner at (x, y)."""
    return _poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


# ---------------------------------------------------------------------------
# Configuration: RANDOM  (original behaviour)
# ---------------------------------------------------------------------------

def config_random() -> ObstacleConfig:
    """3–5 random convex polygons, non-overlapping."""
    n = random.randint(3, 5)
    obstacles: ObstacleList = []
    attempts = 0
    while len(obstacles) < n and attempts < 500:
        attempts += 1
        cx = random.uniform(25, 75)
        cy = random.uniform(25, 75)
        nv = random.randint(3, 6)
        poly = _random_convex_polygon(cx, cy, 18.0, nv)
        bx0, by0, bx1, by1 = poly.bounds
        if bx0 < WORKSPACE_MIN or by0 < WORKSPACE_MIN:
            continue
        if bx1 > WORKSPACE_MAX or by1 > WORKSPACE_MAX:
            continue
        if any(poly.intersects(prev) for prev in obstacles):
            continue
        obstacles.append(poly)
    return _pack(obstacles)


# ---------------------------------------------------------------------------
# Configuration: NARROW_PASSAGE
# Two large walls with a small gap in the middle.
# ---------------------------------------------------------------------------

def config_narrow_passage() -> ObstacleConfig:
    """Two horizontal walls with a narrow gap — tests tight manoeuvring."""
    wall_top = _rect(0, 55, 100, 8)
    wall_bot = _rect(0, 37, 100, 8)
    # gap: remove a section from each wall
    gap_left = 42.0
    gap_right = 58.0
    # clip walls to leave a passage
    from shapely.geometry import box
    clip = box(gap_left, 30, gap_right, 65)
    top_clipped = wall_top.difference(clip)
    bot_clipped = wall_bot.difference(clip)
    # convert MultiPolygon to list if needed
    obstacles: ObstacleList = []
    for geom in [top_clipped, bot_clipped]:
        if geom.geom_type == "MultiPolygon":
            obstacles.extend(list(geom.geoms))
        else:
            obstacles.append(geom)
    return _pack(obstacles,
                 start_region=(5.0, 30.0),
                 goal_region=(70.0, 95.0))


# ---------------------------------------------------------------------------
# Configuration: MAZE
# Grid of rectangular blocks forming corridors.
# ---------------------------------------------------------------------------

def config_maze() -> ObstacleConfig:
    """Simple grid-based maze with corridors between blocks."""
    blocks: ObstacleList = []
    # 3×3 grid of blocks with gaps
    block_w, block_h = 18.0, 18.0
    gap = 10.0
    x0, y0 = 12.0, 12.0
    for row in range(3):
        for col in range(3):
            # skip some blocks to create passages
            if (row, col) in [(0, 0), (1, 1), (2, 2), (0, 2)]:
                continue
            bx = x0 + col * (block_w + gap)
            by = y0 + row * (block_h + gap)
            blocks.append(_rect(bx, by, block_w, block_h))
    return _pack(blocks,
                 start_region=(2.0, 20.0),
                 goal_region=(80.0, 98.0))


# ---------------------------------------------------------------------------
# Configuration: CLUTTERED
# Many small obstacles scattered across the workspace.
# ---------------------------------------------------------------------------

def config_cluttered() -> ObstacleConfig:
    """12–18 small obstacles densely spread across the space."""
    n = random.randint(12, 18)
    obstacles: ObstacleList = []
    attempts = 0
    while len(obstacles) < n and attempts < 1000:
        attempts += 1
        cx = random.uniform(15, 85)
        cy = random.uniform(15, 85)
        nv = random.randint(3, 5)
        poly = _random_convex_polygon(cx, cy, 8.0, nv)
        bx0, by0, bx1, by1 = poly.bounds
        if bx0 < WORKSPACE_MIN or by0 < WORKSPACE_MIN:
            continue
        if bx1 > WORKSPACE_MAX or by1 > WORKSPACE_MAX:
            continue
        if any(poly.intersects(prev) for prev in obstacles):
            continue
        obstacles.append(poly)
    return _pack(obstacles,
                 start_region=(2.0, 15.0),
                 goal_region=(85.0, 98.0))


# ---------------------------------------------------------------------------
# Configuration: L_SHAPED
# Large L-shaped wall that forces a detour around it.
# ---------------------------------------------------------------------------

def config_l_shaped() -> ObstacleConfig:
    """One large L-shaped obstacle blocking the direct path."""
    # vertical bar
    vert = _rect(40, 10, 12, 60)
    # horizontal bar
    horiz = _rect(40, 55, 40, 12)
    # extra small blocks for variety
    block1 = _rect(15, 60, 10, 10)
    block2 = _rect(70, 20, 12, 12)
    return _pack([vert, horiz, block1, block2],
                 start_region=(5.0, 30.0),
                 goal_region=(70.0, 95.0))


# ---------------------------------------------------------------------------
# Configuration: DIAGONAL_WALLS
# Rotated walls creating diagonal corridors.
# ---------------------------------------------------------------------------

def config_diagonal_walls() -> ObstacleConfig:
    """Diagonal walls crossing the workspace at ~45°."""
    def _rotated_rect(cx: float, cy: float, w: float, h: float,
                      angle: float) -> ShapelyPolygon:
        """Rectangle centred at (cx,cy) rotated by *angle* radians."""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        hw, hh = w / 2, h / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        rotated = [(cx + x * cos_a - y * sin_a,
                     cy + x * sin_a + y * cos_a) for x, y in corners]
        return _poly(rotated)

    wall1 = _rotated_rect(35, 35, 50, 6, math.radians(45))
    wall2 = _rotated_rect(65, 65, 50, 6, math.radians(45))
    wall3 = _rotated_rect(50, 80, 30, 5, math.radians(-30))
    return _pack([wall1, wall2, wall3],
                 start_region=(2.0, 20.0),
                 goal_region=(75.0, 98.0))


# ---------------------------------------------------------------------------
# Configuration: CONCENTRIC
# Concentric polygonal rings with gaps — tests navigation through layers.
# ---------------------------------------------------------------------------

def config_concentric() -> ObstacleConfig:
    """Two concentric octagonal rings with gaps."""
    def _ring_segment(cx: float, cy: float, r_in: float, r_out: float,
                      a_start: float, a_end: float,
                      n_pts: int = 8) -> ShapelyPolygon:
        angles_out = [a_start + i * (a_end - a_start) / n_pts
                      for i in range(n_pts + 1)]
        angles_in = list(reversed(angles_out))
        outer = [(cx + r_out * math.cos(a), cy + r_out * math.sin(a))
                 for a in angles_out]
        inner = [(cx + r_in * math.cos(a), cy + r_in * math.sin(a))
                 for a in angles_in]
        return _poly(outer + inner)

    cx, cy = 50.0, 50.0
    obstacles: ObstacleList = []

    # Outer ring: 3 segments with 2 gaps
    for a0, a1 in [(0.4, 1.8), (2.2, 3.8), (4.2, 5.9)]:
        obstacles.append(_ring_segment(cx, cy, 32, 38, a0, a1))

    # Inner ring: 2 segments with 2 gaps
    for a0, a1 in [(0.8, 2.5), (3.5, 5.5)]:
        obstacles.append(_ring_segment(cx, cy, 16, 22, a0, a1))

    return _pack(obstacles,
                 start_region=(2.0, 18.0),
                 goal_region=(45.0, 55.0))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONFIGURATIONS: Dict[str, ConfigFactory] = {
    "random":          config_random,
    "narrow_passage":  config_narrow_passage,
    "maze":            config_maze,
    "cluttered":       config_cluttered,
    "l_shaped":        config_l_shaped,
    "diagonal_walls":  config_diagonal_walls,
    "concentric":      config_concentric,
}


def list_configs() -> List[str]:
    """Return sorted list of available configuration names."""
    return sorted(CONFIGURATIONS.keys())


def get_config(name: str) -> ObstacleConfig:
    """Build and return the named obstacle configuration.

    Raises KeyError if *name* is not in the registry.
    """
    if name not in CONFIGURATIONS:
        available = ", ".join(list_configs())
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    return CONFIGURATIONS[name]()
