"""
unicycle.py – Unicycle motion model and the steering arcs it can follow.

This module finds the circular arc that is tangent to the robot's current
heading and reaches a sampled point in the plane. It also steps that arc
forward in time with a discrete unicycle model:
    x' = v·cos θ,  y' = v·sin θ,  θ' = ω  (with ω = v / R).

When the goal lies almost straight ahead, the radius goes very large and
we treat the path as a near-straight segment instead of a tight turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RADIUS: float = 1e6          # pretend the turn is straight when the radius gets huge
MIN_RADIUS: float = 3.0          # don’t allow turns tighter than this so arcs stay sane
DEFAULT_VELOCITY: float = 1.0    # robot speed in world units per second
DEFAULT_DT: float = 0.2          # integration timestep in seconds
MAX_STEPS_PER_ARC: int = 200     # don’t simulate more than this many steps per arc
STEP_SIZE: float = 1.0          # max arc length we try in one planning step
MAX_ARC_ANGLE: float = math.pi * 0.75  # never turn more than 135° in one shot
COLLINEAR_TOL: float = 1e-6      # how close to a straight line we call “collinear”

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
Pose = Tuple[float, float, float]


@dataclass
class ArcParams:
    """Parameters describing one circular arc."""
    center_x: float        # centre of the turning circle
    center_y: float
    radius: float          # turning radius  (always > 0)
    omega: float           # angular rate  v / R, signed
    arc_length: float      # total arc length to travel
    direction: int         # +1 left (CCW), -1 right (CW)
    degenerate: bool = False  # True when target ≈ on heading line


@dataclass
class PropagationResult:
    """Sequence of poses produced by discrete propagation."""
    poses: List[Pose] = field(default_factory=list)
    arc_params: Optional[ArcParams] = None
    reached_goal: bool = False


# ---------------------------------------------------------------------------
# Arc geometry
# ---------------------------------------------------------------------------

def compute_arc(
    x_s: float, y_s: float, theta_s: float,
    x_g: float, y_g: float,
) -> Optional[ArcParams]:
    """Compute the unique circle tangent to θ_start at (x_s, y_s) passing
    through (x_g, y_g).

    **Geometry** (no shortcuts):

    1. The tangent direction at start is  t = (cos θ, sin θ).
    2. The centre must lie on the **perpendicular** to t through start:
       C = (x_s, y_s) + R · n,  where n = (-sin θ, cos θ)  (left normal).
    3. The constraint  |C − G| = R  yields a linear equation in R.
       Expanding and simplifying:
           |S − G|² − 2 R · d_perp = 0
       so  R = |S − G|² / (2 · d_perp)
       where d_perp = (G − S) · n  is the signed perpendicular projection.

    If the target is (almost) on the tangent line the radius diverges;
    we clamp to ``MAX_RADIUS`` and flag the arc as *degenerate*.

    Returns None only when the geometry is truly impossible (start ≡ goal
    or target behind the robot on the heading line).
    """
    dx = x_g - x_s
    dy = y_g - y_s
    dist_sq = dx * dx + dy * dy
    if dist_sq < 1e-12:
        return None  # start ≡ goal

    # Perpendicular (left-normal) to heading
    nx = -math.sin(theta_s)
    ny =  math.cos(theta_s)

    # Signed distance of goal from the heading line through start
    d_perp = dx * nx + dy * ny   # positive → goal is to the left

    if abs(d_perp) < COLLINEAR_TOL:
        # Goal lies (almost) on the heading line → degenerate straight arc
        # Check that the goal is *ahead* (positive tangent projection)
        d_tang = dx * math.cos(theta_s) + dy * math.sin(theta_s)
        if d_tang <= 0:
            return None  # goal is behind

        arc_len = min(math.sqrt(dist_sq), STEP_SIZE)
        return ArcParams(
            center_x=x_s + MAX_RADIUS * nx,
            center_y=y_s + MAX_RADIUS * ny,
            radius=MAX_RADIUS,
            omega=DEFAULT_VELOCITY / MAX_RADIUS,
            arc_length=arc_len,
            direction=1,
            degenerate=True,
        )

    # ---- Non-degenerate case ------------------------------------------------
    R_signed = dist_sq / (2.0 * d_perp)  # positive → left turn
    radius = abs(R_signed)

    # Reject arcs tighter than the minimum turning radius
    if radius < MIN_RADIUS:
        return None

    if radius > MAX_RADIUS:
        radius = MAX_RADIUS
        R_signed = math.copysign(MAX_RADIUS, R_signed)

    direction = 1 if R_signed > 0 else -1

    cx = x_s + R_signed * nx
    cy = y_s + R_signed * ny

    # Arc angle from start to goal
    angle_start = math.atan2(y_s - cy, x_s - cx)
    angle_goal  = math.atan2(y_g - cy, x_g - cx)

    if direction == 1:  # CCW (left turn)
        arc_angle = angle_goal - angle_start
        if arc_angle < 0:
            arc_angle += 2.0 * math.pi
    else:               # CW (right turn)
        arc_angle = angle_start - angle_goal
        if arc_angle < 0:
            arc_angle += 2.0 * math.pi

    # Clamp arc angle to avoid spiralling past the target
    if arc_angle > MAX_ARC_ANGLE:
        arc_angle = MAX_ARC_ANGLE

    arc_length = radius * arc_angle
    omega = DEFAULT_VELOCITY / radius * direction  # signed angular rate

    return ArcParams(
        center_x=cx,
        center_y=cy,
        radius=radius,
        omega=omega,
        arc_length=arc_length,
        direction=direction,
        degenerate=False,
    )


# ---------------------------------------------------------------------------
# Discrete propagation  (Euler integration of the unicycle model)
# ---------------------------------------------------------------------------

def propagate(
    x: float, y: float, theta: float,
    arc: ArcParams,
    dt: float = DEFAULT_DT,
    max_steps: int = MAX_STEPS_PER_ARC,
    velocity: float = DEFAULT_VELOCITY,
    override_length: Optional[float] = None,
) -> PropagationResult:
    """Propagate the unicycle model along *arc* with fixed time-step *dt*.

    Kinematic equations (Euler integration)::

        x_{k+1}     = x_k  + v · cos(θ_k) · Δt
        y_{k+1}     = y_k  + v · sin(θ_k) · Δt
        θ_{k+1}     = θ_k  + ω · Δt           (ω = v / R)

    Parameters
    ----------
    override_length : if given, use this as the arc length instead of
                      clamping to STEP_SIZE (used for goal connection).

    Returns a ``PropagationResult`` containing every intermediate pose.
    """
    poses: List[Pose] = [(x, y, theta)]
    travelled = 0.0
    step_dist = velocity * dt

    # Clamp the arc length to STEP_SIZE unless overridden
    if override_length is not None:
        effective_length = override_length
    else:
        effective_length = min(arc.arc_length, STEP_SIZE)
    n_steps = min(int(math.ceil(effective_length / step_dist)), max_steps)

    for _ in range(n_steps):
        remaining = effective_length - travelled
        if remaining <= 0:
            break
        actual_dt = dt if remaining >= step_dist else remaining / velocity

        x = x + velocity * math.cos(theta) * actual_dt
        y = y + velocity * math.sin(theta) * actual_dt
        theta = theta + arc.omega * actual_dt
        travelled += velocity * actual_dt
        poses.append((x, y, theta))

    result = PropagationResult(poses=poses, arc_params=arc)
    result.reached_goal = (travelled >= effective_length - 1e-6)
    return result
