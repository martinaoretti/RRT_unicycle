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


# alias
Pose = Tuple[float, float, float]
ObstacleList = List[ShapelyPolygon]


# controllo confini del workspace 
def _in_bounds(x: float, y: float,
               x_min: float, y_min: float,
               x_max: float, y_max: float) -> bool: #riceve limiti della mappa e una posizione
    """Return True if (x, y) is within the workspace."""
    return x_min <= x <= x_max and y_min <= y <= y_max



#controlla se in una collisione il robot collide con qualsiasi ostacolo o confine -->se collide ritorna True
def check_pose_collision(
    robot: Robot,
    pose: Pose,
    obstacles: ObstacleList,
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
) -> bool:
    x, y, _ = pose
    if not _in_bounds(x, y, *bounds):
        return True

    footprint: ShapelyPolygon = robot.footprint_at(pose)
    for obs in obstacles:
        if footprint.intersects(obs):   #se robot tocca l'ostacolo
            return True
    return False


#controllo ogni posa lungo l'arco per collisioni
#ritorna:
#collision_free True se l'intero arco va bene
#last_free_index indice dell'ultima posa corretta (-1 se anche la prima colide-->arco inutilizzabile)

def check_arc_collision(
    robot: Robot,
    poses: List[Pose],
    obstacles: ObstacleList,
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
) -> Tuple[bool, int]:
    
    last_free: int = -1

    for idx, pose in enumerate(poses): #scorre tutte le pose
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
    """ True when the pose is collision-free."""
    return not check_pose_collision(robot, pose, obstacles, bounds)
