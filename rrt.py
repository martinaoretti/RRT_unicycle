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


# costanti
GOAL_BIAS: float = 0.15           # probability of sampling the goal 15%
GOAL_TOLERANCE: float = 4.0       # tolleranza per dichiarare successo
NEAREST_FALLBACK_K: int = 5       # prova i K nearest nodes (se il primo fallisce)
MIN_ARC_STEPS: int = 2            # rifiuta archi troppo corti


# strutture dati
@dataclass
class RRTNode: #definisco nodo
    """Un nodo nell'albero RRT che memorizza lo stato completo dell'uniciclo"""
    x: float
    y: float
    theta: float
    parent: Optional[int] = None          # indice nella lista albero(nodo padre)
    arc_poses: List[Tuple[float, float, float]] = field(default_factory=list)
    # lista delle pose intermedie lungo l'arco che collega il padre( per disegno)


@dataclass
class RRTResult:    #oggetto che contiene il risultato finale
    """Container returned by the planner."""
    tree: List[RRTNode]
    path_indices: Optional[List[int]]     # indici dei nodi da start a goal 
    success: bool
    snapshots: List[List[RRTNode]]        # istantanee albero per animazione



# funzioni-----------------------------------------------------
def _euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    """Distanza euclidea tra due punti """
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _k_nearest(tree: List[RRTNode], px: float, py: float, k: int) -> List[int]:
    """Ritorna gli indici dei k closest nodes con la distanza euclidea"""
    dists = [(_euclidean(n.x, n.y, px, py), idx) for idx, n in enumerate(tree)]
    dists.sort(key=lambda t: t[0]) #ordina per distanza
    return [idx for _, idx in dists[:k]] #restituisce i primi k


def _extract_path(tree: List[RRTNode], goal_idx: int) -> List[int]:
    """Ripercorre i puntatori del genitore fino alla radice e restituisce gli indici in ordine progressivo. 
    Restituisce il percorso dalla destinazione al punto di partenza."""
    path: List[int] = []
    idx: Optional[int] = goal_idx #parto da goal
    while idx is not None: #seguo i genitori
        path.append(idx)
        idx = tree[idx].parent
    path.reverse()
    return path



# Main RRT
def rrt(    #costruisce l'albero fino a trovare il goal
    robot: Robot,
    obstacles: ObstacleList,
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    num_iterations: int = 2000, #massimo numero campionamento
    workspace: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
    dt: float = DEFAULT_DT, #passo temporale di propagazione
    velocity: float = DEFAULT_VELOCITY, #velocità lineare
    max_steps: int = MAX_STEPS_PER_ARC, #limite al numero di passi di propagazione per arco
    snapshot_interval: int = 10, #salva istantanea albero ogni N iterazione
    goal_check_interval: int = 50, #ogni N iterazione prova a connettersi direttamente al goal 

) -> RRTResult:
   
    x_min, y_min, x_max, y_max = workspace

    root = RRTNode(x=start[0], y=start[1], theta=start[2])
    tree: List[RRTNode] = [root]
    snapshots: List[List[RRTNode]] = [_snapshot(tree)]

    for iteration in range(1, num_iterations + 1):
        #sample 
        if random.random() < GOAL_BIAS: #con pr 15% scelgo il goal
            sx, sy = goal[0], goal[1]
        else: #altrimenti punto casuale
            sx = random.uniform(x_min, x_max)
            sy = random.uniform(y_min, y_max)

        #estensione con campionamento
        new_node = _try_extend(tree, sx, sy, robot, obstacles, workspace, dt, velocity, max_steps)
        if new_node is not None:
            tree.append(new_node)

        #tentativo periodico di connessione al goal ogni N iterazioni
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
            snapshots.append(_snapshot(tree)) #salva stato albero

    # iterazioni esaurite - nessun path trovato 
    snapshots.append(_snapshot(tree))
    return RRTResult(tree=tree, path_indices=None,
                     success=False, snapshots=snapshots)



# funzioni interne------------------------------------------

def _snapshot(tree: List[RRTNode]) -> List[RRTNode]:
    """copia superficiale lista nodi per animazione """
    return list(tree)


def _try_extend(
    tree: List[RRTNode],
    sx: float, sy: float,
    robot: Robot,
    obstacles: ObstacleList,
    bounds: Tuple[float, float, float, float],
    dt: float, velocity: float, max_steps: int,
) -> Optional[RRTNode]:
    """prova a far crescere l'albero verso il punto campionato, provando i k-nearest nodes"""
    candidates = _k_nearest(tree, sx, sy, NEAREST_FALLBACK_K)

    for near_idx in candidates:
        near = tree[near_idx]
        arc = compute_arc(near.x, near.y, near.theta, sx, sy)
        if arc is None: #se impossibile -->salta
            continue

        prop = propagate(near.x, near.y, near.theta, arc, dt=dt, velocity=velocity, max_steps=max_steps) #simula movimento
        if len(prop.poses) < MIN_ARC_STEPS:
            continue

        ok, last_free = check_arc_collision(robot, prop.poses, obstacles, bounds) 

        if ok:
            # l'arco è completamente collision-free
            final = prop.poses[-1]
            node = RRTNode(x=final[0], y=final[1], theta=final[2],
                           parent=near_idx, arc_poses=list(prop.poses))
            return node
        elif last_free >= MIN_ARC_STEPS:
            # l'arco è parzialmente collision-free 
            final = prop.poses[last_free]
            node = RRTNode(x=final[0], y=final[1], theta=final[2],
                           parent=near_idx,
                           arc_poses=list(prop.poses[:last_free + 1]))
            return node

    return None  # nessuno dei k era estendibile


def _try_connect_goal(
    tree: List[RRTNode],
    goal: Tuple[float, float, float],
    robot: Robot,
    obstacles: ObstacleList,
    bounds: Tuple[float, float, float, float],
    dt: float, velocity: float, max_steps: int,
) -> Optional[RRTNode]:
    """prova a connettere il nodo piu vicino direttamente con il goal"""

    gx, gy, g_theta = goal
    goal_max_dist = STEP_SIZE * 3.0  # Per la connessione al goal si permette un arco più lungo (STEP_SIZEx3)
    candidates = _k_nearest(tree, gx, gy, min(len(tree), 10)) #10 nearest al goal

    for near_idx in candidates:
        near = tree[near_idx]
        dist_to_goal = _euclidean(near.x, near.y, gx, gy)
        if dist_to_goal > goal_max_dist: #se troppo lontano salta
            continue

        arc = compute_arc(near.x, near.y, near.theta, gx, gy)
        if arc is None:
            continue

        # per la connessione con il goal: propaga l'arco completo (non limitato da STEP_SIZE)
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
        if _euclidean(final[0], final[1], gx, gy) < GOAL_TOLERANCE: #se vicino al goal-->successo
            node = RRTNode(x=final[0], y=final[1], theta=final[2],
                           parent=near_idx, arc_poses=list(prop.poses))
            return node

    return None
