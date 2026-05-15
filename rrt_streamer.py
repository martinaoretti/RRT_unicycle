"""
rrt_streamer.py  -  invia aggiornamenti passo-passo per visualizzazione web/animazione.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from robot import Robot
from collision import ObstacleList, is_pose_free, check_arc_collision
from unicycle import (
    compute_arc, propagate,
    DEFAULT_DT, DEFAULT_VELOCITY, MAX_STEPS_PER_ARC, STEP_SIZE,
)
from rrt import (
    RRTNode, _k_nearest, _euclidean, _extract_path,
    GOAL_BIAS, GOAL_TOLERANCE, NEAREST_FALLBACK_K, MIN_ARC_STEPS,
)
from obstacles import get_config #carica scenari ostacoli

Event = Dict[str, Any]

#conversione nodo in dizionario: trasforma nodo RRT in formato JSON (per inviarlo al frontend)
def _node_dict(node: RRTNode, idx: int) -> Dict[str, Any]:
    arc = []
    if node.arc_poses: #se il nodo ha un arco
        arc = [[round(p[0], 3), round(p[1], 3), round(p[2], 4)] for p in node.arc_poses] #arrotonda pose per JSON
    return {
        "idx":    idx,
        "x":      round(node.x, 3),
        "y":      round(node.y, 3),
        "theta":  round(node.theta, 4),
        "parent": node.parent,
        "arc":    arc,
    }

#unisce tutte le pose degli archi del percorso finale. Scorre tutti gli archi evitando duplicati e restituisce tutte le pose
def _flatten_path_poses(path_nodes: List[Dict[str, Any]]) -> List[List[float]]:
    poses: List[List[float]] = []
    for node in path_nodes:
        for pose in node.get('arc', []):
            if not poses or poses[-1] != pose:
                poses.append(pose)
    return poses


#Sceglie dove provare a estendere l'albero. Trova una pos casuale dove il robot non collide.
def _sample_free(
    robot,
    obstacles,
    lo: float,
    hi: float,
    workspace,
    target: Optional[Tuple[float, float]] = None,
):
    for _ in range(3000):
        x = random.uniform(lo, hi)
        y = random.uniform(lo, hi)
        if target is None: #se non c'è un goal verso a cui puntare-->orientazione casuale (tra -180° e +180°)
            t = random.uniform(-math.pi, math.pi)
        else:
            dx = target[0] - x
            dy = target[1] - y
            if abs(dx) < 1e-9 and abs(dy) < 1e-9: #se sono praticamente nulle
                t = random.uniform(-math.pi, math.pi)
            else:
                t = math.atan2(dy, dx) #usa atan2 per orientare l'arco verso il goal
        if is_pose_free(robot, (x, y, t), obstacles, workspace):
            return (x, y, t)
    raise RuntimeError(f"Cannot find free pose in [{lo}, {hi}]")


#calcola percorso e invia dati in tempo reale mentre l'albero cresce
def rrt_stream(
    config_name: str,
    seed: int = 42,
    num_iterations: int = 4000,
    goal_bias: float = GOAL_BIAS,
    goal_check_interval: int = 20,
    workspace: Tuple = (0.0, 0.0, 100.0, 100.0),
) -> Generator[Event, None, None]:
 
    random.seed(seed) #se uso lo stesso seme il robot farà sempre stesso percorso 
    np.random.seed(seed)

    x_min, y_min, x_max, y_max = workspace

    try:
        obstacles, vert_arrays, start_r, goal_r = get_config(config_name) #carica mappa scelta
    except KeyError as e:
        yield {"type": "error", "msg": str(e)}
        return

    robot = Robot()

    try:
        #crea una posizione libera per partenza e arrivo
        goal  = _sample_free(robot, obstacles, goal_r[0], goal_r[1], workspace) 
        start = _sample_free(robot, obstacles, start_r[0], start_r[1], workspace,
                             target=(goal[0], goal[1]))
    except RuntimeError as e:
        yield {"type": "error", "msg": str(e)}
        return

    obs_data = [
        [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in va]
        for va in vert_arrays
    ]

    #manda al browser un primo messaggio con ostacoli, mappa start e goal
    yield { 
        "type":      "init",
        "obstacles": obs_data,
        "start":     [round(start[0], 3), round(start[1], 3), round(start[2], 4)],
        "goal":      [round(goal[0],  3), round(goal[1],  3), round(goal[2],  4)],
        "workspace": list(workspace),
    }

    root = RRTNode(x=start[0], y=start[1], theta=start[2])
    tree: List[RRTNode] = [root] #istanzia albero

    def try_extend(sx, sy):
        for near_idx in _k_nearest(tree, sx, sy, NEAREST_FALLBACK_K):
            near = tree[near_idx]
            arc  = compute_arc(near.x, near.y, near.theta, sx, sy)
            if arc is None:
                continue
            prop = propagate(near.x, near.y, near.theta, arc,
                             dt=DEFAULT_DT, velocity=DEFAULT_VELOCITY,
                             max_steps=MAX_STEPS_PER_ARC)
            if len(prop.poses) < MIN_ARC_STEPS:
                continue
            ok, last_free = check_arc_collision(
                robot, prop.poses, obstacles, workspace)
            if ok:
                final = prop.poses[-1]
                return RRTNode(x=final[0], y=final[1], theta=final[2], parent=near_idx, arc_poses=list(prop.poses))
            
            elif last_free >= MIN_ARC_STEPS:
                final = prop.poses[last_free]
                return RRTNode(x=final[0], y=final[1], theta=final[2],
                               parent=near_idx,
                               arc_poses=list(prop.poses[:last_free + 1]))
        return None

    def try_connect_goal():
        gx, gy, _ = goal
        max_dist = STEP_SIZE * 3.0
        for near_idx in _k_nearest(tree, gx, gy, min(len(tree), 10)):
            near = tree[near_idx]
            if _euclidean(near.x, near.y, gx, gy) > max_dist:
                continue
            arc = compute_arc(near.x, near.y, near.theta, gx, gy)
            if arc is None:
                continue
            gl = min(arc.arc_length, max_dist)
            prop = propagate(near.x, near.y, near.theta, arc,
                             dt=DEFAULT_DT, velocity=DEFAULT_VELOCITY,
                             max_steps=MAX_STEPS_PER_ARC,
                             override_length=gl)
            if len(prop.poses) < MIN_ARC_STEPS:
                continue
            ok, _ = check_arc_collision(robot, prop.poses, obstacles, workspace)
            if not ok:
                continue
            final = prop.poses[-1]
            if _euclidean(final[0], final[1], gx, gy) < GOAL_TOLERANCE:
                return RRTNode(x=final[0], y=final[1], theta=final[2],
                               parent=near_idx, arc_poses=list(prop.poses))
        return None

    for iteration in range(1, num_iterations + 1):
        if random.random() < goal_bias: #implementa goal bias per spingere l'albero verso l'obbiettivo
            sx, sy = goal[0], goal[1]
        else: #se non scatta il bias-->punto casuale
            sx = random.uniform(x_min, x_max)
            sy = random.uniform(y_min, y_max)

        new_node = try_extend(sx, sy)
        if new_node is not None:
            new_idx = len(tree)
            tree.append(new_node)
            yield { #invia dati nuovo nodo al browser
                "type":      "node",
                "node":      _node_dict(new_node, new_idx),
                "iter":      iteration,
                "tree_size": len(tree),
            }

        if iteration % goal_check_interval == 0:
            gn = try_connect_goal() #tenta di creare arco finale che arriva al goal
            if gn is not None:
                goal_idx   = len(tree)
                tree.append(gn)
                path_idx   = _extract_path(tree, goal_idx)
                path_nodes = [_node_dict(tree[i], i) for i in path_idx]
                path_poses = _flatten_path_poses(path_nodes)
                yield {"type": "path", "path_nodes": path_nodes,
                       "path_poses": path_poses,
                       "length": len(path_idx)}
                yield {"type": "done", "success": True,
                       "iterations": iteration, "tree_size": len(tree)}
                return

    yield {"type": "done", "success": False,
           "iterations": num_iterations, "tree_size": len(tree)}
