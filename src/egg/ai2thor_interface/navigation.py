# pyright: reportExplicitAny=none, reportAny=none
import math
import logging
from typing import Any, TypeAlias
from collections.abc import Iterable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from egg.utils.geometry import Position, Rotation
from egg.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="ai2thor_interfaceh/navigation.log",
)

OrientedNode: TypeAlias = tuple[
    float, float, int
]  # (x, z, yaw_idx), yaw_idx in {0,1,2,3}
Key2D: TypeAlias = tuple[float, float]  # (x, z) snapped to grid
Command: TypeAlias = dict[
    str, str
]  # e.g., {"action": "MoveAhead"} or {"action": "RotateRight"}


def normalize_angle_deg(a_deg: float) -> float:
    """Normalize angle to [0, 360)."""
    return a_deg % 360.0


def select_best_reachable_and_yaw(
    reachable_positions: list[Position],
    target_pos: Position,
    current_pos: Position,
) -> tuple[Position, int, Rotation]:
    if not reachable_positions:
        raise ValueError("reachable_positions is empty")

    R = np.asarray(
        [p.as_numpy_2d().reshape(-1)[:2] for p in reachable_positions], float
    )
    target_xz = target_pos.as_numpy_2d().reshape(-1)[:2]
    robot_xz = current_pos.as_numpy_2d().reshape(-1)[:2]

    # Distances
    d_target = cdist(R, target_xz[None, :], metric="euclidean").ravel()
    d_robot = cdist(R, robot_xz[None, :], metric="euclidean").ravel()

    # Primary: min d_target; Secondary: min d_robot
    best_idx = int(np.lexsort((d_robot, d_target))[0])
    best_pos = reachable_positions[best_idx].as_numpy_2d()

    # Bearing from best_pos to target (XY), then quantize to nearest 90°
    dx = float(target_xz[0] - best_pos[0])
    dy = float(target_xz[1] - best_pos[1])
    bearing_deg = normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
    desired_yaw_deg = normalize_angle_deg(
        round(bearing_deg / 90.0) * 90.0
    )  # in [0, 360)

    return (
        reachable_positions[best_idx],
        best_idx,
        Rotation(x=0, y=desired_yaw_deg, z=0),
    )


def visualize(
    reachable_positions: list[Position],
    target_position: Position,
    agent_position: Position,
    nav_position: Position,
    nav_angle: float,
    path: list[Position] | None = None,
) -> tuple[Figure, Axes]:
    xs = [rp.x for rp in reachable_positions]
    zs = [rp.z for rp in reachable_positions]

    fig, ax = plt.subplots(1, 1)
    _ = ax.scatter(xs, zs, label="Reachable")
    _ = ax.scatter(target_position.x, target_position.z, label="Target")
    _ = ax.scatter(agent_position.x, agent_position.z, label="Agent")

    if path:
        xp = [p.x for p in path]
        zp = [p.z for p in path]
        _ = ax.scatter(xp, zp, label="Path")

    theta = np.deg2rad(nav_angle)
    # Direction components in XZ plane
    dx = np.cos(theta)
    dz = np.sin(theta)
    # Draw arrow using quiver
    q = ax.quiver(
        [nav_position.x],
        [nav_position.z],
        [dx],
        [dz],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        label="Nav point",
    )

    _ = ax.set_xlabel("$x$")
    _ = ax.set_ylabel("$z$")
    _ = ax.set_title("Reachable Positions in the Scene")
    ax.legend()
    ax.set_aspect("equal")
    return fig, ax


def quantize(value: float, grid_size: float) -> float:
    """
    Snap a coordinate value to the nearest grid point given grid_size.
    """
    return round(
        round(value / grid_size) * grid_size, int(max(0, -math.log10(grid_size)) + 1)
    )


def pos_to_key(pos: Position, grid_size: float) -> Key2D:
    """
    Convert a position dict (with x,z) to a 2D key snapped to grid.
    """
    x = quantize(pos.x, grid_size)
    z = quantize(pos.z, grid_size)
    return (x, z)


def key_to_pos(key: Key2D, y: float = 0.0) -> Position:
    """
    Convert a 2D key back to a position dict (with x,z and optional y).
    """
    x, z = key
    return Position(x=x, y=y, z=z)


def build_graph(
    reachable_positions: list[Position],
    grid_size: float,
    allow_diagonals: bool = False,
) -> nx.Graph:
    """
    Build an undirected grid graph from AI2-THOR reachable positions.
    Nodes are (x,z) tuples snapped to the grid. Edges connect 4-neighbors
    (or 8-neighbors if allow_diagonals=True).
    """
    G: nx.Graph = nx.Graph()
    nodes: set[Key2D] = set()

    # Add nodes snapped to grid
    for p in reachable_positions:
        nodes.add(pos_to_key(p, grid_size))

    # Connect neighbors
    # Directions are expressed in grid steps
    if allow_diagonals:
        neighbor_steps = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
    else:
        neighbor_steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for x, z in nodes:
        G.add_node((x, z))
        for dx, dz in neighbor_steps:
            nx_pos: Key2D = (round(x + dx * grid_size, 6), round(z + dz * grid_size, 6))
            # Snap neighbor to grid too, to be safe
            nx_pos = (quantize(nx_pos[0], grid_size), quantize(nx_pos[1], grid_size))
            if nx_pos in nodes:
                # Weight = Euclidean distance
                dist = math.hypot(nx_pos[0] - x, nx_pos[1] - z)
                G.add_edge((x, z), nx_pos, weight=dist)

    return G


def euclidean(a: Key2D, b: Key2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def find_closest_node(G: nx.Graph, key: Key2D) -> Key2D | None:
    """
    Return the graph node closest to the given (x,z) key.
    Useful if start/goal is slightly off the grid.
    """
    if key in G:
        return key
    if len(G) == 0:
        return None
    best: Key2D | None = None
    best_d = float("inf")
    for n in G.nodes:
        d = euclidean(n, key)  # type: ignore[arg-type]
        if d < best_d:
            best_d = d
            best = n  # type: ignore[assignment]
    return best


def angle_diff_deg(a: float, b: float) -> float:
    """
    Shortest signed difference a - b in degrees in (-180, 180]
    """
    d = (a - b + 180) % 360 - 180
    return d if d != -180 else 180


def yaw_to_idx(yaw_deg: float) -> int:
    """
    Map a yaw angle (degrees) to the nearest 90-degree index: 0->0°, 1->90°, 2->180°, 3->270°.
    """
    yaw = normalize_angle_deg(yaw_deg)
    return int(round(yaw / 90.0)) % 4


def idx_to_yaw(idx: int) -> float:
    return float((idx % 4) * 90)


def dir_to_yaw_idx(dir_vec: tuple[int, int]) -> int:
    """
    Map a cardinal dir vector to yaw index:
    (0,1)->0, (1,0)->1, (0,-1)->2, (-1,0)->3
    """
    mapping: dict[tuple[int, int], int] = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
    return mapping[dir_vec]


def yaw_idx_to_dir(idx: int) -> tuple[int, int]:
    mapping: dict[int, tuple[int, int]] = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
    return mapping[idx % 4]


# ---------------------------
# Base spatial graph (positions only)
# ---------------------------
def build_position_graph(
    reachable_positions: list[Position],
    grid_size: float,
    allow_diagonals: bool = False,
) -> nx.Graph:
    """
    Undirected graph over positions only. Nodes: (x,z). Edges: 4- or 8-connected neighbors.
    Edge weight = Euclidean distance.
    """
    G: nx.Graph = nx.Graph()
    nodes: set[Key2D] = set(pos_to_key(p, grid_size) for p in reachable_positions)

    if not nodes:
        return G

    # Neighbor steps (in grid units)
    if allow_diagonals:
        neighbor_steps: list[tuple[int, int]] = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
    else:
        neighbor_steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for x, z in nodes:
        G.add_node((x, z))
        for dx, dz in neighbor_steps:
            nx_pos: Key2D = (
                quantize(x + dx * grid_size, grid_size),
                quantize(z + dz * grid_size, grid_size),
            )
            if nx_pos in nodes:
                G.add_edge(
                    (x, z), nx_pos, weight=math.hypot(nx_pos[0] - x, nx_pos[1] - z)
                )

    return G


def find_closest_node(G: nx.Graph, key: Key2D) -> Key2D | None:
    if key in G:
        return key
    if len(G) == 0:
        return None
    best: Key2D | None = None
    best_d = float("inf")
    for n in G.nodes:
        d = euclidean(n, key)  # type: ignore[arg-type]
        if d < best_d:
            best_d = d
            best = n  # type: ignore[assignment]
    return best


# ---------------------------
# Orientation-aware graph for A* with NetworkX
# ---------------------------
def build_oriented_graph(
    base_graph: nx.Graph,
    grid_size: float,
    move_cost_scale: float = 1.0,
    rotation_cost_per_90: float = 0.1,
) -> nx.DiGraph:
    """
    Build a directed graph whose nodes are (x, z, yaw_idx), yaw_idx in {0,1,2,3}.
    Edges:
      - RotateLeft / RotateRight at same (x,z) with cost = rotation_cost_per_90
      - MoveAhead from (x,z,yaw_idx) to (x+dx, z+dz, yaw_idx) if facing that neighbor,
        with cost = move_cost_scale * distance (typically grid_size)
    """
    OG: nx.DiGraph = nx.DiGraph()

    # Add nodes for each orientation
    for x, z in base_graph.nodes:
        for yaw_idx in range(4):
            OG.add_node((x, z, yaw_idx))

    # Rotation edges
    for x, z in base_graph.nodes:
        for yaw_idx in range(4):
            OG.add_edge(
                (x, z, yaw_idx),
                (x, z, (yaw_idx + 1) % 4),
                action="RotateRight",
                weight=rotation_cost_per_90,
            )
            OG.add_edge(
                (x, z, yaw_idx),
                (x, z, (yaw_idx - 1) % 4),
                action="RotateLeft",
                weight=rotation_cost_per_90,
            )

    # MoveAhead edges along cardinal neighbors
    neighbors: dict[Key2D, set[Key2D]] = {
        n: set(base_graph.neighbors(n)) for n in base_graph.nodes
    }
    for x, z in base_graph.nodes:
        for yaw_idx in range(4):
            step_dir = yaw_idx_to_dir(yaw_idx)  # which way we face
            target: Key2D = (
                quantize(x + step_dir[0] * grid_size, grid_size),
                quantize(z + step_dir[1] * grid_size, grid_size),
            )
            if target in neighbors[(x, z)]:
                dist = euclidean((x, z), target)
                OG.add_edge(
                    (x, z, yaw_idx),
                    (target[0], target[1], yaw_idx),
                    action="MoveAhead",
                    weight=move_cost_scale * dist,
                )

    return OG


def admissible_heuristic(a: OrientedNode, b: OrientedNode) -> float:
    """
    Heuristic ignoring rotations: straight-line distance between positions.
    This is admissible since rotations have non-negative costs.
    """
    ax, az, _ = a
    bx, bz, _ = b
    return math.hypot(ax - bx, az - bz)


def astar_best_to_any_goal(
    OG: nx.DiGraph,
    start: OrientedNode,
    goal_positions: Iterable[Key2D],
) -> list[OrientedNode]:
    """
    Run A* from a start oriented node to the best goal (any yaw) at any of the goal positions.
    Tries four yaw targets at each goal position and returns the best full path.
    """
    best_path: list[OrientedNode] = []
    best_len = float("inf")

    # Consider all (goal_pos, yaw_idx)
    for gx, gz in goal_positions:
        for gyaw in range(4):
            target: OrientedNode = (gx, gz, gyaw)
            if target not in OG:
                continue
            try:
                path = nx.astar_path(
                    OG,
                    start,
                    target,
                    heuristic=lambda a, b: admissible_heuristic(a, b),  # type: ignore[arg-type]
                    weight="weight",
                )
                path_len = nx.path_weight(OG, path, weight="weight")
                if path_len < best_len:
                    best_len = path_len
                    best_path = path
            except nx.NetworkXNoPath:
                continue

    if not best_path:
        raise RuntimeError(
            "No path found to the goal with the given reachable positions and orientation constraints."
        )
    return best_path


def plan_path_and_command(
    start_pos: Position,
    reachable_positions: list[Position],
    goal_pos: Position,
    grid_size: float = 0.25,
    initial_yaw_deg: float = 0.0,
    allow_diagonals: bool = False,
    rotation_cost_per_90: float = 0.1,
    move_cost_scale: float = 1.0,
) -> tuple[list[Position], list[Command]]:
    """
    Plan an orientation-aware path using NetworkX A* over an oriented state graph.

    Inputs:
      - start_pos: current agent position
      - reachable_positions: list from AI2-THOR (GetReachablePositions)
      - goal_pos: target position to navigate to
      - initial_yaw_deg: agent's starting yaw angle in degrees (0=+Z, 90=+X, etc.)
      - grid_size: movement discretization; match AI2-THOR controller gridSize
      - allow_diagonals: if True, base graph includes diagonal adjacency; agent still moves via cardinal headings
      - rotation_cost_per_90: cost for a 90-degree rotation (tune to penalize turning)
      - move_cost_scale: multiplier for move cost (usually 1.0)

    Returns:
      - waypoints: list of positions (x,y,z) to visit (includes start and goal)
      - commands: list of AI2-THOR commands: {"action":"RotateLeft"/"RotateRight"/"MoveAhead"}
    """
    if not reachable_positions:
        raise ValueError("reachable_positions is empty")

    # Use y from start if available, else from first reachable
    y_floor = start_pos.y

    # 1) Build base position graph
    base_G = build_position_graph(
        reachable_positions, grid_size, allow_diagonals=allow_diagonals
    )
    if len(base_G) == 0:
        raise ValueError("Base graph is empty after building from reachable_positions.")

    # 2) Snap start and goal to nearest nodes
    start_key = pos_to_key(start_pos, grid_size)
    goal_key = pos_to_key(goal_pos, grid_size)
    s_xy = find_closest_node(base_G, start_key)
    g_xy = find_closest_node(base_G, goal_key)
    if s_xy is None or g_xy is None:
        raise ValueError("Could not map start or goal to graph nodes.")

    # 3) Build oriented graph with rotation edges and forward moves
    OG = build_oriented_graph(
        base_graph=base_G,
        grid_size=grid_size,
        move_cost_scale=move_cost_scale,
        rotation_cost_per_90=rotation_cost_per_90,
    )

    # 4) Run A* from oriented start to best oriented goal (any yaw at goal position)
    start_yaw_idx = yaw_to_idx(initial_yaw_deg)
    start_node: OrientedNode = (s_xy[0], s_xy[1], start_yaw_idx)
    oriented_path: list[OrientedNode] = astar_best_to_any_goal(
        OG,
        start=start_node,
        goal_positions=[g_xy],
    )

    # 5) Convert oriented path to commands and waypoints
    commands: list[Command] = []
    waypoints: list[Position] = []

    def append_waypoint(x: float, z: float) -> None:
        if not waypoints or (waypoints[-1].x, waypoints[-1].z) != (x, z):
            waypoints.append(key_to_pos((x, z), y=y_floor))

    # Start waypoint
    append_waypoint(oriented_path[0][0], oriented_path[0][1])

    # Translate edges to commands; collect waypoints when position changes
    for (x1, z1, y1), (x2, z2, y2) in zip(oriented_path[:-1], oriented_path[1:]):
        if (x1, z1) == (x2, z2) and y1 != y2:
            # Rotation step
            # Determine whether this is a left or right 90 step (graph uses single 90-degree increments)
            if (y2 - y1) % 4 == 1:  # right
                commands.append({"action": "RotateRight"})
            elif (y2 - y1) % 4 == 3:  # left (equiv to -1 mod 4)
                commands.append({"action": "RotateLeft"})
            else:
                # Multi-step rotation shouldn't occur because our graph adds only +/-1 edges
                steps = (y2 - y1) % 4
                if steps == 2:
                    commands.extend(
                        [{"action": "RotateRight"}, {"action": "RotateRight"}]
                    )
                else:
                    # Fallback
                    for _ in range(steps):
                        commands.append({"action": "RotateRight"})
        elif (x1, z1) != (x2, z2) and y1 == y2:
            # Move ahead step
            commands.append({"action": "MoveAhead"})
            append_waypoint(x2, z2)
        else:
            # Shouldn't happen in our construction
            raise RuntimeError("Unexpected transition in oriented path.")

    return waypoints, commands
