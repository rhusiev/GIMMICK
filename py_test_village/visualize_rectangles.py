# visualize_rectangles.py
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import heapq
from village_objects import House, Coord, rect_bounds

def get_door(h: House) -> Coord:
    x0, x1, y0, y1 = rect_bounds(h)
    # pick the wall midpoint according to orientation
    if not h.orientation:
        # door on the corner1 side
        if h.corner1.x in (x0, x1):
            cx = h.corner1.x
            cy = (y0 + y1) // 2
        else:
            cy = h.corner1.y
            cx = (x0 + x1) // 2
    else:
        # door on the corner2 side
        if h.corner2.x in (x0, x1):
            cx = h.corner2.x
            cy = (y0 + y1) // 2
        else:
            cy = h.corner2.y
            cx = (x0 + x1) // 2
    return Coord(cx, cy)

def astar(start: Coord,
          goal: Coord,
          blocked: np.ndarray,
          max_steps: int = 8000) -> list[Coord] | None:
    """
    A* on grid of shape blocked.shape (True=wall), 4‐neighbors,
    returns list of Coords from start→goal, or None if no path
    within max_steps expansions.
    """
    H, W = blocked.shape
    def h(a: tuple[int,int], b: tuple[int,int]) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    start_t = (start.x, start.y)
    goal_t  = (goal.x, goal.y)

    open_heap: list[tuple[int,int,int,int]] = []
    # heap entries: (f, g, x, y)
    g_scores: dict[tuple[int,int],int] = {start_t: 0}
    heapq.heappush(open_heap, (h(start_t, goal_t), 0, start_t[0], start_t[1]))

    came_from: dict[tuple[int,int], tuple[int,int]] = {}
    steps = 0

    while open_heap and steps < max_steps:
        f, g, x, y = heapq.heappop(open_heap)
        pos = (x, y)
        if pos == goal_t:
            # reconstruct
            path = []
            cur = pos
            while cur != start_t:
                path.append(Coord(cur[0], cur[1]))
                cur = came_from[cur]
            path.append(start)
            return list(reversed(path))

        # already found better
        if g > g_scores.get(pos, 1_000_000):
            continue

        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if not (0 <= nx < W and 0 <= ny < H): continue
            if blocked[ny, nx]:
                continue
            ng = g + 1
            np_t = (nx, ny)
            if ng < g_scores.get(np_t, 1_000_000):
                g_scores[np_t] = ng
                came_from[np_t] = pos
                heapq.heappush(open_heap, (ng + h(np_t, goal_t), ng, nx, ny))
        steps += 1

    return None  # no path found in budget

def draw_village(houses: set[House], path_width: int = 3, max_astar_steps: int = 10000):
    # build raw grids
    rects = [(h.corner1.x, h.corner1.y, h.corner2.x, h.corner2.y) for h in houses]
    max_x = max(max(x1, x2) for x1, _, x2, _ in rects) + path_width + 2
    max_y = max(max(y1, y2) for _, y1, _, y2 in rects) + path_width + 2

    house_mask = np.zeros((max_y, max_x), dtype=bool)
    path_mask  = np.zeros_like(house_mask)

    # mark houses
    for x1, y1, x2, y2 in rects:
        xs, xe = sorted((x1, x2))
        ys, ye = sorted((y1, y2))
        house_mask[ys:ye+1, xs:xe+1] = True

    # build blocked = house_inflated
    half = path_width // 2
    # add padding around houses so paths stay at least 'padding' cells away
    padding = 2  # tweak this as you like
    inflate = half + padding
    blocked = np.zeros_like(house_mask)
    ys, xs = np.where(house_mask)
    for y, x in zip(ys, xs):
        y0 = max(0, y - inflate)
        y1 = min(max_y, y + inflate + 1)
        x0 = max(0, x - inflate)
        x1 = min(max_x, x + inflate + 1)
        blocked[y0:y1, x0:x1] = True

    # gather doors and carve 3×3 hole around each
    doors = [get_door(h) for h in houses]
    for d in doors:
        for dy in range(-inflate, inflate+1):
            for dx in range(-inflate, inflate+1):
                ny, nx = d.y + dy, d.x + dx
                if 0 <= nx < max_x and 0 <= ny < max_y:
                    blocked[ny, nx] = False

    # build MST edges
    def dist(i, j):
        dx = doors[i].x - doors[j].x
        dy = doors[i].y - doors[j].y
        return (dx*dx + dy*dy)**0.5

    n = len(doors)
    in_t = {0}
    mst = []
    while len(in_t) < n:
        best = None
        for u in in_t:
            for v in range(n):
                if v in in_t: continue
                d = dist(u, v)
                if best is None or d < best[0]:
                    best = (d, u, v)
        _, u, v = best
        in_t.add(v)
        mst.append((u, v))

    # A* each edge
    for u, v in mst:
        start, goal = doors[u], doors[v]
        path = astar(start, goal, blocked, max_steps=max_astar_steps)
        if not path:
            continue
        for c in path:
            for dy in range(-half, half+1):
                for dx in range(-half, half+1):
                    x, y = c.x+dx, c.y+dy
                    if 0 <= x < max_x and 0 <= y < max_y:
                        if not house_mask[y, x]:
                            path_mask[y, x] = True

    # combine and plot
    grid = np.zeros((max_y, max_x), dtype=int)
    grid[house_mask] = 1
    grid[path_mask ] = 2

    cmap = plt.cm.get_cmap("Greys", 3)
    plt.figure(figsize=(8,8))
    plt.imshow(grid, cmap=cmap, origin="lower")
    plt.title("Village: houses (light) + paths (dark)")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True, color="lightgray")
    plt.xticks(np.arange(0, max_x, max(1, max_x//20)))
    plt.yticks(np.arange(0, max_y, max(1, max_y//20)))
    plt.show()

