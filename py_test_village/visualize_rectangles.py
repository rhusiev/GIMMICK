# visualize_rectangles.py
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from village_objects import House, rect_bounds, Coord


def get_door(h: House) -> Coord:
    """
    Returns the grid cell at the center of the wall 
    where the door sits. orientation=False → corner1 side; True → corner2 side.
    """
    x0, x1, y0, y1 = rect_bounds(h)
    if not h.orientation:
        # door on the wall that includes corner1
        cx = (h.corner1.x == x0 and x0 or x1)
        cy = (h.corner1.y == y0 and (y0 + y1)//2 or
              h.corner1.y == y1 and (y0 + y1)//2 or
              (h.corner1.y + h.corner2.y)//2)
    else:
        # door on the opposite wall (corner2)
        cx = (h.corner2.x == x0 and x0 or x1)
        cy = (h.corner2.y == y0 and (y0 + y1)//2 or
              h.corner2.y == y1 and (y0 + y1)//2 or
              (h.corner1.y + h.corner2.y)//2)
    return Coord(cx, cy)


def mst_edges(points: list[Coord]) -> list[tuple[int,int]]:
    """
    Return a list of index-pairs forming a minimum spanning tree
    over 'points' by Euclidean distance (Prim’s algorithm).
    """
    import math
    n = len(points)
    if n < 2:
        return []
    in_tree = {0}
    edges = []
    # distances
    def d(i,j):
        dx = points[i].x - points[j].x
        dy = points[i].y - points[j].y
        return math.hypot(dx,dy)
    while len(in_tree) < n:
        best = None
        for u in in_tree:
            for v in range(n):
                if v in in_tree: continue
                dist = d(u,v)
                if best is None or dist < best[0]:
                    best = (dist, u, v)
        _, u, v = best
        in_tree.add(v)
        edges.append((u,v))
    return edges


def draw_village(houses: set[House], path_width: int = 4) -> None:
    # build grid
    rects = [(h.corner1.x, h.corner1.y, h.corner2.x, h.corner2.y) for h in houses]
    max_x = max(max(x1, x2) for x1, _, x2, _ in rects) + path_width + 1
    max_y = max(max(y1, y2) for _, y1, _, y2 in rects) + path_width + 1
    grid = np.zeros((max_y, max_x), dtype=int)

    # mark houses as 1
    for x1, y1, x2, y2 in rects:
        xs, xe = sorted((x1, x2))
        ys, ye = sorted((y1, y2))
        grid[ys:ye+1, xs:xe+1] = 1

    # compute doors
    doors = [get_door(h) for h in houses]

    # get MST over doors
    edges = mst_edges(doors)

    # draw each path
    for u, v in edges:
        a, b = doors[u], doors[v]
        # decide bend point (try H→V first)
        bend1 = Coord(b.x, a.y)
        path_pts = [(a.x, a.y, bend1.x, bend1.y),
                    (bend1.x, bend1.y, b.x, b.y)]
        # test if either segment would collide house
        def collides(x0,y0,x1,y1):
            xs, xe = sorted((x0,x1))
            ys, ye = sorted((y0,y1))
            width = path_width//2
            # check a thickened corridor
            for xx in range(xs-width, xe+width+1):
                for yy in range(ys-width, ye+width+1):
                    if xx<0 or yy<0 or xx>=max_x or yy>=max_y: continue
                    if grid[yy,xx] == 1:
                        return True
            return False
        # if collision, swap to V→H
        if collides(*path_pts[0]):
            # redefine bend
            bend2 = Coord(a.x, b.y)
            path_pts = [(a.x, a.y, bend2.x, bend2.y),
                        (bend2.x, bend2.y, b.x, b.y)]
        # finally, draw both segments
        for x0,y0,x1,y1 in path_pts:
            xs, xe = sorted((x0, x1))
            ys, ye = sorted((y0, y1))
            # expand by half-width
            w = path_width//2
            for xx in range(xs-w, xe+w+1):
                for yy in range(ys-w, ye+w+1):
                    if 0 <= xx < max_x and 0 <= yy < max_y:
                        # only draw on empty space
                        if grid[yy,xx] == 0:
                            grid[yy,xx] = 2

    # plot: 0=white,1=grey(house),2=darkgrey(path)
    cmap = plt.cm.get_cmap("Greys", 3)
    plt.figure(figsize=(8,8))
    plt.imshow(grid, cmap=cmap, origin="lower")
    plt.title("Village: houses (light) + paths (dark)")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True, color="lightgray")
    plt.xticks(np.arange(0, max_x, max(1, max_x//20)))
    plt.yticks(np.arange(0, max_y, max(1, max_y//20)))
    plt.show()

