from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from village_objects import House


def houses_to_rectangles(houses: set[House]) -> list[tuple[int, int, int, int]]:
    return [
        (house.corner1.x, house.corner1.y, house.corner2.x, house.corner2.y)
        for house in houses
    ]


def draw_rectangles(rectangles: list[tuple[int, int, int, int]]) -> None:
    max_x = max(max(x1, x2) for x1, _, x2, _ in rectangles) + 1
    max_y = max(max(y1, y2) for _, y1, _, y2 in rectangles) + 1

    grid = np.zeros((max_y, max_x), dtype=int)

    for x1, y1, x2, y2 in rectangles:
        x_start, x_end = sorted((x1, x2))
        y_start, y_end = sorted((y1, y2))
        grid[y_start : y_end + 1, x_start : x_end + 1] = 1

    plt.imshow(grid, cmap="Greys", origin="lower")
    plt.title("Occupied Grid Cells")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, color="lightgray")
    plt.xticks(np.arange(0, max_x))
    plt.yticks(np.arange(0, max_y))
    plt.show()
