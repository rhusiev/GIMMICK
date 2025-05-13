from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Coord:
    x: int
    y: int

    def __add__(self, other: Coord) -> Coord:
        return Coord(self.x + other.x, self.y + other.y)


@dataclass(frozen=True)
class House:
    corner1: Coord
    corner2: Coord
    orientation: bool


def rect_bounds(h: House) -> tuple[int, int, int, int]:
    x0, x1 = sorted((h.corner1.x, h.corner2.x))
    y0, y1 = sorted((h.corner1.y, h.corner2.y))
    return x0, x1, y0, y1


def intersects(a: House, b: House) -> bool:
    ax0, ax1, ay0, ay1 = rect_bounds(a)
    bx0, bx1, by0, by1 = rect_bounds(b)
    return not (
        ax1 + 2 < bx0 - 3 or bx1 + 2 < ax0 - 3 or ay1 + 2 < by0 - 3 or by1 + 2 < ay0 - 3
    )


def is_valid(new: House, existing: Iterable[House]) -> bool:
    for h in existing:
        if intersects(new, h):
            return False
    return True
