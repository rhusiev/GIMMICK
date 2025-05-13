import random
import math
import numpy as np
from village_objects import House, Coord, is_valid
from visualize_rectangles import draw_village


if __name__ == "__main__":
    probability = 1.0
    start = Coord(100, 100)
    sizes = [Coord(10, 15), Coord(20, 20), Coord(16, 20), Coord(25, 38)]
    houses: set[House] = {
        House(start, start + random.choice(sizes), bool(random.randint(0, 1)))
    }

    while random.random() < probability:
        base = random.choice(tuple(houses))

        origin = base.corner1 if random.randint(0, 1) == 0 else base.corner2
        step = int(np.random.exponential(scale=40))
        angle = random.random() * 2 * math.pi
        offset = Coord(int(step * math.cos(angle)), int(step * math.sin(angle)))
        origin += offset

        size = random.choice(sizes)
        if random.randint(0, 1):
            size = Coord(size.y, size.x)
        new = House(origin, origin + size, bool(random.randint(0, 1)))
        if is_valid(new, houses):
            houses.add(new)
            probability *= (probability * 0.995) ** 0.07

    draw_village(houses, path_width=4)
