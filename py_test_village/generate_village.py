import random
import math
import numpy as np
from village_objects import House, Coord, is_valid
from visualize_rectangles import draw_rectangles, houses_to_rectangles

probability = 1.0
start = Coord(100, 100)
sizes = [Coord(10, 15), Coord(20, 20), Coord(16, 20)]
houses: set[House] = {House(start, start + random.choice(sizes))}

while random.random() < probability:
    base = random.choice(tuple(houses))

    origin = base.corner1 if random.randint(0, 1) == 0 else base.corner2
    step = int(np.random.exponential(scale=40))
    angle = random.random() * 2 * math.pi
    offset = Coord(int(step * math.cos(angle)), int(step * math.sin(angle)))
    origin += offset

    new = House(origin, origin + random.choice(sizes))
    if is_valid(new, houses):
        houses.add(new)
        probability *= (probability * 0.995)**0.07

draw_rectangles(houses_to_rectangles(houses))
