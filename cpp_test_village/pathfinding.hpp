#ifndef PATHFINDING_HPP
#define PATHFINDING_HPP

#include <vector>
#include <optional>
#include "village_objects.hpp"

std::optional<std::vector<Coord>> astar(
    const Coord& start,
    const Coord& goal,
    const std::vector<std::vector<bool>>& blocked,
    int max_steps = 8000
);

#endif // PATHFINDING_HPP
