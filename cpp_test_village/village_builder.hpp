#ifndef VILLAGE_BUILDER_HPP
#define VILLAGE_BUILDER_HPP

#include "village_objects.hpp"
#include <vector>
#include <set>

using VillageGrid = std::vector<std::vector<int>>;

struct VillageLayout {
    VillageGrid grid;
    int max_x;
    int max_y;
};


VillageLayout generate_village_layout(
    const std::set<House>& houses,
    int path_width = 3,
    int max_astar_steps = 10000
);

#endif // VILLAGE_BUILDER_HPP
