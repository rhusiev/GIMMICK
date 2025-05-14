#include "village_builder.hpp"
#include "village_objects.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <vector>

void print_grid(const VillageGrid &grid) {
    if (grid.empty()) {
        std::cout << "Grid is empty." << std::endl;
        return;
    }
    for (int r = 0; r < grid.height; ++r) {
        for (int c = 0; c < grid.width; ++c) {
            char ch = ' ';
            int cell_value = grid(r, c);
            if (cell_value == 1)
                ch = 'H';
            else if (cell_value == 2)
                ch = '#';
            std::cout << ch;
        }
        std::cout << std::endl;
    }
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    const int terrain_map_width = 50;
    const int terrain_map_height = 50;
    IntGrid terrain_heights_map(terrain_map_width, terrain_map_height);
    for (int y = 0; y < terrain_map_height; ++y) {
        for (int x = 0; x < terrain_map_width; ++x) {
            terrain_heights_map(y, x) = (x / 10 + y / 15) % 5;
        }
    }

    Coord start_coord(terrain_map_width / 2, terrain_map_height / 2);
    std::vector<Coord> house_sizes = {Coord(15, 40), Coord(20, 20),
                                      Coord(20, 36), Coord(8, 16)};
    int max_houses_to_generate = 40;
    double mean_step_for_new_house = 40.0;

    std::set<House> houses_set = generate_village_houses(
        max_houses_to_generate, start_coord, house_sizes, terrain_heights_map,
        mean_step_for_new_house, gen);

    std::cout << "Generated " << houses_set.size() << " houses." << std::endl;
    if (!houses_set.empty()) {
        std::cout
            << "Example house (first in set by sort order) terrain height: "
            << houses_set.begin()->terrain_height << std::endl;
    }

    VillageLayout layout = generate_village_paths(houses_set);

    std::cout << "Village grid dimensions: " << layout.grid.width << "x"
              << layout.grid.height << std::endl;
    print_grid(layout.grid);

    return 0;
}
