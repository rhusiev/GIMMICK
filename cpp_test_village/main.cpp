#include "village_builder.hpp"
#include "village_objects.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <vector>

int calculate_max_terrain_height_for_house(
    const Coord &c1, const Coord &c2,
    const std::vector<std::vector<int>> &heights_map) {

    if (heights_map.empty() || heights_map[0].empty()) {
        return 0;
    }
    size_t map_rows = heights_map.size();
    size_t map_cols = heights_map[0].size();

    int x_min_world = std::min(c1.x, c2.x);
    int x_max_world = std::max(c1.x, c2.x);
    int y_min_world = std::min(c1.y, c2.y);
    int y_max_world = std::max(c1.y, c2.y);

    int max_h = 0;
    bool found_valid_point_on_map = false;

    for (int y_world = y_min_world; y_world <= y_max_world; ++y_world) {
        for (int x_world = x_min_world; x_world <= x_max_world; ++x_world) {
            if (x_world >= 0 && static_cast<size_t>(x_world) < map_cols &&
                y_world >= 0 && static_cast<size_t>(y_world) < map_rows) {

                int current_terrain_height =
                    heights_map[static_cast<size_t>(y_world)]
                               [static_cast<size_t>(x_world)];

                if (!found_valid_point_on_map) {
                    max_h = current_terrain_height;
                    found_valid_point_on_map = true;
                } else {
                    max_h = std::max(max_h, current_terrain_height);
                }
            }
        }
    }

    if (!found_valid_point_on_map) {
        return 0;
    }
    return max_h;
}

void print_grid(const VillageGrid &grid) {
    if (grid.empty()) {
        std::cout << "Grid is empty." << std::endl;
        return;
    }
    for (const auto &row : grid) {
        for (int cell : row) {
            char c = ' ';
            if (cell == 1)
                c = 'H';
            else if (cell == 2)
                c = '#';
            std::cout << c;
        }
        std::cout << std::endl;
    }
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    const int terrain_map_width = 500;
    const int terrain_map_height = 500;
    std::vector<std::vector<int>> terrain_heights_map(
        terrain_map_height, std::vector<int>(terrain_map_width));

    for (int y = 0; y < terrain_map_height; ++y) {
        for (int x = 0; x < terrain_map_width; ++x) {
            terrain_heights_map[y][x] = (x / 20 + y / 30) % 15;
        }
    }

    double probability = 1.0;
    Coord start_coord(250, 250);
    std::vector<Coord> sizes = {Coord(10, 15), Coord(20, 20), Coord(16, 20),
                                Coord(25, 38)};

    std::uniform_int_distribution<> bool_dist(0, 1);
    std::uniform_int_distribution<size_t> size_choice_dist(0, sizes.size() - 1);

    // Initial house
    Coord initial_size = sizes[size_choice_dist(gen)];
    bool initial_orientation = static_cast<bool>(bool_dist(gen));
    int initial_house_terrain_height = calculate_max_terrain_height_for_house(
        start_coord, start_coord + initial_size, terrain_heights_map);
    House initial_house(start_coord, start_coord + initial_size,
                        initial_orientation, initial_house_terrain_height);

    std::set<House> houses_set;
    houses_set.insert(initial_house);

    std::vector<House> houses_vec;
    houses_vec.push_back(initial_house);

    std::exponential_distribution<double> step_dist(1.0 / 40.0);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    int max_houses = 50;
    int houses_generated_count = 1;

    while (std::uniform_real_distribution<double>(0.0, 1.0)(gen) <
               probability &&
           houses_generated_count < max_houses) {
        if (houses_vec.empty())
            break;

        std::uniform_int_distribution<size_t> base_choice_dist(
            0, houses_vec.size() - 1);
        const House &base_house = houses_vec[base_choice_dist(gen)];

        Coord origin =
            (bool_dist(gen) == 0) ? base_house.corner1 : base_house.corner2;

        double step_val = step_dist(gen);
        double angle = angle_dist(gen);

        Coord offset(static_cast<int>(step_val * std::cos(angle)),
                     static_cast<int>(step_val * std::sin(angle)));
        origin += offset;

        Coord size = sizes[size_choice_dist(gen)];
        if (bool_dist(gen) == 0) {
            size = Coord(size.y, size.x);
        }

        bool new_house_orientation = static_cast<bool>(bool_dist(gen));
        Coord new_house_c1 = origin;
        Coord new_house_c2 = origin + size;

        int new_house_terrain_height = calculate_max_terrain_height_for_house(
            new_house_c1, new_house_c2, terrain_heights_map);

        House new_house(new_house_c1, new_house_c2, new_house_orientation,
                        new_house_terrain_height);

        if (is_valid(new_house, houses_vec)) {
            auto insert_result = houses_set.insert(new_house);
            if (insert_result.second) {
                houses_vec.push_back(new_house);
                probability *= std::pow(0.995, 0.07);
                houses_generated_count++;
            }
        }
    }

    std::cout << "Generated " << houses_set.size() << " houses." << std::endl;
    if (!houses_set.empty()) {
        std::cout << "Example house (first in set) terrain height: "
                  << houses_set.begin()->terrain_height << std::endl;
    }

    VillageLayout layout = generate_village_layout(houses_set, 3, 10000);

    std::cout << "Village grid dimensions: " << layout.max_x << "x"
              << layout.max_y << std::endl;
    print_grid(layout.grid);

    return 0;
}
