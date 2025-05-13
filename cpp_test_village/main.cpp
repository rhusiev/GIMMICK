#include "village_objects.hpp"
#include "village_builder.hpp"
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <cmath>

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

    double probability = 1.0;
    Coord start_coord(100, 100);
    std::vector<Coord> sizes = {Coord(10, 15), Coord(20, 20), Coord(16, 20),
                                Coord(25, 38)};

    std::uniform_int_distribution<> bool_dist(0, 1);
    std::uniform_int_distribution<size_t> size_choice_dist(0, sizes.size() - 1);

    Coord initial_size = sizes[size_choice_dist(gen)];
    House initial_house(start_coord, start_coord + initial_size,
                        static_cast<bool>(bool_dist(gen)));

    std::set<House> houses_set;
    houses_set.insert(initial_house);

    std::exponential_distribution<double> step_dist(1.0 / 40.0);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    while (std::uniform_real_distribution<double>(0.0, 1.0)(gen) <
           probability) {
        std::vector<House> current_houses_vec(houses_set.begin(),
                                              houses_set.end());
        if (current_houses_vec.empty())
            break;

        std::uniform_int_distribution<size_t> base_choice_dist(
            0, current_houses_vec.size() - 1);
        const House &base_house = current_houses_vec[base_choice_dist(gen)];

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

        House new_house(origin, origin + size,
                        static_cast<bool>(bool_dist(gen)));

        if (is_valid(new_house, current_houses_vec)) {
            houses_set.insert(new_house);
            probability *= std::pow(probability * 0.995, 0.07);
        }
    }

    std::cout << "Generated " << houses_set.size() << " houses." << std::endl;

    VillageLayout layout = generate_village_layout(houses_set);

    std::cout << "Village grid dimensions: " << layout.max_x << "x"
              << layout.max_y << std::endl;
    print_grid(layout.grid);

    return 0;
}
