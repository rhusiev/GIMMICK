#include "village_builder.hpp"
#include "pathfinding.hpp"
#include "village_objects.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>

#ifndef M_PI // Ensure M_PI is defined
#define M_PI 3.14159265358979323846
#endif

static int calculate_max_terrain_height_for_house(const Coord &c1,
                                                  const Coord &c2,
                                                  const IntGrid &heights_map) {
    if (heights_map.empty()) {
        return 0;
    }

    int x_min_world = std::min(c1.x, c2.x);
    int x_max_world = std::max(c1.x, c2.x);
    int y_min_world = std::min(c1.y, c2.y);
    int y_max_world = std::max(c1.y, c2.y);

    int max_h = 0;
    bool found_valid_point_on_map = false;

    for (int y_world = y_min_world; y_world <= y_max_world; ++y_world) {
        for (int x_world = x_min_world; x_world <= x_max_world; ++x_world) {
            // Check if the current world coordinate is within the terrain map
            // bounds
            if (heights_map.is_valid(y_world, x_world)) {
                int current_terrain_height = heights_map(y_world, x_world);

                if (!found_valid_point_on_map) {
                    max_h = current_terrain_height;
                    found_valid_point_on_map = true;
                } else {
                    max_h = std::max(max_h, current_terrain_height);
                }
            }
        }
    }
    return max_h;
}

std::set<House> generate_village_houses(
    int max_num_houses, const Coord &initial_start_coord,
    const std::vector<Coord> &possible_house_sizes, const IntGrid &terrain_map,
    double step_dist_mean, std::mt19937 &gen, double initial_probability,
    double probability_base_decay, double probability_exponent_decay) {
    std::set<House> houses_set;
    if (possible_house_sizes.empty() || max_num_houses <= 0) {
        return houses_set;
    }

    std::uniform_int_distribution<> bool_dist(0, 1);
    std::uniform_int_distribution<size_t> size_choice_dist(
        0, possible_house_sizes.size() - 1);

    Coord initial_size = possible_house_sizes[size_choice_dist(gen)];
    bool initial_orientation = static_cast<bool>(bool_dist(gen));
    int initial_house_terrain_height = calculate_max_terrain_height_for_house(
        initial_start_coord, initial_start_coord + initial_size, terrain_map);
    House initial_house(initial_start_coord, initial_start_coord + initial_size,
                        initial_orientation, initial_house_terrain_height);

    houses_set.insert(initial_house);

    std::vector<House> houses_vec;
    houses_vec.push_back(initial_house);

    std::exponential_distribution<double> step_dist(1.0 / step_dist_mean);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    int houses_generated_count = 1;
    double current_probability = initial_probability;

    while (std::uniform_real_distribution<double>(0.0, 1.0)(gen) <
               current_probability &&
           houses_generated_count < max_num_houses) {
        if (houses_vec.empty()) {
            break;
        }

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

        Coord size = possible_house_sizes[size_choice_dist(gen)];
        if (bool_dist(gen) == 0) {
            size = Coord(size.y, size.x);
        }

        bool new_house_orientation = static_cast<bool>(bool_dist(gen));
        Coord new_house_c1 = origin;
        Coord new_house_c2 = origin + size;

        int new_house_terrain_height = calculate_max_terrain_height_for_house(
            new_house_c1, new_house_c2, terrain_map);

        House new_house(new_house_c1, new_house_c2, new_house_orientation,
                        new_house_terrain_height);

        if (is_valid(new_house, houses_vec)) {
            auto insert_result = houses_set.insert(new_house);
            if (insert_result.second) {
                houses_vec.push_back(new_house);
                current_probability *= std::pow(probability_base_decay,
                                                probability_exponent_decay);
                houses_generated_count++;
            }
        }
    }
    return houses_set;
}

VillageLayout generate_village_paths(const std::set<House> &houses_set,
                                     int path_width, int max_astar_steps) {
    if (houses_set.empty()) {
        return {VillageGrid()};
    }

    std::vector<House> houses(houses_set.begin(), houses_set.end());

    int min_world_x = std::numeric_limits<int>::max();
    int max_world_x = std::numeric_limits<int>::min();
    int min_world_y = std::numeric_limits<int>::max();
    int max_world_y = std::numeric_limits<int>::min();

    for (const auto &h : houses) {
        RectBounds r = get_rect_bounds(h);
        min_world_x = std::min(min_world_x, r.x0);
        max_world_x = std::max(max_world_x, r.x1);
        min_world_y = std::min(min_world_y, r.y0);
        max_world_y = std::max(max_world_y, r.y1);
    }

    if (min_world_x > max_world_x) {
        min_world_x = 0;
        max_world_x = 0;
        min_world_y = 0;
        max_world_y = 0;
    }

    Coord grid_offset(0, 0);
    const int border_padding = path_width + 2;

    grid_offset.x = -min_world_x + border_padding;
    grid_offset.y = -min_world_y + border_padding;

    int world_span_x = max_world_x - min_world_x;
    int world_span_y = max_world_y - min_world_y;

    int grid_width_dim = world_span_x + 2 * border_padding + 1;
    int grid_height_dim = world_span_y + 2 * border_padding + 1;

    if (grid_width_dim <= 0 || grid_height_dim <= 0) {
        return {VillageGrid()};
    }

    BoolGrid house_mask(grid_width_dim, grid_height_dim, false);
    BoolGrid path_mask(grid_width_dim, grid_height_dim, false);

    for (const auto &h : houses) {
        RectBounds r_world = get_rect_bounds(h);
        // world coordinates to grid coordinates
        int grid_r_x0 = r_world.x0 + grid_offset.x;
        int grid_r_x1 = r_world.x1 + grid_offset.x;
        int grid_r_y0 = r_world.y0 + grid_offset.y;
        int grid_r_y1 = r_world.y1 + grid_offset.y;

        for (int y_grid = grid_r_y0; y_grid <= grid_r_y1; ++y_grid) {
            for (int x_grid = grid_r_x0; x_grid <= grid_r_x1; ++x_grid) {
                if (house_mask.is_valid(y_grid, x_grid)) {
                    house_mask(y_grid, x_grid) = true;
                }
            }
        }
    }

    int half_path_width = path_width / 2;
    int padding_for_astar_blocked = 2; // Extra space around houses for A*
    int inflate_radius = half_path_width + padding_for_astar_blocked;

    BoolGrid blocked(grid_width_dim, grid_height_dim, false);

    for (int r_grid = 0; r_grid < grid_height_dim; ++r_grid) {
        for (int c_grid = 0; c_grid < grid_width_dim; ++c_grid) {
            if (house_mask(r_grid, c_grid)) {
                for (int dy = -inflate_radius; dy <= inflate_radius; ++dy) {
                    for (int dx = -inflate_radius; dx <= inflate_radius; ++dx) {
                        int nr_grid = r_grid + dy;
                        int nc_grid = c_grid + dx;
                        if (blocked.is_valid(nr_grid, nc_grid)) {
                            blocked(nr_grid, nc_grid) = true;
                        }
                    }
                }
            }
        }
    }

    // Get door coordinates in both world and grid space
    std::vector<Coord> doors_world_coords;
    std::vector<Coord> doors_grid_coords;
    for (const auto &h : houses) {
        Coord door_world = get_door(h);
        doors_world_coords.push_back(door_world);
        Coord door_grid =
            Coord(door_world.x + grid_offset.x, door_world.y + grid_offset.y);
        doors_grid_coords.push_back(door_grid);
    }

    for (const auto &d_grid : doors_grid_coords) {
        if (blocked.is_valid(d_grid.y, d_grid.x)) {
            for (int dy = -inflate_radius; dy <= inflate_radius; ++dy) {
                for (int dx = -inflate_radius; dx <= inflate_radius; ++dx) {
                    int ny_grid = d_grid.y + dy;
                    int nx_grid = d_grid.x + dx;
                    if (blocked.is_valid(ny_grid, nx_grid)) {
                        blocked(ny_grid, nx_grid) = false;
                    }
                }
            }
        } else {
            std::cerr << "Warning: Door at grid (" << d_grid.x << ","
                      << d_grid.y << "), translated from world ("
                      << d_grid.x - grid_offset.x << ","
                      << d_grid.y - grid_offset.y
                      << "), is outside calculated grid bounds. Max dims: ("
                      << grid_width_dim - 1 << "," << grid_height_dim - 1
                      << ")." << std::endl;
        }
    }

    // Build Minimum Spanning Tree of doors using Prim's algorithm
    std::vector<std::pair<int, int>> mst_edges;
    size_t num_doors = doors_grid_coords.size();

    if (num_doors > 1) {
        std::set<size_t> in_tree_nodes;
        in_tree_nodes.insert(0);

        std::vector<double> min_dist_to_tree(
            num_doors, std::numeric_limits<double>::max());
        std::vector<int> edge_to_tree_node(num_doors, -1);

        auto calculate_distance = [&](size_t door_idx1, size_t door_idx2) {
            long long dx_val =
                static_cast<long long>(doors_world_coords[door_idx1].x) -
                doors_world_coords[door_idx2].x;
            long long dy_val =
                static_cast<long long>(doors_world_coords[door_idx1].y) -
                doors_world_coords[door_idx2].y;
            return std::sqrt(
                static_cast<double>(dx_val * dx_val + dy_val * dy_val));
        };

        for (size_t v_idx = 1; v_idx < num_doors; ++v_idx) {
            min_dist_to_tree[v_idx] = calculate_distance(0, v_idx);
            edge_to_tree_node[v_idx] = 0;
        }

        while (in_tree_nodes.size() < num_doors) {
            double current_min_found_dist = std::numeric_limits<double>::max();
            int next_node_to_add = -1;

            for (size_t v_idx = 0; v_idx < num_doors; ++v_idx) {
                if (in_tree_nodes.count(v_idx))
                    continue;

                if (min_dist_to_tree[v_idx] < current_min_found_dist) {
                    current_min_found_dist = min_dist_to_tree[v_idx];
                    next_node_to_add = static_cast<int>(v_idx);
                }
            }

            if (next_node_to_add == -1)
                break;

            in_tree_nodes.insert(static_cast<size_t>(next_node_to_add));
            if (edge_to_tree_node[next_node_to_add] != -1) { // Add edge to MST
                mst_edges.push_back(
                    {edge_to_tree_node[next_node_to_add], next_node_to_add});
            }

            for (size_t w_idx = 0; w_idx < num_doors; ++w_idx) {
                if (in_tree_nodes.count(w_idx))
                    continue;

                double dist_to_new_node = calculate_distance(
                    static_cast<size_t>(next_node_to_add), w_idx);
                if (dist_to_new_node < min_dist_to_tree[w_idx]) {
                    min_dist_to_tree[w_idx] = dist_to_new_node;
                    edge_to_tree_node[w_idx] = next_node_to_add;
                }
            }
        }
    }

    // Generate paths along MST edges using A*
    for (const auto &edge : mst_edges) {
        const Coord &start_node_grid =
            doors_grid_coords[static_cast<size_t>(edge.first)];
        const Coord &goal_node_grid =
            doors_grid_coords[static_cast<size_t>(edge.second)];

        auto path_opt =
            astar(start_node_grid, goal_node_grid, blocked, max_astar_steps);

        if (path_opt) {
            for (const auto &p_coord_grid : *path_opt) {
                for (int dy = -half_path_width; dy <= half_path_width; ++dy) {
                    for (int dx = -half_path_width; dx <= half_path_width;
                         ++dx) {
                        int x_path_cell = p_coord_grid.x + dx;
                        int y_path_cell = p_coord_grid.y + dy;
                        if (path_mask.is_valid(y_path_cell, x_path_cell)) {
                            if (!house_mask(y_path_cell,
                                            x_path_cell)) { // Don't draw paths
                                                            // over houses
                                path_mask(y_path_cell, x_path_cell) = true;
                            }
                        }
                    }
                }
            }
        }
    }

    VillageGrid final_grid(grid_width_dim, grid_height_dim, 0);
    for (int r_grid = 0; r_grid < grid_height_dim; ++r_grid) {
        for (int c_grid = 0; c_grid < grid_width_dim; ++c_grid) {
            if (house_mask(r_grid, c_grid)) {
                final_grid(r_grid, c_grid) = 1;
            } else if (path_mask(r_grid, c_grid)) {
                final_grid(r_grid, c_grid) = 2;
            }
        }
    }

    return {final_grid};
}
