#include "village_builder.hpp"
#include "pathfinding.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

VillageLayout generate_village_layout(const std::set<House> &houses_set,
                                      int path_width, int max_astar_steps) {
    if (houses_set.empty()) {
        return {{}, 0, 0};
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

    if (min_world_x == std::numeric_limits<int>::max()) {
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
        return {{}, 0, 0};
    }

    std::vector<std::vector<bool>> house_mask(
        static_cast<size_t>(grid_height_dim),
        std::vector<bool>(static_cast<size_t>(grid_width_dim), false));
    std::vector<std::vector<bool>> path_mask(
        static_cast<size_t>(grid_height_dim),
        std::vector<bool>(static_cast<size_t>(grid_width_dim), false));

    for (const auto &h : houses) {
        RectBounds r_world = get_rect_bounds(h);
        int grid_r_x0 = r_world.x0 + grid_offset.x;
        int grid_r_x1 = r_world.x1 + grid_offset.x;
        int grid_r_y0 = r_world.y0 + grid_offset.y;
        int grid_r_y1 = r_world.y1 + grid_offset.y;

        for (int y_grid = grid_r_y0; y_grid <= grid_r_y1; ++y_grid) {
            for (int x_grid = grid_r_x0; x_grid <= grid_r_x1; ++x_grid) {
                if (y_grid >= 0 && y_grid < grid_height_dim && x_grid >= 0 &&
                    x_grid < grid_width_dim) {
                    house_mask[static_cast<size_t>(y_grid)]
                              [static_cast<size_t>(x_grid)] = true;
                }
            }
        }
    }

    int half_path_width = path_width / 2;
    int padding_for_astar_blocked = 2;
    int inflate_radius = half_path_width + padding_for_astar_blocked;

    std::vector<std::vector<bool>> blocked(
        static_cast<size_t>(grid_height_dim),
        std::vector<bool>(static_cast<size_t>(grid_width_dim), false));

    for (int r_grid = 0; r_grid < grid_height_dim; ++r_grid) {
        for (int c_grid = 0; c_grid < grid_width_dim; ++c_grid) {
            if (house_mask[static_cast<size_t>(r_grid)]
                          [static_cast<size_t>(c_grid)]) {
                for (int dy = -inflate_radius; dy <= inflate_radius; ++dy) {
                    for (int dx = -inflate_radius; dx <= inflate_radius; ++dx) {
                        int nr_grid = r_grid + dy;
                        int nc_grid = c_grid + dx;
                        if (nr_grid >= 0 && nr_grid < grid_height_dim &&
                            nc_grid >= 0 && nc_grid < grid_width_dim) {
                            blocked[static_cast<size_t>(nr_grid)]
                                   [static_cast<size_t>(nc_grid)] = true;
                        }
                    }
                }
            }
        }
    }

    std::vector<Coord> doors_world_coords;
    std::vector<Coord> doors_grid_coords;
    for (const auto &h : houses) {
        Coord door_world = get_door(h);
        doors_world_coords.push_back(door_world);
        doors_grid_coords.push_back(
            Coord(door_world.x + grid_offset.x, door_world.y + grid_offset.y));
    }

    for (const auto &d_grid : doors_grid_coords) {
        if (d_grid.y >= 0 && d_grid.y < grid_height_dim && d_grid.x >= 0 &&
            d_grid.x < grid_width_dim) {
            for (int dy = -inflate_radius; dy <= inflate_radius; ++dy) {
                for (int dx = -inflate_radius; dx <= inflate_radius; ++dx) {
                    int ny_grid = d_grid.y + dy;
                    int nx_grid = d_grid.x + dx;
                    if (nx_grid >= 0 && nx_grid < grid_width_dim &&
                        ny_grid >= 0 && ny_grid < grid_height_dim) {
                        blocked[static_cast<size_t>(ny_grid)]
                               [static_cast<size_t>(nx_grid)] = false;
                    }
                }
            }
        } else {
            std::cerr << "Warning: Door at grid (" << d_grid.x << ","
                      << d_grid.y << ") translated from world ("
                      << d_grid.x - grid_offset.x << ","
                      << d_grid.y - grid_offset.y
                      << ") is outside grid bounds. Max dims: "
                      << grid_width_dim - 1 << "," << grid_height_dim - 1
                      << std::endl;
        }
    }

    // Build Minimum Spanning Tree (MST) of doors using Prim's algorithm
    // Edges connect door indices
    std::vector<std::pair<int, int>> mst_edges;
    size_t num_doors = doors_world_coords.size();

    if (num_doors > 1) {
        std::set<size_t> in_tree;
        in_tree.insert(0);

        auto calculate_dist_sq = [&](size_t i, size_t j) {
            long long dx_val = static_cast<long long>(doors_world_coords[i].x) -
                               doors_world_coords[j].x;
            long long dy_val = static_cast<long long>(doors_world_coords[i].y) -
                               doors_world_coords[j].y;
            return dx_val * dx_val + dy_val * dy_val;
        };

        std::vector<double> min_edge_dist(num_doors,
                                          std::numeric_limits<double>::max());
        std::vector<int> edge_to(num_doors, -1);

        for (size_t v = 1; v < num_doors; ++v) {
            double d_sq = static_cast<double>(calculate_dist_sq(0, v));
            min_edge_dist[v] = std::sqrt(d_sq);
            edge_to[v] = 0;
        }

        while (in_tree.size() < num_doors) {
            double current_min_dist = std::numeric_limits<double>::max();
            int next_v = -1;

            for (size_t v_idx = 0; v_idx < num_doors; ++v_idx) {
                if (in_tree.count(v_idx))
                    continue;
                if (min_edge_dist[v_idx] < current_min_dist) {
                    current_min_dist = min_edge_dist[v_idx];
                    next_v = static_cast<int>(v_idx);
                }
            }

            if (next_v == -1)
                break;

            in_tree.insert(static_cast<size_t>(next_v));
            if (edge_to[next_v] != -1) {
                mst_edges.push_back({edge_to[next_v], next_v});
            }

            for (size_t w_idx = 0; w_idx < num_doors; ++w_idx) {
                if (in_tree.count(w_idx))
                    continue;
                double d_sq = static_cast<double>(
                    calculate_dist_sq(static_cast<size_t>(next_v), w_idx));
                double d = std::sqrt(d_sq);
                if (d < min_edge_dist[w_idx]) {
                    min_edge_dist[w_idx] = d;
                    edge_to[w_idx] = next_v;
                }
            }
        }
    }

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
                        int x_grid = p_coord_grid.x + dx;
                        int y_grid = p_coord_grid.y + dy;
                        if (x_grid >= 0 && x_grid < grid_width_dim &&
                            y_grid >= 0 && y_grid < grid_height_dim) {
                            if (!house_mask[static_cast<size_t>(y_grid)]
                                           [static_cast<size_t>(x_grid)]) {
                                path_mask[static_cast<size_t>(y_grid)]
                                         [static_cast<size_t>(x_grid)] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    VillageGrid final_grid(
        static_cast<size_t>(grid_height_dim),
        std::vector<int>(static_cast<size_t>(grid_width_dim), 0));
    for (int r_grid = 0; r_grid < grid_height_dim; ++r_grid) {
        for (int c_grid = 0; c_grid < grid_width_dim; ++c_grid) {
            if (house_mask[static_cast<size_t>(r_grid)]
                          [static_cast<size_t>(c_grid)]) {
                final_grid[static_cast<size_t>(r_grid)]
                          [static_cast<size_t>(c_grid)] = 1;
            } else if (path_mask[static_cast<size_t>(r_grid)]
                                [static_cast<size_t>(c_grid)]) {
                final_grid[static_cast<size_t>(r_grid)]
                          [static_cast<size_t>(c_grid)] = 2;
            }
        }
    }

    return {final_grid, grid_width_dim, grid_height_dim};
}
