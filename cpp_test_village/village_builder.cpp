#include "village_builder.hpp"
#include "pathfinding.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

VillageLayout generate_village_layout(const std::set<House> &houses_set,
                                      int path_width, int max_astar_steps) {
    if (houses_set.empty()) {
        return {{}, 0, 0};
    }

    std::vector<House> houses(houses_set.begin(), houses_set.end());

    int current_max_x = 0;
    int current_max_y = 0;
    for (const auto &h : houses) {
        RectBounds r = get_rect_bounds(h);
        current_max_x = std::max({current_max_x, r.x0, r.x1});
        current_max_y = std::max({current_max_y, r.y0, r.y1});
    }

    int grid_width_dim = current_max_x + path_width + 2;
    int grid_height_dim = current_max_y + path_width + 2;

    if (grid_width_dim <= 0 || grid_height_dim <= 0) {
        // all coordinates were <=0 and padding wasn't enough.
        return {{}, 0, 0};
    }

    std::vector<std::vector<bool>> house_mask(
        grid_height_dim, std::vector<bool>(grid_width_dim, false));
    std::vector<std::vector<bool>> path_mask(
        grid_height_dim, std::vector<bool>(grid_width_dim, false));

    for (const auto &h : houses) {
        RectBounds r = get_rect_bounds(h);
        for (int y = r.y0; y <= r.y1; ++y) {
            for (int x = r.x0; x <= r.x1; ++x) {
                if (y >= 0 && y < grid_height_dim && x >= 0 &&
                    x < grid_width_dim) {
                    house_mask[y][x] = true;
                }
            }
        }
    }

    int half_path_width = path_width / 2;
    int padding = 2;
    int inflate = half_path_width + padding;
    std::vector<std::vector<bool>> blocked(
        grid_height_dim, std::vector<bool>(grid_width_dim, false));

    for (int r = 0; r < grid_height_dim; ++r) {
        for (int c = 0; c < grid_width_dim; ++c) {
            if (house_mask[r][c]) {
                for (int dy = -inflate; dy <= inflate; ++dy) {
                    for (int dx = -inflate; dx <= inflate; ++dx) {
                        int nr = r + dy;
                        int nc = c + dx;
                        if (nr >= 0 && nr < grid_height_dim && nc >= 0 &&
                            nc < grid_width_dim) {
                            blocked[nr][nc] = true;
                        }
                    }
                }
            }
        }
    }

    // Gather doors and carve 3x3 (actually (2*inflate+1)x(2*inflate+1)) hole
    // around each otherwise there were some strange behaviours, when a path was
    // stuck inside a house
    std::vector<Coord> doors;
    for (const auto &h : houses) {
        doors.push_back(get_door(h));
    }

    for (const auto &d : doors) {
        for (int dy = -inflate; dy <= inflate; ++dy) {
            for (int dx = -inflate; dx <= inflate; ++dx) {
                int ny = d.y + dy;
                int nx = d.x + dx;
                if (nx >= 0 && nx < grid_width_dim && ny >= 0 &&
                    ny < grid_height_dim) {
                    blocked[ny][nx] = false;
                }
            }
        }
    }

    // Build MST edges (Prim's algorithm)
    int num_doors = doors.size();
    std::vector<std::pair<int, int>> mst_edges;

    if (num_doors > 1) {
        std::set<int> in_tree;
        in_tree.insert(0);

        auto calculate_dist_sq = [&](int i, int j) {
            long long dx_val = doors[i].x - doors[j].x;
            long long dy_val = doors[i].y - doors[j].y;
            return dx_val * dx_val + dy_val * dy_val;
        };

        while (in_tree.size() < num_doors) {
            double min_dist = std::numeric_limits<double>::max();
            int best_u = -1, best_v = -1;

            for (int u : in_tree) {
                for (int v = 0; v < num_doors; ++v) {
                    if (in_tree.count(v))
                        continue;

                    double d_sq = static_cast<double>(calculate_dist_sq(u, v));
                    double d = std::sqrt(d_sq);
                    if (d < min_dist) {
                        min_dist = d;
                        best_u = u;
                        best_v = v;
                    }
                }
            }
            if (best_v != -1) {
                in_tree.insert(best_v);
                mst_edges.push_back({best_u, best_v});
            } else {
                break;
            }
        }
    }

    // A* for each MST edge
    for (const auto &edge : mst_edges) {
        const Coord &start_node = doors[edge.first];
        const Coord &goal_node = doors[edge.second];

        auto path_opt = astar(start_node, goal_node, blocked, max_astar_steps);

        if (path_opt) {
            for (const auto &p_coord : *path_opt) {
                for (int dy = -half_path_width; dy <= half_path_width; ++dy) {
                    for (int dx = -half_path_width; dx <= half_path_width;
                         ++dx) {
                        int x = p_coord.x + dx;
                        int y = p_coord.y + dy;
                        if (x >= 0 && x < grid_width_dim && y >= 0 &&
                            y < grid_height_dim) {
                            if (!house_mask[y][x]) {
                                path_mask[y][x] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    VillageGrid final_grid(grid_height_dim,
                           std::vector<int>(grid_width_dim, 0));
    for (int r = 0; r < grid_height_dim; ++r) {
        for (int c = 0; c < grid_width_dim; ++c) {
            if (house_mask[r][c]) {
                final_grid[r][c] = 1;
            } else if (path_mask[r][c]) {
                final_grid[r][c] = 2;
            }
        }
    }

    return {final_grid, grid_width_dim, grid_height_dim};
}
