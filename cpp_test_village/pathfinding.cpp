#include "pathfinding.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

int heuristic(const Coord &a, const Coord &b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

struct AStarNode {
    int f_score;
    int g_score;
    Coord pos;

    bool operator>(const AStarNode &other) const {
        if (f_score != other.f_score) {
            return f_score > other.f_score;
        }
        if (g_score != other.g_score) {
            return g_score > other.g_score;
        }
        return pos > other.pos;
    }
};

std::optional<std::vector<Coord>> astar(const Coord &start, const Coord &goal,
                                        const BoolGrid &blocked,
                                        int max_steps) {

    if (blocked.empty()) {
        return std::nullopt;
    }
    int grid_height = blocked.height;
    int grid_width = blocked.width;

    if (!blocked.is_valid(start.y, start.x) || blocked(start.y, start.x)) {
        return std::nullopt;
    }
    if (!blocked.is_valid(goal.y, goal.x)) {
        return std::nullopt;
    }

    std::priority_queue<AStarNode, std::vector<AStarNode>,
                        std::greater<AStarNode>>
        open_set;

    Grid1D<int> g_scores(grid_width, grid_height,
                         std::numeric_limits<int>::max());
    Grid1D<Coord> came_from(grid_width, grid_height, Coord{-1, -1});

    g_scores(start.y, start.x) = 0;
    open_set.push({heuristic(start, goal), 0, start});

    int steps = 0;
    static const int dx[] = {1, -1, 0, 0};
    static const int dy[] = {0, 0, 1, -1};

    while (!open_set.empty() && steps < max_steps) {
        AStarNode current_node = open_set.top();
        open_set.pop();
        Coord current_pos = current_node.pos;

        if (current_pos == goal) {
            std::vector<Coord> path;
            Coord path_curr = goal;
            while (path_curr != start) {
                path.push_back(path_curr);
                path_curr = came_from(path_curr.y, path_curr.x);
                if (path_curr.x == -1 && path_curr.y == -1 &&
                    path_curr != start) {
                    return std::nullopt;
                }
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }

        // If a shorter path already found
        if (current_node.g_score > g_scores(current_pos.y, current_pos.x)) {
            continue;
        }

        steps++;

        for (int i = 0; i < 4; ++i) {
            Coord neighbor_pos(current_pos.x + dx[i], current_pos.y + dy[i]);

            if (!blocked.is_valid(neighbor_pos.y, neighbor_pos.x)) {
                continue;
            }
            if (blocked(neighbor_pos.y, neighbor_pos.x)) {
                continue;
            }

            int tentative_g_score = g_scores(current_pos.y, current_pos.x) + 1;

            if (tentative_g_score < g_scores(neighbor_pos.y, neighbor_pos.x)) {
                came_from(neighbor_pos.y, neighbor_pos.x) = current_pos;
                g_scores(neighbor_pos.y, neighbor_pos.x) = tentative_g_score;
                int f_score = tentative_g_score + heuristic(neighbor_pos, goal);
                open_set.push({f_score, tentative_g_score, neighbor_pos});
            }
        }
    }

    return std::nullopt;
}
