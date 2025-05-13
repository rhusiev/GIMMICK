#include "pathfinding.hpp"
#include <queue>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>

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

std::optional<std::vector<Coord>>
astar(const Coord &start, const Coord &goal,
      const std::vector<std::vector<bool>> &blocked, int max_steps) {
    if (blocked.empty() || blocked[0].empty()) {
        return std::nullopt;
    }
    size_t grid_height = blocked.size();
    size_t grid_width = blocked[0].size();

    if (!(start.x >= 0 && start.x < grid_width && start.y >= 0 &&
          start.y < grid_height && !blocked[start.y][start.x])) {
        return std::nullopt;
    }
    if (!(goal.x >= 0 && goal.x < grid_width && goal.y >= 0 &&
          goal.y < grid_height && !blocked[goal.y][goal.x])) {
        if (!(goal.x >= 0 && goal.x < grid_width && goal.y >= 0 &&
              goal.y < grid_height))
            return std::nullopt;
    }

    std::priority_queue<AStarNode, std::vector<AStarNode>,
                        std::greater<AStarNode>>
        open_set;
    std::map<Coord, int> g_scores;
    std::map<Coord, Coord> came_from;

    g_scores[start] = 0;
    open_set.push({heuristic(start, goal), 0, start});

    int steps = 0;
    const int infinity = std::numeric_limits<int>::max();

    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};

    while (!open_set.empty() && steps < max_steps) {
        AStarNode current_node = open_set.top();
        open_set.pop();
        Coord current_pos = current_node.pos;

        if (current_pos == goal) {
            std::vector<Coord> path;
            Coord path_curr = goal;
            while (path_curr != start) {
                path.push_back(path_curr);
                path_curr = came_from[path_curr];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }

        if (current_node.g_score > g_scores.at(current_pos)) {
            continue;
        }

        steps++;

        for (int i = 0; i < 4; ++i) {
            Coord neighbor_pos(current_pos.x + dx[i], current_pos.y + dy[i]);

            if (neighbor_pos.x < 0 || neighbor_pos.x >= grid_width ||
                neighbor_pos.y < 0 || neighbor_pos.y >= grid_height) {
                continue;
            }

            if (blocked[neighbor_pos.y][neighbor_pos.x]) {
                continue;
            }

            int tentative_g_score = g_scores[current_pos] + 1;

            auto it_g = g_scores.find(neighbor_pos);
            int neighbor_g_val =
                (it_g == g_scores.end()) ? infinity : it_g->second;

            if (tentative_g_score < neighbor_g_val) {
                came_from[neighbor_pos] = current_pos;
                g_scores[neighbor_pos] = tentative_g_score;
                int f_score = tentative_g_score + heuristic(neighbor_pos, goal);
                open_set.push({f_score, tentative_g_score, neighbor_pos});
            }
        }
    }

    return std::nullopt; // No path found
}
