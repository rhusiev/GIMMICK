#ifndef VILLAGE_OBJECTS_HPP
#define VILLAGE_OBJECTS_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

struct Coord {
    int x, y;

    explicit Coord(int x_val = 0, int y_val = 0) : x(x_val), y(y_val) {}

    Coord operator+(const Coord &other) const {
        return Coord(x + other.x, y + other.y);
    }

    Coord &operator+=(const Coord &other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    bool operator==(const Coord &other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Coord &other) const { return !(*this == other); }

    bool operator<(const Coord &other) const {
        return std::tie(x, y) < std::tie(other.x, other.y);
    }

    bool operator>(const Coord &other) const { return other < (*this); }
};

struct CoordHash {
    std::size_t operator()(const Coord &c) const {
        std::size_t h1 = std::hash<int>{}(c.x);
        std::size_t h2 = std::hash<int>{}(c.y);
        return h1 ^ (h2 << 1);
    }
};

struct House {
    Coord corner1;
    Coord corner2;
    bool orientation;
    int terrain_height;

    House(Coord c1, Coord c2, bool orient, int height = 0)
        : corner1(c1), corner2(c2), orientation(orient),
          terrain_height(height) {}

    bool operator==(const House &other) const {
        return std::tie(corner1, corner2, orientation, terrain_height) ==
               std::tie(other.corner1, other.corner2, other.orientation,
                        other.terrain_height);
    }

    bool operator<(const House &other) const {
        return std::tie(corner1, corner2, orientation, terrain_height) <
               std::tie(other.corner1, other.corner2, other.orientation,
                        other.terrain_height);
    }
};

struct HouseHash {
    std::size_t operator()(const House &h) const {
        std::size_t h1 = CoordHash{}(h.corner1);
        std::size_t h2 = CoordHash{}(h.corner2);
        std::size_t h3 = std::hash<bool>{}(h.orientation);
        std::size_t h4 = std::hash<int>{}(h.terrain_height);
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct RectBounds {
    int x0, x1, y0, y1;
};

RectBounds get_rect_bounds(const House &h);
bool intersects(const House &a, const House &b);
bool is_valid(const House &new_house,
              const std::vector<House> &existing_houses);
Coord get_door(const House &h);

#endif // VILLAGE_OBJECTS_HPP
