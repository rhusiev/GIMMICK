#ifndef VILLAGE_OBJECTS_HPP
#define VILLAGE_OBJECTS_HPP

#include <cmath>
#include <functional>
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
        if (x != other.x) {
            return x < other.x;
        }
        return y < other.y;
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

    House(Coord c1, Coord c2, bool orient)
        : corner1(c1), corner2(c2), orientation(orient) {}

    bool operator==(const House &other) const {
        return corner1 == other.corner1 && corner2 == other.corner2 &&
               orientation == other.orientation;
    }

    bool operator<(const House &other) const {
        if (corner1 < other.corner1)
            return true;
        if (other.corner1 < corner1)
            return false; // handles !(a<b) && !(b<a)
        if (corner2 < other.corner2)
            return true;
        if (other.corner2 < corner2)
            return false;
        return orientation < other.orientation;
    }
};

struct HouseHash {
    std::size_t operator()(const House &h) const {
        std::size_t h1 = CoordHash{}(h.corner1);
        std::size_t h2 = CoordHash{}(h.corner2);
        std::size_t h3 = std::hash<bool>{}(h.orientation);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
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
