#include "village_objects.hpp"
#include <algorithm>
#include <vector>

RectBounds get_rect_bounds(const House &h) {
    int x_coords[] = {h.corner1.x, h.corner2.x};
    int y_coords[] = {h.corner1.y, h.corner2.y};
    std::sort(std::begin(x_coords), std::end(x_coords));
    std::sort(std::begin(y_coords), std::end(y_coords));
    return {x_coords[0], x_coords[1], y_coords[0], y_coords[1]};
}

// Check if two houses intersect, considering a margin
// The constants +2 and -3 define a specific kind of margin/gap.
bool intersects(const House &a, const House &b) {
    RectBounds bounds_a = get_rect_bounds(a);
    RectBounds bounds_b = get_rect_bounds(b);

    bool no_x_intersect = (bounds_a.x1 + 2 < bounds_b.x0 - 3) ||
                          (bounds_b.x1 + 2 < bounds_a.x0 - 3);
    bool no_y_intersect = (bounds_a.y1 + 2 < bounds_b.y0 - 3) ||
                          (bounds_b.y1 + 2 < bounds_a.y0 - 3);

    return !(no_x_intersect || no_y_intersect);
}

// Check if a new house is valid (doesn't intersect with existing houses)
bool is_valid(const House &new_house,
              const std::vector<House> &existing_houses) {
    for (const auto &h : existing_houses) {
        if (intersects(new_house, h)) {
            return false;
        }
    }
    return true;
}

// Determine the door's coordinate based on house corners and orientation
Coord get_door(const House &h) {
    RectBounds bounds = get_rect_bounds(h);
    int cx, cy;

    if (!h.orientation) {
        if (h.corner1.x == bounds.x0 || h.corner1.x == bounds.x1) {
            cx = h.corner1.x;
            cy = (bounds.y0 + bounds.y1) / 2;
        } else {
            cy = h.corner1.y;
            cx = (bounds.x0 + bounds.x1) / 2;
        }
    } else {
        if (h.corner2.x == bounds.x0 || h.corner2.x == bounds.x1) {
            cx = h.corner2.x;
            cy = (bounds.y0 + bounds.y1) / 2;
        } else {
            cy = h.corner2.y;
            cx = (bounds.x0 + bounds.x1) / 2;
        }
    }
    return Coord(cx, cy);
}
