#include "village_objects.hpp"
#include <algorithm> // For std::min, std::max
#include <vector>    // For std::vector in is_valid

RectBounds get_rect_bounds(const House &h) {
    return {
        std::min(h.corner1.x, h.corner2.x), std::max(h.corner1.x, h.corner2.x),
        std::min(h.corner1.y, h.corner2.y), std::max(h.corner1.y, h.corner2.y)};
}

bool intersects(const House &a, const House &b) {
    RectBounds bounds_a = get_rect_bounds(a);
    RectBounds bounds_b = get_rect_bounds(b);

    bool no_x_intersect = (bounds_a.x1 + 2 < bounds_b.x0 - 3) ||
                          (bounds_b.x1 + 2 < bounds_a.x0 - 3);
    bool no_y_intersect = (bounds_a.y1 + 2 < bounds_b.y0 - 3) ||
                          (bounds_b.y1 + 2 < bounds_a.y0 - 3);

    return !(no_x_intersect || no_y_intersect);
}

bool is_valid(const House &new_house,
              const std::vector<House> &existing_houses) {
    for (const auto &h : existing_houses) {
        if (intersects(new_house, h)) {
            return false;
        }
    }
    return true;
}

Coord get_door(const House &h) {
    RectBounds bounds = get_rect_bounds(h);
    int cx, cy;

    if (!h.orientation) {
        // If corner1.x is one of the vertical extent of the house (bounds.x0 or
        // bounds.x1) then the wall is vertical, so door is at (corner1.x,
        // midpoint_y)
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
